"""合成 SCADA 数据生成器：在拿到真实 Kelmarsh 数据之前，用它把整条流程跑通。

    python -m datagen.synth            # 生成到 data/synth/

模仿 Kelmarsh 的结构：6 台 MM92（额定 2.05 MW）、10 分钟分辨率、2019–2023 五年。
生成三样东西：
  scada.parquet  宽表。每台机 8 个量（命名模仿 Kelmarsh 的常用字段）：
                 p_ 有功功率(MW)  w_ 风速计1(m/s)  w2_ 风速计2(m/s)  ws_ 风速标准差(m/s)
                 wx_ 最大风速  wn_ 最小风速  pt_ 桨距角(度)  rpm_ 转子转速
  events.csv     运行事件日志（限功率、停机），工具可以查询；约 8% 的事件是事后补录的
  truth.csv      只有出题人能看的“真值”：未记录的欠发、自然掉线、自然冻结。工具不能读它

注意：这是合成数据，只用来调通代码，任何数值结论都不能写进报告。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "synth"

TURBINES = [f"T0{i}" for i in range(1, 7)]
# 假设 6 台机排成一条线，相邻的互为邻机；两端各取两台最近的
NEIGHBORS = {
    "T01": ["T02", "T03"], "T02": ["T01", "T03"], "T03": ["T02", "T04"],
    "T04": ["T03", "T05"], "T05": ["T04", "T06"], "T06": ["T05", "T04"],
}
RATED_MW = 2.05
FIELDS = ["p", "w", "w2", "ws", "wx", "wn", "pt", "rpm"]
# 近似 MM92 功率曲线（风速 m/s → 功率 kW），只用于生成数据；工具使用的是从训练段拟合的经验曲线
CURVE_WS = [0, 2.99, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12.5, 24, 24.01, 40]
CURVE_KW = [0, 0, 0, 60, 175, 330, 540, 820, 1150, 1500, 1800, 1990, 2050, 2050, 0, 0]

START, END = "2019-01-01 00:00", "2023-12-31 23:50"
STEP = pd.Timedelta(minutes=10)


def true_power_curve(ws):
    return np.interp(ws, CURVE_WS, CURVE_KW) / 1000.0


def _ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(n) * np.sqrt(1 - phi**2)
    x = lfilter([1.0], [1.0, -phi], e)
    return x / x.std()


def _farm_wind(index: pd.DatetimeIndex, rng) -> np.ndarray:
    n = len(index)
    z = 0.88 * _ar1(n, 0.997, rng) + 0.42 * _ar1(n, 0.95, rng)   # 天气尺度 + 小时尺度
    z /= z.std()
    u = np.clip(0.5 * (1 + erf(z / np.sqrt(2))), 1e-6, 1 - 1e-6)
    doy = index.dayofyear.to_numpy()
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    scale = 7.6 + 1.8 * np.cos(2 * np.pi * (doy - 15) / 365.25) + 0.4 * np.cos(2 * np.pi * (hour - 15) / 24)
    return np.clip(scale * (-np.log(1 - u)) ** (1 / 2.1), 0, 30)   # Weibull 边缘分布，k=2.1


def _segments(rng, n_steps: int, count: int, dur_steps: tuple[int, int]):
    for _ in range(count):
        s = int(rng.integers(0, n_steps - dur_steps[1] - 1))
        yield s, s + int(rng.integers(dur_steps[0], dur_steps[1] + 1))


def generate(seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range(START, END, freq=STEP)
    n = len(index)
    years = index.year.nunique()
    farm = _farm_wind(index, rng)
    rng2 = np.random.default_rng(seed + 1)    # 新增字段与风况事件用独立随机流，不改变原有量的随机序列
    ramps = []
    for st, en in _segments(rng2, n, 30 * years, (6, 30)):   # 全场风速突变（突降/突升）后维持一段
        f = float(rng2.choice([0.5, 0.6, 0.7, 1.3, 1.45]))
        prof = np.ones(en - st)
        k = int(rng2.integers(1, 4))
        prof[:k] = np.linspace(1, f, k + 1)[1:]
        prof[k:] = f
        prof[-3:] = np.linspace(f, 1, 4)[1:]
        farm[st:en] = farm[st:en] * prof
        ramps.append((st, en, f))

    frame = {}
    events, truth = [], []
    scale = [1.00, 0.98, 1.02, 0.97, 1.01, 0.99]
    lag = [0, 1, 2, 0, 1, 2]
    ev_id = tr_id = 0

    for i, t in enumerate(TURBINES):
        u = np.roll(farm, lag[i]) * scale[i] * (1 + 0.03 * _ar1(n, 0.85, rng))   # 机组各自的湍流
        u = np.clip(u, 0, None)
        # 单机阵风：只有本机风速短时抬高（邻机不同步），湍流强度同时变大
        gust = np.zeros(n)
        for st, en in _segments(rng2, n, 15 * years, (2, 4)):
            u[st:en] *= 1 + float(rng2.uniform(0.3, 0.5)) * np.hanning(en - st + 2)[1:-1]
            gust[st:en] = 1
            tr_id += 1
            truth.append(dict(id=f"G{tr_id:04d}", turbine=t, type="gust", start=index[st], end=index[en]))
        power = true_power_curve(u) * (1 + 0.015 * _ar1(n, 0.8, rng)) + 0.005 * rng.standard_normal(n)
        du = np.abs(np.diff(u, prepend=u[0]))
        wstd = np.clip(0.25 + 0.08 * u + 0.9 * du + 0.5 * gust + 0.08 * rng2.standard_normal(n), 0.05, None)
        pitch = np.where(u > 12, np.clip((u - 12) * 2.2, 0, 25), 0.0) + 0.15 * rng2.standard_normal(n)
        rpm = np.interp(u, [0, 3, 3.5, 12, 30], [0.5, 4.0, 9.0, 15.0, 15.0]) + 0.08 * rng2.standard_normal(n)
        idle = u < 3
        power[idle] = -0.006 + 0.002 * rng.standard_normal(idle.sum())   # 待机时的厂用电
        power = np.minimum(power, RATED_MW * (1.004 + 0.003 * rng.standard_normal(n)))

        # 限功率事件（有日志）
        for s, e in _segments(rng, n, 15 * years, (6, 36)):
            sp = round(float(rng.uniform(0.6, 1.4)), 1)
            raw = power[s:e].copy()
            power[s:e] = np.minimum(power[s:e], sp * (1 + 0.004 * rng.standard_normal(e - s)))
            binding = raw > sp * 1.02                      # 限功率真正压住功率时，变桨会把叶片调开
            pitch[s:e] += np.where(binding, 3 + 12 * (1 - power[s:e] / np.maximum(raw, 1e-6)), 0)
            late = rng.random() < 0.08
            avail = index[e] + pd.Timedelta(hours=int(rng.integers(1, 48))) if late \
                else index[s] + STEP * int(rng.integers(1, 4))
            ev_id += 1
            events.append(dict(id=f"E{ev_id:04d}", turbine=t, type="curtailment", start=index[s], end=index[e],
                               available_at=avail, setpoint_mw=sp, detail=f"限功率运行，设定值 {sp} MW"))
        # 停机事件（有日志）
        for s, e in _segments(rng, n, 5 * years, (12, 144)):
            power[s:e] = -0.008 + 0.002 * rng.standard_normal(e - s)
            rpm[s:e] = np.abs(0.15 * rng2.standard_normal(e - s))      # 停机：转子基本不转，叶片顺桨
            pitch[s:e] = 88 + 0.5 * rng2.standard_normal(e - s)
            ev_id += 1
            events.append(dict(id=f"E{ev_id:04d}", turbine=t, type="stop", start=index[s], end=index[e],
                               available_at=index[s] + STEP, setpoint_mw=np.nan, detail="机组停机"))
        # 未记录的欠发（如偏航对风偏差），只在真值表里
        for s, e in _segments(rng, n, 4 * years, (18, 72)):
            seg = power[s:e]
            power[s:e] = np.where(seg > 0, seg * 0.65, seg)
            tr_id += 1
            truth.append(dict(id=f"G{tr_id:04d}", turbine=t, type="hidden_derate", start=index[s], end=index[e]))

        wind = np.round(u + 0.06 * rng.standard_normal(n), 2)
        wind = np.clip(wind, 0, None)
        wind2 = np.clip(np.round(u + 0.08 * rng2.standard_normal(n), 2), 0, None)    # 第二个风速计，独立噪声
        wmax = np.round(wind + 1.6 * wstd, 2)
        wmin = np.round(np.clip(wind - 1.6 * wstd, 0, None), 2)
        wstd = np.round(wstd, 2)
        power = np.round(power, 4)

        # 自然数据问题：掉线、风速冻结、零星缺失
        for s, e in _segments(rng, n, 8 * years, (3, 24)):
            for arr in (power, wind, wind2, wstd, wmax, wmin, pitch, rpm):
                arr[s:e] = np.nan
            tr_id += 1
            truth.append(dict(id=f"G{tr_id:04d}", turbine=t, type="dropout", start=index[s], end=index[e]))
        for s, e in _segments(rng, n, 2 * years, (6, 18)):
            wind[s:e] = wind[s]
            wstd[s:e] = 0.0                      # 风速计 1 卡住：读数、标准差、最大最小都不再变化，风速计 2 仍正常
            wmax[s:e] = wn_v = wind[s]
            wmin[s:e] = wn_v
            tr_id += 1
            truth.append(dict(id=f"G{tr_id:04d}", turbine=t, type="wind_freeze", start=index[s], end=index[e]))
        power[rng.random(n) < 0.003] = np.nan
        wind[rng.random(n) < 0.003] = np.nan

        frame[f"p_{t}"] = power
        frame[f"w_{t}"] = wind
        frame[f"w2_{t}"] = wind2
        frame[f"ws_{t}"] = wstd
        frame[f"wx_{t}"] = wmax
        frame[f"wn_{t}"] = wmin
        frame[f"pt_{t}"] = np.round(pitch, 2)
        frame[f"rpm_{t}"] = np.round(rpm, 2)

    df = pd.DataFrame(frame, index=index)
    df.index.name = "time"
    cols = [f"{k}_{t}" for k in FIELDS for t in TURBINES]
    for st, en, f in ramps:
        truth.append(dict(id=f"R{len(truth) + 1:04d}", turbine="ALL", type="wind_ramp", start=index[st], end=index[en]))
    return df[cols], pd.DataFrame(events), pd.DataFrame(truth)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, events, truth = generate()
    df.to_parquet(OUT_DIR / "scada.parquet")
    events.to_csv(OUT_DIR / "events.csv", index=False)
    truth.to_csv(OUT_DIR / "truth.csv", index=False)
    meta = {"turbines": TURBINES, "neighbors": NEIGHBORS, "rated_mw": RATED_MW,
            "resolution_minutes": 10, "source": "synthetic (datagen/synth.py)"}
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"生成 {len(df)} 个时刻 × {len(TURBINES)} 台机组；事件 {len(events)} 条；真值记录 {len(truth)} 条 → {OUT_DIR}")


if __name__ == "__main__":
    main()
