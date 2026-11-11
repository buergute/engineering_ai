"""在 val 段上确定“明显欠发”的判定阈值（负责人：B 统计，C 使用）

    python -m eval.thresholds

另外估计复核用的四个阈值（功率突变（30 分钟净变化）、两个风速计不一致、湍流偏大、桨距角偏大），写入同一文件的 "review" 项：
都取正常时刻统计量的 99.5%（湍流 99%）分位数，即正常时刻被判可疑的比例约 0.5%–1%。
做法：在 val 段随机抽取正常运行时刻（前后 24 小时没有任何事件或真值异常），
分别用本机风速、两台邻机推算、只剩一台邻机推算三种方式计算功率曲线残差；阈值取正常时刻残差的 1% 分位数（即误报率约 1%），
再看同一阈值在 val 段隐性欠发时段上的检出率。结果写入 artifacts/thresholds.json，状态机读取。
val 段允许看真值表（这是调参用的段）；test 段的真值只用于出题，不参与这里的任何计算。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from agent.toolbox import real_toolbox
from agent_eval.pick import Picker
from common.config import load_split
from common.store import DataStore
from tools.check_power_curve import check_power_curve
from tools.reference import ART_DIR

FALSE_ALARM = 0.01


def residuals(store, tools, turbine, t):
    d = tools.read_asof(turbine, t.isoformat(timespec="minutes"))
    out = {}
    for key, src, excl in (("own", "own", ()), ("neighbors", "neighbors", ()),
                           ("neighbors_1", "neighbors", (store.neighbors(turbine)[0],))):
        c = check_power_curve(d, turbine, wind_source=src, exclude_neighbors=excl)
        out[key] = (c.result["residual"], c.result["expected_power"]) if c.status == "ok" else (np.nan, np.nan)
    return out


def main(n_clean: int = 3000) -> None:
    store = DataStore.load(with_truth=True)
    tools = real_toolbox(store)
    split = load_split()["splits"]["val"]
    pk = Picker(store)
    rng = np.random.default_rng(0)
    times = pd.date_range(pd.Timestamp(split["start"]) + pd.Timedelta(days=2),
                          pd.Timestamp(split["end"]) - pd.Timedelta(days=1), freq="10min")

    clean = {"own": [], "neighbors": [], "neighbors_1": []}
    rv = {"jump": [], "wdiff": [], "ti": [], "pitch": [], "ws": [], "wm": []}
    F = store.frame
    tries = 0
    while len(clean["own"]) < n_clean and tries < 50 * n_clean:
        tries += 1
        t, tb = times[rng.integers(len(times))], store.turbines[rng.integers(6)]
        i = pk._i(t)
        group = [tb, *store.neighbors(tb)]
        if not pk._complete(group, i, 48, 0) or not pk._quiet(group, i, 144, 0):
            continue
        if np.mean(pk.W[tb][i - 2:i + 1]) < 4:          # 低风速时残差没有意义
            continue
        r = residuals(store, tools, tb, t)
        for k in clean:
            clean[k].append(r[k])
        w, ws = F[f"w_{tb}"].to_numpy()[i - 2:i + 1], F[f"ws_{tb}"].to_numpy()[i - 2:i + 1]
        x = pk.P[tb][i - 6:i + 1]
        rv["jump"].append(float(np.nanmax(np.abs(x[3:] - x[:-3]))))
        rv["wdiff"].append(float(np.nanmean(np.abs(w - F[f"w2_{tb}"].to_numpy()[i - 2:i + 1]))))
        rv["ti"].append(float(np.nanmean(ws / w)))
        rv["ws"].append(float(np.nanmean(ws)))
        rv["wm"].append(float(np.nanmean(w)))
        if np.mean(w) < 10.5:
            rv["pitch"].append(float(np.nanmean(F[f"pt_{tb}"].to_numpy()[i - 2:i + 1])))

    derate = {"own": [], "neighbors": [], "neighbors_1": []}
    tr = store.truth[(store.truth.type == "hidden_derate") & (store.truth.start >= split["start"])
                     & (store.truth.end <= split["end"])]
    for e in tr.itertuples(index=False):
        t = (e.start + pd.Timedelta(minutes=60)).ceil("10min")
        if t > e.end:
            continue
        i = pk._i(t)
        if np.mean(pk.W[e.turbine][i - 2:i + 1]) < 6 or not pk._complete([e.turbine, *store.neighbors(e.turbine)], i, 6, 0):
            continue
        r = residuals(store, tools, e.turbine, t)
        for k in derate:
            derate[k].append(r[k])

    result = {}
    for src in clean:
        c = np.array(clean[src], dtype=float)
        c = c[~np.isnan(c[:, 0])]
        thr = float(np.quantile(c[:, 0], FALSE_ALARM))
        d = np.array(derate[src], dtype=float)
        d = d[~np.isnan(d[:, 0])]
        result[src] = {"residual_alert_mw": round(thr, 3), "n_clean": len(c), "clean_p50": round(float(np.median(c[:, 0])), 3),
                       "clean_p01": round(thr, 3), "clean_p99": round(float(np.quantile(c[:, 0], 0.99)), 3),
                       "n_derate": len(d), "derate_detect_rate": round(float((d[:, 0] < thr).mean()), 3) if len(d) else None}
    q = lambda k, p_: round(float(np.nanquantile(rv[k], p_)), 3)   # noqa: E731
    result["review"] = {
        "power_jump_mw": q("jump", 0.995), "anemometer_diff_ms": q("wdiff", 0.995), "turbulence_ti": q("ti", 0.99),
        "pitch_deg": round(max(q("pitch", 0.995), 2.0) + 0.5, 3), "rpm_stopped": 1.0, "sync_ratio": 0.4,
        "n_clean": len(rv["jump"])}
    # 湍流：风速标准差的 99% 分位随风速变化（低风速时相对值大），按风速分箱，复核时按本机风速插值
    wm, wsd = np.array(rv["wm"]), np.array(rv["ws"])
    edges = [0, 4, 6, 8, 10, 12, 30]
    cen, q99 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (wm >= lo) & (wm < hi) & ~np.isnan(wsd)
        if m.sum() >= 30:
            cen.append(round(float(wm[m].mean()), 2))
            q99.append(round(float(np.quantile(wsd[m], 0.99)), 3))
    result["review"]["turbulence_curve"] = {"wind": cen, "std_q99": q99}
    result["review"]["anemometer_diff_ms"] = round(1.5 * result["review"]["anemometer_diff_ms"], 3)   # 邻机也要查，6 台机组共同作用，取更严的阈值
    print("复核阈值:", result["review"])
    (ART_DIR / "thresholds.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    for src, r in ((k, v) for k, v in result.items() if k != "review"):
        print(f"{src:11s} 阈值 {r['residual_alert_mw']:+.3f} MW（正常时刻 n={r['n_clean']}，中位数 {r['clean_p50']:+.3f}，"
              f"99% 分位 {r['clean_p99']:+.3f}）；隐性欠发检出率 {r['derate_detect_rate']}（n={r['n_derate']}）")


if __name__ == "__main__":
    main()
