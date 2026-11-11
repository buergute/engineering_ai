"""扰动注入（负责人：A）

所有函数只改模型输入（宽表里的 p_/w_ 列或事件日志），从不改预测目标。
评测时，目标值始终取自未扰动的原始数据。
两种用法：
  1. 按测试题注入：apply_task_injections()，扰动位置相对于起报时刻 as_of
  2. 训练增强：random_augment()，在整段数据上随机撒扰动（刻意不含缓慢偏移，留作“没见过的扰动”）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEP = pd.Timedelta(minutes=10)


# ───────── 基本扰动（区间均为左开右闭 (start, end]）─────────

def _mask(df: pd.DataFrame, start, end) -> np.ndarray:
    return (df.index > pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))


def random_missing(df: pd.DataFrame, cols: list[str], rate: float, start, end, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = _mask(df, start, end)
    for c in cols:
        hit = m & (rng.random(len(df)) < rate)
        df.loc[hit, c] = np.nan
    return df


def contiguous_dropout(df: pd.DataFrame, cols: list[str], start, end) -> pd.DataFrame:
    df.loc[_mask(df, start, end), cols] = np.nan
    return df


def freeze(df: pd.DataFrame, col: str, start, end) -> pd.DataFrame:
    """读数冻结：区间内保持为区间开始时刻的值。"""
    df.loc[_mask(df, start, end), col] = df.at[pd.Timestamp(start), col]
    return df


WIND_BLOCK = ["w", "w2", "ws", "wx", "wn"]      # 风速计相关字段
ALL_FIELDS = ["p", "w", "w2", "ws", "wx", "wn", "pt", "rpm"]


def freeze_wind_sensor1(df: pd.DataFrame, tb: str, start, end) -> pd.DataFrame:
    """风速计 1 卡住：读数、标准差（=0）、最大最小值都停住；风速计 2 不受影响。"""
    m = _mask(df, start, end)
    v = df.at[pd.Timestamp(start), f"w_{tb}"]
    df.loc[m, f"w_{tb}"] = v
    df.loc[m, f"ws_{tb}"] = 0.0
    df.loc[m, f"wx_{tb}"] = v
    df.loc[m, f"wn_{tb}"] = v
    return df


def anemometer_mismatch(df: pd.DataFrame, tb: str, factor: float, start, end) -> pd.DataFrame:
    """风速计 2 系统性偏高/偏低 factor 倍（例如 0.15 表示高 15%），风速计 1 不变。"""
    m = _mask(df, start, end)
    df.loc[m, f"w2_{tb}"] = df.loc[m, f"w2_{tb}"] * (1 + factor)
    return df


def slow_drift(df: pd.DataFrame, col: str, drift_per_day: float, start, end) -> pd.DataFrame:
    m = _mask(df, start, end)
    days = (df.index[m] - pd.Timestamp(start)) / pd.Timedelta(days=1)
    df.loc[m, col] = df.loc[m, col] + drift_per_day * np.asarray(days)
    return df


def add_event(events: pd.DataFrame, **row) -> pd.DataFrame:
    return pd.concat([events, pd.DataFrame([row])], ignore_index=True)


# ───────── 按测试题注入 ─────────

def _resolve(name: str, turbine: str, neighbors: list[str]) -> str:
    if name == "$turbine":
        return turbine
    if name.startswith("$neighbor"):
        return neighbors[int(name[len("$neighbor"):])]
    return name


def apply_task_injections(store, task_injects: list[dict], turbine: str, as_of: str, seed: int = 0) -> list[str]:
    """在 store（已复制）上执行测试题的 inject 列表，返回注入事件的 id（评测时用来检查有没有被误用）。"""
    t = pd.Timestamp(as_of)
    df = store.frame
    nbrs = store.neighbors(turbine)
    injected_events = []
    for k, spec in enumerate(task_injects or []):
        kind = spec["type"]
        tb = _resolve(spec.get("turbine", "$turbine"), turbine, nbrs)
        if kind == "freeze":
            t0 = t - pd.Timedelta(minutes=spec["duration_min"])
            if spec["field"] == "wind_speed":
                freeze_wind_sensor1(df, tb, t0, t)
            else:
                freeze(df, f"p_{tb}", t0, t)
        elif kind == "contiguous_dropout":
            if spec["fields"] == "all":
                fields = ALL_FIELDS
            else:
                fields = [x for f in spec["fields"] for x in (WIND_BLOCK if f == "wind_speed" else ["p"])]
            contiguous_dropout(df, [f"{f}_{tb}" for f in fields], t - pd.Timedelta(minutes=spec["duration_min"]), t)
        elif kind == "random_missing":
            cols = [f"{x}_{tb}" for f in spec["fields"] for x in (["w", "w2"] if f == "wind_speed" else ["p"])]
            random_missing(df, cols, spec["rate"], t - pd.Timedelta(hours=spec["window_hours"]), t, seed + k)
        elif kind == "slow_drift":
            col = f"w_{tb}" if spec["field"] == "wind_speed" else f"p_{tb}"
            slow_drift(df, col, spec["drift_per_day"], t - pd.Timedelta(days=spec["days"]), t)
            if spec["field"] == "wind_speed":      # 风速计 1 漂移：最大/最小风速同步偏移，风速计 2 不动
                for c in (f"wx_{tb}", f"wn_{tb}"):
                    slow_drift(df, c, spec["drift_per_day"], t - pd.Timedelta(days=spec["days"]), t)
        elif kind == "hide_active_event":
            # 把此刻正在发生的某类事件改成“事后才补录”：起报时刻还看不到它，只能靠转速、桨距角等字段判断
            ev = store.events
            hit = (ev.turbine == tb) & (ev.type == spec["event_type"]) & (ev.start <= t) & (ev.end >= t)
            if not hit.any():
                raise ValueError(f"{tb} 在 {as_of} 没有正在进行的 {spec['event_type']} 事件可隐藏")
            store.events.loc[hit, "available_at"] = t + pd.Timedelta(minutes=spec.get("available_after_as_of_min", 80))
        elif kind == "anemometer_mismatch":
            anemometer_mismatch(df, tb, spec["factor"], t - pd.Timedelta(minutes=spec["duration_min"]), t)
        elif kind in ("add_event", "late_event"):
            eid = f"X{k + 1:03d}"
            start = t - pd.Timedelta(minutes=spec["start_before_as_of_min"])
            avail = (start + pd.Timedelta(minutes=1) if kind == "add_event"
                     else t + pd.Timedelta(minutes=spec["available_after_as_of_min"]))
            store.events = add_event(
                store.events, id=eid, turbine=tb, type=spec["event_type"], start=start,
                end=t + pd.Timedelta(hours=3), available_at=avail, setpoint_mw=spec.get("setpoint_mw"),
                detail=f"限功率运行，设定值 {spec.get('setpoint_mw')} MW")
            injected_events.append(eid)
        else:
            raise ValueError(f"未知扰动类型 {kind}")
    return injected_events


# ───────── 训练增强 ─────────

def random_augment(df: pd.DataFrame, turbines: list[str], seed: int, per_turbine: int = 3000) -> pd.DataFrame:
    """在整段数据上随机撒三类扰动：风速冻结、功率冻结、连续掉线，外加零星缺失。
    刻意不加 slow_drift：评测时把它当作“训练没见过的扰动”。"""
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    for tb in turbines:
        p = out[f"p_{tb}"].to_numpy(copy=True)
        w = out[f"w_{tb}"].to_numpy(copy=True)
        for _ in range(per_turbine):
            s = int(rng.integers(0, n - 40))
            e = s + int(rng.integers(3, 25))
            kind = rng.choice(3, p=[0.45, 0.2, 0.35])
            if kind == 0:
                w[s + 1:e + 1] = w[s]
            elif kind == 1:
                p[s + 1:e + 1] = p[s]
            else:
                p[s + 1:e + 1] = np.nan
                w[s + 1:e + 1] = np.nan
        p[rng.random(n) < 0.02] = np.nan
        w[rng.random(n) < 0.02] = np.nan
        out[f"p_{tb}"] = p
        out[f"w_{tb}"] = w
    return out
