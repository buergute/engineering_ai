"""特征工程（负责人：B）。训练和推理共用 build_features，保证两边的特征完全一致。

输入是宽表（p_<机组>, w_<机组>），只用当前及以前的数据（全部是 shift 和向后看的 rolling）。
三组特征：
  历史量    本机功率/风速滞后、1 小时均值；邻机均值
  质量量    6 小时缺失数、末尾连续缺失长度、末尾连续相同值长度（冻结）——“质量感知”指的就是这一组
  时间量    小时、年内日的正余弦
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_STEPS = {10: 1, 30: 3, 60: 6}
OWN_WIND_COLS = ["w_lag0", "w_lag1", "w_lag2", "w_lag3", "w_mean_1h", "w_nan_6h", "w_gap_tail", "w_frozen_run"]
QUALITY_COLS = ["p_nan_6h", "w_nan_6h", "p_gap_tail", "w_gap_tail", "p_frozen_run", "w_frozen_run", "nb_valid"]


def _tail_run(flag: pd.Series) -> pd.Series:
    """flag 连续为 True 的长度（截至当前时刻）。"""
    flag = flag.astype(bool)
    return flag.groupby((~flag).cumsum()).cumsum().astype(float)


def _frozen_run(x: pd.Series) -> pd.Series:
    same = (x == x.shift(1)) & x.notna()
    run = _tail_run(same) + 1
    return run.where(x.notna(), 0.0)


def build_features(frame: pd.DataFrame, turbine: str, neighbors: list[str], turbine_code: int,
                   kind: str = "quality_aware") -> pd.DataFrame:
    """kind: quality_aware（完整）、fallback（不用本机风速）、imputed（先前向补值，不含质量特征）。"""
    if kind == "imputed":
        frame = frame.ffill()
    p, w = frame[f"p_{turbine}"], frame[f"w_{turbine}"]
    nb_p = frame[[f"p_{n}" for n in neighbors]].mean(axis=1)
    nb_w = frame[[f"w_{n}" for n in neighbors]].mean(axis=1)

    f = {}
    for k in (0, 1, 2, 3, 6):
        f[f"p_lag{k}"] = p.shift(k)
    for k in (0, 1, 2, 3):
        f[f"w_lag{k}"] = w.shift(k)
    f["p_mean_1h"] = p.rolling(6, min_periods=1).mean()
    f["w_mean_1h"] = w.rolling(6, min_periods=1).mean()
    f["nb_p_lag0"] = nb_p
    f["nb_p_lag1"] = nb_p.shift(1)
    f["nb_p_mean_1h"] = nb_p.rolling(6, min_periods=1).mean()
    f["nb_w_lag0"] = nb_w
    f["nb_w_mean_1h"] = nb_w.rolling(6, min_periods=1).mean()
    f["nb_w_diff_1h"] = nb_w - nb_w.shift(6)

    f["p_nan_6h"] = p.isna().rolling(36, min_periods=1).sum()
    f["w_nan_6h"] = w.isna().rolling(36, min_periods=1).sum()
    f["p_gap_tail"] = _tail_run(p.isna())
    f["w_gap_tail"] = _tail_run(w.isna())
    p_frozen = _frozen_run(p)
    f["p_frozen_run"] = p_frozen.where(p.abs() > 0.05, 0.0)   # 停机时功率本来就是平的，不算冻结
    f["w_frozen_run"] = _frozen_run(w)
    f["nb_valid"] = frame[[f"w_{n}" for n in neighbors]].notna().sum(axis=1).astype(float)

    idx = frame.index
    hour = idx.hour + idx.minute / 60
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * idx.dayofyear / 365.25), np.cos(2 * np.pi * idx.dayofyear / 365.25)
    f["turbine"] = float(turbine_code)

    X = pd.DataFrame(f, index=idx)
    if kind == "fallback":
        X = X.drop(columns=OWN_WIND_COLS)
    elif kind == "imputed":
        X = X.drop(columns=[c for c in QUALITY_COLS if c in X])
    return X


def is_degraded(X: pd.DataFrame) -> pd.Series:
    """区间校准的分组依据：近期有缺失或冻结的样本单独校准（数据越差，区间越宽）。"""
    cond = (X["p_nan_6h"] > 0) | (X["p_frozen_run"] >= 6)
    if "w_nan_6h" in X:
        cond |= (X["w_nan_6h"] > 0) | (X["w_frozen_run"] >= 6)
    cond |= X["nb_valid"] < 2
    return cond


def targets(frame: pd.DataFrame, turbine: str) -> dict[int, pd.Series]:
    p = frame[f"p_{turbine}"]
    return {h: p.shift(-s) for h, s in HORIZON_STEPS.items()}
def features_at(frame: pd.DataFrame, as_of, turbine: str, *, fallback: bool = False,
                quality_features: bool = True, window_steps: int = 36) -> pd.Series:
    """Build features from observations available at or before as_of.

    The fallback excludes every own-wind-derived feature, including missingness.
    """
    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.has_duplicates:
        raise ValueError("frame must have a unique DatetimeIndex")
    origin = pd.Timestamp(as_of)
    history = frame.loc[frame.index <= origin].sort_index().tail(window_steps)
    features = {}
    own_wind = f"w_{turbine}"
    columns = [f"p_{turbine}"]
    if not fallback:
        columns.append(own_wind)
    columns += sorted(c for c in frame.columns if c.startswith(("p_", "w_"))
                      and c not in (f"p_{turbine}", own_wind))
    for col in columns:
        values = pd.to_numeric(history[col], errors="coerce") if col in history else pd.Series(dtype=float)
        valid = values.dropna()
        features[f"{col}_last"] = float(valid.iloc[-1]) if len(valid) else np.nan
        features[f"{col}_mean"] = float(valid.mean()) if len(valid) else np.nan
        if quality_features:
            features[f"{col}_missing_fraction"] = float(values.isna().mean()) if len(values) else 1.0
            features[f"{col}_age_steps"] = float(len(values) - 1 - np.flatnonzero(values.notna())[-1]) if len(valid) else np.nan
    return pd.Series(features, dtype=float)


def make_examples(raw: pd.DataFrame, inputs: pd.DataFrame, turbine: str, origins,
                  horizon_minutes: int, split_start, split_end, **feature_options):
    """Construct aligned examples with targets exclusively from raw, within a split.

    Origins and target timestamps must both lie in the same inclusive split.
    Inputs may be perturbed independently; raw must be the unperturbed source.
    """
    if horizon_minutes <= 0:
        raise ValueError("horizon_minutes must be positive")
    if not isinstance(raw.index, pd.DatetimeIndex) or raw.index.has_duplicates:
        raise ValueError("raw must have a unique DatetimeIndex")
    start, end = pd.Timestamp(split_start), pd.Timestamp(split_end)
    if start > end:
        raise ValueError("invalid split")
    target_col = f"p_{turbine}"
    rows, targets, kept = [], [], []
    for item in origins:
        origin = pd.Timestamp(item)
        target_time = origin + pd.Timedelta(minutes=horizon_minutes)
        if not (start <= origin <= end and start <= target_time <= end):
            continue
        if target_time not in raw.index or pd.isna(raw.at[target_time, target_col]):
            continue
        rows.append(features_at(inputs, origin, turbine, **feature_options))
        targets.append(float(raw.at[target_time, target_col]))
        kept.append(origin)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(kept)), pd.Series(targets, index=pd.DatetimeIndex(kept), dtype=float, name="target")
