"""Causal features for a single forecast origin.

Input is a time-indexed frame with p_<turbine> and w_<turbine> columns.
No statistics are fitted here: missing values remain NaN for a downstream model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
