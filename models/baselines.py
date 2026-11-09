"""预测基线（负责人：B）"""
from __future__ import annotations

import pandas as pd


def persistence(power: pd.Series, horizons_steps: list[int]) -> dict[int, float]:
    """持续性预测：未来各时距都等于起报时刻最后一个有效观测值。

    power 必须只包含起报时刻及以前的数据。
    """
    last = power.dropna()
    if last.empty:
        raise ValueError("起报窗口内没有有效功率观测")
    value = float(last.iloc[-1])
    return {h: value for h in horizons_steps}


def imputed_lightgbm(*args, **kwargs):
    """统一补值的 LightGBM 基线（截止 11/12）。补值只能用起报时刻以前的数据。"""
    raise NotImplementedError("TODO(B)")
