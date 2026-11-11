"""预测基线（负责人：B）

- 持续性：persistence()，也在 eval/numeric.py 里直接用前向填充实现
- 统一补值 LightGBM：在 models/train.py 中以 kind="imputed" 训练（先前向补值，去掉质量特征）
"""
from __future__ import annotations

import pandas as pd


def persistence(power: pd.Series, horizons_steps: list[int]) -> dict[int, float]:
    """持续性预测：未来各时距都等于起报时刻最后一个有效观测值。power 只能包含起报时刻及以前的数据。"""
    last = power.dropna()
    if last.empty:
        raise ValueError("起报窗口内没有有效功率观测")
    value = float(last.iloc[-1])
    return {h: value for h in horizons_steps}
