"""数值评测指标（负责人：B）"""
from __future__ import annotations

import numpy as np


def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def nmae(y, yhat, rated_power: float) -> float:
    """按额定容量归一化的 MAE。"""
    return mae(y, yhat) / rated_power


def pinball(y, q_pred, tau: float) -> float:
    y, q_pred = np.asarray(y, float), np.asarray(q_pred, float)
    diff = y - q_pred
    return float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))


def coverage(y, lower, upper) -> float:
    """经验覆盖率：真实值落在 [lower, upper] 内的比例。要按机组、工况、缺失模式分组报告。"""
    y, lower, upper = (np.asarray(a, float) for a in (y, lower, upper))
    return float(np.mean((y >= lower) & (y <= upper)))


def mean_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper, float) - np.asarray(lower, float)))
def daily_block_bootstrap(dates, *arrays, statistic, n_bootstrap: int = 2000,
                          confidence: float = 0.95, seed: int = 0):
    """按日期整块重采样，返回统计量及百分位置信区间。

    同一天的所有记录（包括不同机组）一起抽取；数组必须逐行对齐。
    statistic 接收重采样后的数组，返回有限标量。
    """
    import pandas as pd

    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap 必须为正，confidence 必须在 0 和 1 之间")
    days = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    values = [np.asarray(a) for a in arrays]
    if len(days) == 0 or days.isna().any() or not values:
        raise ValueError("日期和数据不能为空，日期不能缺失")
    if any(a.ndim != 1 or len(a) != len(days) for a in values):
        raise ValueError("所有数据必须是一维且与日期等长")
    blocks = [np.flatnonzero(days == day) for day in days.unique()]

    def calculate(indices):
        result = float(statistic(*(a[indices] for a in values)))
        if not np.isfinite(result):
            raise ValueError("统计量必须为有限数值")
        return result

    estimate = calculate(np.arange(len(days)))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        chosen = rng.integers(len(blocks), size=len(blocks))
        samples[i] = calculate(np.concatenate([blocks[j] for j in chosen]))
    alpha = (1 - confidence) / 2
    low, high = np.quantile(samples, [alpha, 1 - alpha])
    return estimate, float(low), float(high)
