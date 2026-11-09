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
