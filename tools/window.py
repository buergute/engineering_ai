"""把 read_asof 的结果还原成宽表，供其他工具和模型使用。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.schema import ToolResult


def to_frame(data: ToolResult) -> pd.DataFrame:
    r = data.result
    cols = {}
    names = {"power": "p", "wind_speed": "w", "wind_speed_2": "w2", "wind_std": "ws", "wind_max": "wx",
             "wind_min": "wn", "pitch": "pt", "rpm": "rpm"}
    for t, s in r["series"].items():
        for key, k in names.items():
            if key in s:
                cols[f"{k}_{t}"] = np.array([np.nan if v is None else v for v in s[key]], dtype=float)
    return pd.DataFrame(cols, index=pd.to_datetime(r["time"]))


def tail_run(x: np.ndarray, pred) -> int:
    """从末尾往前数，连续满足 pred 的点数。"""
    n = 0
    for v in x[::-1]:
        if not pred(v):
            break
        n += 1
    return n


def frozen_tail(x: np.ndarray) -> int:
    """末尾连续相同值的长度（跳过末尾的缺失）。"""
    vals = x[~np.isnan(x)]
    if len(vals) == 0:
        return 0
    last = vals[-1]
    return tail_run(vals, lambda v: v == last)
