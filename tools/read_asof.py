"""② 读数据（负责人：A）"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from common.schema import ToolResult, ensure_not_future
from common.store import DataStore


EXTRA = ("p", "w", "w2", "ws", "wx", "wn", "pt", "rpm")
FIELD_NAMES = {"wind_speed_2": "w2", "wind_std": "ws", "wind_max": "wx", "wind_min": "wn", "pitch": "pt", "rpm": "rpm"}


def _clean(values) -> list:
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4) for v in values]


def read_asof(store: DataStore, turbine: str, as_of: str, window_hours: float = 7.0,
              include_neighbors: bool = True) -> ToolResult:
    """返回 as_of 及以前 window_hours 小时的数据（本机 + 邻机）。as_of 之后的数据一律不返回。"""
    if turbine not in store.turbines:
        return ToolResult(tool="read_asof", status="error", as_of=as_of,
                          warnings=[f"机组 {turbine} 不存在，可选 {', '.join(store.turbines)}"])
    t = pd.Timestamp(as_of)
    nbrs = store.neighbors(turbine) if include_neighbors else []
    # 本机读全部状态字段，邻机读功率、风速和第二风速计；数据里没有的字段（真实数据可能缺）自动略过
    cols = [c for x in [turbine, *nbrs] for k in (EXTRA if x == turbine else ("p", "w", "w2"))
            if (c := f"{k}_{x}") in store.frame.columns]
    win = store.frame.loc[t - pd.Timedelta(hours=window_hours):t, cols]
    if len(win):
        ensure_not_future(win.index.max().isoformat(), as_of)

    series = {}
    for x in [turbine, *nbrs]:
        series[x] = {"power": _clean(win[f"p_{x}"]), "wind_speed": _clean(win[f"w_{x}"])}
        for key, k in FIELD_NAMES.items():
            if f"{k}_{x}" in win.columns:
                series[x][key] = _clean(win[f"{k}_{x}"])
    own_empty = win[[f"p_{turbine}", f"w_{turbine}"]].isna().all().all() if len(win) else True
    digest = hashlib.sha1(pd.util.hash_pandas_object(win, index=True).values.tobytes()).hexdigest()[:12]
    return ToolResult(
        tool="read_asof", status="empty" if own_empty else "ok", as_of=as_of, data_hash=digest,
        tool_version="0.1.0",
        result={
            "turbine": turbine, "neighbors": nbrs,
            "time": [ts.isoformat(timespec="minutes") for ts in win.index],
            "latest_time": win.index.max().isoformat(timespec="minutes") if len(win) else None,
            "series": series,
            "missing_count": {x: int(win[[f"p_{x}", f"w_{x}"]].isna().sum().sum()) for x in [turbine, *nbrs]},
        },
        warnings=[f"{turbine} 在窗口内没有任何有效数据"] if own_empty else [],
    )
