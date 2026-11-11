"""⑥ 查证据（负责人：A）

只返回起报时刻已经“可见”的事件（available_at <= as_of）。
如果事件在起报时还没结束，结束时间在当时是未知的，返回 end=None，而不是把事后才知道的结束时间泄露出来。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.schema import ToolResult
from common.store import DataStore


def query_evidence(store: DataStore, turbine: str, as_of: str, lookback_hours: float = 24.0) -> ToolResult:
    t = pd.Timestamp(as_of)
    ev = store.events
    sel = ev[(ev.turbine == turbine) & (ev.available_at <= t) & (ev.start <= t)
             & (ev.end >= t - pd.Timedelta(hours=lookback_hours))]
    events = []
    for _, e in sel.sort_values("start").iterrows():
        ended = e.end <= t
        events.append({
            "id": e.id, "type": e.type, "start": e.start.isoformat(timespec="minutes"),
            "end": e.end.isoformat(timespec="minutes") if ended else None,
            "active_at_as_of": not ended,
            "available_at": e.available_at.isoformat(timespec="minutes"),
            "setpoint_mw": None if pd.isna(e.setpoint_mw) else float(e.setpoint_mw),
            "detail": e.detail,
        })
    return ToolResult(tool="query_evidence", status="ok" if events else "empty", as_of=as_of,
                      tool_version="0.1.0", result={"events": events},
                      evidence_ids=[e["id"] for e in events])
