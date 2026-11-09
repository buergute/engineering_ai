"""⑥ 查证据（负责人：A，截止 11/16）"""
from __future__ import annotations

from common.schema import ToolResult


def query_evidence(turbine: str, as_of: str, lookback_hours: float = 24.0) -> ToolResult:
    """查询起报前“已经可见”的事件和邻机对照。

    result 应包含：
      - "events": [{"id", "type", "start", "available_at", "end", "detail"}]
        只返回 available_at <= as_of 的事件（事后补录的不能用）
      - "neighbor_comparison": {邻机: {"power", "wind_speed"}}
      - "supports": [支持某解释的证据 id], "conflicts": [冲突证据 id]
    查不到事件时 status="empty"，不要编造。
    """
    raise NotImplementedError("TODO(A): 接入事件日志")
