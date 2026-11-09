"""② 读数据（负责人：A，截止 11/12）"""
from __future__ import annotations

from common.schema import ToolResult


def read_asof(turbine: str, as_of: str, window_hours: float = 6.0, include_neighbors: bool = True) -> ToolResult:
    """返回 as_of 及以前 window_hours 小时的 SCADA 数据。

    result 应包含：
      - "series": {机组: {"time": [...], "power": [...], "wind_speed": [...], ...}}
      - "missing_mask": {机组: {字段: [0/1, ...]}}
      - "neighbors": [相邻机组 id]
    data_hash：本次读取的数据快照哈希，用于复现。
    as_of 之后的任何数据都不能出现在结果里（用 common.schema.ensure_not_future 检查）。
    """
    raise NotImplementedError("TODO(A): 接入 Kelmarsh 数据")
