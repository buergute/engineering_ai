"""③ 数据质检（负责人：A，截止 11/12）"""
from __future__ import annotations

from common.schema import ToolResult


def check_quality(data: ToolResult) -> ToolResult:
    """输入 read_asof 的结果，检出质量问题。

    result 应包含：
      - "issues": [{"turbine", "field", "type": "frozen|gap|out_of_range", "start", "end", "detail"}]
      - "quality_flags": {机组: {"missing_ratio", "max_gap_steps", "frozen": bool, ...}}  ← 给 B 的模型当输入特征
      - "to_check": [待人工核查项]
    没有问题时 issues 为空列表，status 仍为 "ok"。
    """
    raise NotImplementedError("TODO(A): 实现冻结、连续缺失、越界检测")
