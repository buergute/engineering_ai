"""⑤ 功率曲线复核（负责人：A，截止 11/16）"""
from __future__ import annotations

from typing import Literal

from common.schema import ToolResult


def check_power_curve(
    data: ToolResult,
    turbine: str,
    wind_source: Literal["own", "neighbors"] = "own",
) -> ToolResult:
    """用训练期分箱经验曲线，比较“该风速下应发多少”和“实际发了多少”。

    wind_source="neighbors" 时用邻机风速推算本机风速（本机风速不可信时用）。
    result 应包含：
      - "wind_speed_used", "expected_power", "actual_power", "residual"
      - "curve_source": 例如 "train-period binned curve, 2019"
      - "valid_domain": 曲线适用的工况范围
    注意：输出的是“异常证据”，不是故障结论。曲线只能用 train 段拟合。
    """
    raise NotImplementedError("TODO(A): 拟合经验功率曲线")
