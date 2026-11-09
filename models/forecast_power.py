"""⑦ 预测工具（负责人：B，截止 11/17）"""
from __future__ import annotations

from common.schema import ToolResult


def forecast_power(
    data: ToolResult,
    quality: ToolResult,
    turbine: str,
    horizons_minutes: tuple[int, ...] = (10, 30, 60),
    use_fallback: bool = False,
) -> ToolResult:
    """质量感知 LightGBM 分位数模型 + 区间校准。

    use_fallback=True 时用不依赖本机风速的后备模型（本机风速不可信时）。
    result 应包含：
      - "forecasts": [{"horizon_min", "p50", "lower", "upper", "coverage": 0.8}]
      - "model": 模型名, "model_version", "calibration_version"
    约束：lower <= p50 <= upper；单位 MW。
    """
    raise NotImplementedError("TODO(B): 训练并封装质量感知模型")
