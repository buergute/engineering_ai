"""把各工具组装成一个 Toolbox，状态机只通过它调用工具。

真实工具（A、B 实现）和假工具（agent/mock_tools.py）提供同样的六个方法，
所以 C 可以先用假工具开发，A、B 做完后一行切换。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Toolbox:
    read_asof: Callable
    check_quality: Callable
    check_power_curve: Callable
    query_evidence: Callable
    forecast_power: Callable


def real_toolbox() -> Toolbox:
    from models.forecast_power import forecast_power
    from tools import check_power_curve, check_quality, query_evidence, read_asof

    return Toolbox(read_asof, check_quality, check_power_curve, query_evidence, forecast_power)
