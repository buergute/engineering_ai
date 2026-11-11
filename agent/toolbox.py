"""把各工具组装成一个 Toolbox，状态机只通过它调用工具。

真实工具（A、B 实现）和假工具（agent/mock_tools.py）提供同样的方法，
所以 C 可以用假工具开发，真实工具完成后一行切换。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable


@dataclass
class Toolbox:
    read_asof: Callable
    check_quality: Callable
    check_power_curve: Callable
    query_evidence: Callable
    forecast_power: Callable
    review_context: Callable | None = None     # 假工具可以不提供，状态机会跳过复核上下文
    turbines: list[str] = field(default_factory=lambda: [f"T0{i}" for i in range(1, 7)])
    rated_mw: float = 2.05


def real_toolbox(store=None) -> Toolbox:
    from common.store import DataStore
    from models.forecast_power import forecast_power
    from tools import check_power_curve, check_quality, query_evidence, read_asof, review_context

    store = store or DataStore.load()
    return Toolbox(
        read_asof=partial(read_asof, store),
        check_quality=partial(check_quality, rated_mw=store.rated),
        check_power_curve=check_power_curve,
        query_evidence=partial(query_evidence, store),
        review_context=review_context,
        forecast_power=partial(forecast_power, rated_mw=store.rated),
        turbines=store.turbines,
        rated_mw=store.rated,
    )
