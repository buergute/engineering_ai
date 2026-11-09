"""全组共用的工具返回格式（接口约定 v1，11/8 冻结后改动需三人同意）。

每个工具都返回 ToolResult：公共字段放在外层，工具自己的内容放在 result 里。
空结果必须返回 status="empty"，不允许编造数据。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

Status = Literal["ok", "empty", "error"]

DEFAULT_UNITS = {"power": "MW", "wind_speed": "m/s"}


class FutureDataError(ValueError):
    """请求了起报时刻 as_of 之后的数据。所有工具遇到这种请求都必须拒绝。"""


@dataclass
class ToolResult:
    tool: str
    status: Status
    as_of: str
    result: dict[str, Any] = field(default_factory=dict)
    data_hash: str | None = None
    tool_version: str = "0.0.0"
    units: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_UNITS))
    evidence_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def ensure_not_future(ts: str, as_of: str, what: str = "数据") -> None:
    """ts 晚于 as_of 就抛 FutureDataError。读数据、查事件前都要调用。"""
    if parse_time(ts) > parse_time(as_of):
        raise FutureDataError(f"{what}时间 {ts} 晚于起报时刻 {as_of}，拒绝读取")
