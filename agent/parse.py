"""① 理解请求（负责人：C）

目前是规则版解析，作为大模型的占位。接入大模型后，让它输出同样的 ParsedRequest 字段即可，
状态机和后面的工具都不用改。规则版的好处是：评测时可以对照“有无大模型”看它到底贡献了什么。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CN_NUM = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "十二": 12, "二十四": 24}
SUPPORTED = (10, 30, 60)


@dataclass
class ParsedRequest:
    text: str
    turbine: str | None
    as_of: str | None
    horizons: tuple[int, ...] = SUPPORTED
    unit: str = "MW"
    asks_reason: bool = False
    future_request: bool = False
    clarify: list[str] = field(default_factory=list)   # 需要向用户追问的点
    out_of_scope: str | None = None


def _to_minutes(num: str | None, unit: str) -> int:
    if num in (None, ""):
        n = 1
    elif num == "半":
        return 30
    else:
        n = int(num) if num.isdigit() else CN_NUM.get(num, 1)
    return n * (60 if unit == "小时" else 1)


def parse_request(text: str, as_of: str | None, valid_turbines: list[str]) -> ParsedRequest:
    future_pat = r"(\d+|[一两二三四五六半])\s*(?:个)?\s*(分钟|小时)后的实际"
    future = re.search(future_pat, text) is not None or "实际值" in text
    core = re.sub(future_pat + r"\S*", "", text)

    m = re.search(r"\b(T\d{2})\b", core)
    turbine = m.group(1) if m else None
    req = ParsedRequest(text=text, turbine=turbine, as_of=as_of, future_request=future)
    req.unit = "kW" if re.search(r"kW|千瓦", core) else "MW"
    req.asks_reason = re.search(r"为什么|原因|出问题|故障|限电|怎么回事", core) is not None

    hm = re.search(r"未来\s*(\d+|[一两二三四五六半]|十二|二十四)?\s*(?:个)?\s*(分钟|小时)", core)
    if hm:
        minutes = _to_minutes(hm.group(1), hm.group(2))
        if minutes > max(SUPPORTED):
            req.out_of_scope = f"系统只支持 10 / 30 / 60 分钟的预测，无法给出未来 {minutes // 60 if minutes >= 60 else minutes} {'小时' if minutes >= 60 else '分钟'}的结果"
        elif minutes == 60:
            req.horizons = SUPPORTED
        else:
            req.horizons = tuple(h for h in SUPPORTED if h == minutes) or (minutes,)

    if turbine is None:
        req.clarify.append(f"请问要预测哪台机组？可选：{', '.join(valid_turbines)}")
    elif turbine not in valid_turbines:
        req.clarify.append(f"机组 {turbine} 不存在。本风场的机组是：{', '.join(valid_turbines)}，请确认编号")
    if as_of is None:
        req.clarify.append("请提供起报时刻（例如 2023-03-04T14:00），系统只使用该时刻及以前的数据")
    return req
