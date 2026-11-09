"""⑧ 独立校验（负责人：C）：发布前的硬检查，不依赖大模型。"""
from __future__ import annotations

from common.schema import ToolResult, parse_time


def validate_result(as_of: str, results: list[ToolResult], cited_evidence: list[str]) -> dict:
    errors: list[str] = []
    known_ids: set[str] = set()

    for r in results:
        known_ids.update(r.evidence_ids)
        if r.as_of != as_of:
            errors.append(f"{r.tool}: as_of={r.as_of} 与请求 {as_of} 不一致")
        if r.units.get("power") != "MW":
            errors.append(f"{r.tool}: 功率单位不是 MW")
        for ev in r.result.get("events", []):
            if parse_time(ev["available_at"]) > parse_time(as_of):
                errors.append(f"{r.tool}: 事件 {ev['id']} 在起报后才可见，不能使用")
        for f in r.result.get("forecasts", []):
            if not (f["lower"] <= f["p50"] <= f["upper"]):
                errors.append(f"{r.tool}: {f['horizon_min']} 分钟预测区间顺序错误")

    for eid in cited_evidence:
        if eid not in known_ids:
            errors.append(f"报告引用了不存在的证据 {eid}")

    return {"ok": not errors, "errors": errors}
