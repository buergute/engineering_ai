"""④ 编排决策（负责人：C）

固定的状态流：读数据 → 质检 → [需要时补查] → 预测 → 复核结论 → 独立校验 → 输出。
程序决定“允许做什么”；之后接入大模型时，只让它在允许的步骤里选择补查、组织说明，
数字一律来自工具。

预算：补查最多 MAX_FOLLOWUPS 次；单个工具失败重试 1 次，仍失败则降级。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.toolbox import Toolbox
from agent.validate import validate_result
from common.schema import ToolResult, parse_time

MAX_FOLLOWUPS = 3
RESIDUAL_ALERT_MW = 0.3   # 功率曲线残差超过它才算“明显偏低”。TODO(C/A)：在 val 段上定阈值
FIELD_NAMES = {"wind_speed": "风速读数", "power": "功率读数"}
ISSUE_NAMES = {"frozen": "冻结", "gap": "连续缺失", "out_of_range": "越界"}


@dataclass
class Request:
    turbine: str
    as_of: str
    horizons_minutes: tuple[int, ...] = (10, 30, 60)


@dataclass
class RunState:
    request: Request
    results: list[ToolResult] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    followups: int = 0
    failed_tools: list[str] = field(default_factory=list)


def _call(state: RunState, name: str, fn, *args, followup: bool = False, **kwargs) -> ToolResult | None:
    if followup:
        if state.followups >= MAX_FOLLOWUPS:
            state.trace.append({"step": name, "status": "skipped", "reason": "补查预算用完"})
            return None
        state.followups += 1
    for attempt in (1, 2):
        try:
            res = fn(*args, **kwargs)
            state.results.append(res)
            state.trace.append({"step": name, "status": res.status, "attempt": attempt})
            return res
        except Exception as e:  # noqa: BLE001  工具失败统一在这里接住
            state.trace.append({"step": name, "status": "error", "attempt": attempt, "error": str(e)})
    state.failed_tools.append(name)
    return None


def _active_curtailment(evidence: ToolResult | None, as_of: str) -> dict | None:
    if evidence is None:
        return None
    t = parse_time(as_of)
    for ev in evidence.result.get("events", []):
        if ev["type"] != "curtailment":
            continue
        started = parse_time(ev["start"]) <= t
        not_ended = ev.get("end") is None or parse_time(ev["end"]) > t
        if started and not_ended:
            return ev
    return None


def diagnose(state: RunState, quality: ToolResult, curve: ToolResult | None, evidence: ToolResult | None) -> dict:
    """规则化的复核结论：分开报告“低功率原因”和“数据质量问题”。"""
    req = state.request
    findings: list[dict] = []
    cited: list[str] = []

    for issue in quality.result.get("issues", []):
        if issue["turbine"] == req.turbine:
            field_name = FIELD_NAMES.get(issue["field"], issue["field"])
            issue_name = ISSUE_NAMES.get(issue["type"], issue["type"])
            findings.append({"kind": "data_quality", "strength": "中",
                             "text": f"{field_name}疑似{issue_name}：{issue['detail']}",
                             "evidence": quality.evidence_ids})
            cited += quality.evidence_ids

    low_power = curve is not None and curve.result["residual"] < -RESIDUAL_ALERT_MW
    curtail = _active_curtailment(evidence, req.as_of)

    if not low_power:
        cause = {"verdict": "normal", "text": "实际功率与功率曲线基本一致，未见明显少发"}
    elif curtail is not None:
        cause = {"verdict": "confirmed_cause", "cause": "curtailment",
                 "text": f"主因是限功率运行（{curtail['detail']}，{curtail['start']} 起）",
                 "evidence": [curtail["id"], *curve.evidence_ids]}
        cited += cause["evidence"]
    else:
        missing = "事件日志查询失败" if "query_evidence" in state.failed_tools else "未查到相关事件记录"
        cause = {"verdict": "undetermined",
                 "text": f"功率明显偏低，但{missing}，原因无法确定；候选：未记录的限电或降额、对风偏差",
                 "evidence": curve.evidence_ids}
        cited += curve.evidence_ids
    return {"cause": cause, "findings": findings, "cited": sorted(set(cited))}


def run(request: Request, tools: Toolbox) -> dict:
    state = RunState(request)
    req = request

    data = _call(state, "read_asof", tools.read_asof, req.turbine, req.as_of)
    if data is None:
        return {"status": "failed", "reason": "无法读取数据", "trace": state.trace}
    quality = _call(state, "check_quality", tools.check_quality, data)
    if quality is None:
        return {"status": "failed", "reason": "质检工具失败", "trace": state.trace}

    own_wind_bad = any(i["turbine"] == req.turbine and i["field"] == "wind_speed"
                       for i in quality.result.get("issues", []))

    curve = _call(state, "check_power_curve", tools.check_power_curve, data, req.turbine,
                  wind_source="neighbors" if own_wind_bad else "own", followup=own_wind_bad)

    need_evidence = own_wind_bad or (curve is not None and curve.result["residual"] < -RESIDUAL_ALERT_MW)
    evidence = None
    if need_evidence:
        evidence = _call(state, "query_evidence", tools.query_evidence, req.turbine, req.as_of, followup=True)

    forecast = _call(state, "forecast_power", tools.forecast_power, data, quality, req.turbine,
                     horizons_minutes=req.horizons_minutes, use_fallback=own_wind_bad)

    diagnosis = diagnose(state, quality, curve, evidence)
    cited = diagnosis["cited"] + (forecast.evidence_ids if forecast else [])
    validation = validate_result(req.as_of, state.results, cited)
    state.trace.append({"step": "validate_result", "status": "ok" if validation["ok"] else "error",
                        "errors": validation["errors"]})

    status = "ok" if validation["ok"] and forecast is not None else "degraded"
    return {
        "status": status,
        "request": req,
        "forecast": forecast,
        "used_fallback": own_wind_bad,
        "diagnosis": diagnosis,
        "validation": validation,
        "failed_tools": state.failed_tools,
        "trace": state.trace,
    }
