"""④ 编排决策（负责人：C）

固定的状态流：理解请求 → 读数据 → 检测（数据质量检查 + 功率突变/偏离曲线）→ [需要时补查] → 复核 → 预测 → 独立校验 → 输出。
复核按固定顺序给出四类结论：数据问题 → 运行状态变化 → 风况变化（正常）→ 无法解释；每一类都要有正面证据，
没有证据就归为“无法解释”，不靠排除法下结论。
程序决定“允许做什么”；数字一律来自工具。
预算：补查最多 MAX_FOLLOWUPS 次；单个工具失败重试 1 次，仍失败则降级。

结论（verdict）取值：normal（含 primary_cause=wind_change：可疑点只是风况变化）/ data_issue_only /
                     operating_change（限电、停机等运行状态变化）/ undetermined（无法解释）/
                     need_clarification / out_of_scope / failed_explicit
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.parse import ParsedRequest, parse_request
from agent.toolbox import Toolbox
from agent.validate import validate_result
from common.schema import ToolResult

MAX_FOLLOWUPS = 3
RESIDUAL_ALERT_MW = 0.3     # 默认的“明显欠发”阈值；运行过 eval.thresholds 后改用 val 段上估计的值
LOW_WIND_MS = 5.0           # 低于它且没有欠发，就把低功率解释为风小
NEIGHBOR_STRONG_FACTOR = 2.0  # 用邻机推算风速时，残差超过 2 倍阈值才算“明显欠发”
CURTAIL_TOL_MW = 0.1        # 实际功率 ≤ 设定值 + 0.1 才算“被限电压住”
CONFLICT_TOL_MW = 0.15      # 实际功率 > 设定值 + 0.15 视为与限电记录矛盾


@lru_cache(maxsize=1)
def residual_thresholds() -> dict[str, float]:
    """按风速来源分别取阈值（val 段上正常时刻残差的 1% 分位数）。"""
    path = Path(__file__).resolve().parents[1] / "artifacts" / "thresholds.json"
    if path.exists():
        t = json.loads(path.read_text(encoding="utf-8"))
        return {k: -float(v["residual_alert_mw"]) for k, v in t.items() if k != "review"}
    return {"own": RESIDUAL_ALERT_MW, "neighbors": RESIDUAL_ALERT_MW, "neighbors_1": RESIDUAL_ALERT_MW}


def _threshold(curve: ToolResult) -> float:
    r = curve.result
    key = r["wind_source"] if r["wind_source"] == "own" or len(r["neighbors_used"]) >= 2 else "neighbors_1"
    return residual_thresholds()[key]


@dataclass
class Request:
    text: str
    as_of: str | None


@dataclass
class RunState:
    request: ParsedRequest
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
            state.trace.append({"step": name, "status": res.status, "attempt": attempt, "followup": followup})
            return res
        except Exception as e:  # noqa: BLE001  工具失败统一在这里接住
            state.trace.append({"step": name, "status": "error", "attempt": attempt, "error": str(e)})
    state.failed_tools.append(name)
    return None


def _early(state: RunState, verdict: str, message: str, **extra) -> dict:
    return {"verdict": verdict, "primary_cause": extra.pop("primary_cause", "none"), "flags": extra.pop("flags", []),
            "message": message, "request": state.request, "forecast": None, "used_fallback": False,
            "diagnosis": None, "validation": {"ok": True, "errors": []}, "failed_tools": state.failed_tools,
            "followups": state.followups, "trace": state.trace, **extra}


ISSUE_FLAG = {("wind_speed", "anemometer_mismatch"): "wind_sensors_disagree", ("wind_speed", "frozen"): "wind_speed_frozen", ("power", "frozen"): "power_frozen",
              ("wind_speed", "drift"): "wind_speed_drift_suspected", ("power", "gap"): "data_gap",
              ("wind_speed", "gap"): "data_gap", ("power", "random_missing"): "random_missing",
              ("wind_speed", "random_missing"): "random_missing", ("power", "out_of_range"): "out_of_range",
              ("wind_speed", "out_of_range"): "out_of_range"}
WIND_BAD = ("frozen", "gap", "drift", "anemometer_mismatch")    # 这些问题出现时，该机组的风速读数不再可信
FIELD_CN = {"wind_speed": "风速读数", "power": "功率读数"}
TYPE_CN = {"anemometer_mismatch": "两个风速计不一致", "frozen": "冻结", "gap": "连续缺失", "random_missing": "零星缺失", "out_of_range": "越界", "drift": "漂移"}


def run(request: Request, tools: Toolbox) -> dict:
    req = parse_request(request.text, request.as_of, tools.turbines)
    state = RunState(req)
    state.trace.append({"step": "parse_request", "status": "ok"})

    if req.out_of_scope:
        return _early(state, "out_of_scope", req.out_of_scope)
    if req.clarify:
        return _early(state, "need_clarification", "；".join(req.clarify))

    data = _call(state, "read_asof", tools.read_asof, req.turbine, req.as_of)
    if data is None or data.status != "ok":
        why = "数据读取工具失败" if data is None else "；".join(data.warnings) or "没有可用数据"
        return _early(state, "failed_explicit", f"无法完成：{why}。不会凭空给出 {req.turbine} 的预测。",
                      primary_cause="data_gap", flags=["no_data"])
    quality = _call(state, "check_quality", tools.check_quality, data)
    issues = quality.result.get("issues", []) if quality else []

    own = [i for i in issues if i["turbine"] == req.turbine]
    nbr_bad = sorted({i["turbine"] for i in issues if i["turbine"] != req.turbine
                      and i["field"] == "wind_speed" and i["type"] in WIND_BAD})
    flags = sorted({ISSUE_FLAG[(i["field"], i["type"])] for i in own if (i["field"], i["type"]) in ISSUE_FLAG})
    if nbr_bad:
        flags.append("neighbor_unreliable")
    own_wind_bad = any(i["field"] == "wind_speed" and i["type"] in WIND_BAD for i in own)
    own_power_bad = any(i["field"] == "power" and i["type"] in ("frozen", "gap") for i in own)

    curve = _call(state, "check_power_curve", tools.check_power_curve, data, req.turbine,
                  wind_source="neighbors" if own_wind_bad else "own", exclude_neighbors=tuple(nbr_bad),
                  followup=own_wind_bad)
    curve_ok = curve is not None and curve.status == "ok"
    # 证据强弱：本机风速可信时，残差超过阈值即为强证据；用邻机推算时误差更大，
    # 1–2 倍阈值之间只算弱证据（可能偏低，但在推算误差范围内），超过 2 倍才算明显欠发
    low_power = weak_deficit = False
    if curve_ok and not own_power_bad:
        thr = _threshold(curve)
        factor = 1.0 if curve.result["wind_source"] == "own" else NEIGHBOR_STRONG_FACTOR
        low_power = curve.result["residual"] < -thr * factor
        weak_deficit = not low_power and curve.result["residual"] < -thr

    ctx = _call(state, "review_context", tools.review_context, data, req.turbine) if tools.review_context else None
    wind_now = curve.result["wind_speed_used"] if curve_ok else None
    op = ctx.result["operation"] if ctx and ctx.status == "ok" else {}
    jump_sus = bool(ctx and ctx.status == "ok" and ctx.result["power_jump"]["suspicious"])
    op_sus = bool(op.get("available")) and (_stopped(op, wind_now) or _pitch_large(op, wind_now))

    evidence = None
    if own_wind_bad or low_power or weak_deficit or req.asks_reason or jump_sus or op_sus:
        evidence = _call(state, "query_evidence", tools.query_evidence, req.turbine, req.as_of, followup=True)

    # 限电已确认且仍在持续：告诉预测工具按设定值封顶（模型本身不知道限电）
    curt = _active_event(evidence, "curtailment")
    cap = None
    if curt and curve_ok and curve.result["actual_power"] <= curt["setpoint_mw"] + CURTAIL_TOL_MW:
        cap = curt["setpoint_mw"]
    forecast = _call(state, "forecast_power", tools.forecast_power, data, quality, req.turbine,
                     horizons_minutes=req.horizons, use_fallback=own_wind_bad,
                     exclude_neighbors=tuple(nbr_bad), cap_mw=cap, followup=own_wind_bad)

    diag = diagnose(state, own, flags, curve if curve_ok else None, evidence, own_power_bad, low_power, weak_deficit,
                    ctx if ctx and ctx.status == "ok" else None)
    if req.future_request:
        diag["flags"].append("future_request_refused")
    cited = diag["cited"] + (forecast.evidence_ids if forecast else [])
    if ctx:
        diag["cited"] = sorted(set(diag["cited"]))
    validation = validate_result(req.as_of, state.results, cited, tools.rated_mw)
    state.trace.append({"step": "validate_result", "status": "ok" if validation["ok"] else "error",
                        "errors": validation["errors"]})
    if not validation["ok"]:
        diag["verdict"], diag["primary_cause"] = "undetermined", "unknown"

    return {"verdict": diag["verdict"], "primary_cause": diag["primary_cause"], "flags": sorted(set(diag["flags"])),
            "message": None, "request": req, "forecast": forecast, "used_fallback": own_wind_bad,
            "excluded_neighbors": nbr_bad, "curve": curve if curve_ok else None, "evidence": evidence,
            "context": ctx, "diagnosis": diag, "validation": validation, "failed_tools": state.failed_tools,
            "followups": state.followups, "trace": state.trace, "cited": cited}


def _active_event(evidence: ToolResult | None, kind: str) -> dict | None:
    if evidence is None:
        return None
    for ev in evidence.result.get("events", []):
        if ev["type"] == kind and ev["active_at_as_of"]:
            return ev
    return None


def _stopped(op: dict, wind: float | None) -> bool:
    """转子基本停转，而风速足以发电（低风速时转子本来就慢）。"""
    return bool(op.get("stopped")) and (wind is None or wind > 3.5)


def _pitch_large(op: dict, wind: float | None) -> bool:
    """风速低于额定附近，桨距角却偏大：叶片被调开，符合限功率运行。高风速时变桨是正常的。"""
    return bool(op.get("pitch_large")) and (wind is None or wind < 11.5)


def diagnose(state: RunState, own_issues: list, flags: list, curve: ToolResult | None,
             evidence: ToolResult | None, own_power_bad: bool, low_power: bool, weak_deficit: bool = False,
             ctx: ToolResult | None = None) -> dict:
    """规则化的复核：把可疑点归入四类之一，且每一类都要有正面证据。
    顺序：数据问题 → 运行状态变化（日志或转速/桨距角）→ 风况变化（功率跟着风走且邻机同步或湍流偏大）→ 无法解释。"""
    flags = list(flags)
    findings = [{"text": f"{FIELD_CN[i['field']]}疑似{TYPE_CN[i['type']]}：{i['detail']}", "evidence": [i["id"]]}
                for i in own_issues]
    cited = [i["id"] for i in own_issues]
    curt, stop = _active_event(evidence, "curtailment"), _active_event(evidence, "stop")
    evidence_failed = "query_evidence" in state.failed_tools
    if evidence_failed:
        flags.append("tool_failed")
    actual = curve.result["actual_power"] if curve else None
    wind_now = curve.result["wind_speed_used"] if curve else None
    rc = ctx.result if ctx else None
    op = rc["operation"] if rc else {}
    jump = rc["power_jump"] if rc else {}
    jump_sus = bool(jump.get("suspicious"))
    stopped, pitched = _stopped(op, wind_now), _pitch_large(op, wind_now)

    if weak_deficit and curt and actual <= curt["setpoint_mw"] + CURTAIL_TOL_MW:
        low_power = True                    # 限电日志是直接证据，弱残差也足以确认
    if own_power_bad:
        cause = "data_gap" if "data_gap" in flags else "sensor_fault"
        verdict, text = "data_issue_only", "本机功率读数不可用（缺失或冻结），无法判断当前是否欠发；预测已降低置信度"
    elif low_power:
        cited += curve.evidence_ids
        res = curve.result
        gap_text = f"按{'邻机推算' if res['wind_source'] == 'neighbors' else '本机'}风速 {res['wind_speed_used']} m/s，应发约 {res['expected_power']} MW，实际 {res['actual_power']} MW"
        if rc:
            cited += ctx.evidence_ids
        if curt and actual <= curt["setpoint_mw"] + CURTAIL_TOL_MW:
            verdict, cause = "operating_change", "curtailment"
            text = f"主因是限功率运行（{curt['detail']}，{curt['start']} 起，结束时间未知）。{gap_text}"
            cited.append(curt["id"])
        elif stop:
            verdict, cause = "operating_change", "stop"
            text = f"机组处于停机状态（{stop['start']} 起）。{gap_text}"
            cited.append(stop["id"])
        elif stopped:
            verdict, cause = "operating_change", "stop"
            flags.append("evidence_from_fields_only")
            text = (f"转子转速仅 {op['rpm_mean']} rpm，机组基本停转，但事件日志里查不到对应的停机记录（可能尚未补录）。{gap_text}")
        elif pitched:
            verdict, cause = "operating_change", "curtailment"
            flags.append("evidence_from_fields_only")
            text = (f"桨距角 {op['pitch_mean']}°，而风速只有 {wind_now} m/s（低于额定），叶片被调开，符合限功率运行；"
                    f"事件日志没有记录，设定值未知。{gap_text}")
        elif evidence_failed:
            verdict, cause = "undetermined", "unknown"
            text = f"功率明显偏低，但事件日志查询失败，原因判断未完成。{gap_text}"
        else:
            verdict, cause = "undetermined", "unknown"
            text = (f"功率明显偏低，但没有找到任何解释：事件日志没有记录，转速和桨距角正常，风速也不低。{gap_text}。"
                    "原因无法解释，建议人工核查（调度指令、偏航与桨距记录）")
    else:
        serious = [f for f in flags if f in ("wind_speed_frozen", "wind_speed_drift_suspected", "data_gap",
                                             "neighbor_unreliable", "out_of_range", "wind_sensors_disagree")]
        if curve:
            cited += curve.evidence_ids
        ended = _recent_ended(evidence, state.request.as_of) if jump_sus else None
        if curt and actual is not None and actual > curt["setpoint_mw"] + CONFLICT_TOL_MW:
            verdict, cause = "undetermined", "unknown"
            flags.append("evidence_conflict")
            cited.append(curt["id"])
            text = (f"证据矛盾：日志显示 {curt['start']} 起限功率 {curt['setpoint_mw']} MW，但实际功率 {actual} MW 高于设定值。"
                    "可能是指令未执行、已解除但未记录，或记录有误，建议向调度核实")
        elif serious:
            verdict = "data_issue_only"
            cause = "data_gap" if serious == ["data_gap"] else "sensor_fault"
            if weak_deficit:
                flags.append("weak_deficit_evidence")
                r = curve.result
                text = (f"发现数据质量问题。按邻机推算风速 {r['wind_speed_used']} m/s，功率可能比曲线低 "
                        f"{-r['residual']:.2f} MW，但仍在邻机推算的误差范围内，不足以判定欠发；建议修复风速计后复核")
            else:
                text = "发现数据质量问题，但功率本身与功率曲线相符，未见欠发"
        elif ended:
            verdict, cause = "operating_change", ended["type"]
            cited.append(ended["id"])
            text = (f"最近 1 小时内的功率突变与运行状态变化吻合：事件日志记录的{'限功率' if ended['type'] == 'curtailment' else '停机'}"
                    f"已在 {ended['end']} 结束（{ended['id']}）")
        elif jump_sus:
            cited += ctx.evidence_ids
            sy, tb = rc["neighbor_sync"], rc["turbulence"]
            sync_txt = "、".join(f"{n['turbine']} 同期变化 {n['change_mw']:+.2f} MW" for n in sy["neighbors"]
                                if n["in_sync"])
            dir_cn = "下降" if jump["direction"] == "down" else "上升"
            base = f"{jump['time']} 前后30 分钟内功率{dir_cn} {abs(jump['max_change_mw']):.2f} MW，但功率与功率曲线相符（风速 {wind_now} m/s）"
            if sy["synced"] or tb["high"]:
                verdict, cause = "normal", "wind_change"
                flags.append("suspicious_but_wind_change")
                why = []
                if sy["synced"]:
                    why.append(f"本机变化 {sy['own_change_mw']:+.2f} MW，邻机同步变化（{sync_txt}）")
                if tb["high"]:
                    why.append(f"风速标准差为同风速正常水平的 {tb['vs_normal_q99']:.1f} 倍（湍流偏大）" if tb.get("vs_normal_q99")
                               else f"风速标准差 / 风速 = {tb['ti']}，湍流偏大")
                text = f"{base}。判断为风况变化，不是设备异常：" + "；".join(why)
            else:
                verdict, cause = "undetermined", "unknown"
                flags.append("unexplained_jump")
                text = (f"{base}，但邻机没有同步变化、湍流也不大，没有证据说明这是风况变化。"
                        "原因无法解释，建议人工核查")
        elif curve and curve.result["wind_speed_used"] < LOW_WIND_MS:
            verdict, cause = "normal", "low_wind"
            text = f"风速仅 {curve.result['wind_speed_used']} m/s，功率与功率曲线相符，低功率由风小造成，未发现故障迹象"
        else:
            verdict, cause = "normal", "none"
            text = "功率与功率曲线基本一致，未见异常"
    return {"verdict": verdict, "primary_cause": cause, "flags": flags, "text": text,
            "findings": findings, "cited": sorted(set(cited))}


def _recent_ended(evidence: ToolResult | None, as_of: str, minutes: int = 60) -> dict | None:
    """最近 minutes 分钟内刚结束的限电/停机事件（结束时间在起报时刻已可见）。"""
    import pandas as pd
    if evidence is None:
        return None
    t = pd.Timestamp(as_of)
    for ev in evidence.result.get("events", []):
        if ev["end"] and ev["type"] in ("curtailment", "stop") and t - pd.Timedelta(minutes=minutes) <= pd.Timestamp(ev["end"]) <= t:
            return ev
    return None
