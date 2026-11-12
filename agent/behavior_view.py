"""把一次 run() 的结果整理成界面可用的「思考链 / 调用过程」。"""
from __future__ import annotations

from typing import Any

from common.schema import ToolResult

STEP_THINK = {
    "parse_request": "理解用户请求，提取机组、时距，并检查是否超范围或在索取未来真值",
    "read_asof": "按起报时刻读取历史窗口（本机 + 邻机），禁止读未来数据",
    "check_quality": "检查冻结、缺测、越界、漂移、风速计矛盾等数据问题",
    "check_power_curve": "对照经验功率曲线，估计应发与实发残差；必要时改用邻机风速",
    "review_context": "汇总功率跳变、邻机同步、湍流与运行状态，作为复核上下文",
    "query_evidence": "查询起报前可见的停机/限电等事件证据",
    "forecast_power": "生成 10/30/60 分钟点预测与区间；必要时切换后备模型或限电封顶",
    "validate_result": "独立校验单位、引用证据与未来泄漏，决定能否发布",
}


def _tool_summary(name: str, result_obj: Any) -> str:
    if result_obj is None:
        return "无返回"
    if isinstance(result_obj, ToolResult):
        r = result_obj.result or {}
        status = result_obj.status
    elif isinstance(result_obj, dict) and "result" in result_obj:
        r = result_obj.get("result") or {}
        status = result_obj.get("status")
    else:
        return str(result_obj)[:200]

    if name == "check_quality":
        issues = r.get("issues") or []
        types = [i.get("type") for i in issues]
        return f"status={status}, issues={len(issues)}, types={types}"
    if name == "check_power_curve":
        return (
            f"status={status}, residual={r.get('residual')}, "
            f"wind_source={r.get('wind_source')}, expected={r.get('expected_power')}, "
            f"actual={r.get('actual_power')}"
        )
    if name == "query_evidence":
        events = r.get("events") or []
        return f"status={status}, events={len(events)}, types={[e.get('type') for e in events]}"
    if name == "forecast_power":
        fcs = r.get("forecasts") or []
        return f"status={status}, model={r.get('model')}, horizons={len(fcs)}"
    if name == "review_context":
        return f"status={status}, keys={list(r.keys())}"
    if name == "read_asof":
        return f"status={status}, latest_time={r.get('latest_time')}, missing={r.get('missing_count')}"
    return f"status={status}"


def build_behavior_chain(out: dict[str, Any]) -> list[dict[str, str]]:
    """返回 [{kind, title, detail}, ...]，供 UI 时间线展示。"""
    chain: list[dict[str, str]] = []
    req = out.get("request")
    if req is not None:
        text = getattr(req, "text", None) or ""
        turbine = getattr(req, "turbine", None)
        as_of = getattr(req, "as_of", None)
        chain.append(
            {
                "kind": "think",
                "title": "接到任务",
                "detail": f"原文：{text}；解析机组={turbine}，起报={as_of}",
            }
        )

    # 从结果推断关键决策（即使 trace 里没有单独的 decide 步）
    if out.get("used_fallback"):
        chain.append(
            {
                "kind": "decide",
                "title": "路由：使用后备模型",
                "detail": "本机风速不可信，预测改为不依赖本机风速的后备路径",
            }
        )
    if out.get("excluded_neighbors"):
        chain.append(
            {
                "kind": "decide",
                "title": "路由：排除不可靠邻机",
                "detail": f"排除邻机：{', '.join(out['excluded_neighbors'])}",
            }
        )

    tool_map = {
        "read_asof": None,
        "check_quality": None,
        "check_power_curve": out.get("curve"),
        "review_context": out.get("context"),
        "query_evidence": out.get("evidence"),
        "forecast_power": out.get("forecast"),
    }

    for step in out.get("trace") or []:
        name = step.get("step", "")
        status = step.get("status", "")
        if name == "parse_request":
            chain.append(
                {
                    "kind": "think",
                    "title": "理解请求",
                    "detail": STEP_THINK["parse_request"],
                }
            )
            continue
        if name == "validate_result":
            chain.append(
                {
                    "kind": "think",
                    "title": "独立校验",
                    "detail": STEP_THINK["validate_result"],
                }
            )
            ok = status == "ok"
            errs = step.get("errors") or (out.get("validation") or {}).get("errors") or []
            chain.append(
                {
                    "kind": "tool",
                    "title": f"调用 validate_result → {status}",
                    "detail": "通过" if ok else f"失败：{errs}",
                }
            )
            continue

        think = STEP_THINK.get(name)
        if think:
            chain.append({"kind": "think", "title": f"准备：{name}", "detail": think})

        attempt = step.get("attempt", 1)
        followup = step.get("followup")
        extra = []
        if attempt and attempt > 1:
            extra.append(f"第{attempt}次")
        if followup:
            extra.append("补查")
        if step.get("error"):
            extra.append(f"error={step['error']}")
        suffix = f"（{'；'.join(extra)}）" if extra else ""

        summary = _tool_summary(name, tool_map.get(name))
        if status == "skipped":
            summary = step.get("reason") or "跳过"
        chain.append(
            {
                "kind": "tool",
                "title": f"调用 {name} → {status}{suffix}",
                "detail": summary,
            }
        )

    diag = out.get("diagnosis") or {}
    if diag:
        chain.append(
            {
                "kind": "decide",
                "title": f"复核结论：{out.get('verdict')}",
                "detail": diag.get("text") or "",
            }
        )
    elif out.get("message"):
        chain.append(
            {
                "kind": "decide",
                "title": f"提前结束：{out.get('verdict')}",
                "detail": out.get("message") or "",
            }
        )
    return chain
