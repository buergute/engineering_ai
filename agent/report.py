"""⑨ 生成报告（负责人：C）。目前是模板输出；之后可让大模型润色措辞，但不许改数字。"""
from __future__ import annotations

VERDICT_CN = {"normal": "正常", "data_issue_only": "数据问题（已调整预测）", "operating_change": "运行状态变化",
              "undetermined": "无法解释（建议人工核查）", "need_clarification": "需要补充信息", "out_of_scope": "超出能力范围",
              "failed_explicit": "未能完成"}


def render_report(out: dict) -> str:
    req = out["request"]
    if out["message"]:
        return f"【{VERDICT_CN[out['verdict']]}】{out['message']}"

    k, unit = (1000, "kW") if req.unit == "kW" else (1, "MW")
    lines = [f"# {req.turbine} 功率预测与复核（起报 {req.as_of}）", f"结论：{VERDICT_CN[out['verdict']]}", ""]
    if not out["validation"]["ok"]:
        lines += ["**校验未通过，以下结果不可发布：**", *[f"- {e}" for e in out["validation"]["errors"]], ""]

    lines.append("## 预测")
    fc = out["forecast"]
    if fc is None:
        lines.append("- 预测工具失败，暂无结果")
    else:
        for f in fc.result["forecasts"]:
            lines.append(f"- {f['horizon_min']} 分钟后：{f['p50'] * k:.{0 if k > 1 else 2}f} {unit}"
                         f"（{int(f['coverage'] * 100)}% 区间 {f['lower'] * k:.{0 if k > 1 else 2}f}–"
                         f"{f['upper'] * k:.{0 if k > 1 else 2}f} {unit}）")
        if out["used_fallback"]:
            lines.append("- 说明：本机风速不可信，已切换到不使用本机风速的后备模型")
        if fc.result.get("curtailment_cap_mw") is not None:
            lines.append(f"- 说明：限电仍在持续，中位数按设定值 {fc.result['curtailment_cap_mw']} MW 封顶；"
                         "上界保留，表示限电解除后的可能出力")
        if out.get("excluded_neighbors"):
            lines.append(f"- 说明：邻机 {', '.join(out['excluded_neighbors'])} 风速不可信，已排除")
    if req.future_request:
        lines.append("- 起报时刻之后的实际值属于未来数据，系统不能读取，也不会提供")

    d = out["diagnosis"]
    lines += ["", "## 复核结论", f"- {d['text']}"]
    for f in d["findings"]:
        lines.append(f"- 数据质量：{f['text']}")
    if out["failed_tools"]:
        lines.append(f"- 未完成：{', '.join(out['failed_tools'])} 调用失败")

    lines += ["", "## 执行轨迹"]
    for t in out["trace"]:
        extra = f"（第 {t['attempt']} 次）" if t.get("attempt", 1) > 1 else ""
        lines.append(f"- {t['step']}: {t['status']}{extra}")
    return "\n".join(lines)
