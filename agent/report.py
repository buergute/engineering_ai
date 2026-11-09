"""⑨ 生成报告（负责人：C）。目前是模板输出；之后可让大模型润色措辞，但不许改数字。"""
from __future__ import annotations


def render_report(out: dict) -> str:
    if out["status"] == "failed":
        return f"任务失败：{out['reason']}"

    req = out["request"]
    lines = [f"# {req.turbine} 功率预测与复核（起报 {req.as_of}）", ""]

    if not out["validation"]["ok"]:
        lines += ["**校验未通过，以下结果不可发布：**", *[f"- {e}" for e in out["validation"]["errors"]], ""]

    lines.append("## 预测")
    fc = out["forecast"]
    if fc is None:
        lines.append("- 预测工具失败，暂无结果")
    else:
        for f in fc.result["forecasts"]:
            lines.append(f"- {f['horizon_min']} 分钟后：{f['p50']:.2f} MW"
                         f"（{int(f['coverage'] * 100)}% 区间 {f['lower']:.2f}–{f['upper']:.2f} MW）")
        if out["used_fallback"]:
            lines.append("- 说明：本机风速不可信，已切换到不使用本机风速的后备模型")

    d = out["diagnosis"]
    lines += ["", "## 复核结论", f"- {d['cause']['text']}"]
    for f in d["findings"]:
        lines.append(f"- 另有数据问题（证据强度：{f['strength']}）：{f['text']}")
    if out["failed_tools"]:
        lines.append(f"- 未完成：{', '.join(out['failed_tools'])} 调用失败")

    lines += ["", "## 执行轨迹"]
    for t in out["trace"]:
        extra = f"（第 {t['attempt']} 次）" if t.get("attempt", 1) > 1 else ""
        lines.append(f"- {t['step']}: {t['status']}{extra}")
    return "\n".join(lines)
