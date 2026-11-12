"""行为日志（C）：开发调试时记录思考、决策与工具调用。

每次 run(..., verbose/save_log) 写入::

    agent/logs/<run_id>/
      behavior.md   # 人类可读时间线
      trace.json    # 完整结构化事件

控制台可同步打印。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from common.schema import ToolResult

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def _short(obj: Any, limit: int = 800) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except TypeError:
        text = str(obj)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def _as_dict(output: ToolResult | dict[str, Any] | None) -> dict[str, Any] | None:
    if output is None:
        return None
    if isinstance(output, ToolResult):
        return output.to_dict()
    return output


class BehaviorLogger:
    def __init__(self, run_id: str | None = None, *, verbose: bool = True) -> None:
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:10]}"
        self.verbose = verbose
        self.started_at = _now_iso()
        self.events: list[dict[str, Any]] = []
        self.dir = LOGS_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._echo(f"[log] 行为归档目录: {self.dir}")

    def _echo(self, line: str) -> None:
        if self.verbose:
            print(line, flush=True)

    def think(self, step: str, thought: str, **extra: Any) -> None:
        self.events.append(
            {"ts": _now_iso(), "kind": "think", "step": step, "thought": thought, **extra}
        )
        self._echo(f"  💭 [{step}] {thought}")

    def decide(self, step: str, action: str, reason: str, **extra: Any) -> None:
        self.events.append(
            {
                "ts": _now_iso(),
                "kind": "decide",
                "step": step,
                "action": action,
                "reason": reason,
                **extra,
            }
        )
        self._echo(f"  🧠 [{step}] 决定 {action}")
        self._echo(f"     原因: {reason}")

    def note(self, message: str, **extra: Any) -> None:
        self.events.append({"ts": _now_iso(), "kind": "note", "message": message, **extra})
        self._echo(f"  • {message}")

    def tool_call(
        self,
        step: str,
        tool: str,
        *,
        inputs: dict[str, Any] | None = None,
        output: ToolResult | dict[str, Any] | None = None,
        error: str | None = None,
        attempt: int | None = None,
    ) -> None:
        out = _as_dict(output)
        status = (out or {}).get("status") if out else ("error" if error else None)
        summary = _summarize_tool_output(tool, out) if out else {"error": error}
        event = {
            "ts": _now_iso(),
            "kind": "tool",
            "step": step,
            "tool": tool,
            "inputs": inputs or {},
            "status": status,
            "attempt": attempt,
            "output_summary": summary,
        }
        self.events.append(event)
        att = f" attempt={attempt}" if attempt is not None else ""
        self._echo(f"  🔧 [{step}] 调用 {tool}{att} → status={status}")
        if inputs:
            self._echo(f"     in : {_short(inputs, 240)}")
        self._echo(f"     out: {_short(summary, 240)}")

    def finalize(self, result: dict[str, Any]) -> Path:
        # Request / ToolResult 等转可序列化
        safe_result = _jsonable(result)
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": _now_iso(),
            "status": safe_result.get("status"),
            "used_fallback": safe_result.get("used_fallback"),
            "failed_tools": safe_result.get("failed_tools"),
            "events": self.events,
            "pipeline_trace": safe_result.get("trace"),
            "diagnosis": safe_result.get("diagnosis"),
            "validation": safe_result.get("validation"),
            "request": safe_result.get("request"),
            "forecast": safe_result.get("forecast"),
        }
        (self.dir / "trace.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        (self.dir / "behavior.md").write_text(_to_markdown(payload), encoding="utf-8")
        self._echo(f"[log] 已写入 behavior.md / trace.json")
        return self.dir


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, ToolResult):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


def _summarize_tool_output(tool: str, output: dict[str, Any] | None) -> dict[str, Any]:
    if not output:
        return {}
    result = output.get("result") or {}
    base: dict[str, Any] = {
        "status": output.get("status"),
        "tool_version": output.get("tool_version"),
        "evidence_ids": output.get("evidence_ids") or [],
        "warnings": output.get("warnings") or [],
    }
    if tool == "read_asof":
        base["turbine"] = result.get("turbine")
        base["latest_keys"] = list((result.get("latest") or {}).keys())
        base["has_series"] = "series" in result
    elif tool == "check_quality":
        issues = result.get("issues") or []
        base["n_issues"] = len(issues)
        base["issue_types"] = [i.get("type") for i in issues]
    elif tool == "check_power_curve":
        base["residual"] = result.get("residual")
        base["expected_power"] = result.get("expected_power")
        base["actual_power"] = result.get("actual_power")
        base["wind_source"] = result.get("wind_source") or result.get("curve_source")
    elif tool == "query_evidence":
        events = result.get("events") or []
        base["n_events"] = len(events)
        base["event_types"] = [e.get("type") for e in events]
    elif tool == "forecast_power":
        forecasts = result.get("forecasts") or []
        base["n_horizons"] = len(forecasts)
        base["model"] = result.get("model")
        if forecasts:
            base["first"] = forecasts[0]
    else:
        base["result_keys"] = list(result.keys())[:20]
    return base


def _to_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# 智能体行为日志 · `{payload['run_id']}`",
        "",
        f"- 开始: {payload['started_at']}",
        f"- 结束: {payload['finished_at']}",
        f"- 运行状态: **{payload.get('status')}**",
        f"- 使用后备模型: {payload.get('used_fallback')}",
        f"- 失败工具: {payload.get('failed_tools')}",
        "",
        "## 时间线",
        "",
    ]
    for i, ev in enumerate(payload.get("events") or [], 1):
        kind = ev.get("kind")
        ts = ev.get("ts", "")
        if kind == "think":
            lines += [f"### {i}. 思考 · {ev.get('step')} ({ts})", f"- {ev.get('thought')}", ""]
        elif kind == "decide":
            lines += [
                f"### {i}. 决策 · {ev.get('step')} ({ts})",
                f"- action: `{ev.get('action')}`",
                f"- reason: {ev.get('reason')}",
                "",
            ]
        elif kind == "tool":
            lines += [
                f"### {i}. 工具 · {ev.get('tool')} ({ts})",
                f"- step: `{ev.get('step')}`",
                f"- status: `{ev.get('status')}`",
                f"- inputs: `{_short(ev.get('inputs'), 400)}`",
                f"- output: `{_short(ev.get('output_summary'), 400)}`",
                "",
            ]
        else:
            lines += [f"### {i}. 备注 ({ts})", f"- {ev.get('message')}", ""]

    if payload.get("diagnosis"):
        lines += [
            "## 复核结论",
            "",
            "```json",
            json.dumps(payload["diagnosis"], ensure_ascii=False, indent=2, default=str)[:3000],
            "```",
            "",
        ]
    lines += ["完整结构化数据见同目录 `trace.json`。", ""]
    return "\n".join(lines)
