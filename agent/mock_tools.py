"""假工具（负责人：C）：返回固定的示意数据，用于在真实工具完成前搭建和测试流程。

所有数值都是编的，只用来演示格式和流程，不能出现在报告结果里。

场景：
  curtailment  T03 风速读数冻结 + 起报前有限功率事件（默认演示场景）
  no_event     T03 风速读数冻结，但查不到任何事件 → 应输出“原因无法确定”
  normal       数据正常 → 走常规流程，不补查
  tool_failure query_evidence 持续失败 → 应重试一次后降级
"""
from __future__ import annotations

from common.schema import ToolResult, ensure_not_future
from agent.toolbox import Toolbox

SCENARIOS = ("curtailment", "no_event", "normal", "tool_failure")


def mock_toolbox(scenario: str = "curtailment") -> Toolbox:
    if scenario not in SCENARIOS:
        raise ValueError(f"未知场景 {scenario}，可选：{SCENARIOS}")
    frozen = scenario != "normal"

    def read_asof(turbine, as_of, window_hours=6.0, include_neighbors=True):
        ensure_not_future(as_of, as_of)
        return ToolResult(
            tool="read_asof", status="ok", as_of=as_of, data_hash="mock-0001",
            tool_version="mock",
            result={
                "turbine": turbine,
                "neighbors": ["T02", "T04"],
                "latest": {
                    turbine: {"power": 0.9 if frozen else 1.6, "wind_speed": 7.32 if frozen else 9.4},
                    "T02": {"power": 1.6, "wind_speed": 9.6},
                    "T04": {"power": 1.65, "wind_speed": 9.5},
                },
            },
            warnings=["mock data"],
        )

    def check_quality(data):
        issues = []
        if frozen:
            issues.append({
                "turbine": data.result["turbine"], "field": "wind_speed", "type": "frozen",
                "start": "2019-07-14T12:40", "end": data.as_of,
                "detail": "风速连续 9 个点保持 7.32 m/s，同期邻机风速 7.1→9.6 m/s",
            })
        return ToolResult(
            tool="check_quality", status="ok", as_of=data.as_of, data_hash=data.data_hash,
            tool_version="mock", result={"issues": issues}, evidence_ids=["Q1"] if issues else [],
        )

    def check_power_curve(data, turbine, wind_source="own"):
        latest = data.result["latest"][turbine]
        if wind_source == "neighbors":
            wind, expected = 9.5, 1.7
        else:
            wind = latest["wind_speed"]
            expected = 1.0 if frozen else 1.62
        actual = latest["power"]
        return ToolResult(
            tool="check_power_curve", status="ok", as_of=data.as_of, data_hash=data.data_hash,
            tool_version="mock", evidence_ids=["C1"],
            result={
                "wind_source": wind_source, "wind_speed_used": wind,
                "expected_power": expected, "actual_power": actual,
                "residual": round(actual - expected, 3),
                "curve_source": "mock binned curve",
            },
        )

    def query_evidence(turbine, as_of, lookback_hours=24.0):
        if scenario == "tool_failure":
            raise ConnectionError("mock: 事件日志服务无响应")
        if scenario in ("no_event", "normal"):
            return ToolResult(tool="query_evidence", status="empty", as_of=as_of, tool_version="mock",
                              result={"events": []})
        event = {
            "id": "E1", "type": "curtailment", "setpoint_mw": 1.0,
            "start": "2019-07-14T12:30", "available_at": "2019-07-14T12:31", "end": None,
            "detail": "限功率运行，设定值 1.0 MW",
        }
        ensure_not_future(event["available_at"], as_of, "事件可见")
        return ToolResult(tool="query_evidence", status="ok", as_of=as_of, tool_version="mock",
                          result={"events": [event]}, evidence_ids=["E1"])

    def forecast_power(data, quality, turbine, horizons_minutes=(10, 30, 60), use_fallback=False):
        if use_fallback:
            rows = [(10, 0.95, 0.85, 1.05), (30, 1.00, 0.80, 1.60), (60, 1.10, 0.80, 1.85)]
        else:
            rows = [(10, 1.60, 1.48, 1.72), (30, 1.58, 1.35, 1.80), (60, 1.55, 1.20, 1.90)]
        rows = [r for r in rows if r[0] in horizons_minutes]
        return ToolResult(
            tool="forecast_power", status="ok", as_of=data.as_of, data_hash=data.data_hash,
            tool_version="mock", evidence_ids=["F1"],
            result={
                "model": "fallback" if use_fallback else "quality_aware",
                "forecasts": [{"horizon_min": h, "p50": m, "lower": lo, "upper": hi, "coverage": 0.8}
                              for h, m, lo, hi in rows],
            },
        )

    return Toolbox(read_asof, check_quality, check_power_curve, query_evidence, forecast_power)
