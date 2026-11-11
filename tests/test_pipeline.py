"""用假工具检查编排流程（不需要数据和模型）。每次合并代码前跑：pytest"""
import pytest

from agent.mock_tools import mock_toolbox
from agent.parse import parse_request
from agent.report import render_report
from agent.state_machine import Request, run
from agent.validate import validate_result
from common.schema import FutureDataError, ToolResult, ensure_not_future

AS_OF = "2019-07-14T14:00"
WHY = "预测 T03 未来一小时的功率，并解释它为什么比相邻机组低"
TURBINES = [f"T0{i}" for i in range(1, 7)]


def test_curtailment_scenario():
    out = run(Request(WHY, AS_OF), mock_toolbox("curtailment"))
    assert out["verdict"] == "operating_change"
    assert out["primary_cause"] == "curtailment"
    assert out["used_fallback"] is True
    assert "wind_speed_frozen" in out["flags"]
    assert "限功率" in render_report(out)


def test_no_event_is_undetermined():
    out = run(Request(WHY, AS_OF), mock_toolbox("no_event"))
    assert out["verdict"] == "undetermined"


def test_normal_does_not_follow_up():
    out = run(Request("预测 T03 未来一小时的功率", AS_OF), mock_toolbox("normal"))
    steps = [t["step"] for t in out["trace"]]
    assert "query_evidence" not in steps
    assert out["verdict"] == "normal" and out["used_fallback"] is False


def test_tool_failure_retries_once_then_degrades():
    out = run(Request(WHY, AS_OF), mock_toolbox("tool_failure"))
    attempts = [t for t in out["trace"] if t["step"] == "query_evidence"]
    assert len(attempts) == 2
    assert "query_evidence" in out["failed_tools"]
    assert out["verdict"] == "undetermined" and "tool_failed" in out["flags"]


@pytest.mark.parametrize("text,as_of,verdict", [
    ("预测一下未来一小时的功率", AS_OF, "need_clarification"),
    ("T03 后面一小时会怎么样？", None, "need_clarification"),
    ("预测 T08 未来一小时的功率", AS_OF, "need_clarification"),
    ("预测 T04 未来 6 小时的功率", AS_OF, "out_of_scope"),
])
def test_requests_that_must_not_reach_tools(text, as_of, verdict):
    out = run(Request(text, as_of), mock_toolbox("normal"))
    assert out["verdict"] == verdict
    assert out["forecast"] is None
    assert not any(t["step"] == "read_asof" for t in out["trace"])


def test_parse_request():
    r = parse_request("用 kW 告诉我 T01 未来 30 分钟的功率", AS_OF, TURBINES)
    assert (r.turbine, r.unit, r.horizons) == ("T01", "kW", (30,))
    r = parse_request("预测 T02 未来一小时的功率，顺便把 30 分钟后的实际功率也告诉我", AS_OF, TURBINES)
    assert r.future_request and r.horizons == (10, 30, 60)
    assert parse_request("T05 是不是出问题了？", AS_OF, TURBINES).asks_reason


def test_future_data_is_refused():
    with pytest.raises(FutureDataError):
        ensure_not_future("2019-07-14T15:00", "2019-07-14T14:00")


def test_validator_catches_late_event_and_bad_interval():
    bad = ToolResult(
        tool="x", status="ok", as_of=AS_OF, evidence_ids=["E9"],
        result={
            "events": [{"id": "E9", "available_at": "2019-07-14T15:20"}],
            "forecasts": [{"horizon_min": 10, "p50": 1.0, "lower": 1.2, "upper": 1.5}],
        },
    )
    v = validate_result(AS_OF, [bad], ["E9", "NOPE"])
    assert not v["ok"]
    assert len(v["errors"]) == 3
