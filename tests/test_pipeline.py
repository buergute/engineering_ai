"""用假工具检查整条流程。每次合并代码前跑：pytest"""
import pytest

from agent.mock_tools import mock_toolbox
from agent.report import render_report
from agent.state_machine import Request, run
from agent.validate import validate_result
from common.schema import FutureDataError, ToolResult, ensure_not_future

REQ = Request(turbine="T03", as_of="2019-07-14T14:00")


def test_curtailment_scenario():
    out = run(REQ, mock_toolbox("curtailment"))
    assert out["status"] == "ok"
    assert out["used_fallback"] is True
    assert out["diagnosis"]["cause"]["cause"] == "curtailment"
    assert any(f["kind"] == "data_quality" for f in out["diagnosis"]["findings"])
    assert "限功率" in render_report(out)


def test_no_event_is_undetermined():
    out = run(REQ, mock_toolbox("no_event"))
    assert out["diagnosis"]["cause"]["verdict"] == "undetermined"


def test_normal_does_not_follow_up():
    out = run(REQ, mock_toolbox("normal"))
    steps = [t["step"] for t in out["trace"]]
    assert "query_evidence" not in steps
    assert out["used_fallback"] is False


def test_tool_failure_retries_once_then_degrades():
    out = run(REQ, mock_toolbox("tool_failure"))
    attempts = [t for t in out["trace"] if t["step"] == "query_evidence"]
    assert len(attempts) == 2
    assert "query_evidence" in out["failed_tools"]
    assert out["diagnosis"]["cause"]["verdict"] == "undetermined"


def test_future_data_is_refused():
    with pytest.raises(FutureDataError):
        ensure_not_future("2019-07-14T15:00", "2019-07-14T14:00")


def test_validator_catches_late_event_and_bad_interval():
    bad = ToolResult(
        tool="x", status="ok", as_of="2019-07-14T14:00", evidence_ids=["E9"],
        result={
            "events": [{"id": "E9", "available_at": "2019-07-14T15:20"}],
            "forecasts": [{"horizon_min": 10, "p50": 1.0, "lower": 1.2, "upper": 1.5}],
        },
    )
    v = validate_result("2019-07-14T14:00", [bad], ["E9", "NOPE"])
    assert not v["ok"]
    assert len(v["errors"]) == 3
