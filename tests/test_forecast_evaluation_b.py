from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from common.schema import ToolResult
from eval.metrics import paired_daily_means
from eval.numeric import METHODS, summarize

tool = import_module("models.forecast_power")


def test_paired_day_weights_preserve_day_and_method_alignment():
    dates = ["2021-01-01"] * 2 + ["2021-01-02"]
    values = np.array([[0, 1], [10, 11], [20, 21]])
    point, draws = paired_daily_means(values, dates, 200, 4)
    np.testing.assert_allclose(point, [10, 11])
    assert set(draws[:, 0]) == {5, 10, 20}
    np.testing.assert_allclose(draws[:, 1] - draws[:, 0], 1)


def test_metrics_compare_on_same_origins():
    frame = pd.DataFrame(index=pd.date_range("2021-01-01", periods=6, freq="D"))
    for m in METHODS:
        frame[m+"_error"] = np.arange(1, 7) * (1 if m == "persistence" else .5)
        frame[m+"_covered"] = 1.
        frame[m+"_width"] = .4
    row = summarize("clean", 10, frame, 2., {"seed": 1, "bootstrap_draws": 100})[1]
    np.testing.assert_allclose([row["skill_pct_lo"], row["skill_pct_hi"]], 50)
    assert row["nMAE"] == row["MAE"] / 2


@pytest.mark.parametrize("power, predicted, expected", [(2.05, 2.6, 2.05), (0, 0, 0)])
def test_forecast_bounds_with_crossed_quantiles(monkeypatch, power, predicted, expected):
    def model(v):
        return SimpleNamespace(predict=lambda x: np.array([v]))
    bundle = {"turbines": ["T01"], "version": "fixture", "coverage": .8,
              "quantiles": [.1, .5, .9], "features": {"quality_aware": ["p_lag0", "p_nan_6h",
              "p_frozen_run", "nb_valid"]}, "models": {}, "cqr": {}}
    for tau, v in zip(bundle["quantiles"], [3, predicted, -1]):
        bundle["models"]["quality_aware", 10, tau] = model(v)
    bundle["cqr"]["quality_aware", 10, "degraded"] = 0
    monkeypatch.setattr(tool, "load_models", lambda: bundle)
    as_of = "2021-01-01T00:00"
    d = ToolResult("read_asof", "ok", as_of, result={"time": [as_of], "neighbors": [],
                   "series": {"T01": {"power": [power], "wind_speed": [9]}}})
    row = tool.forecast_power(d, d, "T01", (10,)).result["forecasts"][0]
    assert -.05 <= row["lower"] <= row["p50"] == expected <= row["upper"] <= 2.05


def test_all_missing_window_returns_empty_without_loading_model():
    d = ToolResult("read_asof", "ok", "2021-01-01", result={"time": ["2021-01-01"],
                   "series": {"T01": {"power": [None], "wind_speed": [None]}}})
    assert tool.forecast_power(d, d, "T01").status == "empty"


def test_ten_minute_target_does_not_jump_across_a_missing_timestamp():
    from models.features import targets

    frame = pd.DataFrame({"p_T01": [1., 2., 0.]}, index=pd.to_datetime(
        ["2021-01-01T00:00", "2021-01-01T00:20", "2021-01-01T00:30"]))
    labels = targets(frame, "T01")
    assert np.isnan(labels[10].iloc[0])
    assert labels[30].iloc[0] == 0.
    assert np.isnan(labels[10].iloc[-1])
