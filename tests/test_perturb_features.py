"""扰动注入和特征工程的单元测试（不需要数据文件）。"""
import numpy as np
import pandas as pd

from models.features import build_features
from perturb.inject import contiguous_dropout, freeze, slow_drift


def _frame(n=60):
    idx = pd.date_range("2023-01-01", periods=n, freq="10min")
    rng = np.random.default_rng(0)
    cols = {}
    for t in ("T01", "T02", "T03"):
        cols[f"p_{t}"] = rng.uniform(0.5, 1.5, n).round(4)
        cols[f"w_{t}"] = rng.uniform(6, 10, n).round(2)
    return pd.DataFrame(cols, index=idx)


def test_freeze_and_frozen_run_feature():
    df = _frame()
    t = df.index[-1]
    freeze(df, "w_T01", t - pd.Timedelta(minutes=80), t)
    assert df["w_T01"].iloc[-9:].nunique() == 1
    X = build_features(df, "T01", ["T02", "T03"], 0)
    assert X["w_frozen_run"].iloc[-1] == 9


def test_dropout_and_gap_feature():
    df = _frame()
    t = df.index[-1]
    contiguous_dropout(df, ["p_T01", "w_T01"], t - pd.Timedelta(minutes=120), t)
    X = build_features(df, "T01", ["T02", "T03"], 0)
    assert X["p_gap_tail"].iloc[-1] == 12 and X["w_gap_tail"].iloc[-1] == 12


def test_drift_only_changes_target_column():
    df = _frame()
    before = df.copy()
    t = df.index[-1]
    slow_drift(df, "w_T01", 0.5, t - pd.Timedelta(hours=6), t)
    assert df["w_T01"].iloc[-1] > before["w_T01"].iloc[-1]
    assert df.drop(columns="w_T01").equals(before.drop(columns="w_T01"))


def test_features_do_not_look_ahead():
    df = _frame()
    X1 = build_features(df, "T01", ["T02", "T03"], 0)
    df2 = df.copy()
    df2.iloc[-1] = 99.0                       # 改动最后一行，不应影响之前任何一行的特征
    X2 = build_features(df2, "T01", ["T02", "T03"], 0)
    pd.testing.assert_frame_equal(X1.iloc[:-1], X2.iloc[:-1])
