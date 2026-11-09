import pandas as pd

from eval.metrics import coverage, mae, pinball, rmse
from models.baselines import persistence


def test_metrics_basic():
    y, yhat = [1.0, 2.0, 3.0], [1.0, 2.0, 4.0]
    assert mae(y, yhat) == 1 / 3
    assert round(rmse(y, yhat), 4) == round((1 / 3) ** 0.5, 4)
    assert coverage(y, [0, 0, 0], [2, 2, 2]) == 2 / 3
    assert pinball([1.0], [1.0], 0.9) == 0.0


def test_persistence_uses_last_valid_value():
    s = pd.Series([1.0, 1.2, None])
    assert persistence(s, [1, 3, 6]) == {1: 1.2, 3: 1.2, 6: 1.2}
