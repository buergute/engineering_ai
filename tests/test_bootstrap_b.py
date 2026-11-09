import numpy as np
import pytest

from eval.metrics import daily_block_bootstrap, mae


def test_bootstrap_samples_entire_days_and_is_reproducible():
    dates = ["2020-01-01"] * 2 + ["2020-01-02"] * 2
    # Within each day the two observations sum to 10. Point sampling breaks this.
    values = [0, 10, 4, 6]
    def stat(x):
        return np.mean(x)
    result = daily_block_bootstrap(dates, values, statistic=stat, seed=17)
    assert result == (5.0, 5.0, 5.0)
    assert result == daily_block_bootstrap(dates, values, statistic=stat, seed=17)


def test_bootstrap_paired_arrays_and_validation():
    result = daily_block_bootstrap(["2020-01-01", "2020-01-02"], [1, 2], [1, 2],
                                   statistic=mae, n_bootstrap=20)
    assert result == (0, 0, 0)
    with pytest.raises(ValueError):
        daily_block_bootstrap(["2020-01-01"], [1, 2], statistic=np.mean)
