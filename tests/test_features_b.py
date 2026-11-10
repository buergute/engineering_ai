import numpy as np
import pandas as pd
import pytest

from models.features import features_at, make_examples
from models.calibration import cqr_offset, calibrated_interval


def frame():
    return pd.DataFrame({'p_T01': [0., 1., 2., 1.5], 'w_T01': [2., 3., 4., 5.], 'w_T02': [3., 4., 5., 6.]}, index=pd.date_range('2020-01-01', periods=4, freq='10min'))


def test_future_cannot_change_features_and_fallback_excludes_own_wind():
    data = frame()
    origin = data.index[1]
    altered = data.copy()
    altered.loc[altered.index > origin] = 999
    pd.testing.assert_series_equal(features_at(data, origin, 'T01'), features_at(altered, origin, 'T01'))
    fallback = features_at(data, origin, 'T01', fallback=True)
    altered['w_T01'] = -999
    pd.testing.assert_series_equal(fallback, features_at(altered, origin, 'T01', fallback=True))
    assert not any(x.startswith('w_T01_') for x in fallback.index)


def test_targets_are_raw_and_do_not_cross_split():
    raw = frame()
    dirty = raw.copy()
    dirty['p_T01'] = -50
    x, y = make_examples(raw, dirty, 'T01', raw.index, 10, raw.index[0], raw.index[2])
    assert list(y) == [1, 2]
    assert list(x.index) == list(raw.index[:2])
    assert x.iloc[0]['p_T01_last'] == -50


def test_cqr_and_physical_bounds():
    assert cqr_offset([0, 0, 0, 0], [-1]*4, [1]*4, .8) == -1
    lo, mid, hi = calibrated_interval([-4, 0], [5, 0], [6, 0], 0)
    np.testing.assert_allclose(lo, [-.05, 0])
    np.testing.assert_allclose(mid, [2.05, 0])
    np.testing.assert_allclose(hi, [2.05, 0])
    with pytest.raises(ValueError):
        cqr_offset([0], [-1], [1], .8)


def test_empty_window_and_quality_ablation():
    data = frame()
    empty = features_at(data, '2019-01-01', 'T01')
    assert np.isnan(empty['p_T01_last'])
    assert empty['p_T01_missing_fraction'] == 1
    ablated = features_at(data, data.index[1], 'T01', quality_features=False)
    assert not any('missing_fraction' in k or 'age_steps' in k for k in ablated.index)
