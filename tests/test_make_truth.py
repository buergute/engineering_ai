"""真实数据真值候选的分段规则。"""
from __future__ import annotations

import pandas as pd

from prep.make_truth import mask_to_episodes


def test_every_other_sample_does_not_form_a_two_hour_episode():
    """每隔一个 10 分钟点命中时，中间缺口必须切断事件。"""
    index = pd.date_range("2024-01-01", periods=25, freq="10min")
    alternating = pd.Series(False, index=index)
    alternating.iloc[::2] = True

    episodes = mask_to_episodes(
        alternating,
        index[0],
        index[-1],
        min_duration=pd.Timedelta(hours=2),
    )

    assert episodes == []


def test_continuous_ten_minute_samples_form_a_two_hour_episode():
    """连续 10 分钟命中满两小时，才允许形成两小时真值。"""
    index = pd.date_range("2024-01-01", periods=13, freq="10min")
    continuous = pd.Series(True, index=index)

    episodes = mask_to_episodes(
        continuous,
        index[0],
        index[-1],
        min_duration=pd.Timedelta(hours=2),
    )

    assert episodes == [(index[0], index[-1])]
