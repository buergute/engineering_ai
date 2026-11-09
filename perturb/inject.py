"""扰动注入（负责人：A，截止 11/17）

四类扰动只改模型输入，不改预测目标（真实功率）。产出的扰动测试集交给 B 做压力测试。
每个函数都要接收随机种子，保证可复现。
"""
from __future__ import annotations

import pandas as pd


def random_missing(df: pd.DataFrame, cols: list[str], rate: float, seed: int) -> pd.DataFrame:
    """随机把 rate 比例的点置为缺失。"""
    raise NotImplementedError("TODO(A)")


def contiguous_dropout(df: pd.DataFrame, cols: list[str], n_segments: int, length_steps: int, seed: int) -> pd.DataFrame:
    """连续掉线：随机选 n_segments 段，每段 length_steps 个点置为缺失。"""
    raise NotImplementedError("TODO(A)")


def freeze(df: pd.DataFrame, col: str, n_segments: int, length_steps: int, seed: int) -> pd.DataFrame:
    """读数冻结：某段内读数保持为段首值不变。"""
    raise NotImplementedError("TODO(A)")


def slow_drift(df: pd.DataFrame, col: str, drift_per_day: float, start: str, seed: int) -> pd.DataFrame:
    """缓慢偏移：从 start 起，读数按 drift_per_day 线性漂移。"""
    raise NotImplementedError("TODO(A)")
