"""数据访问层。工具只通过 DataStore 读数据，换成真实 Kelmarsh 时只需重写 load()。

宽表约定：索引为时间（10 分钟），列为 p_<机组>（MW）与 w_<机组>（m/s）。
时间戳表示该 10 分钟区间的结束时刻，到达这个时刻即视为可见。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SYNTH_DIR = ROOT / "data" / "synth"


@dataclass
class DataStore:
    frame: pd.DataFrame
    events: pd.DataFrame
    meta: dict
    truth: pd.DataFrame | None = None   # 只给出题和评测用，工具不得访问

    @classmethod
    def load(cls, path: Path = SYNTH_DIR, with_truth: bool = False) -> "DataStore":
        frame = pd.read_parquet(path / "scada.parquet")
        events = pd.read_csv(path / "events.csv", parse_dates=["start", "end", "available_at"])
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        truth = None
        if with_truth and (path / "truth.csv").exists():
            truth = pd.read_csv(path / "truth.csv", parse_dates=["start", "end"])
        return cls(frame, events, meta, truth)

    @property
    def turbines(self) -> list[str]:
        return self.meta["turbines"]

    @property
    def rated(self) -> float:
        return float(self.meta["rated_mw"])

    def neighbors(self, turbine: str) -> list[str]:
        return list(self.meta["neighbors"][turbine])

    def copy(self) -> "DataStore":
        """给单个测试题注入扰动前先复制，避免污染原始数据。"""
        return DataStore(self.frame.copy(), self.events.copy(), copy.deepcopy(self.meta), self.truth)
