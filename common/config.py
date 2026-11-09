"""读取全组唯一的时间切分 configs/split.yaml（由成员 A 维护）。"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / "configs" / "split.yaml"


def load_split(path: Path = SPLIT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
