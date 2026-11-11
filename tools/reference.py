"""读取 models/train.py 从 train 段拟合的参考量（经验功率曲线、邻机风速比例）。"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def load_reference() -> dict:
    path = ART_DIR / "reference.json"
    if not path.exists():
        raise FileNotFoundError("找不到 artifacts/reference.json，请先运行 python -m models.train")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_power(turbine: str, wind: float) -> float:
    c = load_reference()["curve"][turbine]
    return float(np.interp(wind, c["ws"], c["mw"]))


def implied_wind(turbine: str, power):
    """沿经验功率曲线反推：发出这么多功率，对应多大的风。只在切入到额定之间有意义。"""
    c = load_reference()["curve"][turbine]
    ws, mw = np.array(c["ws"]), np.maximum.accumulate(np.array(c["mw"]))
    keep = np.r_[True, np.diff(mw) > 1e-4]
    return np.interp(power, mw[keep], ws[keep])


# 复核用的阈值：eval/thresholds.py 在 val 段正常时刻上估计，写入 artifacts/thresholds.json 的 "review" 项
REVIEW_DEFAULTS = {
    "power_jump_mw": 0.8,         # 30 分钟内功率净变化超过它，算“功率突变”
    "anemometer_diff_ms": 0.35,   # 两个风速计最近 30 分钟平均差超过它，算“不一致”
    "turbulence_ti": 0.45,        # 风速标准差 / 风速 超过它，算“湍流偏大（阵风）”
    "pitch_deg": 4.0,             # 风速低于额定附近而桨距角超过它，说明叶片被调开（限功率迹象）
    "rpm_stopped": 1.0,           # 转速低于它视为转子基本停转
    "sync_ratio": 0.4,            # 邻机同期功率变化 ≥ 本机的这个比例，算“同步变化”
}


def review_thresholds() -> dict:
    path = ART_DIR / "thresholds.json"
    out = dict(REVIEW_DEFAULTS)
    if path.exists():
        out.update(json.loads(path.read_text(encoding="utf-8")).get("review", {}))
    return out
