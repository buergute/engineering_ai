"""⑦ 预测工具（负责人：B）

质量感知 LightGBM 分位数模型（10% / 50% / 90%）+ 分组 CQR 校准，给出 80% 区间。
use_fallback=True 时换用不含本机风速的后备模型。
物理约束：输出裁剪到 [-0.05, 额定功率]（额定边界是机理约束，不是调参）。
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from common.schema import ToolResult
from models.features import build_features, is_degraded
from tools.window import to_frame

ART_DIR = Path(__file__).resolve().parents[1] / "artifacts"


@lru_cache(maxsize=1)
def load_models() -> dict:
    path = ART_DIR / "models.pkl"
    if not path.exists():
        raise FileNotFoundError("找不到 artifacts/models.pkl，请先运行 python -m models.train")
    with open(path, "rb") as f:
        return pickle.load(f)


def forecast_power(data: ToolResult, quality: ToolResult, turbine: str,
                   horizons_minutes: tuple[int, ...] = (10, 30, 60), use_fallback: bool = False,
                   exclude_neighbors: tuple[str, ...] = (), rated_mw: float = 2.05,
                   cap_mw: float | None = None) -> ToolResult:
    """cap_mw：已确认且仍在持续的限功率设定值。模型本身不知道限电，所以中位数和下界按“限电持续”压到设定值以内；
    上界保持不变，因为限电结束时间未知、随时可能解除。"""
    b = load_models()
    kind = "fallback" if use_fallback else "quality_aware"
    df = to_frame(data)
    for n in exclude_neighbors:                     # 不可信的邻机当作缺失
        df[f"p_{n}"] = np.nan
        df[f"w_{n}"] = np.nan
    X = build_features(df, turbine, data.result["neighbors"], b["turbines"].index(turbine), kind)
    x = X.iloc[[-1]][b["features"][kind]]
    group = "degraded" if bool(is_degraded(x).iloc[0]) or exclude_neighbors else "clean"

    rows = []
    for h in horizons_minutes:
        lo, mid, hi = (float(b["models"][(kind, h, q)].predict(x)[0]) for q in b["quantiles"])
        pad = b["cqr"][(kind, h, group)]
        lo, hi = lo - pad, hi + pad
        lo, mid, hi = np.clip(sorted([lo, mid, hi]), -0.05, rated_mw)
        if cap_mw is not None:
            hi = max(hi, cap_mw)
            mid, lo = min(mid, cap_mw), min(lo, cap_mw)
        rows.append({"horizon_min": h, "p50": round(float(mid), 3), "lower": round(float(lo), 3),
                     "upper": round(float(hi), 3), "coverage": b["coverage"]})
    return ToolResult(tool="forecast_power", status="ok", as_of=data.as_of, data_hash=data.data_hash,
                      tool_version=b["version"], evidence_ids=["F1"],
                      result={"model": kind, "calibration_group": group, "forecasts": rows,
                              "excluded_neighbors": list(exclude_neighbors), "curtailment_cap_mw": cap_mw})
