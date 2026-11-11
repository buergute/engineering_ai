"""⑤ 功率曲线复核（负责人：A）

用 train 段拟合的经验曲线，比较“这个风速下应该发多少”和“实际发了多少”（取最近 30 / 60 分钟平均）。
wind_source="neighbors"：本机风速不可信时，用邻机风速 × 训练期两两风速比例来推算本机风速。
输出的是“异常证据”，不是故障结论。
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from common.schema import ToolResult
from tools.reference import expected_power, load_reference
from tools.window import to_frame

# 取最近多少个点做平均：本机风速 30 分钟；邻机推算 60 分钟（机组之间有时间差，平均更久误差更小）
LAST_STEPS = {"own": 3, "neighbors": 6}


def check_power_curve(data: ToolResult, turbine: str, wind_source: Literal["own", "neighbors"] = "own",
                      exclude_neighbors: tuple[str, ...] = ()) -> ToolResult:
    ref = load_reference()
    df = to_frame(data).iloc[-LAST_STEPS[wind_source]:]
    p_own = df[f"p_{turbine}"]
    used = []
    if wind_source == "own":
        w_est = df[f"w_{turbine}"]
    else:
        ests = []
        for n in data.result["neighbors"]:
            if n in exclude_neighbors or df[f"w_{n}"].isna().all():
                continue
            ests.append(df[f"w_{n}"] * ref["wind_ratio"][turbine][n])
            used.append(n)
        w_est = pd.concat(ests, axis=1).mean(axis=1) if ests else pd.Series(np.nan, index=df.index)
    both = p_own.notna() & w_est.notna()          # 只比较风速和功率同时有值的时刻
    wind = float(w_est[both].mean()) if both.any() else float("nan")
    actual = float(p_own[both].mean()) if both.any() else float("nan")

    if np.isnan(wind) or np.isnan(actual):
        return ToolResult(tool="check_power_curve", status="empty", as_of=data.as_of, data_hash=data.data_hash,
                          result={"wind_source": wind_source, "neighbors_used": used},
                          warnings=["缺少可用的风速或功率，无法复核"])
    exp = expected_power(turbine, wind)
    return ToolResult(
        tool="check_power_curve", status="ok", as_of=data.as_of, data_hash=data.data_hash, tool_version="0.1.0",
        evidence_ids=["C1"],
        result={"wind_source": wind_source, "neighbors_used": used, "wind_speed_used": round(wind, 2),
                "expected_power": round(exp, 3), "actual_power": round(actual, 3),
                "residual": round(actual - exp, 3), "curve_source": ref["curve_source"],
                "valid_domain": "正常运行工况；不适用于限电、停机期间"},
    )
