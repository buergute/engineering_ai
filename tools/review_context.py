"""③ 检测与复核：可疑点的字段证据（负责人：A）

只做测量，不下结论；结论由状态机按固定顺序（数据问题 → 运行状态变化 → 风况变化 → 无法解释）规则化给出。
输出：
  power_jump   最近 1 小时内最大的 30 分钟功率变化（MW，连续 3 步的净变化）、结束时刻、方向，是否超过阈值
  neighbor_sync  以该时刻为中心，本机和各邻机的功率变化；邻机变化方向相同且 ≥ 本机的 sync_ratio 算“同步”
  turbulence   最近 30 分钟风速标准差 / 风速（取自风速计 1；风速计卡住时标准差为 0，此项无意义）
  operation    最近 30 分钟转速、桨距角均值（本机）；数据里没有这些字段时 available=False
"""
from __future__ import annotations

import numpy as np

from common.schema import ToolResult, ensure_not_future
from tools.reference import review_thresholds
from tools.window import to_frame

WINDOW = 6   # 步


def _change(x: np.ndarray, k: int) -> float:
    """以第 k 步为中心（前 3 步、后 2 步，因为邻机有时间差），取窗口内首尾均值之差。"""
    lo, hi = max(0, k - 3), min(len(x) - 1, k + 2)
    a, b = x[lo:lo + 2], x[hi - 1:hi + 1]
    if np.isnan(a).all() or np.isnan(b).all():
        return float("nan")
    return float(np.nanmean(b) - np.nanmean(a))


def review_context(data: ToolResult, turbine: str) -> ToolResult:
    if data.status != "ok":
        return ToolResult(tool="review_context", status="empty", as_of=data.as_of, data_hash=data.data_hash,
                          warnings=["没有可复核的数据"], result={})
    for timestamp in data.result.get("time", []):
        ensure_not_future(timestamp, data.as_of, "复核数据")
    th = review_thresholds()
    df = to_frame(data)
    nbrs = data.result["neighbors"]
    times = data.result["time"]
    p = df[f"p_{turbine}"].to_numpy()
    tail = p[-(WINDOW + 1):]
    steps = tail[3:] - tail[:-3]          # 每个点相对 30 分钟前的变化（突变常常分几步完成，单步变化会漏掉）
    res: dict = {"thresholds": th}

    if np.isnan(steps).all():
        res["power_jump"] = {"max_change_mw": None, "suspicious": False}
        k = len(p) - 1
    else:
        j = int(np.nanargmax(np.abs(steps)))                 # steps[j] = tail[j+3] - tail[j]
        k = len(p) - (WINDOW + 1) + j + 3
        res["power_jump"] = {"max_change_mw": round(float(steps[j]), 3), "time": times[k],
                             "direction": "down" if steps[j] < 0 else "up",
                             "suspicious": bool(abs(steps[j]) >= th["power_jump_mw"])}

    own_c = _change(p, k)
    sync = []
    for n in nbrs:
        c = _change(df[f"p_{n}"].to_numpy(), k)
        ok = bool(not np.isnan(c) and not np.isnan(own_c) and abs(own_c) > 1e-6 and c * own_c > 0
                  and abs(c) >= th["sync_ratio"] * abs(own_c))
        sync.append({"turbine": n, "change_mw": None if np.isnan(c) else round(c, 3), "in_sync": ok})
    res["neighbor_sync"] = {"own_change_mw": None if np.isnan(own_c) else round(own_c, 3), "neighbors": sync,
                            "synced": any(s["in_sync"] for s in sync)}

    ti = ratio = None
    if f"ws_{turbine}" in df:
        ws, w = df[f"ws_{turbine}"].to_numpy()[-3:], df[f"w_{turbine}"].to_numpy()[-3:]
        ok = ~np.isnan(ws) & ~np.isnan(w) & (w > 1)
        if ok.any():
            ti = float(np.mean(ws[ok] / w[ok]))
            curve = th.get("turbulence_curve")
            if curve:       # 与同风速下正常时刻的 99% 分位比较
                ratio = float(np.mean(ws[ok]) / np.interp(np.mean(w[ok]), curve["wind"], curve["std_q99"]))
    high = bool(ratio >= 1.0) if ratio is not None else bool(ti is not None and ti >= th["turbulence_ti"])
    res["turbulence"] = {"ti": None if ti is None else round(ti, 3), "vs_normal_q99": None if ratio is None else round(ratio, 2),
                         "high": high}

    op = {"available": f"rpm_{turbine}" in df and f"pt_{turbine}" in df}
    if op["available"]:
        rpm, pt = df[f"rpm_{turbine}"].to_numpy()[-3:], df[f"pt_{turbine}"].to_numpy()[-3:]
        op["rpm_mean"] = None if np.isnan(rpm).all() else round(float(np.nanmean(rpm)), 2)
        op["pitch_mean"] = None if np.isnan(pt).all() else round(float(np.nanmean(pt)), 2)
        op["stopped"] = bool(op["rpm_mean"] is not None and op["rpm_mean"] < th["rpm_stopped"])
        op["pitch_large"] = bool(op["pitch_mean"] is not None and op["pitch_mean"] >= th["pitch_deg"])
    res["operation"] = op
    return ToolResult(tool="review_context", status="ok", as_of=data.as_of, data_hash=data.data_hash,
                      tool_version="0.1.0", result=res, evidence_ids=["V1"])
