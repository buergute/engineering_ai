"""③ 数据质量检查（负责人：A）

对本机和邻机逐一检查：
  gap             末尾或窗口内连续缺失 ≥ 30 分钟
  random_missing  零星缺失比例 > 5%，但没有长缺口
  frozen          末尾连续 ≥ 6 个点（1 小时）读数完全相同；功率接近 0 时不算（停机本来就是平的）
  out_of_range    风速 < 0 或 > 40 m/s；功率 > 1.1 倍额定或 < -0.2 MW
  anemometer_mismatch  风速计 1、2 最近 30 分钟平均差超过 val 段正常时刻的 99.5% 分位
  drift           （仅本机）风速与邻机的比例偏离训练期 > 8%，且比按功率反推的等效风速多偏 8% 以上
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.schema import ToolResult
from tools.reference import implied_wind, load_reference, review_thresholds
from tools.window import frozen_tail, tail_run, to_frame

FROZEN_STEPS = 6
GAP_STEPS = 3
DRIFT_TOL = 0.08


def _longest_nan_run(x: np.ndarray) -> int:
    best = cur = 0
    for v in np.isnan(x):
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def check_quality(data: ToolResult, rated_mw: float = 2.05) -> ToolResult:
    if data.status != "ok":
        return ToolResult(tool="check_quality", status="empty", as_of=data.as_of, data_hash=data.data_hash,
                          result={"issues": []}, warnings=["没有可检查的数据"])
    df = to_frame(data)
    own, nbrs = data.result["turbine"], data.result["neighbors"]
    times = data.result["time"]
    issues = []

    def add(turbine, field, kind, steps, detail):
        start = times[-steps] if 0 < steps <= len(times) else times[0]
        issues.append({"id": f"Q{len(issues) + 1}", "turbine": turbine, "field": field, "type": kind,
                       "start": start, "end": times[-1], "steps": int(steps), "detail": detail})

    for t in [own, *nbrs]:
        for field, col in (("power", f"p_{t}"), ("wind_speed", f"w_{t}")):
            x = df[col].to_numpy()
            tail_gap = tail_run(x, np.isnan)
            longest = _longest_nan_run(x)
            miss = float(np.isnan(x).mean())
            if tail_gap >= GAP_STEPS:
                add(t, field, "gap", tail_gap, f"截至起报已连续缺失 {tail_gap * 10} 分钟")
            elif longest >= GAP_STEPS:
                add(t, field, "gap", longest, f"窗口内出现 {longest * 10} 分钟的连续缺失")
            elif miss > 0.05:
                add(t, field, "random_missing", 1, f"窗口内零星缺失 {miss:.0%}")
            fz = frozen_tail(x)
            last = x[~np.isnan(x)][-1] if (~np.isnan(x)).any() else np.nan
            if fz >= FROZEN_STEPS and not (field == "power" and abs(last) < 0.05):
                add(t, field, "frozen", fz, f"读数连续 {fz} 个点保持 {last:g}，持续 {fz * 10} 分钟")
            if field == "wind_speed" and f"w2_{t}" in df:
                w2 = df[f"w2_{t}"].to_numpy()
                d = np.abs(x[-3:] - w2[-3:])
                if (~np.isnan(d)).any() and np.nanmean(d) > review_thresholds()["anemometer_diff_ms"]:
                    add(t, field, "anemometer_mismatch", 3,
                        f"风速计 1 与 2 最近 30 分钟平均相差 {np.nanmean(d):.2f} m/s（正常波动 < {review_thresholds()['anemometer_diff_ms']:.2f}）")
            bad = (x < 0) | (x > 40) if field == "wind_speed" else (x > 1.1 * rated_mw) | (x < -0.2)
            if np.nansum(bad) > 0:
                add(t, field, "out_of_range", 1, f"有 {int(np.nansum(bad))} 个读数超出合理范围")

    # 漂移：本机实测风速相对邻机的比例偏离训练期，但用本机功率沿功率曲线反推的“等效风速”并不偏。
    # 说明机组实际吃到的风和邻机一致，偏的是风速计本身。用比例而不是功率差，避开曲线陡峭段的放大效应。
    R = load_reference()
    ref = R["wind_ratio"][own]
    unreliable = {i["turbine"] for i in issues if i["field"] == "wind_speed" and i["type"] in ("frozen", "gap")}
    w_own, p_own = df[f"w_{own}"], df[f"p_{own}"]
    implied = pd.Series(implied_wind(own, p_own.to_numpy()), index=df.index)
    dev_w, dev_p = [], []
    for n in nbrs:
        if n in unreliable:
            continue
        w_n = df[f"w_{n}"]
        ok = (w_own > 4) & (w_n > 4) & (p_own > 0.1) & (p_own < 0.93 * rated_mw)
        if ok.sum() >= 12:
            dev_w.append(float((w_own[ok] / w_n[ok]).median() / ref[n] - 1))
            dev_p.append(float((implied[ok] / w_n[ok]).median() / ref[n] - 1))
    gap = [w - p for w, p in zip(dev_w, dev_p)]
    if own not in unreliable and dev_w and min(abs(d) for d in dev_w) > DRIFT_TOL \
            and len({np.sign(d) for d in dev_w}) == 1 and min(abs(g) for g in gap) > DRIFT_TOL \
            and max(abs(d) for d in dev_p) < 0.1:
        add(own, "wind_speed", "drift", 36,
            f"风速比邻机系统性偏{'高' if dev_w[0] > 0 else '低'}约 {abs(np.mean(dev_w)):.0%}，"
            f"但按功率反推的等效风速与邻机一致（偏差 {abs(np.mean(dev_p)):.0%}），疑似风速计漂移")

    flags = {t: sorted({f"{i['field']}:{i['type']}" for i in issues if i["turbine"] == t}) for t in [own, *nbrs]}
    return ToolResult(tool="check_quality", status="ok", as_of=data.as_of, data_hash=data.data_hash,
                      tool_version="0.1.0", result={"issues": issues, "quality_flags": flags},
                      evidence_ids=[i["id"] for i in issues])
