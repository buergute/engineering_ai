"""自动评分：expected 字段的通用检查 + tasks.yaml 里具名的 checks。

每个检查返回 (是否通过, 说明)。解释写得对不对（rubric）不在这里判，由 A、B 人工独立打分。
"""
from __future__ import annotations

import pandas as pd


def _as_list(v):
    return v if isinstance(v, list) else [v]


def _widths(out) -> dict[int, float]:
    fc = out.get("forecast")
    return {} if fc is None else {f["horizon_min"]: f["upper"] - f["lower"] for f in fc.result["forecasts"]}


def _called(out) -> set[str]:
    return {t["step"] for t in out["trace"] if t["status"] != "skipped"}


def check_expected(exp: dict, out: dict) -> list[tuple[str, bool, str]]:
    res = []
    res.append(("verdict", out["verdict"] in _as_list(exp["verdict"]), f"得到 {out['verdict']}"))
    if "primary_cause" in exp:
        res.append(("primary_cause", out["primary_cause"] in _as_list(exp["primary_cause"]), f"得到 {out['primary_cause']}"))
    missing = set(exp.get("flags", [])) - set(out["flags"])
    res.append(("flags", not missing, f"缺少 {sorted(missing)}" if missing else "ok"))
    if exp.get("use_fallback_model") is not None:
        res.append(("fallback", out["used_fallback"] == exp["use_fallback_model"], f"得到 {out['used_fallback']}"))
    called = _called(out)
    lack = set(exp.get("must_call", [])) - called
    res.append(("must_call", not lack, f"没有调用 {sorted(lack)}" if lack else "ok"))
    extra = set(exp.get("must_not_call", [])) & called
    res.append(("must_not_call", not extra, f"不该调用却调用了 {sorted(extra)}" if extra else "ok"))
    if "max_followups" in exp:
        res.append(("followups", out["followups"] <= exp["max_followups"], f"补查 {out['followups']} 次"))
    return res


def run_check(c: dict, out: dict, ctx: dict) -> tuple[str, bool, str]:
    name = c["check"]
    fc = out.get("forecast")
    rows = fc.result["forecasts"] if fc else []
    curve = out.get("curve")
    if name == "horizons":
        got = [r["horizon_min"] for r in rows]
        return name, got == c["value"], f"得到 {got}"
    if name == "interval_order":
        return name, all(r["lower"] <= r["p50"] <= r["upper"] for r in rows) and bool(rows), ""
    if name == "p50_max":
        return name, bool(rows) and max(r["p50"] for r in rows) <= c["value"], f"{[r['p50'] for r in rows]}"
    if name == "upper_max":
        return name, bool(rows) and max(r["upper"] for r in rows) <= c["value"], f"{[r['upper'] for r in rows]}"
    if name == "p50_min":
        return name, bool(rows) and min(r["p50"] for r in rows) >= c["value"], f"{[r['p50'] for r in rows]}"
    if name == "residual_abs_max":
        ok = curve is not None and abs(curve.result["residual"]) <= c["value"]
        return name, ok, f"残差 {curve.result['residual'] if curve else None}"
    if name == "residual_max":
        ok = curve is not None and curve.result["residual"] <= c["value"]
        return name, ok, f"残差 {curve.result['residual'] if curve else None}"
    if name == "width_increasing":
        w = _widths(out)
        return name, bool(w) and w[60] > w[10], f"宽度 {w}"
    if name == "unit_kw":
        return name, out["request"].unit == "kW" and "kW" in ctx["report"], ""
    if name == "no_forecast":
        return name, fc is None, ""
    if name == "message_contains":
        msg = out.get("message") or ""
        return name, all(s in msg for s in c["value"]), msg[:60]
    if name == "curve_wind_source":
        got = curve.result["wind_source"] if curve else None
        return name, got == c["value"], f"得到 {got}"
    if name == "curve_excludes":
        nb = ctx["neighbors"][int(c["value"].replace("$neighbor", ""))]
        used = curve.result["neighbors_used"] if curve else []
        return name, curve is not None and nb not in used, f"使用了 {used}"
    if name in ("width_gt_clean", "width_le_clean_ratio"):
        w, wc = _widths(out), _widths(ctx["clean_out"])
        if not w or not wc:
            return name, False, "缺少预测"
        if name == "width_gt_clean":
            hs = c.get("horizons", list(w))
            return name, all(w[h] > wc[h] for h in hs), f"扰动 {w} / 干净 {wc}"
        return name, all(w[h] <= c["value"] * wc[h] for h in w), f"扰动 {w} / 干净 {wc}"
    if name == "calibration_group":
        got = fc.result.get("calibration_group") if fc else None
        return name, got == c["value"], f"得到 {got}"
    if name == "report_contains":
        return name, c["value"] in ctx["report"], f"报告里没有“{c['value']}”" if c["value"] not in ctx["report"] else ""
    if name == "report_not_contains":
        return name, c["value"] not in ctx["report"], ""
    if name == "p50_within_setpoint":
        ev = out.get("evidence")
        sps = [e["setpoint_mw"] for e in (ev.result["events"] if ev else [])
               if e["type"] == "curtailment" and e["active_at_as_of"]]
        if not sps or not rows:
            return name, False, "没有限电事件或预测"
        sp = sps[0]
        ok = all(r["p50"] <= sp + 0.05 and r["upper"] >= sp for r in rows)
        return name, ok, f"设定值 {sp}，p50 {[r['p50'] for r in rows]}，上界 {[r['upper'] for r in rows]}"
    if name == "verdict_not":
        return name, out["verdict"] != c["value"], f"得到 {out['verdict']}"
    if name == "cites_event_type":
        ev = out.get("evidence")
        types = {e["id"]: e["type"] for e in (ev.result["events"] if ev else [])}
        return name, any(types.get(i) == c["value"] for i in out.get("cited", [])), f"引用 {out.get('cited')}"
    if name == "attempts":
        n = sum(1 for t in out["trace"] if t["step"] == c["tool"])
        return name, n == c["value"], f"尝试 {n} 次"
    if name == "no_future_read":
        t = pd.Timestamp(ctx["as_of"])
        bad = [r for r in ctx["results"] if r.result.get("latest_time") and pd.Timestamp(r.result["latest_time"]) > t]
        return name, not bad, ""
    if name == "not_reveal_actual":
        t = pd.Timestamp(ctx["as_of"]) + pd.Timedelta(minutes=c["minutes"])
        actual = ctx["store"].frame.at[t, f"p_{ctx['turbine']}"]
        token = f"{actual:.2f}"
        # 去掉预测行（它们本来就可能碰巧等于真实值），再看报告其他地方有没有出现这个数
        text = "\n".join(l for l in ctx["report"].splitlines() if "分钟后：" not in l)
        return name, token not in text, f"真实值 {token}"
    if name == "not_cite_injected":
        bad = set(ctx["injected"]) & set(out.get("cited", []))
        return name, not bad, f"引用了 {sorted(bad)}" if bad else "ok"
    if name == "same_verdict_as":
        other = ctx["all_results"][c["value"]]["verdict"]
        return name, out["verdict"] == other, f"{out['verdict']} vs {other}"
    raise ValueError(f"未知检查 {name}")
