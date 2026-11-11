"""结果表 2：检测与复核评测（负责人：A 出题、C 运行）

    python -m eval.review_eval [--n 60] [--seed 0]

在 test 段按真值挑出六类时刻，各取 n 个，让智能体复核，统计“结论对不对”和“报没报警”：
  normal      数据正常、没有任何事件的随机时刻                      应判 normal，误报 = 判成了别的
  wind_change 阵风、全场风速突变（功率变化大，但只是风变了）           应判 normal（风况变化），误报 = 判成了别的
  data_issue  人工注入：风速冻结、功率冻结、连续掉线、两个风速计矛盾、风速计缓慢漂移   应判 data_issue_only
  operating   日志里的限电、停机；以及“事后才补录”的限电（起报时只能靠桨距角）        应判 operating_change
  unexplained 没有任何记录和迹象的欠发                               应判 undetermined（无法解释），给出原因不算对

为了快，预测工具换成空壳（本表只看复核结论，预测精度见结果表 1）。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from agent.state_machine import Request, run
from agent.toolbox import real_toolbox
from agent_eval.pick import Picker
from common.schema import ToolResult
from common.store import DataStore
from perturb.inject import apply_task_injections

OUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
REQUEST = "{t} 的功率有没有问题？"          # 不含“为什么”，不诱导补查；是否补查由系统自己决定
WANT = {"normal": ("normal",), "wind_change": ("normal",), "data_issue": ("data_issue_only",),
        "operating": ("operating_change",), "unexplained": ("undetermined",)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def fast_toolbox(store):
    tb = real_toolbox(store)

    def fake_forecast(data, quality, turbine, horizons_minutes=(10, 30, 60), **kw):
        return ToolResult(tool="forecast_power", status="ok", as_of=data.as_of, data_hash=data.data_hash,
                          result={"forecasts": [], "model": "stub"})
    tb.forecast_power = fake_forecast
    return tb


def sample_clean(pk: Picker, rng, n, wind=None, ptp=None, quiet_steps=144):
    out, seen = [], set()
    t0, t1 = pk.t0, pk.t1
    guard = 0
    while len(out) < n and guard < 200000:
        guard += 1
        t = t0 + (t1 - t0) * float(rng.random())
        t = t.floor("10min")
        tb = pk.s.turbines[int(rng.integers(6))]
        i = pk._i(t)
        group = [tb, *pk.s.neighbors(tb)]
        if (tb, t) in seen or not pk._complete(group, i, 48, 0) or not pk._quiet(group, i, quiet_steps, 0):
            continue
        w = float(np.mean(pk.W[tb][i - 2:i + 1]))
        if wind and not wind[0] <= w <= wind[1]:
            continue
        if ptp and np.ptp(pk.W[tb][i - 12:i + 1]) < ptp:
            continue
        seen.add((tb, t))
        out.append((tb, t.isoformat(timespec="minutes")))
    return out


def sample_events(pick_fn, rule, n, base_seed):
    out, seen = [], set()
    for s in range(base_seed, base_seed + 20 * n):
        if len(out) >= n:
            break
        hit = pick_fn(rule, s)
        if hit and hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n, rng = args.n, np.random.default_rng(args.seed)

    store = DataStore.load(with_truth=True)
    pk = Picker(store)
    cases: list[dict] = []

    def add(kind, sub, picks, inject=None):
        cases.extend({"cls": kind, "sub": sub, "turbine": tb, "as_of": t, "inject": inject} for tb, t in picks)

    add("normal", "随机正常时刻", sample_clean(pk, rng, 5 * n))
    add("wind_change", "单机阵风",
        sample_events(pk.wind_event, {"event": "gust", "direction": "up", "min_jump_mw": 1.1, "wind": [3.5, 12]}, n, args.seed * 1000))
    add("wind_change", "全场风速突降",
        sample_events(pk.wind_event, {"event": "wind_ramp", "direction": "down", "min_jump_mw": 1.1, "wind": [3.5, 13]}, n, args.seed * 1000))
    add("wind_change", "全场风速突升",
        sample_events(pk.wind_event, {"event": "wind_ramp", "direction": "up", "min_jump_mw": 1.1, "wind": [3.5, 13]}, n, args.seed * 1000))
    inj = {
        "风速计冻结": ([{"type": "freeze", "field": "wind_speed", "duration_min": 80}], dict(wind=(5, 11), ptp=1.5)),
        "功率冻结": ([{"type": "freeze", "field": "power", "duration_min": 60}], dict(wind=(5, 11))),
        "连续掉线": ([{"type": "contiguous_dropout", "fields": "all", "duration_min": 120}], dict(wind=(5, 11))),
        "两个风速计矛盾": ([{"type": "anemometer_mismatch", "factor": 0.25, "duration_min": 60}], dict(wind=(5, 11))),
        "风速计缓慢漂移": ([{"type": "slow_drift", "field": "wind_speed", "drift_per_day": 0.5, "days": 3}],
                     dict(wind=(6, 10), quiet_steps=76 * 6)),
    }
    for sub, (spec, kw) in inj.items():
        add("data_issue", sub, sample_clean(pk, rng, n, **kw), spec)
    add("operating", "日志里的限电",
        sample_events(pk.curtailment_active, {"min_elapsed_min": 90, "min_remaining_min": 60, "deficit_min": 0.5, "jitter_min": 240}, n, args.seed * 1000))
    add("operating", "日志里的停机",
        sample_events(pk.stop_active, {"min_elapsed_min": 40, "min_remaining_min": 30, "deficit_min": 0.5, "jitter_min": 240}, n, args.seed * 1000))
    add("operating", "限电日志事后才补录",
        sample_events(pk.curtailment_active, {"min_elapsed_min": 90, "min_remaining_min": 60, "deficit_min": 0.5, "jitter_min": 240}, n, args.seed * 1000 + 500),
        [{"type": "hide_active_event", "event_type": "curtailment", "available_after_as_of_min": 80}])
    add("unexplained", "无记录的欠发",
        sample_events(pk.hidden_derate, {"min_elapsed_min": 60, "deficit_min": 0.25, "jitter_min": 240}, n, args.seed * 1000))

    store.truth = None
    rows = []
    for k, c in enumerate(cases):
        work = store.copy()
        apply_task_injections(work, c["inject"], c["turbine"], c["as_of"], seed=k)
        out = run(Request(REQUEST.format(t=c["turbine"]), c["as_of"]), fast_toolbox(work))
        rows.append({**{x: c[x] for x in ("cls", "sub", "turbine", "as_of")}, "verdict": out["verdict"],
                     "cause": out["primary_cause"], "flags": out["flags"]})
    df = pd.DataFrame(rows)

    lines = ["# 结果表 2：检测与复核（合成数据，仅说明流程可用）", "",
             "“结论正确”= 结论与真值一致；“报警”= 结论不是 normal。95% 区间为 Wilson 区间。", "",
             "| 类别 | 子类 | 题数 | 结论正确 | 报警 | 错判成 |", "| --- | --- | --- | --- | --- | --- |"]
    summary = defaultdict(lambda: [0, 0, 0])
    for (cls, sub), g in df.groupby(["cls", "sub"], sort=False):
        ok = int(g.verdict.isin(WANT[cls]).sum())
        alarm = int((g.verdict != "normal").sum())
        wrong = g[~g.verdict.isin(WANT[cls])].verdict.value_counts().to_dict()
        lines.append(f"| {cls} | {sub} | {len(g)} | {ok / len(g):.0%} | {alarm / len(g):.0%} | {wrong or '-'} |")
        s = summary[cls]
        s[0] += len(g); s[1] += ok; s[2] += alarm
    lines += ["", "## 按类别汇总", "", "| 类别 | 题数 | 结论正确率（95% 区间） | 报警率 |", "| --- | --- | --- | --- |"]
    for cls, (m, ok, alarm) in summary.items():
        lo, hi = wilson(ok, m)
        lines.append(f"| {cls} | {m} | {ok / m:.0%}（{lo:.0%}–{hi:.0%}） | {alarm / m:.0%} |")
    lines += ["", "说明：normal 和 wind_change 的“报警率”就是误报率；其余类的“报警率”是检出率。"]
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "review_results.md").write_text("\n".join(lines), encoding="utf-8")
    df.to_json(OUT_DIR / "review_results.json", orient="records", force_ascii=False, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
