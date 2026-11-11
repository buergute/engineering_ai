"""智能体评测（负责人：C）

    python -m agent_eval.run_tasks                 # 跑全部 24 题
    python -m agent_eval.run_tasks --only C01 D03  # 只跑部分题目
    python -m agent_eval.run_tasks --show C01      # 同时打印某题的完整报告

流程：挑起报时刻 → 复制数据并注入扰动 → 包装工具故障 → 运行智能体 → 自动评分。
结果写到 outputs/task_results.md 和 outputs/task_results.json。

TODO(C)：加上“固定流水线”“去掉校验”两组对照，与完整系统用同一套题、同样的数据和工具。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from agent.report import render_report
from agent.state_machine import Request, run
from agent.toolbox import Toolbox, real_toolbox
from agent_eval.checks import check_expected, run_check
from agent_eval.pick import pick_all
from common.schema import ToolResult
from common.store import DataStore
from perturb.inject import apply_task_injections

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"


def with_faults(tools: Toolbox, faults: list[dict]) -> Toolbox:
    for f in faults or []:
        name, mode = f["tool"], f["mode"]
        orig = getattr(tools, name)
        if mode == "fail_always":
            def fn(*a, _n=name, **k):
                raise RuntimeError(f"模拟故障：{_n} 无响应")
        elif mode == "fail_once":
            state = {"n": 0}

            def fn(*a, _o=orig, _s=state, _n=name, **k):
                _s["n"] += 1
                if _s["n"] == 1:
                    raise RuntimeError(f"模拟故障：{_n} 第一次调用失败")
                return _o(*a, **k)
        elif mode == "empty":
            def fn(turbine, as_of, *a, _n=name, **k):
                return ToolResult(tool=_n, status="empty", as_of=as_of, warnings=[f"{turbine} 在窗口内没有任何有效数据"])
        else:
            raise ValueError(mode)
        setattr(tools, name, fn)
    return tools


def run_task(task: dict, pick: dict, store: DataStore, all_results: dict) -> dict:
    turbine, as_of = pick["turbine"], pick["as_of"]
    text = task["request"].format(turbine=turbine)
    work = store.copy()
    injected = apply_task_injections(work, task.get("inject"), turbine, as_of) if turbine and as_of else []
    out = run(Request(text, as_of), with_faults(real_toolbox(work), task.get("tool_faults")))
    clean_out = run(Request(text, as_of), real_toolbox(store)) if task.get("paired_clean") else None
    report = render_report(out)

    ctx = {"report": report, "clean_out": clean_out, "as_of": as_of, "turbine": turbine, "store": store,
           "injected": injected, "all_results": all_results,
           "neighbors": store.neighbors(turbine) if turbine in store.turbines else [],
           "results": [r for r in [out.get("forecast"), out.get("curve"), out.get("evidence")] if r]}
    checks = check_expected(task["expected"], out) + [run_check(c, out, ctx) for c in task.get("checks", [])]
    failed = [f"{n}: {d}" for n, ok, d in checks if not ok]
    return {"id": task["id"], "category": task["category"], "title": task["title"], "turbine": turbine,
            "as_of": as_of, "request": text, "verdict": out["verdict"], "primary_cause": out["primary_cause"],
            "flags": out["flags"], "passed": not failed, "failed_checks": failed,
            "n_checks": len(checks), "report": report}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(ROOT / "tasks" / "tasks.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--show", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0, help="换一个种子就换一批起报时刻")
    args = ap.parse_args()

    tasks = yaml.safe_load(open(args.tasks, encoding="utf-8"))
    store = DataStore.load(with_truth=True)
    picks = pick_all(store, tasks, args.seed)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "picked.json").write_text(json.dumps(picks, ensure_ascii=False, indent=1), encoding="utf-8")

    store.truth = None                       # 挑完时刻后立即收回真值表，运行智能体时绝对看不到
    results: dict[str, dict] = {}
    for task in tasks:
        if args.only and task["id"] not in args.only and not any(
                task["id"] == t.get("pick_rule", {}).get("id") for t in tasks if t["id"] in args.only):
            continue
        r = run_task(task, picks[task["id"]], store, results)
        results[task["id"]] = r
        mark = "通过" if r["passed"] else "未通过"
        print(f"{r['id']:4s} {mark:4s} {r['verdict']:18s} {r['title']}"
              + ("" if r["passed"] else f"\n       ↳ {'; '.join(r['failed_checks'])}"))
        if r["id"] in args.show:
            print("\n" + r["report"] + "\n")

    shown = [r for r in results.values() if not args.only or r["id"] in args.only]
    by_cat = Counter(r["category"] for r in shown)
    ok_cat = Counter(r["category"] for r in shown if r["passed"])
    print("\n按题型：" + "，".join(f"{c} {ok_cat[c]}/{n}" for c, n in by_cat.items()))
    print(f"合计通过 {sum(ok_cat.values())}/{len(shown)}")

    (OUT_DIR / "task_results.json").write_text(json.dumps(shown, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# 智能体评测结果（合成数据）", "", "| 题号 | 题型 | 机组 | 起报时刻 | 结论 | 通过 | 未通过的检查 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r['id']} | {r['category']} | {r['turbine'] or '-'} | {r['as_of'] or '-'} | {r['verdict']} | "
           f"{'是' if r['passed'] else '否'} | {'; '.join(r['failed_checks'])} |" for r in shown]
    md += ["", f"合计通过 {sum(ok_cat.values())}/{len(shown)}", "", "## 各题报告", ""]
    md += [f"### {r['id']} {r['title']}\n\n请求：{r['request']}\n\n```\n{r['report']}\n```\n" for r in shown]
    (OUT_DIR / "task_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
