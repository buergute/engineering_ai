"""智能体评测（负责人：C，截止 11/22）

    python -m agent_eval.run_tasks tasks/tasks.example.yaml

TODO(C)：
  - 按题目的 inject / simulate_tool_failure 构造工具（真实工具 + 扰动）
  - 三组对照：完整系统 / 固定流水线 / 去掉校验；三组拿到的数据和工具必须相同
  - 记录调用次数、时延；解释正确性留给 A、B 人工打分
"""
from __future__ import annotations

import sys

import yaml

from agent.mock_tools import mock_toolbox
from agent.state_machine import Request, run


def score_auto(task: dict, out: dict) -> dict:
    """只评能自动判的部分：该调用的工具是否调用了、是否用了后备模型、结论类别。"""
    exp = task["expected"]
    called = {t["step"] for t in out["trace"] if t["status"] != "skipped"}
    return {
        "id": task["id"],
        "calls_ok": set(exp.get("must_call", [])) <= called,
        "fallback_ok": exp.get("use_fallback_model") in (None, out["used_fallback"]),
        "verdict_ok": out["diagnosis"]["cause"]["verdict"] in exp.get("allowed_verdicts", []),
        "validated": out["validation"]["ok"],
    }


def main(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        tasks = yaml.safe_load(f)
    for task in tasks:
        out = run(Request(task["turbine"], task["as_of"]), mock_toolbox("curtailment"))
        print(score_auto(task, out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tasks/tasks.example.yaml")
