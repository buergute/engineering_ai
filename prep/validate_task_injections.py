"""
验证 D01-D06 在各自真实 test 时刻的扰动注入与质量检查结果。

运行：
    python -m prep.validate_task_injections

不运行智能体，不写 outputs，不产生表 2 或表 3 数字。
"""

from __future__ import annotations

import yaml

from agent.toolbox import real_toolbox
from agent_eval.pick import Picker, pick_all
from common.store import DataStore
from perturb.inject import apply_task_injections


def quality_flags_with_turbine(quality_result: dict) -> set[str]:
    flags = set()

    for turbine, values in quality_result["quality_flags"].items():
        for value in values:
            flags.add(f"{turbine}:{value}")

    return flags


def expected_flags(task_id: str, turbine: str, store: DataStore) -> set[str]:
    neighbor0 = store.neighbors(turbine)[0]

    expected = {
        "D01": {f"{turbine}:wind_speed:frozen"},
        "D02": {
            f"{turbine}:power:gap",
            f"{turbine}:wind_speed:gap",
        },
        "D03": {f"{turbine}:wind_speed:drift"},
        "D04": {f"{turbine}:power:frozen"},
        "D05": {f"{turbine}:wind_speed:anemometer_mismatch"},
        "D06": {
            f"{turbine}:wind_speed:frozen",
            f"{neighbor0}:wind_speed:frozen",
        },
    }

    return expected[task_id]


def main() -> None:
    with open("tasks/tasks.yaml", encoding="utf-8") as file:
        tasks = yaml.safe_load(file)

    if isinstance(tasks, dict):
        tasks = tasks["tasks"]

    store = DataStore.load(with_truth=True)
    picks = pick_all(store, tasks, seed=0)
    quality_tasks = [task for task in tasks if task["id"].startswith("D")]

    print("TASK_INJECTION_VALIDATION")
    failures = []

    for task in quality_tasks:
        task_id = task["id"]
        picked = picks[task_id]
        turbine = picked["turbine"]
        as_of = picked["as_of"]

        work = store.copy()
        apply_task_injections(
            work,
            task.get("inject", []),
            turbine,
            as_of,
            seed=0,
        )

        toolbox = real_toolbox(work)
        data = toolbox.read_asof(turbine, as_of)
        quality = toolbox.check_quality(data)

        actual = quality_flags_with_turbine(quality.result)
        expected = expected_flags(task_id, turbine, store)
        missing = expected - actual

        passed = not missing
        print(
            f"{task_id}: {'PASS' if passed else 'FAIL'} | "
            f"expected={sorted(expected)} | actual={sorted(actual)}"
        )

        if not passed:
            failures.append((task_id, sorted(missing)))

    print()
    print("CLEAN_FALSE_ALERT_CHECK")

    picker = Picker(store)
    toolbox = real_toolbox(store)
    seen = set()
    checked = 0
    drift_alerts = 0
    mismatch_alerts = 0

    for seed in range(1000, 3000):
        if checked >= 50:
            break

        hit = picker.clean(
            {
                "turbine": "T03",
                "wind": [7, 10],
                "power_min": 0.5,
                "clean_hours_before": 24,
            },
            seed=seed,
        )

        if hit is None or hit in seen:
            continue

        seen.add(hit)
        turbine, as_of = hit

        data = toolbox.read_asof(turbine, as_of)
        quality = toolbox.check_quality(data)
        flags = set(quality.result["quality_flags"][turbine])

        drift_alerts += "wind_speed:drift" in flags
        mismatch_alerts += "wind_speed:anemometer_mismatch" in flags
        checked += 1

    print("clean_cases_checked =", checked)
    print("clean_drift_alerts =", drift_alerts)
    print("clean_mismatch_alerts =", mismatch_alerts)

    if checked < 50:
        failures.append(("clean_sample", [f"only {checked} clean cases found"]))

    if drift_alerts != 0:
        failures.append(("clean_false_alert_check", ["drift was alerted on clean data"]))

    if failures:
        raise SystemExit(f"VALIDATION_FAILED: {failures}")

    print("ALL_TASK_INJECTIONS_PASS")


if __name__ == "__main__":
    main()