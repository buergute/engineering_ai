"""
在真实 Kelmarsh 宽表上验证扰动注入可被 check_quality 检出。

运行：
    python -m prep.validate_injections

只在 DataStore.copy() 的内存副本上注入，不写 SCADA、truth.csv 或评测结果。
"""

from __future__ import annotations

from agent.toolbox import real_toolbox
from agent_eval.pick import Picker
from common.store import DataStore
from perturb.inject import apply_task_injections


SCENARIOS = {
    "wind_freeze": {
        "inject": [
            {"type": "freeze", "field": "wind_speed", "duration_min": 80},
        ],
        "expected": {"wind_speed:frozen"},
    },
    "power_freeze": {
        "inject": [
            {"type": "freeze", "field": "power", "duration_min": 60},
        ],
        "expected": {"power:frozen"},
    },
    "contiguous_dropout": {
        "inject": [
            {"type": "contiguous_dropout", "fields": "all", "duration_min": 120},
        ],
        "expected": {"power:gap", "wind_speed:gap"},
    },
    "anemometer_mismatch": {
        "inject": [
            {
                "type": "anemometer_mismatch",
                "factor": -0.25,
                "duration_min": 60,
            },
        ],
        "expected": {"wind_speed:anemometer_mismatch"},
    },
    "slow_drift": {
        "inject": [
            {
                "type": "slow_drift",
                "field": "wind_speed",
                "drift_per_day": 0.5,
                "days": 3,
            },
        ],
        "expected": {"wind_speed:drift"},
    },
}


def main() -> None:
    store = DataStore.load(with_truth=True)
    picker = Picker(store)

    # 选择 test 段中干净、风速足够、功率非零的真实时刻。
    hit = picker.clean(
        {
            "turbine": "T03",
            "wind": [7, 10],
            "power_min": 0.5,
            "clean_hours_before": 24,
        },
        seed=0,
    )

    if hit is None:
        raise RuntimeError("找不到适合进行真实数据扰动验证的干净时刻")

    turbine, as_of = hit
    print(f"base_case = {turbine} @ {as_of}")

    failures = []

    for name, scenario in SCENARIOS.items():
        work = store.copy()
        apply_task_injections(
            work,
            scenario["inject"],
            turbine,
            as_of,
            seed=0,
        )

        toolbox = real_toolbox(work)
        data = toolbox.read_asof(turbine, as_of)
        quality = toolbox.check_quality(data)

        actual = set(quality.result["quality_flags"][turbine])
        expected = scenario["expected"]
        missing = expected - actual
        passed = not missing

        print(
            f"{name}: {'PASS' if passed else 'FAIL'} | "
            f"expected={sorted(expected)} | actual={sorted(actual)}"
        )

        if not passed:
            failures.append((name, sorted(missing)))

    if failures:
        raise SystemExit(f"Injection validation failed: {failures}")

    print("ALL_SCENARIOS_PASS")


if __name__ == "__main__":
    main()