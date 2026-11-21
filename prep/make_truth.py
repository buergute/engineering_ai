"""
Kelmarsh 真实数据真值候选生成。

默认仅 dry run，打印候选事件，不写文件：
    python -m prep.make_truth

人工确认后才写 data/kelmarsh/truth.csv：
    python -m prep.make_truth --write
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_split
from tools.reference import expected_power

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "kelmarsh"

TURBINES = [f"T{i:02d}" for i in range(1, 7)]
NEIGHBORS = {
    "T01": ["T02"],
    "T02": ["T01", "T03"],
    "T03": ["T02", "T04"],
    "T04": ["T03", "T05"],
    "T05": ["T04", "T06"],
    "T06": ["T05"],
}

STEP = pd.Timedelta(minutes=10)
EVENT_GUARD = pd.Timedelta(minutes=60)
WINDOW_STEPS = 3  # 30 分钟

# 单机阵风：本机功率/风速明显同向变化，邻机风速基本稳定。
GUST_POWER_JUMP_MW = 1.10
GUST_WIND_JUMP_MS = 0.40
GUST_NEIGHBOR_WIND_MAX_MS = 0.50

# 全场风速突变：门槛高于此前版本，避免把日常风况变化全纳入。
RAMP_WIND_JUMP_MS = 1.50
RAMP_POWER_JUMP_MW = 0.80
RAMP_MIN_TURBINES = 5
RAMP_POWER_MIN_TURBINES = 4

# 功率曲线欠发候选。
DERATE_WIND_MIN_MS = 6.0
DERATE_WIND_MAX_MS = 11.0
DERATE_MIN_DEFICIT_MW = 0.35
DERATE_NEIGHBOR_RESIDUAL_MAX_MW = 0.20

# “无日志停机”要同时具备两类正面字段证据。
STOP_PITCH_MIN_DEG = 60.0
STOP_RPM_MAX = 1.0

# 真正的 hidden_derate 必须没有停机/羽化证据。
HIDDEN_PITCH_MAX_DEG = 5.0
HIDDEN_RPM_MIN = 1.0
HIDDEN_MIN_DURATION = pd.Timedelta(minutes=120)


def build_busy_masks(index: pd.DatetimeIndex, events: pd.DataFrame) -> dict[str, pd.Series]:
    """标出日志事件及其前后 60 分钟。"""
    busy = {tb: pd.Series(False, index=index) for tb in TURBINES}

    for event in events.itertuples(index=False):
        targets = TURBINES if event.turbine == "ALL" else [event.turbine]
        within = (
            (index >= event.start - EVENT_GUARD)
            & (index <= event.end + EVENT_GUARD)
        )
        for tb in targets:
            busy[tb].loc[within] = True

    return busy


def complete_window(power: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """当前点及 30 分钟前的功率、风速都必须存在。"""
    now_ok = power.notna() & wind.notna()
    return now_ok & now_ok.shift(WINDOW_STEPS).fillna(False).astype(bool)


def group_is_safe(
    turbines: list[str],
    complete: pd.DataFrame,
    busy: dict[str, pd.Series],
) -> pd.Series:
    """目标机组及邻机完整，且均没有记录在案的运行事件。"""
    data_ok = complete[turbines].all(axis=1)
    no_logged_event = ~pd.concat([busy[tb] for tb in turbines], axis=1).any(axis=1)
    return data_ok & no_logged_event


def mask_to_episodes(
    mask: pd.Series,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    *,
    pre_minutes: int = 0,
    min_duration: pd.Timedelta = pd.Timedelta(0),
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    连续候选点合并为事件段，并进一步合并彼此重叠或相接的时间窗。
    事件严格裁剪在当前 split 内。
    """
    times = mask.index[mask.to_numpy()]
    if len(times) == 0:
        return []

    cuts = np.where(np.diff(times.asi8) > 2 * STEP.value)[0] + 1
    groups = np.split(times, cuts)

    raw = []
    for group in groups:
        start = max(group[0] - pd.Timedelta(minutes=pre_minutes), split_start)
        end = min(group[-1], split_end)
        raw.append((start, end))

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for start, end in raw:
        if merged and start <= merged[-1][1] + STEP:
            old_start, old_end = merged[-1]
            merged[-1] = (old_start, max(old_end, end))
        else:
            merged.append((start, end))

    return [
        (start, end)
        for start, end in merged
        if end - start >= min_duration
    ]


def curve_residuals(power: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """残差 = 经验功率曲线期望功率 - 实际功率；正值代表欠发。"""
    residual = pd.DataFrame(index=power.index)

    for tb in TURBINES:
        expected = [
            expected_power(tb, value) if pd.notna(value) else np.nan
            for value in wind[tb]
        ]
        residual[tb] = pd.Series(expected, index=power.index) - power[tb]

    return residual


def find_gusts(
    power: pd.DataFrame,
    wind: pd.DataFrame,
    complete: pd.DataFrame,
    busy: dict[str, pd.Series],
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    split_name: str,
) -> list[dict]:
    dp = power.diff(WINDOW_STEPS)
    dw = wind.diff(WINDOW_STEPS)
    rows = []

    for tb in TURBINES:
        neighbors = NEIGHBORS[tb]
        candidate = (
            (dp[tb].abs() >= GUST_POWER_JUMP_MW)
            & (dw[tb].abs() >= GUST_WIND_JUMP_MS)
            & ((dp[tb] * dw[tb]) > 0)
            & (dw[neighbors].abs().median(axis=1) <= GUST_NEIGHBOR_WIND_MAX_MS)
            & group_is_safe([tb, *neighbors], complete, busy)
        )

        for start, end in mask_to_episodes(
            candidate, split_start, split_end, pre_minutes=30
        ):
            rows.append(
                {
                    "split": split_name,
                    "turbine": tb,
                    "start": start,
                    "end": end,
                    "type": "gust",
                }
            )

    return rows


def find_wind_ramps(
    power: pd.DataFrame,
    wind: pd.DataFrame,
    complete: pd.DataFrame,
    busy: dict[str, pd.Series],
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    split_name: str,
) -> list[dict]:
    dp = power.diff(WINDOW_STEPS)
    dw = wind.diff(WINDOW_STEPS)
    all_safe = group_is_safe(TURBINES, complete, busy)

    ramp_up = (
        ((dw >= RAMP_WIND_JUMP_MS).sum(axis=1) >= RAMP_MIN_TURBINES)
        & ((dp >= RAMP_POWER_JUMP_MW).sum(axis=1) >= RAMP_POWER_MIN_TURBINES)
        & all_safe
    )
    ramp_down = (
        ((dw <= -RAMP_WIND_JUMP_MS).sum(axis=1) >= RAMP_MIN_TURBINES)
        & ((dp <= -RAMP_POWER_JUMP_MW).sum(axis=1) >= RAMP_POWER_MIN_TURBINES)
        & all_safe
    )

    rows = []
    for candidate in (ramp_up, ramp_down):
        for start, end in mask_to_episodes(
            candidate, split_start, split_end, pre_minutes=30
        ):
            rows.append(
                {
                    "split": split_name,
                    "turbine": "ALL",
                    "start": start,
                    "end": end,
                    "type": "wind_ramp",
                }
            )

    return rows


def find_derate_types(
    power: pd.DataFrame,
    wind: pd.DataFrame,
    pitch: pd.DataFrame,
    rpm: pd.DataFrame,
    complete: pd.DataFrame,
    busy: dict[str, pd.Series],
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    split_name: str,
) -> list[dict]:
    """
    将无日志的曲线欠发区分为：
    - unlogged_stop：字段明确显示停机/羽化；
    - hidden_derate：没有任何该类正面运行证据。
    """
    residual = curve_residuals(power, wind)
    rows = []

    for tb in TURBINES:
        neighbors = NEIGHBORS[tb]
        base = (
            wind[tb].between(DERATE_WIND_MIN_MS, DERATE_WIND_MAX_MS)
            & (residual[tb] >= DERATE_MIN_DEFICIT_MW)
            & (residual[neighbors].median(axis=1).abs() <= DERATE_NEIGHBOR_RESIDUAL_MAX_MW)
            & group_is_safe([tb, *neighbors], complete, busy)
        )

        unlogged_stop = base & (
            (pitch[tb] >= STOP_PITCH_MIN_DEG)
            | (rpm[tb] < STOP_RPM_MAX)
        )
        hidden_derate = base & (
            (pitch[tb] <= HIDDEN_PITCH_MAX_DEG)
            & (rpm[tb] >= HIDDEN_RPM_MIN)
        )

        for start, end in mask_to_episodes(
            unlogged_stop,
            split_start,
            split_end,
            min_duration=pd.Timedelta(minutes=60),
        ):
            rows.append(
                {
                    "split": split_name,
                    "turbine": tb,
                    "start": start,
                    "end": end,
                    "type": "unlogged_stop",
                }
            )

        for start, end in mask_to_episodes(
            hidden_derate,
            split_start,
            split_end,
            min_duration=HIDDEN_MIN_DURATION,
        ):
            rows.append(
                {
                    "split": split_name,
                    "turbine": tb,
                    "start": start,
                    "end": end,
                    "type": "hidden_derate",
                }
            )

    return rows

def merge_overlapping_events(candidates: pd.DataFrame) -> pd.DataFrame:
    """合并同一 split、同一机组、同一类型下重叠或相接的事件段。"""
    if candidates.empty:
        return candidates

    rows = []

    for (split, turbine, event_type), group in candidates.groupby(
        ["split", "turbine", "type"],
        sort=False,
    ):
        group = group.sort_values(["start", "end"])
        current_start = None
        current_end = None

        for event in group.itertuples(index=False):
            if current_start is None:
                current_start = event.start
                current_end = event.end
            elif event.start <= current_end + STEP:
                current_end = max(current_end, event.end)
            else:
                rows.append(
                    {
                        "split": split,
                        "turbine": turbine,
                        "start": current_start,
                        "end": current_end,
                        "type": event_type,
                    }
                )
                current_start = event.start
                current_end = event.end

        rows.append(
            {
                "split": split,
                "turbine": turbine,
                "start": current_start,
                "end": current_end,
                "type": event_type,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["start", "turbine", "type"])
        .reset_index(drop=True)
    )

def print_summary(candidates: pd.DataFrame) -> None:
    if candidates.empty:
        print("No candidate episodes found.")
        return

    out = candidates.copy()
    out["duration_min"] = (
        (out["end"] - out["start"]).dt.total_seconds() / 60
    ).round(1)

    print("\nCandidate episodes:")
    print(
        out.groupby(["split", "type"])
        .agg(
            episodes=("type", "size"),
            median_minutes=("duration_min", "median"),
            max_minutes=("duration_min", "max"),
        )
        .round(1)
        .to_string()
    )

    print("\nFirst 12 candidate episodes:")
    samples = (
        out.sort_values(["split", "type", "start", "turbine"])
        .groupby(["split", "type"], group_keys=False)
        .head(3)
    )

    print(
        samples[["split", "turbine", "type", "start", "end", "duration_min"]]
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="人工确认 dry run 输出后，才写入 data/kelmarsh/truth.csv。",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(DATA_DIR / "scada.parquet").sort_index()
    events = pd.read_csv(
        DATA_DIR / "events.csv",
        parse_dates=["start", "end", "available_at"],
    ).sort_values("start")

    power_all = frame[[f"p_{tb}" for tb in TURBINES]].copy()
    wind_all = frame[[f"w_{tb}" for tb in TURBINES]].copy()
    pitch_all = frame[[f"pt_{tb}" for tb in TURBINES]].copy()
    rpm_all = frame[[f"rpm_{tb}" for tb in TURBINES]].copy()

    for table in (power_all, wind_all, pitch_all, rpm_all):
        table.columns = TURBINES

    split_config = load_split()["splits"]
    rows = []

    # 只处理 val 和 test：分别扫描，确保事件绝不会跨 split。
    for split_name in ("val", "test"):
        split_start = pd.Timestamp(split_config[split_name]["start"])
        split_end = pd.Timestamp(split_config[split_name]["end"])

        select = (frame.index >= split_start) & (frame.index <= split_end)
        index = frame.index[select]
        power = power_all.loc[index]
        wind = wind_all.loc[index]
        pitch = pitch_all.loc[index]
        rpm = rpm_all.loc[index]

        relevant_events = events[
            (events["end"] >= split_start - EVENT_GUARD)
            & (events["start"] <= split_end + EVENT_GUARD)
        ]
        complete = complete_window(power, wind)
        busy = build_busy_masks(index, relevant_events)

        rows += find_gusts(
            power, wind, complete, busy,
            split_start, split_end, split_name,
        )
        rows += find_wind_ramps(
            power, wind, complete, busy,
            split_start, split_end, split_name,
        )
        rows += find_derate_types(
            power, wind, pitch, rpm, complete, busy,
            split_start, split_end, split_name,
        )

    candidates = pd.DataFrame(
        rows,
        columns=["split", "turbine", "start", "end", "type"],
    )
    candidates = candidates.drop_duplicates()
    candidates = merge_overlapping_events(candidates)

    print("Dry-run input splits: val and test")
    print("Logged-event exclusion guard:", EVENT_GUARD)
    print_summary(candidates)

    if args.write:
        truth = candidates[["turbine", "start", "end", "type"]]
        output = DATA_DIR / "truth.csv"
        truth.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        print(f"\nWrote {len(truth)} rows to: {output}")
    else:
        print("\nDry run only: truth.csv was NOT written.")


if __name__ == "__main__":
    main()