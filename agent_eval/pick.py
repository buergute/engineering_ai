"""按测试题的 pick_rule 在 test 段自动挑选起报时刻（出题人工具，可以看真值表和未来数据）。

挑出来的时刻写进 outputs/picked.json，方便人工复核；同一个随机种子每次结果相同。
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from common.config import load_split
from common.store import DataStore
from tools.reference import expected_power

STEP = pd.Timedelta(minutes=10)
WIND_TYPES = ("gust", "wind_ramp")


class Picker:
    def __init__(self, store: DataStore):
        self.s = store
        self.f = store.frame
        self.idx = self.f.index
        test = load_split()["splits"]["test"]
        self.t0 = pd.Timestamp(test["start"]) + pd.Timedelta(days=4)
        self.t1 = pd.Timestamp(test["end"]) - pd.Timedelta(days=1)
        self.P = {t: self.f[f"p_{t}"].to_numpy() for t in store.turbines}
        self.W = {t: self.f[f"w_{t}"].to_numpy() for t in store.turbines}
        # busy[t]：该时刻有任何事件或真值异常（不论是否被记录）；
        # busy_nowind 不含“风况变化”（阵风、全场风速突变），挑风况变化题时用：它们是要找的，别的异常不能有
        self.busy = {t: np.zeros(len(self.idx), bool) for t in store.turbines}
        self.busy_nowind = {t: np.zeros(len(self.idx), bool) for t in store.turbines}
        rows = [(*r, False) for r in store.events[["turbine", "start", "end"]].itertuples(index=False)]
        if store.truth is not None:
            rows += [(*r[:3], r[3] in WIND_TYPES) for r in store.truth[["turbine", "start", "end", "type"]].itertuples(index=False)]
        for tb, s, e, is_wind in rows:
            lo, hi = self.idx.get_indexer([s], method="bfill")[0], self.idx.get_indexer([e], method="ffill")[0] + 1
            for x in (store.turbines if tb == "ALL" else [tb]):      # ALL：全场风速突变
                self.busy[x][lo:hi] = True
                if not is_wind:
                    self.busy_nowind[x][lo:hi] = True

    def _i(self, t) -> int:
        return int(self.idx.get_loc(pd.Timestamp(t)))

    def _complete(self, turbines, i, before_steps, after_steps=6) -> bool:
        lo, hi = i - before_steps, i + after_steps + 1
        return all(not np.isnan(self.P[t][lo:hi]).any() and not np.isnan(self.W[t][lo:hi]).any() for t in turbines)

    def _quiet(self, turbines, i, before_steps=144, after_steps=6) -> bool:
        return not any(self.busy[t][i - before_steps:i + after_steps + 1].any() for t in turbines)

    def _deficit(self, tb, i) -> float:
        w = float(np.mean(self.W[tb][i - 2:i + 1]))
        return expected_power(tb, w) - float(np.mean(self.P[tb][i - 2:i + 1]))

    def clean(self, rule, seed) -> tuple[str, str] | None:
        tb = rule["turbine"]
        group = [tb, *self.s.neighbors(tb)]
        quiet_steps = max(int(rule.get("clean_hours_before", 24) * 6), 144)   # 这段时间内没有任何事件
        cands = pd.date_range(self.t0, self.t1, freq="h")
        rng = np.random.default_rng(seed)
        for t in cands[rng.permutation(len(cands))]:
            i = self._i(t)
            if not self._complete(group, i, 48) or not self._quiet(group, i, quiet_steps):
                continue
            w, p = self.W[tb], self.P[tb]
            wn, pn = float(np.mean(w[i - 2:i + 1])), float(np.mean(p[i - 2:i + 1]))
            if "wind" in rule and not rule["wind"][0] <= wn <= rule["wind"][1]:
                continue
            if "power_min" in rule and pn < rule["power_min"]:
                continue
            if "power_range" in rule and not rule["power_range"][0] <= pn <= rule["power_range"][1]:
                continue
            if "all_wind_max" in rule and max(self.W[x][i] for x in self.s.turbines) >= rule["all_wind_max"]:
                continue
            if "all_wind_range" in rule:
                lo, hi = rule["all_wind_range"]
                if not all(lo <= self.W[x][i] <= hi for x in self.s.turbines):
                    continue
            if "wind_change_1h_max" in rule and abs(w[i] - w[i - 6]) > rule["wind_change_1h_max"]:
                continue
            if "future_wind_rise_min" in rule and np.mean(w[i + 5:i + 7]) - np.mean(w[i - 2:i + 1]) < rule["future_wind_rise_min"]:
                continue
            if "wind_range_2h_min" in rule and np.ptp(w[i - 12:i + 1]) < rule["wind_range_2h_min"]:
                continue
            if "power_range_1h_min" in rule and np.ptp(p[i - 6:i + 1]) < rule["power_range_1h_min"]:
                continue
            return tb, t.isoformat(timespec="minutes")
        return None

    def _episode(self, rows, rule, seed) -> tuple[str, str] | None:
        rows = rows.sample(frac=1.0, random_state=seed % (2**32))
        for r in rows.itertuples(index=False):
            jitter = int(np.random.default_rng(seed + zlib.crc32(str(r.start).encode()) % 9973).integers(0, rule.get("jitter_min", 0) // 10 + 1)) * 10
            t = (r.start + pd.Timedelta(minutes=rule["min_elapsed_min"] + jitter)).ceil("10min")
            if not (self.t0 <= t <= self.t1) or t > r.end - pd.Timedelta(minutes=rule.get("min_remaining_min", 30)):
                continue
            i = self._i(t)
            nbrs = self.s.neighbors(r.turbine)
            if not self._complete([r.turbine, *nbrs], i, 48) or not self._quiet(nbrs, i):
                continue
            # 本机除了这一段以外，前后不能有其他异常
            own_busy = self.busy[r.turbine][i - 144:i + 7].copy()
            s_i, e_i = self._i(r.start.ceil("10min")), self._i(r.end.floor("10min"))
            own_busy[max(0, s_i - (i - 144)):e_i - (i - 144) + 1] = False
            if own_busy.any() or self._deficit(r.turbine, i) < rule["deficit_min"]:
                continue
            return r.turbine, t.isoformat(timespec="minutes")
        return None

    def wind_event(self, rule, seed):
        """风况变化：起报时刻前 1 小时内有明显的功率突变，原因只是风（阵风或全场风速突变），没有任何其他异常。
        rule: event: gust | wind_ramp；direction: down | up；min_jump_mw；wind: 本机风速范围（可选）"""
        tr = self.s.truth
        rows = tr[tr.type == rule["event"]].sample(frac=1.0, random_state=seed % (2**32))
        rng = np.random.default_rng(seed)
        for r in rows.itertuples(index=False):
            tbs = self.s.turbines if r.turbine == "ALL" else [r.turbine]
            for tb in rng.permutation(tbs):
                group = [tb, *self.s.neighbors(tb)]
                times = pd.date_range(r.start.ceil("10min"), r.end.floor("10min") + pd.Timedelta(minutes=40), freq="10min")
                for t in times[rng.permutation(len(times))]:
                    if not (self.t0 <= t <= self.t1):
                        continue
                    i = self._i(t)
                    if not self._complete(group, i, 12, 0):
                        continue
                    if any(self.busy_nowind[x][i - 144:i + 7].any() for x in group):
                        continue
                    x = self.P[tb][i - 6:i + 1]
                    steps = x[3:] - x[:-3]              # 与 review_context 相同：30 分钟净变化
                    j = int(np.argmax(np.abs(steps)))
                    if abs(steps[j]) < rule["min_jump_mw"] or (steps[j] < 0) != (rule["direction"] == "down"):
                        continue
                    if "wind" in rule and not rule["wind"][0] <= float(np.mean(self.W[tb][i - 2:i + 1])) <= rule["wind"][1]:
                        continue
                    return tb, t.isoformat(timespec="minutes")
        return None

    def stop_active(self, rule, seed):
        ev = self.s.events
        return self._episode(ev[ev.type == "stop"], rule, seed)

    def curtailment_active(self, rule, seed):
        ev = self.s.events
        rows = ev[(ev.type == "curtailment") & (ev.available_at <= ev.start + pd.Timedelta(minutes=30))]
        return self._episode(rows, rule, seed)

    def hidden_derate(self, rule, seed):
        tr = self.s.truth
        return self._episode(tr[tr.type == "hidden_derate"], rule, seed)


def pick_all(store: DataStore, tasks: list[dict], seed: int = 0) -> dict[str, dict]:
    """seed 不同，挑出的时刻不同；用多个 seed 重复跑可以看结果是否稳定。"""
    pk = Picker(store)
    out: dict[str, dict] = {}
    for task in tasks:
        rule = task["pick_rule"]
        task_seed = zlib.crc32(task["id"].encode()) + 7919 * seed
        if rule["kind"] == "none":
            out[task["id"]] = {"turbine": None, "as_of": None}
            continue
        if rule["kind"] == "same_as":
            out[task["id"]] = dict(out[rule["id"]])
            continue
        hit = getattr(pk, rule["kind"])(rule, task_seed)
        if hit is None:
            raise RuntimeError(f"{task['id']}：在 test 段找不到满足 {rule} 的时刻，请放宽条件")
        out[task["id"]] = {"turbine": hit[0], "as_of": hit[1]}
    return out
