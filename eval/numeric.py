"""结果表 1：数值评测（负责人：B）

    python -m eval.numeric

在 test 段上比较四种预测方式在六种数据场景下的表现：
  模型    persistence（持续性）/ imputed（统一补值 LightGBM）/ quality_aware（质量感知）/
          routed（质量感知 + 本机风速坏了就切后备模型，即智能体实际的用法）
  场景    clean / random_missing / dropout / wind_frozen / power_frozen / wind_drift（训练没见过）
  指标    MAE（MW）；quality_aware 与 routed 的 80% 区间实测覆盖率和平均宽度
扰动只加在输入上，目标值来自原始数据。每个场景只统计“扰动正在起作用”的样本。
结果写到 outputs/numeric_results.md。合成数据上的数字只用来检查代码，不能写进报告。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.config import load_split
from common.store import DataStore
from eval.metrics import coverage, mae, mean_width
from models.features import HORIZON_STEPS, build_features, is_degraded, targets
from models.forecast_power import load_models

OUT = Path(__file__).resolve().parents[1] / "outputs"
N_PER_SCENARIO = 6000


def _segments(df, turbines, rng, fn, count, length):
    out = df.copy()
    active = {t: np.zeros(len(df), bool) for t in turbines}
    for t in turbines:
        p, w = out[f"p_{t}"].to_numpy(copy=True), out[f"w_{t}"].to_numpy(copy=True)
        for _ in range(count):
            s = int(rng.integers(500, len(df) - 500))
            e = s + int(rng.integers(*length))
            fn(p, w, s, e)
            active[t][s + 7:e + 1] = True          # 扰动已持续 ≥ 1 小时的时刻才算“起作用”
        out[f"p_{t}"], out[f"w_{t}"] = p, w
    return out, active


def scenarios(clean: pd.DataFrame, turbines: list[str], seed: int = 7):
    rng = np.random.default_rng(seed)
    n = len(clean)
    yield "clean", clean, {t: np.ones(n, bool) for t in turbines}

    rm = clean.copy()
    for t in turbines:
        for c in (f"p_{t}", f"w_{t}"):
            rm.loc[rng.random(n) < 0.2, c] = np.nan
    yield "random_missing", rm, {t: np.ones(n, bool) for t in turbines}

    def drop(p, w, s, e):
        p[s + 1:e + 1] = np.nan
        w[s + 1:e + 1] = np.nan

    def wfreeze(p, w, s, e):
        w[s + 1:e + 1] = w[s]

    def pfreeze(p, w, s, e):
        if abs(p[s]) > 0.05:
            p[s + 1:e + 1] = p[s]

    def drift(p, w, s, e):
        w[s + 1:e + 1] = w[s + 1:e + 1] + 0.5 * np.arange(1, e - s + 1) / 144

    yield "dropout", *_segments(clean, turbines, rng, drop, 120, (12, 30))
    yield "wind_frozen", *_segments(clean, turbines, rng, wfreeze, 120, (12, 30))
    yield "power_frozen", *_segments(clean, turbines, rng, pfreeze, 120, (12, 30))
    df, act = _segments(clean, turbines, rng, drift, 30, (300, 500))
    yield "wind_drift", df, act


def evaluate() -> pd.DataFrame:
    store = DataStore.load()
    b = load_models()
    test = load_split()["splits"]["test"]
    clean = store.frame.loc[test["start"]:test["end"]]
    rng = np.random.default_rng(0)
    rows = []
    for name, frame, active in scenarios(clean, store.turbines):
        parts = []
        for code, t in enumerate(store.turbines):
            nb = store.neighbors(t)
            Y = pd.DataFrame(targets(clean, t))
            feats = {k: build_features(frame, t, nb, code, k)[b["features"][k]] for k in ("quality_aware", "fallback", "imputed")}
            persist = frame[f"p_{t}"].ffill()
            m = active[t] & Y.notna().all(axis=1).to_numpy()
            idx = np.flatnonzero(m)
            parts.append((t, Y.iloc[idx], {k: v.iloc[idx] for k, v in feats.items()}, persist.iloc[idx]))
        for h in HORIZON_STEPS:
            ys, preds = [], {k: [] for k in ("persistence", "imputed", "quality_aware", "routed")}
            ivs = {k: ([], []) for k in ("quality_aware", "routed")}
            for t, Y, F, persist in parts:
                q = F["quality_aware"]
                wind_bad = ((q["w_frozen_run"] >= 6) | (q["w_gap_tail"] >= 3)).to_numpy()
                out = {}
                for kind in ("quality_aware", "fallback"):
                    X = F[kind]
                    lo, mid, hi = (b["models"][(kind, h, qq)].predict(X) for qq in b["quantiles"])
                    grp = is_degraded(X).to_numpy()
                    pad = np.where(grp, b["cqr"][(kind, h, "degraded")], b["cqr"][(kind, h, "clean")])
                    out[kind] = (np.clip(lo - pad, -0.05, store.rated), np.clip(mid, -0.05, store.rated),
                                 np.clip(hi + pad, -0.05, store.rated))
                routed = tuple(np.where(wind_bad, f, q_) for f, q_ in zip(out["fallback"], out["quality_aware"]))
                ys.append(Y[h].to_numpy())
                preds["persistence"].append(persist.to_numpy())
                preds["imputed"].append(np.clip(b["models"][("imputed", h, 0.5)].predict(F["imputed"]), -0.05, store.rated))
                preds["quality_aware"].append(out["quality_aware"][1])
                preds["routed"].append(routed[1])
                for k, trip in (("quality_aware", out["quality_aware"]), ("routed", routed)):
                    ivs[k][0].append(trip[0])
                    ivs[k][1].append(trip[2])
            y = np.concatenate(ys)
            sel = rng.choice(len(y), size=min(N_PER_SCENARIO, len(y)), replace=False)
            row = {"scenario": name, "horizon_min": h, "n": len(sel)}
            for k, v in preds.items():
                p = np.concatenate(v)[sel]
                ok = ~np.isnan(p)
                row[f"MAE_{k}"] = round(mae(y[sel][ok], p[ok]), 4)
            for k, (lo, hi) in ivs.items():
                lo, hi = np.concatenate(lo)[sel], np.concatenate(hi)[sel]
                row[f"cov_{k}"] = round(coverage(y[sel], lo, hi), 3)
                row[f"width_{k}"] = round(mean_width(lo, hi), 3)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = evaluate()
    OUT.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    head = ["# 结果表 1：数值评测（合成数据，仅用于检查代码）", "",
            "MAE 单位 MW；cov 为 80% 区间的实测覆盖率；width 为平均区间宽度（MW）。", "",
            df.to_markdown(index=False)]
    (OUT / "numeric_results.md").write_text("\n".join(head), encoding="utf-8")


if __name__ == "__main__":
    main()
