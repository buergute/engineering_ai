"""训练与校准（负责人：B）

    python -m models.train

严格按 configs/split.yaml 的时间切分：
  train  拟合经验功率曲线、邻机风速比例、LightGBM 模型
  val    早停（选树的棵数）
  calib  分组 CQR 区间校准（正常数据一组，近期有缺失/冻结的一组）
  test   这里一概不碰
训练和校准都同时使用“原始数据”和“随机撒了扰动的数据”（perturb.random_augment），
目标值始终取自原始数据。缓慢偏移刻意不放进训练，留作没见过的扰动。

产物写到 artifacts/：reference.json（功率曲线、风速比例）、models.pkl（模型和校准量）
"""
from __future__ import annotations

import json
import pickle
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.config import load_split
from common.store import DataStore
from models.features import HORIZON_STEPS, build_features, is_degraded, targets
from perturb.inject import random_augment

warnings.filterwarnings("ignore", message=".*eval_set.*")
ROOT = Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts"
QUANTILES = (0.1, 0.5, 0.9)
COVERAGE = 0.8
KINDS = ("quality_aware", "fallback", "imputed")
ROWS_PER_SPLIT = {"train": 160_000, "val": 40_000, "calib": 60_000}


def _period(df, split, name):
    s = split["splits"][name]
    return df.loc[s["start"]:s["end"]]


def fit_reference(store: DataStore, split: dict) -> dict:
    """只用 train 段：经验功率曲线（0.5 m/s 分箱中位数），以及机组两两之间的风速比例。"""
    tr = _period(store.frame, split, "train")
    ev = store.events
    bins = np.arange(0, 26, 0.5)
    curves, ratios = {}, {}
    for t in store.turbines:
        p, w = tr[f"p_{t}"].copy(), tr[f"w_{t}"]
        for _, e in ev[ev.turbine == t].iterrows():          # 剔除有记录的限电和停机时段
            p.loc[e.start:e.end] = np.nan
        ok = p.notna() & w.notna()
        cat = pd.cut(w[ok], bins)
        med = p[ok].groupby(cat, observed=True).median()
        centers = [float(iv.mid) for iv in med.index]
        curves[t] = {"ws": centers, "mw": [round(float(v), 4) for v in med.values]}
        ratios[t] = {}
        for n in store.neighbors(t):
            both = (w > 4) & (tr[f"w_{n}"] > 4)
            ratios[t][n] = round(float((w[both] / tr.loc[both, f"w_{n}"]).median()), 4)
    return {"curve": curves, "wind_ratio": ratios,
            "curve_source": f"train 段（{split['splits']['train']['start'][:10]} 至 {split['splits']['train']['end'][:10]}）0.5 m/s 分箱中位数，已剔除有记录的限电/停机"}


def _dataset(frames: dict[str, pd.DataFrame], store: DataStore, kind: str, n_rows: int, seed: int):
    """把 6 台机的特征和目标堆在一起，再随机抽样。frames 包含 clean 与 aug 两份输入。"""
    rng = np.random.default_rng(seed)
    Xs, Ys = [], []
    for code, t in enumerate(store.turbines):
        y = pd.DataFrame(targets(frames["clean"], t))
        for name in ("clean", "aug"):
            X = build_features(frames[name], t, store.neighbors(t), code, kind)
            Xs.append(X)
            Ys.append(y)
    X, Y = pd.concat(Xs), pd.concat(Ys)
    keep = X["p_lag0"].notna() | X["nb_p_lag0"].notna()
    X, Y = X[keep], Y[keep]
    take = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    return X.iloc[take], Y.iloc[take]


def main() -> None:
    t0 = time.time()
    split = load_split()
    store = DataStore.load()
    ART_DIR.mkdir(exist_ok=True)

    ref = fit_reference(store, split)
    (ART_DIR / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[1/3] 参考量（功率曲线、风速比例）已拟合，用时 {time.time() - t0:.0f}s")

    # 每段在自身数据上做增强，避免跨段
    frames = {}
    for i, name in enumerate(("train", "val", "calib")):
        clean = _period(store.frame, split, name)
        frames[name] = {"clean": clean, "aug": random_augment(clean, store.turbines, seed=100 + i)}

    params = dict(objective="quantile", learning_rate=0.05, num_leaves=31, min_child_samples=100,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, n_estimators=400, verbose=-1, n_jobs=2)
    bundle = {"features": {}, "models": {}, "cqr": {}, "coverage": COVERAGE, "quantiles": QUANTILES,
              "turbines": store.turbines, "version": time.strftime("%Y%m%d-%H%M")}

    for kind in KINDS:
        Xtr, Ytr = _dataset(frames["train"], store, kind, ROWS_PER_SPLIT["train"], 1)
        Xva, Yva = _dataset(frames["val"], store, kind, ROWS_PER_SPLIT["val"], 2)
        Xca, Yca = _dataset(frames["calib"], store, kind, ROWS_PER_SPLIT["calib"], 3)
        bundle["features"][kind] = list(Xtr.columns)
        qs = QUANTILES if kind != "imputed" else (0.5,)
        for h in HORIZON_STEPS:
            for q in qs:
                mtr, mva = Ytr[h].notna(), Yva[h].notna()
                m = lgb.LGBMRegressor(alpha=q, **params)
                m.fit(Xtr[mtr], Ytr[h][mtr], eval_set=[(Xva[mva], Yva[h][mva])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                bundle["models"][(kind, h, q)] = m
            if kind == "imputed":
                continue
            # 分组 CQR：分数 E = max(q_lo - y, y - q_hi)，取 (1-α)(1+1/n) 分位数作为区间外扩量
            mca = Yca[h].notna()
            Xc, yc = Xca[mca], Yca[h][mca].to_numpy()
            lo = bundle["models"][(kind, h, 0.1)].predict(Xc)
            hi = bundle["models"][(kind, h, 0.9)].predict(Xc)
            score = np.maximum(lo - yc, yc - hi)
            deg = is_degraded(Xc).to_numpy()
            for group, mask in (("clean", ~deg), ("degraded", deg)):
                s = np.sort(score[mask])
                k = min(len(s) - 1, int(np.ceil(COVERAGE * (len(s) + 1))) - 1)
                bundle["cqr"][(kind, h, group)] = float(s[k])
        print(f"[2/3] {kind} 模型训练完成，用时 {time.time() - t0:.0f}s")

    with open(ART_DIR / "models.pkl", "wb") as f:
        pickle.dump(bundle, f)
    cq = {f"{k[0]}/{k[1]}min/{k[2]}": round(v, 3) for k, v in bundle["cqr"].items()}
    print(f"[3/3] 区间校准外扩量（MW）：{cq}")
    print(f"完成，总用时 {time.time() - t0:.0f}s → {ART_DIR}")


if __name__ == "__main__":
    main()
