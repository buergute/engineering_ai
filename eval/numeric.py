"""结果表 1：数值评测（负责人：B）

    python -m eval.numeric

同一场景和时距共用起报点，按日分块估计指标区间；目标取自原始数据。
正式输出在 outputs/，验证段输出在 outputs/forecast_validation/。
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from common.config import load_split
from common.store import DataStore
from eval.metrics import paired_daily_means
from models.calibration import calibrated_interval
from models.settings import forecast_settings
from models.features import HORIZON_STEPS, build_features, is_degraded, targets
from models.forecast_power import load_models

OUT = Path(__file__).resolve().parents[1] / "outputs"
METHODS = ("persistence", "imputed", "quality_aware", "routed", "no_quality", "global_cqr")


def _segments(df, turbines, rng, fn, count, length):
    out = df.copy()
    active = {t: np.zeros(len(df), bool) for t in turbines}
    for t in turbines:
        p, w = out[f"p_{t}"].to_numpy(copy=True), out[f"w_{t}"].to_numpy(copy=True)
        for _ in range(count):
            s = int(rng.integers(500, len(df) - 500))
            e = s + int(rng.integers(*length))
            if fn(p, w, s, e) is False:
                continue
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
        if not np.isfinite(w[s]):
            return False
        w[s + 1:e + 1] = w[s]

    def pfreeze(p, w, s, e):
        if not abs(p[s]) > 0.05:
            return False
        p[s + 1:e + 1] = p[s]

    def drift(p, w, s, e):
        w[s + 1:e + 1] = w[s + 1:e + 1] + 0.5 * np.arange(1, e - s + 1) / 144

    yield "dropout", *_segments(clean, turbines, rng, drop, 120, (12, 30))
    yield "wind_frozen", *_segments(clean, turbines, rng, wfreeze, 120, (12, 30))
    yield "power_frozen", *_segments(clean, turbines, rng, pfreeze, 120, (12, 30))
    df, act = _segments(clean, turbines, rng, drift, 30, (300, 500))
    yield "wind_drift", df, act


def predict_methods(bundle, h, features, persistence, rated):
    q = features["quality_aware"]
    routed = ((q["w_frozen_run"] >= 6) | (q["w_gap_tail"] >= 3)).to_numpy()
    pad = bundle["cqr"][("persistence", h, "all")]
    ans = {"persistence": calibrated_interval(persistence, persistence, persistence, pad, rated)}
    for kind in ("imputed", "quality_aware", "fallback", "no_quality"):
        source = features["quality_aware" if kind == "no_quality" else kind]
        x = source[bundle["features"][kind]]
        pred = [bundle["models"][(kind, h, tau)].predict(x) for tau in bundle["quantiles"]]
        offset = bundle["cqr"][(kind, h, "all")]
        if kind == "quality_aware":
            ans["global_cqr"] = calibrated_interval(*pred, offset, rated)
        if kind != "imputed":
            offset = np.where(is_degraded(source), bundle["cqr"][(kind, h, "degraded")],
                              bundle["cqr"][(kind, h, "clean")])
        ans[kind] = calibrated_interval(*pred, offset, rated)
    ans["routed"] = tuple(np.where(routed, f, v) for f, v in zip(ans.pop("fallback"), ans["quality_aware"]))
    return ans


def predictions(clean, store, bundle, settings, *, clean_only=False):
    rng = np.random.default_rng(settings["seed"])
    for scene, frame, active in scenarios(clean, store.turbines, settings["seed"]):
        candidates = []
        for turbine in store.turbines:
            power = frame[f"p_{turbine}"].ffill()
            rows = pd.DataFrame(targets(clean, turbine))
            rows["persistence"] = power
            rows["wind"] = clean[f"w_{turbine}"]
            rows["turbine"] = turbine
            candidates.append(rows.loc[active[turbine] & power.notna()])
        pool = pd.concat(candidates)
        for h in HORIZON_STEPS:
            usable = pool.dropna(subset=[h])
            if usable.empty:
                raise ValueError(f"{scene}/{h} 没有可配对起报点")
            sample = usable.iloc[rng.choice(len(usable), min(len(usable), settings["origins_per_scenario"]), replace=False)]
            chunks = []
            for turbine, part in sample.groupby("turbine"):
                features = {kind: build_features(frame, turbine, store.neighbors(turbine),
                                                 store.turbines.index(turbine), kind).loc[part.index]
                            for kind in ("quality_aware", "imputed", "fallback")}
                pred = predict_methods(bundle, h, features, part.persistence.to_numpy(), store.rated)
                part = part[["turbine", "wind", h]].rename(columns={h: "target"}).copy()
                for kind, (lo, mid, hi) in pred.items():
                    part[f"{kind}_error"] = np.abs(part.target - mid)
                    part[f"{kind}_covered"] = ((lo <= part.target) & (part.target <= hi)).astype(float)
                    part[f"{kind}_width"] = hi - lo
                chunks.append(part)
            yield scene, h, pd.concat(chunks)
        if clean_only:
            break


def summarize(scene, h, samples, rated, settings):
    columns = [f"{m}_{k}" for m in METHODS for k in ("error", "covered", "width")]
    mean, draws = paired_daily_means(samples[columns].to_numpy(), samples.index,
                                     settings["bootstrap_draws"], settings["seed"])
    # First row is the sample estimate, subsequent rows are paired replicates.
    stats = np.vstack([mean, draws]).reshape(-1, len(METHODS), 3)
    base = stats[:, 0, 0]
    result = []
    for i, method in enumerate(METHODS):
        error, cov, width = stats[:, i, :].T
        skill = np.full_like(base, np.nan)
        np.divide(100 * (base - error), base, out=skill, where=base > 0)
        row = dict(scenario=scene, horizon_min=h, method=method, n=len(samples),
                   days=samples.index.normalize().nunique())
        for key, vector in dict(MAE=error, nMAE=error/rated, skill_pct=skill,
                                coverage=cov, width=width).items():
            row[key] = vector[0]
            if row["days"] >= 2 and np.isfinite(vector[1:]).all():
                row[key+"_lo"], row[key+"_hi"] = np.quantile(vector[1:], [.025, .975])
            else:
                row[key+"_lo"] = row[key+"_hi"] = np.nan
        result.append(row)
    return result


def split_errors(store, bundle, split, evaluated, split_name):
    """Clean-input MAE, with identical sampling settings in each period."""
    settings = forecast_settings()
    rows = []
    for name in ("train", "val", "calib", "test"):
        if name == split_name:
            scored = evaluated.query("scenario == 'clean' and method == 'quality_aware'")
            rows.extend(dict(split=name, horizon_min=r.horizon_min, n=r.n, MAE=r.MAE)
                        for r in scored.itertuples())
        elif name != "test":
            bounds = split["splits"][name]
            frame = store.frame.loc[bounds["start"]:bounds["end"]]
            for _, h, samples in predictions(frame, store, bundle, settings, clean_only=True):
                rows.append(dict(split=name, horizon_min=h, n=len(samples),
                                 MAE=samples["quality_aware_error"].mean()))
    return pd.DataFrame(rows)


def _evaluate(split_name):
    split = load_split()
    b = load_models()
    if b.get("dataset") != split["dataset"] or b.get("splits") != split["splits"] or "no_quality" not in b["features"]:
        raise ValueError("模型与当前数据集、切分或评测版本不匹配，请运行 python -m models.train")
    store = DataStore.load()
    df, strata = evaluate_split(store, b, split, split_name)
    return df, strata, b, split_errors(store, b, split, df, split_name)


def evaluate_split(store, b, split, split_name):
    settings = forecast_settings()
    interval = split["splits"][split_name]
    clean = store.frame.loc[interval["start"]:interval["end"]]
    rows, strata = [], []
    for scene, h, samples in predictions(clean, store, b, settings):
        rows.extend(summarize(scene, h, samples, store.rated, settings))
        bins = pd.cut(samples.wind, [-np.inf, *settings["wind_cuts_ms"], np.inf],
                      labels=["low", "mid", "high"], right=False)
        for band, group in samples.groupby(bins.astype(object).fillna("missing")):
            for method in METHODS[:4]:
                strata.append(dict(scenario=scene, horizon_min=h, method=method, wind_band=band,
                                   n=len(group), coverage=group[f"{method}_covered"].mean(),
                                   width=group[f"{method}_width"].mean()))
    return pd.DataFrame(rows), pd.DataFrame(strata)


def evaluate() -> pd.DataFrame:
    return _evaluate("test")[0]


def report(df, strata, bundle, split_name):
    keys = ["scenario", "horizon_min"]
    by_method = {m: df[df.method == m].set_index(keys) for m in METHODS}
    qa = by_method["quality_aware"]
    imputed = by_method["imputed"]
    h1 = pd.DataFrame({"ΔMAE": qa.MAE-imputed.MAE, "Δ覆盖率": qa.coverage-imputed.coverage})
    h1["满足"] = (h1["ΔMAE"] < 0) & (h1["Δ覆盖率"] >= 0)
    degraded = h1.drop(index="clean", level="scenario")
    passed = degraded.groupby(level="scenario")["满足"].all()
    table = df[df.method.isin(METHODS[:4])][[*keys, "method", "n", "days"]].copy()
    for metric in ("MAE", "nMAE", "skill_pct", "coverage", "width"):
        table[metric] = [f"{r[metric]:.4f} [{r[metric+'_lo']:.4f}, {r[metric+'_hi']:.4f}]"
                         for _, r in df.loc[table.index].iterrows()]
    ablations = []
    for m in METHODS[4:]:
        delta = by_method[m][["coverage", "width"]] - qa[["coverage", "width"]]
        ablations.append(delta.assign(ablation=m).reset_index())
    heading = f"# 预测评测 · {bundle['dataset']} · {split_name}"
    if split_name == "test" and bundle["dataset"] == "kelmarsh_v4":
        period = bundle["splits"]["test"]
        if period["start"][:4] == period["end"][:4] == "2023":
            heading += "\n\nA 在此前汇报的基线评测中已使用 Kelmarsh 2023 测试段。据组内确认，A 未依据该次基线结果调整模型或阈值。后续在同一时段得到的结果应标为复评。"
        elif period["start"][:4] == period["end"][:4] == "2024":
            heading += "\n\n本次按用户授权在 2024 年评测。2024 是否曾被其他组员用于开发尚未确认。2023 已被 A 用于基线评测，组内确认未据此调整模型或阈值。"
    return "\n\n".join([heading,
        f"模型 {bundle['version']}。MAE 与宽度单位 MW，nMAE 为额定容量比例，skill_pct 为相对持续性的 MAE 提升百分比。方括号为按起报日整块抽样的 95% 区间。",
        table.to_markdown(index=False), "## H1",
        "按配对点估计判断：质量感知 MAE 更小且覆盖率不低于补值模型；三个时距均满足才计入成立场景。此判定不代表差异显著。",
        "成立场景：" + ("、".join(passed[passed].index) or "无"),
        "未成立场景：" + ("、".join(passed[~passed].index) or "无"),
        degraded.reset_index().to_markdown(index=False), "## 消融相对质量感知的变化",
        pd.concat(ablations).to_markdown(index=False), "## 风速分层",
        "完整分层覆盖率、样本数与宽度见同目录 wind_coverage.csv。低于目标覆盖率的组如下。",
        strata.query("coverage < 0.8").to_markdown(index=False)]) + "\n"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--render-only", action="store_true", help="从本次运行的 CSV 重新排版")
    args = parser.parse_args()
    dest = OUT if args.split == "test" else OUT / "forecast_validation"
    if args.render_only:
        meta = json.loads((dest / "forecast_run.json").read_text(encoding="utf-8"))
        if meta["evaluation_split"] != args.split:
            raise ValueError("评测切分与快照不一致")
        df = pd.read_csv(dest / "forecast_metrics.csv", float_precision="round_trip")
        strata = pd.read_csv(dest / "wind_coverage.csv", float_precision="round_trip")
        errors = pd.read_csv(dest / "split_mae.csv", float_precision="round_trip")
        write_markdown(df, strata, meta, args.split, dest, errors)
        return
    df, strata, b, errors = _evaluate(args.split)
    write_results(df, strata, b, args.split, dest, errors)


def write_results(df, strata, b, split_name, dest, errors):
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "forecast_metrics.csv", index=False)
    strata.to_csv(dest / "wind_coverage.csv", index=False)
    errors.to_csv(dest / "split_mae.csv", index=False)
    metadata = {k: b[k] for k in ("version", "dataset", "seed", "splits")}
    metadata.update(evaluation_split=split_name, settings=forecast_settings())
    (dest / "forecast_run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_markdown(df, strata, b, split_name, dest, errors)


def write_markdown(df, strata, b, split_name, dest, errors):
    body = report(df, strata, b, split_name)
    body += "\n## 干净输入的分段 MAE（MW）\n\n" + errors.to_markdown(index=False) + "\n"
    (dest / "numeric_results.md").write_text(body, encoding="utf-8")
    if split_name == "test" and b["dataset"] == "kelmarsh_v4":
        path = OUT.parent / "docs" / "results_B.md"
        methods = path.read_text(encoding="utf-8").split("## 正式结果")[0]
        path.write_text(methods + "## 正式结果\n\n" + body, encoding="utf-8")
    print(dest / "numeric_results.md")


if __name__ == "__main__":
    main()
