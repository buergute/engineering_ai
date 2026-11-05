"""Direct, two-day-ahead CNC wind forecast using only IEM observations."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/CNC_IEM_2024-2026.csv"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
ISSUE = pd.Timestamp("2026-10-08 07:15")


def load_grid():
    raw = pd.read_csv(RAW, na_values=["M"], parse_dates=["valid"])
    raw = raw[raw.valid.dt.minute.isin([15, 35, 55])].copy()
    raw = raw.drop_duplicates("valid").set_index("valid").sort_index()
    grid = pd.date_range("2024-01-01 00:15", ISSUE, freq="20min")
    raw = raw.reindex(grid)
    raw.index.name = "valid"
    return raw


def features_for_origin(df, origin):
    # All weather summaries and lagged winds end at, or precede, the origin.
    last = df.loc[:origin]
    if len(last) < 7 * 72:
        return None
    wind = last.sknt
    def mean_tail(n):
        v = wind.iloc[-n:]
        return float(v.mean()) if v.notna().sum() >= n * 0.7 else np.nan
    current = last.iloc[-1]
    state = {
        "wind_now": float(current.sknt) if pd.notna(current.sknt) else np.nan,
        "wind_3h": mean_tail(9),
        "wind_12h": mean_tail(36),
        "wind_24h": mean_tail(72),
        "wind_48h": mean_tail(144),
        "wind_std_24h": float(wind.iloc[-72:].std()),
        "temp_now": float(current.tmpf) if pd.notna(current.tmpf) else np.nan,
        "dew_now": float(current.dwpf) if pd.notna(current.dwpf) else np.nan,
        "pressure_now": float(current.alti) if pd.notna(current.alti) else np.nan,
        "pressure_delta_12h": float(current.alti - last.alti.iloc[-37])
        if pd.notna(current.alti) and pd.notna(last.alti.iloc[-37]) else np.nan,
    }
    target_day = origin.normalize() + pd.Timedelta(days=2)
    targets = pd.date_range(target_day + pd.Timedelta(minutes=15), periods=72, freq="20min")
    rows = []
    for target in targets:
        hour = target.hour + target.minute / 60
        year_phase = 2 * np.pi * target.dayofyear / 365.25
        row = dict(state)
        row.update({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "year_sin": np.sin(year_phase),
            "year_cos": np.cos(year_phase),
            "lead_h": (target - origin).total_seconds() / 3600,
            "lag_72h": df.sknt.get(target - pd.Timedelta(hours=72), np.nan),
            "lag_96h": df.sknt.get(target - pd.Timedelta(hours=96), np.nan),
            "lag_168h": df.sknt.get(target - pd.Timedelta(hours=168), np.nan),
            "origin": origin,
            "target_time": target,
            "y": df.sknt.get(target, np.nan),
        })
        rows.append(row)
    return rows


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "Bias": float(np.mean(p - y))}


def main():
    df = load_grid()
    origins = pd.date_range("2024-01-08 07:15", ISSUE, freq="D")
    rows = []
    for origin in origins:
        f = features_for_origin(df, origin)
        if f is not None:
            rows.extend(f)
    all_data = pd.DataFrame(rows)
    labeled = all_data.dropna(subset=["y"]).copy()
    feature_cols = [x for x in labeled.columns if x not in ["origin", "target_time", "y"]]
    train = labeled[labeled.origin < "2026-08-15"]
    val = labeled[(labeled.origin >= "2026-08-15") & (labeled.origin < "2026-09-16")]
    test = labeled[(labeled.origin >= "2026-09-16") & (labeled.origin <= "2026-10-06")]
    configurations = {
        "HGB_15_0.05": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, learning_rate=0.05, l2_regularization=5, min_samples_leaf=80, random_state=42),
        "HGB_31_0.05": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=31, learning_rate=0.05, l2_regularization=5, min_samples_leaf=80, random_state=42),
        "HGB_15_0.10": HistGradientBoostingRegressor(max_iter=180, max_leaf_nodes=15, learning_rate=0.10, l2_regularization=5, min_samples_leaf=80, random_state=42),
        "RF_100_12": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=20, max_features=0.7, n_jobs=-1, random_state=42),
    }
    val_results = []
    for name, model in configurations.items():
        model.fit(train[feature_cols], train.y)
        pred = np.maximum(0, model.predict(val[feature_cols]))
        val_results.append({"model": name, **metrics(val.y, pred)})
    selected = min(val_results, key=lambda r: r["RMSE"])["model"]
    model = configurations[selected]
    # Refit on train+validation, then evaluate once on the later untouched test period.
    pretest = labeled[labeled.origin < "2026-09-16"]
    model.fit(pretest[feature_cols], pretest.y)
    test_pred = np.maximum(0, model.predict(test[feature_cols]))
    baseline = test.lag_72h.fillna(pretest.y.mean()).clip(lower=0).to_numpy()
    test_results = [
        {"model": "72h_lag_baseline", **metrics(test.y, baseline)},
        {"model": selected, **metrics(test.y, test_pred)},
    ]
    test_predictions = test[["origin", "target_time", "y"]].copy()
    test_predictions["predicted_knots"] = test_pred
    test_predictions["baseline_knots"] = baseline
    test_predictions["error_knots"] = test_pred - test.y
    test_predictions.to_csv(OUT / "test_predictions.csv", index=False)

    # Final model uses every labeled example available as of the issue time.
    final_train = labeled[labeled.target_time <= ISSUE]
    model.fit(final_train[feature_cols], final_train.y)
    future = all_data[all_data.origin == ISSUE].copy()
    forecast = np.maximum(0, model.predict(future[feature_cols]))
    future["forecast_knots"] = forecast
    future[["target_time", "forecast_knots"]].to_csv(OUT / "forecast_2026-10-10_CNC.csv", index=False)
    report = {
        "raw_rows": int(len(df)),
        "raw_missing_wind": int(df.sknt.isna().sum()),
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "test_samples": int(len(test)),
        "final_train_samples": int(len(final_train)),
        "features": feature_cols,
        "validation_results": val_results,
        "selected_model": selected,
        "test_results": test_results,
        "forecast_min": float(forecast.min()),
        "forecast_max": float(forecast.max()),
        "forecast_mean": float(forecast.mean()),
        "issue": str(ISSUE),
    }
    (OUT / "model_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
