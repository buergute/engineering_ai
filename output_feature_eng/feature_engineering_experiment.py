"""FCNN feature-engineering comparison for one-day-ahead Shashi level prediction.

Run from the project directory:
    python output_feature_eng/feature_engineering_experiment.py
The supplied spreadsheet's existing 3T input columns and target are already
normalized. Only engineered features are min-max fitted on the training split.
All reported metrics and prediction exports use the inverse-scaled target.
"""
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------------ Configuration -----------------------------
ROOT = Path(__file__).resolve().parents[1]
REQUESTED_DATA_FILE = ROOT / "data" / "Processed_T7_minmax_prev.xlsx"
# Existing preprocessing workbook name in this workspace; used if requested
# alias above is absent.
DATA_FALLBACKS = [ROOT / "data" / "shashi_T7_minmax_fill_previous.xlsx"]
REQUESTED_PARAM_FILE = ROOT / "data" / "params_T7_minmax_prev.npy"
PARAM_FALLBACKS = [ROOT / "data" / "params_T7_minmax_fill_previous.npy"]
OUTPUT_DIR = ROOT / "output_feature_eng"

SEED = 42
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
# Same FCNN architecture in both arms; only input feature dimension changes.
HIDDEN_SIZES = (128, 64, 32)
DROPOUT = 0.10
# --------------------------------------------------------------------------


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_file(primary, fallbacks):
    if primary.exists():
        return primary
    for candidate in fallbacks:
        if candidate.exists():
            print(f"未找到指定文件 {primary.name}，使用当前工作区文件：{candidate}")
            return candidate
    raise FileNotFoundError(
        f"找不到文件 {primary}；也未找到备用文件：{', '.join(map(str, fallbacks))}"
    )


def read_target_scaling(param_path):
    params = np.load(param_path, allow_pickle=True).item()
    if "target_min" in params and "target_max" in params:
        target_min = float(params["target_min"])
        target_max = float(params["target_max"])
    elif "target_params" in params:
        target_min = float(params["target_params"]["min"])
        target_max = float(params["target_params"]["max"])
    else:
        raise KeyError("参数文件需含 target_min/target_max 或 target_params。")
    if target_max <= target_min:
        raise ValueError("target_max 必须大于 target_min。")
    return target_min, target_max


def load_data():
    data_path = resolve_file(REQUESTED_DATA_FILE, DATA_FALLBACKS)
    param_path = resolve_file(REQUESTED_PARAM_FILE, PARAM_FALLBACKS)
    df = pd.read_excel(data_path, engine="openpyxl")
    if df.shape[1] < 5 or df.columns[0] != "time" or df.columns[-1] != "target":
        raise ValueError("表格必须以 time 开头、target 结尾，并包含至少一组输入特征。")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="raise")
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    feature_names = list(df.columns[1:-1])
    X_flat = df[feature_names].to_numpy(dtype=np.float32)
    y_norm = pd.to_numeric(df["target"], errors="raise").to_numpy(dtype=np.float32)
    times = df["time"].reset_index(drop=True)

    if not np.isfinite(X_flat).all() or not np.isfinite(y_norm).all():
        raise ValueError("X 或 target 中存在 NaN/Inf。")
    if X_flat.shape[1] % 3:
        raise ValueError(f"特征数 {X_flat.shape[1]} 不是 3×T。")
    T = X_flat.shape[1] // 3
    X_seq = X_flat.reshape(len(df), T, 3)
    target_min, target_max = read_target_scaling(param_path)
    return df, times, feature_names, X_flat, X_seq, y_norm, T, target_min, target_max, data_path, param_path


def split_indices(n):
    """Chronological 70/10/20 split before feature engineering."""
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    if not (0 < train_end < val_end < n):
        raise ValueError(f"样本数 {n} 不能按 70/10/20 切分。")
    return {
        "train": np.arange(0, train_end),
        "validation": np.arange(train_end, val_end),
        "test": np.arange(val_end, n),
    }


def engineer_features_after_split(X_seq, times, split):
    """Build 18 features per window, fit their min-max scaler on train only."""
    # Per-channel statistics, ordered by channel then statistic: 3 x 4 = 12.
    stats = []
    stat_names = []
    channels = ["sanxia", "gaobazhou", "shashi"]
    for channel_index, channel in enumerate(channels):
        column = X_seq[:, :, channel_index]
        stats.extend([
            np.mean(column, axis=1),
            np.std(column, axis=1, ddof=0),
            np.max(column, axis=1),
            np.min(column, axis=1),
        ])
        stat_names.extend([
            f"{channel}_mean", f"{channel}_std", f"{channel}_max", f"{channel}_min"
        ])
    stat_features = np.column_stack(stats)

    # Difference between the final and penultimate day within each channel.
    differences = X_seq[:, -1, :] - X_seq[:, -2, :]
    difference_names = [f"{name}_last_day_diff" for name in channels]

    sanxia_mean = stat_features[:, stat_names.index("sanxia_mean")]
    gaobazhou_mean = stat_features[:, stat_names.index("gaobazhou_mean")]
    ratio = np.divide(
        sanxia_mean,
        gaobazhou_mean,
        out=np.zeros_like(sanxia_mean),
        where=np.abs(gaobazhou_mean) > 1e-8,
    )[:, None]

    # Target/sample date's day of year. Leap years retain the requested 365-day period.
    day_of_year = times.dt.dayofyear.to_numpy(dtype=np.float64)
    phase = 2.0 * np.pi * day_of_year / 365.0
    cyclical = np.column_stack([np.sin(phase), np.cos(phase)])

    engineered_raw = np.column_stack([stat_features, differences, ratio, cyclical]).astype(np.float64)
    engineered_names = (
        stat_names + difference_names + ["sanxia_mean_over_gaobazhou_mean"]
        + ["doy_sin", "doy_cos"]
    )
    if engineered_raw.shape[1] != 18:
        raise AssertionError(f"预期新增 18 维，实际得到 {engineered_raw.shape[1]} 维。")

    train_values = engineered_raw[split["train"]]
    feature_min = np.min(train_values, axis=0)
    feature_max = np.max(train_values, axis=0)
    feature_range = feature_max - feature_min
    safe_range = np.where(feature_range == 0, 1.0, feature_range)
    # Do not modify X_seq/X_flat; scale only the 18 newly engineered columns.
    engineered_scaled = (engineered_raw - feature_min) / safe_range
    scaler = {
        "fit_on": "training split only",
        "min": feature_min,
        "max": feature_max,
        "range": feature_range,
        "constant_feature_mask": feature_range == 0,
        "feature_names": engineered_names,
    }
    return engineered_scaled.astype(np.float32), scaler


class FCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        previous = input_dim
        for width in HIDDEN_SIZES:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(DROPOUT)])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def fit_fcnn(name, X, y_norm, split, device):
    """Train exactly 200 epochs with MSE + Adam, as requested."""
    set_seed(SEED)  # repeat seed independently for each experimental arm
    model = FCNN(X.shape[1]).to(device)
    train_idx = split["train"]
    train_ds = TensorDataset(
        torch.from_numpy(X[train_idx]), torch.from_numpy(y_norm[train_idx])
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(xb)
            count += len(xb)
        history.append({"model": name, "epoch": epoch, "train_mse_normalized": loss_sum / count})
        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(f"{name}: epoch {epoch:03d}/{EPOCHS}, train MSE={loss_sum/count:.7f}")
    return model, history


def inverse_target(y_norm, target_min, target_max):
    return y_norm.astype(np.float64) * (target_max - target_min) + target_min


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MSE": float(mse),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "Bias_pred_minus_actual": float(np.mean(y_pred - y_true)),
    }


def predict_splits(model, X, y_norm, times, split, target_min, target_max, device):
    model.eval()
    result = {}
    with torch.no_grad():
        for split_name, indices in split.items():
            y_hat_norm = model(torch.from_numpy(X[indices]).to(device)).cpu().numpy()
            result[split_name] = pd.DataFrame({
                "time": times.iloc[indices].to_numpy(),
                "actual_target": inverse_target(y_norm[indices], target_min, target_max),
                "predicted_target": inverse_target(y_hat_norm, target_min, target_max),
            })
            result[split_name]["residual_pred_minus_actual"] = (
                result[split_name]["predicted_target"] - result[split_name]["actual_target"]
            )
    return result


def save_prediction_files(predictions, model_name):
    for split_name, frame in predictions.items():
        path = OUTPUT_DIR / f"{model_name}_{split_name}_predictions.xlsx"
        frame.to_excel(path, index=False, engine="openpyxl")


def save_comparison_plots(metrics, predictions):
    # 1) Metric bars: one chart containing the four requested metrics.
    metric_names = ["MSE", "MAE", "RMSE", "R2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric in zip(axes.flat, metric_names):
        test_metric = metrics[metrics["split"] == "test"].set_index("model")[metric]
        test_metric.plot(kind="bar", ax=ax, color=["#4C78A8", "#F58518"])
        ax.set_title(f"Test {metric} (original scale)")
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.25)
        if metric == "R2":
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Lower is better")
        for bar in ax.patches:
            ax.annotate(f"{bar.get_height():.4f}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 3), textcoords="offset points")
    fig.suptitle("Feature engineering comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_metric_comparison.png", dpi=160)
    plt.close(fig)

    # 2) Test fit curve.
    base = predictions["baseline"]["test"]
    eng = predictions["feature_eng"]["test"]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(base["time"], base["actual_target"], color="black", lw=1.5, label="Actual")
    ax.plot(base["time"], base["predicted_target"], lw=1.0, label="FCNN baseline")
    ax.plot(eng["time"], eng["predicted_target"], lw=1.0, label="FCNN + engineered features")
    ax.set_title("Test set: actual and predicted Shashi level")
    ax.set_ylabel("Original water-level scale")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_test_fit_curves.png", dpi=160)
    plt.close(fig)

    # 3) Residual vs time on test split.
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axhline(0, color="black", lw=1)
    ax.plot(base["time"], base["residual_pred_minus_actual"], label="FCNN baseline", lw=0.9)
    ax.plot(eng["time"], eng["residual_pred_minus_actual"], label="FCNN + engineered features", lw=0.9)
    ax.set_title("Test residuals over time (prediction − actual)")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_test_residuals.png", dpi=160)
    plt.close(fig)

    # 4) Absolute error against actual level.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(base["actual_target"], np.abs(base["residual_pred_minus_actual"]),
               s=18, alpha=0.60, label="FCNN baseline")
    ax.scatter(eng["actual_target"], np.abs(eng["residual_pred_minus_actual"]),
               s=18, alpha=0.60, label="FCNN + engineered features")
    ax.set_title("Test absolute error vs actual water level")
    ax.set_xlabel("Actual water level (original scale)")
    ax.set_ylabel("Absolute error")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_error_vs_actual.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (df, times, feature_names, X_flat, X_seq, y_norm, T,
     target_min, target_max, data_path, param_path) = load_data()
    split = split_indices(len(df))  # Must happen before feature engineering.
    print(f"Samples={len(df)}, T={T}, original features={X_flat.shape[1]}")
    print("Split sizes:", {k: len(v) for k, v in split.items()})
    print("Split date bounds:", {
        name: [str(times.iloc[idx[0]].date()), str(times.iloc[idx[-1]].date())]
        for name, idx in split.items()
    })

    # Derive and fit engineered feature scaling only after chronological split.
    X_engineered, engineered_scaler = engineer_features_after_split(X_seq, times, split)
    X_augmented = np.column_stack([X_flat, X_engineered]).astype(np.float32)
    assert X_augmented.shape[1] == X_flat.shape[1] + 18

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_inputs = {"baseline": X_flat, "feature_eng": X_augmented}
    predictions = {}
    histories = []
    metric_rows = []
    for model_name, X in model_inputs.items():
        model, history = fit_fcnn(model_name, X, y_norm, split, device)
        histories.extend(history)
        model_predictions = predict_splits(
            model, X, y_norm, times, split, target_min, target_max, device
        )
        predictions[model_name] = model_predictions
        save_prediction_files(model_predictions, model_name)

        torch.save({
            "model": "FCNN",
            "model_name": model_name,
            "state_dict": model.cpu().state_dict(),
            "input_dim": X.shape[1],
            "hidden_sizes": HIDDEN_SIZES,
            "dropout": DROPOUT,
            "feature_names": feature_names + (engineered_scaler["feature_names"] if model_name == "feature_eng" else []),
            "engineered_feature_scaler": engineered_scaler if model_name == "feature_eng" else None,
            "target_min": target_min,
            "target_max": target_max,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
        }, OUTPUT_DIR / f"{model_name}_fcnn.pt")

        for split_name, frame in model_predictions.items():
            metric_rows.append({
                "model": model_name,
                "split": split_name,
                "n_samples": len(frame),
                **evaluate(frame["actual_target"].to_numpy(),
                           frame["predicted_target"].to_numpy()),
            })

    metrics = pd.DataFrame(metric_rows)
    test_metrics = metrics[metrics["split"] == "test"].set_index("model")
    relative_rows = []
    baseline = test_metrics.loc["baseline"]
    engineered = test_metrics.loc["feature_eng"]
    for metric in ["MSE", "MAE", "RMSE", "R2"]:
        denom = abs(float(baseline[metric]))
        if denom == 0:
            improvement = np.nan
        elif metric == "R2":
            improvement = (float(engineered[metric]) - float(baseline[metric])) / denom * 100
        else:
            improvement = (float(baseline[metric]) - float(engineered[metric])) / denom * 100
        relative_rows.append({"metric": metric, "test_relative_improvement_percent": improvement})
    relative = pd.DataFrame(relative_rows)

    metrics_path = OUTPUT_DIR / "metrics_compare.xlsx"
    with pd.ExcelWriter(metrics_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics_all_splits", index=False)
        relative.to_excel(writer, sheet_name="test_relative_improvement", index=False)
        pd.DataFrame({
            "item": ["input_file", "parameter_file", "T", "original_feature_count",
                     "engineered_feature_count", "train_only_engineered_scaling",
                     "epochs", "batch_size", "learning_rate", "optimizer", "loss", "seed"],
            "value": [str(data_path), str(param_path), T, X_flat.shape[1], 18,
                      True, EPOCHS, BATCH_SIZE, LR, "Adam", "MSELoss", SEED],
        }).to_excel(writer, sheet_name="experiment_settings", index=False)
        pd.DataFrame(histories).to_excel(writer, sheet_name="train_loss", index=False)

    # Save fitted engineered-feature scaling values for inference/reproduction.
    np.save(OUTPUT_DIR / "engineered_feature_scaler.npy", engineered_scaler, allow_pickle=True)
    save_comparison_plots(metrics, predictions)

    print("\nTest metrics (original water-level scale):")
    print(test_metrics[["MSE", "MAE", "RMSE", "R2", "Bias_pred_minus_actual"]]
          .to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nFeature-eng relative improvement over baseline (%; positive is better):")
    print(relative.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved artifacts under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
