"""Compare FCNN, LSTM, GRU and TCN for one-day-ahead Shashi level prediction.

Run from the project root with the configured Anaconda environment:
    python output/compare_timeseries_models.py
Inputs are already preprocessed. Target metrics/predictions are inverse-scaled
back to the original water-level units before evaluation.
"""
from pathlib import Path
import copy
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


# ------------------------------- Settings ---------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "shashi_T7_minmax_fill_previous.xlsx"
PARAM_CANDIDATES = [
    ROOT / "data" / "params_T7_minmax_prev.npy",  # requested short name
    ROOT / "data" / "params_T7_minmax_fill_previous.npy",  # existing file
]
OUTPUT_DIR = ROOT / "output" / "model_comparison"
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 300
PATIENCE = 35
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
DROPOUT = 0.10
HIDDEN_SIZE = 64
# --------------------------------------------------------------------------


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small tabular/short-sequence batches are often faster and more repeatable
    # with one CPU thread than with a large BLAS/OpenMP thread pool.
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_target_scale():
    path = next((p for p in PARAM_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "找不到 target 参数文件。查找过：" + ", ".join(map(str, PARAM_CANDIDATES))
        )
    params = np.load(path, allow_pickle=True).item()
    if "target_min" in params and "target_max" in params:
        target_min, target_max = params["target_min"], params["target_max"]
    elif "target_params" in params:
        target_min = params["target_params"]["min"]
        target_max = params["target_params"]["max"]
    else:
        raise KeyError("参数文件应包含 target_min/target_max 或 target_params。")
    target_min, target_max = float(target_min), float(target_max)
    if not target_max > target_min:
        raise ValueError("target_max 必须大于 target_min。")
    # Also return the scaler for the last input feature (day_T_shashi), used by
    # the persistence baseline to invert yesterday's normalized level.
    feature_params = params.get("feature_params", {})
    feature_min = feature_params.get("min")
    feature_max = feature_params.get("max")
    if feature_min is not None and feature_max is not None:
        last_feature_min = float(np.asarray(feature_min)[-1])
        last_feature_max = float(np.asarray(feature_max)[-1])
    else:
        last_feature_min, last_feature_max = target_min, target_max
    return target_min, target_max, path, last_feature_min, last_feature_max


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"找不到数据文件：{DATA_FILE}")
    df = pd.read_excel(DATA_FILE, engine="openpyxl")
    if df.columns[0] != "time" or df.columns[-1] != "target":
        raise ValueError("Excel 首列必须为 time，末列必须为 target。")
    times = pd.to_datetime(df.iloc[:, 0], errors="raise").reset_index(drop=True)
    feature_names = list(df.columns[1:-1])
    X_flat = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)
    y_norm = df["target"].to_numpy(dtype=np.float32)
    if not np.isfinite(X_flat).all() or not np.isfinite(y_norm).all():
        raise ValueError("发现 NaN/Inf。请先完成空值处理及特征预处理。")
    if X_flat.shape[1] % 3:
        raise ValueError(f"特征数 {X_flat.shape[1]} 不能被 3 整除。")
    # Input columns should be day_1_sanxia, day_1_gaobazhou, day_1_shashi, ...
    X_seq = X_flat.reshape(len(X_flat), -1, 3)
    target_min, target_max, param_path, last_x_min, last_x_max = load_target_scale()
    return (df, times, X_flat, X_seq, y_norm, feature_names,
            target_min, target_max, param_path, last_x_min, last_x_max)


def chronological_split(n):
    """First 70% train, next 10% validation, final 20% test."""
    i_train = int(n * 0.70)
    i_val = int(n * 0.80)
    if not (0 < i_train < i_val < n):
        raise ValueError(f"样本量 {n} 不足以按 70/10/20 划分。")
    return {
        "train": np.arange(0, i_train),
        "validation": np.arange(i_train, i_val),
        "test": np.arange(i_val, n),
    }


class FCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features=3):
        super().__init__()
        self.rnn = nn.LSTM(n_features, HIDDEN_SIZE, num_layers=2,
                           batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, n_features=3):
        super().__init__()
        self.rnn = nn.GRU(n_features, HIDDEN_SIZE, num_layers=2,
                          batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(nn.Linear(HIDDEN_SIZE, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        sequence, _ = self.rnn(x)
        return self.head(sequence[:, -1, :]).squeeze(-1)


class Chomp1d(nn.Module):
    """Remove right padding to preserve causality and input sequence length."""
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, x):
        return x[:, :, :-self.amount].contiguous() if self.amount else x


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.block(x) + self.residual(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_features=3):
        super().__init__()
        self.tcn = nn.Sequential(
            TemporalBlock(n_features, 32, kernel_size=3, dilation=1, dropout=DROPOUT),
            TemporalBlock(32, 32, kernel_size=3, dilation=2, dropout=DROPOUT),
            TemporalBlock(32, 32, kernel_size=3, dilation=4, dropout=DROPOUT),
        )
        self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        x = self.tcn(x.transpose(1, 2))
        return self.head(x[:, :, -1]).squeeze(-1)


def train_one_model(name, model, train_X, train_y, val_X, val_y, device):
    train_data = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_X_t = torch.from_numpy(val_X).to(device)
    val_y_t = torch.from_numpy(val_y).to(device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    history = {"epoch": [], "train_loss": [], "validation_loss": []}
    best_val, best_epoch, best_state, waiting = float("inf"), 0, None, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        loss_total, count = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_total += loss.item() * len(xb)
            count += len(xb)
        train_loss = loss_total / count
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_X_t), val_y_t).item()
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        if val_loss < best_val:
            best_val, best_epoch, waiting = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"{name:5s} epoch {epoch:03d}: train={train_loss:.6f}, val={val_loss:.6f}")
        if waiting >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError(f"{name} 未训练出有效权重。")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_val


def inverse_target(y_norm, target_min, target_max):
    return y_norm.astype(np.float64) * (target_max - target_min) + target_min


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    error = y_pred - y_true
    # Peak errors: top 10% target samples provide a simple high-water diagnostic.
    threshold = np.quantile(y_true, 0.90)
    peak_mask = y_true >= threshold
    return {
        "MSE": mse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "Bias_pred_minus_actual": float(np.mean(error)),
        "Peak_MAE_top10pct": float(np.mean(np.abs(error[peak_mask]))),
    }


def run_inference(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X).to(device)).cpu().numpy()


def save_model_loss_plot(histories):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, (name, hist) in zip(axes.flat, histories.items()):
        ax.plot(hist["epoch"], hist["train_loss"], label="Train")
        ax.plot(hist["epoch"], hist["validation_loss"], label="Validation")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (normalized target)")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Training curves (loss calculated on normalized target)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparison_loss_curves.png", dpi=160)
    plt.close(fig)


def save_fit_plots(times, actual_by_split, predictions, indices):
    names = list(predictions)
    colors = dict(zip(names, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]))
    # One full timeline plot, shading the chronological split periods.
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(times, actual_by_split["all"], color="black", linewidth=1, label="Actual")
    for name in names:
        full_pred = np.full(len(times), np.nan)
        full_pred[indices["test"]] = predictions[name]["test"]
        ax.plot(times, full_pred, color=colors[name], linewidth=1, label=f"{name} test")
    ax.set_title("Test predictions on chronological tail")
    ax.set_ylabel("Shashi level (original scale)")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "test_model_comparison.png", dpi=160)
    plt.close(fig)

    # Separate fitted curves for each split, overlaying the four architectures.
    for split in ("train", "validation", "test"):
        idx = indices[split]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(times.iloc[idx], actual_by_split[split], color="black", linewidth=1.5,
                label="Actual")
        for name in names:
            ax.plot(times.iloc[idx], predictions[name][split], color=colors[name],
                    linewidth=1.0, alpha=0.9, label=name)
        ax.set_title(f"{split.title()} fit (original water-level scale)")
        ax.set_ylabel("Shashi level")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"fit_comparison_{split}.png", dpi=160)
        plt.close(fig)


def main():
    seed_everything()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    (df, times, X_flat, X_seq, y_norm, feature_names,
     target_min, target_max, param_path, last_x_min, last_x_max) = load_dataset()
    split_indices = chronological_split(len(df))
    y_original = inverse_target(y_norm, target_min, target_max)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device={device}; n={len(df)}; T={X_seq.shape[1]}; features/day={X_seq.shape[2]}")
    print(f"Target inverse scale: min={target_min:.6f}, max={target_max:.6f}; params={param_path}")
    print("Split sizes:", {k: len(v) for k, v in split_indices.items()})

    # All nets receive identical samples and split indices. FCNN receives the
    # flattened representation; sequence nets receive [sample, T, 3].
    constructors = {
        "FCNN": (FCNN(X_flat.shape[1]), X_flat),
        "LSTM": (LSTMRegressor(3), X_seq),
        "GRU": (GRURegressor(3), X_seq),
        "TCN": (TCNRegressor(3), X_seq),
    }
    histories, best_epochs, predictions, metrics_rows = {}, {}, {}, []
    trained_models = {}

    for model_name, (model, model_X) in constructors.items():
        print(f"\nTraining {model_name}...")
        # Equal, repeatable initialization seeds for every model.
        seed_everything(SEED)
        tr, va = split_indices["train"], split_indices["validation"]
        model, history, best_epoch, best_val = train_one_model(
            model_name, model,
            model_X[tr], y_norm[tr], model_X[va], y_norm[va], device,
        )
        trained_models[model_name] = model
        histories[model_name] = history
        best_epochs[model_name] = {"best_epoch": best_epoch,
                                   "best_validation_loss_normalized": best_val}
        predictions[model_name] = {}
        for split_name, idx in split_indices.items():
            pred_norm = run_inference(model, model_X[idx], device)
            pred_original = inverse_target(pred_norm, target_min, target_max)
            predictions[model_name][split_name] = pred_original
            y_part = y_original[idx]
            metrics_rows.append({"model": model_name, "split": split_name,
                                 **evaluate_predictions(y_part, pred_original)})

        torch.save({
            "model_name": model_name,
            "model_state_dict": model.cpu().state_dict(),
            "feature_names": feature_names,
            "T": X_seq.shape[1],
            "target_min": target_min,
            "target_max": target_max,
            "best_epoch": best_epoch,
            "best_validation_loss_normalized": best_val,
        }, OUTPUT_DIR / f"{model_name.lower()}_shashi.pt")
        model.to(device)
        pd.DataFrame(history).to_csv(OUTPUT_DIR / f"loss_{model_name.lower()}.csv", index=False)
        print(f"{model_name}: best epoch={best_epoch}, normalized val MSE={best_val:.7f}")

    # Persistence baseline: predict tomorrow using the latest observed Shashi
    # in the input window. Current input convention stores day_7_shashi at [-1, 2].
    persistence = {}
    for split_name, idx in split_indices.items():
        pred = X_seq[idx, -1, 2].astype(np.float64)
        # Invert using the scaler for day_T_shashi, which is an input feature.
        persistence[split_name] = pred * (last_x_max - last_x_min) + last_x_min
        metrics_rows.append({"model": "Persistence", "split": split_name,
                             **evaluate_predictions(y_original[idx], persistence[split_name])})

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(OUTPUT_DIR / "comparison_metrics_original_scale.csv", index=False)
    metrics.to_excel(OUTPUT_DIR / "comparison_metrics_original_scale.xlsx",
                     index=False, engine="openpyxl")

    # Save comparable predictions and histories.
    with pd.ExcelWriter(OUTPUT_DIR / "comparison_predictions.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        for split_name, idx in split_indices.items():
            part = pd.DataFrame({"time": times.iloc[idx].to_numpy(),
                                 "actual_shashi": y_original[idx]})
            for model_name in constructors:
                part[f"{model_name}_prediction"] = predictions[model_name][split_name]
            part["Persistence_prediction"] = persistence[split_name]
            part.to_excel(writer, sheet_name=split_name, index=False)
    save_model_loss_plot(histories)
    all_predictions = {**predictions, "Persistence": persistence}
    save_fit_plots(times, {"all": y_original,
                           **{name: y_original[idx] for name, idx in split_indices.items()}},
                   all_predictions, split_indices)

    # Export split bounds, learning details, and inverse-transform constants.
    metadata = {
        "input_file": str(DATA_FILE), "parameter_file": str(param_path),
        "target_min": target_min, "target_max": target_max,
        "split_indices": {k: [int(v[0]), int(v[-1]) + 1] for k, v in split_indices.items()},
        "split_dates": {k: [str(times.iloc[v[0]].date()), str(times.iloc[v[-1]].date())]
                        for k, v in split_indices.items()},
        "best_epochs": best_epochs,
        "seed": SEED,
    }
    np.save(OUTPUT_DIR / "comparison_metadata.npy", metadata, allow_pickle=True)

    print("\nOriginal-scale metrics (lower error / higher R2 is better):")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nAll artifacts saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
