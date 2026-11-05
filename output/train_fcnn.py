from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit  # documents the temporal intent; split below is exact 70/10/20
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------- Configuration -----------------------------
INPUT_FILE = Path("/Users/burgut/Downloads/engineering_ai/data/shashi_T7_minmax_fill_previous.xlsx")
PARAM_FILE = Path("/Users/burgut/Downloads/engineering_ai/data/params_T7_minmax_prev.npy")
# Actual parameter file created alongside the preprocessing outputs; used if the
# requested abbreviated name params_T7_minmax_prev.npy is not present.
PARAM_FALLBACK = Path("/Users/burgut/Downloads/engineering_ai/data/params_T7_minmax_fill_previous.npy")
OUTPUT_DIR = Path("/Users/burgut/Downloads/engineering_ai/output")

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 40
HIDDEN_SIZES = (128, 64, 32)
DROPOUT = 0.10


class FCNN(nn.Module):
    """Fully connected regression network."""

    def __init__(self, input_size, hidden_sizes=HIDDEN_SIZES, dropout=DROPOUT):
        super().__init__()
        layers = []
        previous = input_size
        for width in hidden_sizes:
            layers.extend([
                nn.Linear(previous, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def set_random_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_parameter_file():
    if PARAM_FILE.exists():
        return PARAM_FILE
    if PARAM_FALLBACK.exists():
        print(f"指定参数文件不存在，改用现有文件：{PARAM_FALLBACK}")
        return PARAM_FALLBACK
    raise FileNotFoundError(
        f"未找到参数文件：{PARAM_FILE}，也未找到备用文件：{PARAM_FALLBACK}"
    )


def load_data():
    """Load time, preprocessed X, normalized y, and target inverse parameters."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_FILE}")

    parameter_path = resolve_parameter_file()
    data = pd.read_excel(INPUT_FILE, engine="openpyxl")
    if data.shape[1] < 3:
        raise ValueError("输入 Excel 至少需要 time、一个特征列和 target。")
    if data.columns[0] != "time" or data.columns[-1] != "target":
        raise ValueError(
            f"预期首列为 time、末列为 target，实际列为：{list(data.columns)}"
        )

    times = pd.to_datetime(data.iloc[:, 0], errors="raise")
    feature_columns = list(data.columns[1:-1])
    X = data.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y = pd.to_numeric(data["target"], errors="raise").to_numpy(dtype=np.float32)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("输入特征或 target 中仍有 NaN/Inf，请先检查预处理结果。")
    if len(data) < 3:
        raise ValueError("样本数不足，无法划分训练、验证和测试集。")

    params = np.load(parameter_path, allow_pickle=True).item()
    # Support both the requested flat keys and the nested format in the existing
    # preprocessing parameter files.
    if "target_min" in params and "target_max" in params:
        target_min = float(params["target_min"])
        target_max = float(params["target_max"])
    elif "target_params" in params:
        target_params = params["target_params"]
        target_min = float(target_params["min"])
        target_max = float(target_params["max"])
    else:
        raise KeyError("参数文件中找不到 target_min/target_max 或 target_params。")

    if target_max <= target_min:
        raise ValueError("target_max 必须大于 target_min。")

    return times, X, y, feature_columns, target_min, target_max, parameter_path


def chronological_split(X, y, times):
    """Exact chronological 70% train, 10% validation, 20% test split."""
    n = len(X)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.80)
    if train_end == 0 or validation_end <= train_end or validation_end >= n:
        raise ValueError(f"样本数 {n} 不足以完成 70/10/20 顺序切分。")

    split = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n),
    }
    sets = {
        name: (X[start:end], y[start:end], times.iloc[start:end].reset_index(drop=True))
        for name, (start, end) in split.items()
    }
    return sets, split


def make_dataloaders(sets):
    train_X, train_y, _ = sets["train"]
    train_dataset = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    return train_loader


def train_model(train_loader, X_validation, y_validation, input_size, device):
    """Train using normalized y; early stop on normalized validation MSE."""
    model = FCNN(input_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.MSELoss()
    X_val_tensor = torch.from_numpy(X_validation).to(device)
    y_val_tensor = torch.from_numpy(y_validation).to(device)

    history = {"epoch": [], "train_loss": [], "validation_loss": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_X)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch_X)
            train_count += len(batch_X)

        model.eval()
        with torch.no_grad():
            validation_loss = criterion(model(X_val_tensor), y_val_tensor).item()
        train_loss = train_loss_sum / train_count
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.7f}, "
                f"val_loss={validation_loss:.7f}"
            )
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if best_state is None:
        raise RuntimeError("训练没有得到有效模型参数。")
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def save_loss_curve(history, output_dir):
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "loss_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train MSE")
    ax.plot(history["epoch"], history["validation_loss"], label="Validation MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on normalized target")
    ax.set_title("FCNN Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)


def predict_and_evaluate(model, sets, target_min, target_max, device):
    """Predict each split, inverse-transform y/prediction, then calculate metrics."""
    model.eval()
    results = {}
    rows = []
    with torch.no_grad():
        for name, (X_part, y_normalized, times) in sets.items():
            X_tensor = torch.from_numpy(X_part).to(device)
            prediction_normalized = model(X_tensor).cpu().numpy().astype(np.float64)

            # Min-Max inverse transform. Metrics are computed only after this step.
            y_original = y_normalized.astype(np.float64) * (target_max - target_min) + target_min
            prediction_original = (
                prediction_normalized * (target_max - target_min) + target_min
            )

            mse = mean_squared_error(y_original, prediction_original)
            mae = mean_absolute_error(y_original, prediction_original)
            rmse = float(np.sqrt(mse))
            r2 = r2_score(y_original, prediction_original)
            mape = float(
                np.mean(np.abs((y_original - prediction_original) / y_original)) * 100
            ) if np.all(y_original != 0) else float("nan")

            rows.append({
                "split": name,
                "n_samples": len(y_original),
                "MSE": mse,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_percent": mape,
            })
            results[name] = pd.DataFrame({
                "time": times,
                "actual_shashi": y_original,
                "predicted_shashi": prediction_original,
                "residual": y_original - prediction_original,
            })
    return results, pd.DataFrame(rows)


def save_fit_plots(results, output_dir):
    for split_name, result in results.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(result["time"], result["actual_shashi"], label="Actual", linewidth=1.4)
        ax.plot(result["time"], result["predicted_shashi"], label="Predicted", linewidth=1.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Shashi (original scale)")
        ax.set_title(f"FCNN {split_name.title()} Set: Actual vs Predicted")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_dir / f"fit_{split_name}.png", dpi=160)
        plt.close(fig)


def save_outputs(model, results, metrics, feature_columns, target_min, target_max,
                 parameter_path, split_indices, best_epoch, best_validation_loss,
                 device, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save a portable state_dict plus architecture and inverse-transform metadata.
    model_path = output_dir / "fcnn_shashi.pt"
    torch.save({
        "model_state_dict": model.cpu().state_dict(),
        "input_size": len(feature_columns),
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "feature_columns": feature_columns,
        "target_min": target_min,
        "target_max": target_max,
        "best_epoch": best_epoch,
        "best_validation_loss_normalized": best_validation_loss,
        "parameter_file_used": str(parameter_path),
    }, model_path)

    metrics.to_csv(output_dir / "metrics_original_scale.csv", index=False)
    with pd.ExcelWriter(output_dir / "predictions_and_metrics.xlsx", engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="metrics", index=False)
        for split_name, result in results.items():
            result.to_excel(writer, sheet_name=split_name, index=False)

    split_metadata = {
        "split_indices": split_indices,
        "n_samples": sum(len(part) for part in results.values()),
        "best_epoch": best_epoch,
        "best_validation_loss_normalized": best_validation_loss,
        "target_min": target_min,
        "target_max": target_max,
        "parameter_file_used": str(parameter_path),
        "device": str(device),
    }
    np.save(output_dir / "run_metadata.npy", split_metadata, allow_pickle=True)
    return model_path


def main():
    """Run loading, splitting, training, prediction, evaluation, and plotting."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_random_seed()

    times, X, y, feature_columns, target_min, target_max, parameter_path = load_data()
    sets, split_indices = chronological_split(X, y, times)
    train_loader = make_dataloaders(sets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; samples: {len(X)}; input features: {X.shape[1]}")
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(part[0])}" for name, part in sets.items())
    )

    model, history, best_epoch, best_val_loss = train_model(
        train_loader,
        sets["validation"][0],
        sets["validation"][1],
        input_size=X.shape[1],
        device=device,
    )
    save_loss_curve(history, OUTPUT_DIR)

    results, metrics = predict_and_evaluate(
        model, sets, target_min, target_max, device
    )
    save_fit_plots(results, OUTPUT_DIR)
    model_path = save_outputs(
        model=model,
        results=results,
        metrics=metrics,
        feature_columns=feature_columns,
        target_min=target_min,
        target_max=target_max,
        parameter_path=parameter_path,
        split_indices=split_indices,
        best_epoch=best_epoch,
        best_validation_loss=best_val_loss,
        device=device,
        output_dir=OUTPUT_DIR,
    )

    print("\nMetrics on original target scale:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nSaved model: {model_path}")
    print(f"Saved outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
