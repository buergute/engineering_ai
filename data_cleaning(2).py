from pathlib import Path

import numpy as np
import pandas as pd


# ========= 可修改的设置 =========
INPUT_FILE = Path("data/Processed.xlsx")
OUTPUT_DIR = Path("data")
T = 7  # 使用连续历史 T 天预测未来 1 天
# ===============================


def handle_missing_values(df, method):
    """处理 sanxia 和 gaobazhou 的缺失值，并返回处理后的副本及记录。"""
    result = df.copy()
    columns = ["sanxia", "gaobazhou"]
    info = {}

    if method == "fill0":
        for col in columns:
            info[col] = {"method": "fill0"}
            result[col] = result[col].fillna(0)

    elif method == "fill_mean":
        for col in columns:
            mean_value = result[col].mean()
            info[col] = {
                "method": "fill_mean",
                "mean": float(mean_value),
            }
            result[col] = result[col].fillna(mean_value)

    elif method == "fill_previous":
        for col in columns:
            # 用前一天的值向后填充。
            # 若序列最开始就缺失、没有前一天可用，则用 0 补上。
            before_count = int(result[col].isna().sum())
            result[col] = result[col].ffill()
            remaining_count = int(result[col].isna().sum())
            result[col] = result[col].fillna(0)

            info[col] = {
                "method": "fill_previous",
                "missing_before": before_count,
                "missing_without_previous_fallback_to_0": remaining_count,
            }
    else:
        raise ValueError(f"不支持的空值处理方法：{method}")

    return result, info


def transform_features(feature_array, method):
    """处理输入特征，返回处理后数组和所用参数。"""
    params = {"method": method}

    if method == "raw":
        return feature_array.copy(), params

    if method == "minmax":
        feature_min = np.min(feature_array, axis=0)
        feature_max = np.max(feature_array, axis=0)
        feature_range = feature_max - feature_min

        # 避免某个特征在所有样本中均为常数时除以 0。
        safe_range = np.where(feature_range == 0, 1.0, feature_range)
        transformed = (feature_array - feature_min) / safe_range

        params.update({
            "min": feature_min,
            "max": feature_max,
            "range": feature_range,
            "constant_feature_mask": feature_range == 0,
        })
        return transformed, params

    if method == "standard":
        mean = np.mean(feature_array, axis=0)
        std = np.std(feature_array, axis=0, ddof=0)

        # 常数特征的标准差设为 1，处理后的该特征为 0。
        safe_std = np.where(std == 0, 1.0, std)
        transformed = (feature_array - mean) / safe_std

        params.update({
            "mean": mean,
            "std": std,
            "constant_feature_mask": std == 0,
        })
        return transformed, params

    raise ValueError(f"不支持的特征处理方法：{method}")


def minmax_target(target_array):
    """对 target 单独进行 Min-Max 归一化。"""
    target_min = float(np.min(target_array))
    target_max = float(np.max(target_array))
    target_range = target_max - target_min
    safe_range = target_range if target_range != 0 else 1.0

    normalized = (target_array - target_min) / safe_range
    params = {
        "method": "minmax",
        "min": target_min,
        "max": target_max,
        "range": target_range,
        "constant_target": target_range == 0,
    }
    return normalized, params


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到源文件：{INPUT_FILE}")

    if T < 1:
        raise ValueError("T 必须大于等于 1。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 读取 Excel，并检查列名。
    df = pd.read_excel(INPUT_FILE, engine="openpyxl")

    required_columns = ["time", "sanxia", "gaobazhou", "shashi"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Excel 缺少必要列：{missing_columns}；"
            f"实际列名为：{list(df.columns)}"
        )

    # 只使用所需列，并将时间转换为日期格式。
    df = df[required_columns].copy()
    df["time"] = pd.to_datetime(df["time"], errors="raise")

    # 数据按时间升序排列，确保历史窗口的顺序正确。
    df = df.sort_values("time").reset_index(drop=True)

    if len(df) <= T:
        raise ValueError(
            f"数据行数为 {len(df)}，必须大于 T={T} 才能生成预测样本。"
        )

    # 2. 定义两类处理方法，共 3 × 3 种组合。
    missing_methods = {
        "fill0": "0填充",
        "fill_mean": "均值填充",
        "fill_previous": "前一天填充",
    }
    feature_methods = {
        "raw": "原始值",
        "minmax": "归一化",
        "standard": "标准化",
    }

    # 3. 逐一生成九种结果。
    for missing_key, missing_label in missing_methods.items():
        filled_df, missing_info = handle_missing_values(df, missing_key)

        # 组织连续 T 天的输入特征。
        # 每天的特征顺序为：sanxia、gaobazhou、shashi。
        feature_columns = ["sanxia", "gaobazhou", "shashi"]
        feature_rows = []
        target_rows = []
        sample_times = []

        values = filled_df[feature_columns].to_numpy(dtype=np.float64)
        targets = filled_df["shashi"].to_numpy(dtype=np.float64)
        times = filled_df["time"].to_numpy()

        # i 是目标日的位置；使用 i-T 到 i-1 共 T 天作为输入。
        for i in range(T, len(filled_df)):
            window = values[i - T:i, :]
            feature_rows.append(window.reshape(-1))
            target_rows.append(targets[i])
            # 每条样本记录其预测目标对应的日期。
            sample_times.append(times[i])

        X = np.asarray(feature_rows, dtype=np.float64)
        y = np.asarray(target_rows, dtype=np.float64)
        sample_times = pd.to_datetime(sample_times)

        # 可读的特征列名，例如 day_1_sanxia ... day_7_shashi。
        input_feature_names = [
            f"day_{day}_{column}"
            for day in range(1, T + 1)
            for column in feature_columns
        ]

        # 保存生成窗口后的原始中间数据（尚未做特征缩放）。
        np.save(
            OUTPUT_DIR / f"intermediate_X_T{T}_{missing_key}.npy",
            X,
        )
        np.save(
            OUTPUT_DIR / f"intermediate_y_T{T}_{missing_key}.npy",
            y,
        )
        np.save(
            OUTPUT_DIR / f"intermediate_time_T{T}_{missing_key}.npy",
            sample_times.to_numpy(dtype="datetime64[ns]"),
        )

        # 目标列单独做 Min-Max 归一化，不受输入特征处理方式影响。
        y_normalized, target_params = minmax_target(y)

        for feature_key, feature_label in feature_methods.items():
            X_processed, feature_params = transform_features(X, feature_key)

            # 输出每种组合的 Excel：第一列时间，中间是 3×T 个输入特征，
            # 最后一列为归一化后的 target。
            output_df = pd.DataFrame(X_processed, columns=input_feature_names)
            output_df.insert(0, "time", sample_times)
            output_df["target"] = y_normalized

            output_file = (
                OUTPUT_DIR
                / f"shashi_T{T}_{feature_key}_{missing_key}.xlsx"
            )
            output_df.to_excel(output_file, index=False, engine="openpyxl")

            # 保存处理后矩阵。
            np.save(
                OUTPUT_DIR / f"X_T{T}_{feature_key}_{missing_key}.npy",
                X_processed,
            )
            np.save(
                OUTPUT_DIR / f"target_T{T}_{feature_key}_{missing_key}.npy",
                y_normalized,
            )

            # 将方法参数和中间信息保存为字典型 .npy 文件。
            # 加载该文件时使用 np.load(path, allow_pickle=True).item()
            params = {
                "T": T,
                "source_file": str(INPUT_FILE),
                "output_file": str(output_file),
                "input_feature_names": input_feature_names,
                "feature_method": feature_key,
                "feature_params": feature_params,
                "missing_method": missing_key,
                "missing_params": missing_info,
                "target_params": target_params,
                "sample_count": len(output_df),
                "input_feature_count": X_processed.shape[1],
                "sample_time_semantics": "目标日（预测未来 shashi 的日期）",
            }
            params_file = (
                OUTPUT_DIR
                / f"params_T{T}_{feature_key}_{missing_key}.npy"
            )
            np.save(params_file, params, allow_pickle=True)

            print(f"已生成：{output_file}")
            print(f"已保存参数：{params_file}")

    print("\n全部 9 种处理结果已完成。")


if __name__ == "__main__":
    main()