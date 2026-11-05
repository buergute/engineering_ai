
# Engineering AI Projects

工程数据处理、时间序列建模与风速预测项目集合。

This repository contains projects for engineering data processing, time-series modeling, and wind-speed forecasting.

## 项目内容 | Projects

仓库目前包含两个主要工作流：

This repository currently contains two main workflows:

1. **水位时间序列分析（仓库根目录） / Water-level time-series analysis (repository root)**：清洗三峡、高坝洲和沙市水位数据，构造连续 7 天输入窗口，并比较 FCNN、LSTM、GRU、TCN 模型以及特征工程方案。
2. **天气作业（`weather_assignment/`） / Weather assignment (`weather_assignment/`)**：使用美国 Iowa Environmental Mesonet 的 CNC 站观测数据，对风速进行两天提前量预测，并导出预测结果和作业文件。

## 目录结构 | Repository layout

| 路径 / Path | 内容 / Contents |
| --- | --- |
| `data/` | 水位原始/清洗后表格、预处理数据和缩放参数 / Water-level source and processed tables, model inputs, and scaling parameters |
| `data_cleaning.py` | 清洗原始 Excel 表格，生成 `data/Processed.xlsx` / Clean the source Excel workbook and create `data/Processed.xlsx` |
| `data_cleaning(2).py` | 生成 7 天窗口、缺失值处理和特征缩放的 9 种组合 / Generate seven-day windows across nine missing-value and scaling combinations |
| `output/` | FCNN 与多模型比较脚本、训练结果、图表和预测表 / FCNN and model-comparison scripts, results, plots, and prediction tables |
| `output_feature_eng/` | 特征工程实验脚本及其结果 / Feature-engineering experiment and results |
| `weather_assignment/data/` | CNC 站风速及气象观测 CSV / CNC wind and weather observation CSV |
| `weather_assignment/forecast.py` | 风速预测、时间顺序验证与测试 / Wind forecasting with chronological validation and testing |
| `weather_assignment/output/` | 预测结果、模型报告及导出的作业文件 / Forecasts, model report, and assignment deliverables |

## 环境要求 | Requirements

仓库没有锁定依赖版本或提供依赖清单。水位建模脚本使用 Python、NumPy、pandas、openpyxl、Matplotlib、scikit-learn 和 PyTorch；天气预测脚本使用 Python、NumPy、pandas 和 scikit-learn。请按你的操作系统和硬件安装合适的 PyTorch 版本。

This repository does not pin dependency versions or include a dependency manifest. The water-level modeling scripts use Python, NumPy, pandas, openpyxl, Matplotlib, scikit-learn, and PyTorch. The weather forecasting script uses Python, NumPy, pandas, and scikit-learn. Install the PyTorch build appropriate for your operating system and hardware.

例如，可先安装常用 Python 依赖：

For example, install the common Python dependencies with:

```bash
python -m pip install numpy pandas openpyxl matplotlib scikit-learn
```

PyTorch 请根据设备选择对应安装方式。以下命令均假设当前目录为仓库根目录。

Install PyTorch using the option appropriate for your device. The commands below assume you are in the repository root.

## 运行方式 | Run the workflows

### 水位数据处理 | Water-level data preparation

`data_cleaning.py` 优先查找 `data/Original.xlsx`，如果不存在则使用仓库中提供的 `data/Original(1).xlsx`。运行：

`data_cleaning.py` first looks for `data/Original.xlsx` and falls back to the included `data/Original(1).xlsx`. Run:

```bash
python data_cleaning.py
python 'data_cleaning(2).py'
```

第二个脚本读取 `data/Processed.xlsx`，按时间排序，使用前 7 天的三站水位预测沙市下一天水位，并生成 9 组缺失值处理/特征缩放数据及其参数文件。

The second script reads `data/Processed.xlsx`, sorts rows by time, uses the previous seven days of three-station levels to predict the next day's Shashi level, and generates nine datasets with their preprocessing parameters.

### 水位模型实验 | Water-level model experiments

以下脚本从仓库根目录运行：

Run these scripts from the repository root:

```bash
python output/compare_timeseries_models.py
python output_feature_eng/feature_engineering_experiment.py
```

模型比较脚本训练 FCNN、LSTM、GRU 和 TCN，并在 `output/model_comparison/` 保存指标、预测表和图表。特征工程脚本比较基线 FCNN 与加入窗口统计、差分及日历特征的 FCNN，结果保存在 `output_feature_eng/`。

The model-comparison script trains FCNN, LSTM, GRU, and TCN models and writes metrics, prediction tables, and plots to `output/model_comparison/`. The feature-engineering script compares a baseline FCNN with an FCNN that also uses window statistics, differences, and calendar features; results are written to `output_feature_eng/`.

`output/train_fcnn.py` 是单独的 FCNN 训练脚本，其输入、参数和输出路径目前写成了原开发环境的绝对路径。在其他机器上运行前，需要在脚本顶部将这些路径改为本地路径。

`output/train_fcnn.py` is a standalone FCNN training script whose input, parameter, and output paths are currently absolute paths from the original development environment. Update the paths near the top of the script before running it elsewhere.

### 天气风速预测 | Weather wind-speed forecast

确保 `weather_assignment/data/CNC_IEM_2024-2026.csv` 存在后，在仓库根目录运行：

With `weather_assignment/data/CNC_IEM_2024-2026.csv` in place, run from the repository root:

```bash
python weather_assignment/forecast.py
```

脚本按预测起点顺序划分训练、验证和测试数据，比较直方图梯度提升与随机森林，并将预测及报告写入 `weather_assignment/output/`。当前预测起点和目标日期在脚本中配置；如用于新日期，请先检查并更新 `weather_assignment/forecast.py` 中的 `ISSUE` 和日期范围。

The script splits data chronologically, compares histogram gradient boosting with random forest, and writes forecasts and a report to `weather_assignment/output/`. The forecast origin and target date are configured in the script; review and update `ISSUE` and the date ranges in `weather_assignment/forecast.py` before using it for a new forecast date.

## 已有结果 | Included results

仓库已包含若干预处理数据、训练好的模型、评估表格、图表和作业导出文件。它们是当前实验的保存结果；重新运行脚本可能会覆盖相应输出。

The repository includes prepared datasets, trained models, evaluation tables, plots, and assignment exports. These are saved outputs from the current experiments; rerunning scripts may overwrite the corresponding files.

## 注意事项 | Notes

- `.gitignore` 排除了系统文件、Python 缓存、Node.js 依赖目录，以及天气作业的临时渲染和验证文件。/ `.gitignore` excludes system files, Python caches, Node.js dependencies, and temporary weather-assignment render and validation files.
- `weather_assignment/fill_template.py` 和演示文稿构建脚本依赖原开发环境中的模板或工具路径；使用这些辅助脚本前，请检查脚本中的路径和依赖。/ `weather_assignment/fill_template.py` and the presentation-building scripts depend on template or tool paths from the original development environment. Check their paths and dependencies before using them.
- 结果文件名中的 `姓名`、`布尔古特` 等是当前导出文件的名称，不是脚本自动识别的用户信息。/ Names such as `姓名` and `布尔古特` in the exported filenames are part of the current filenames; scripts do not infer user identity from them.
