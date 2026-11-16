# 界面（负责人：C）

Streamlit 前端：展示请求输入、预测与复核结论，以及可选的思考链 / 工具调用过程。  
编排与数字仍由 `agent/` 状态机和工具产出，界面只负责展示与触发运行。

## 启动

在仓库根目录、已安装依赖的环境中（例如 `eng-ai`）：

```bash
python -m streamlit run ui/app.py
```

Windows 若未激活 conda，可用环境内 Python：

```powershell
D:\Anaconda3\envs\eng-ai\python.exe -m streamlit run ui/app.py
```

浏览器打开终端提示的 Local URL（默认多为 `http://localhost:8501`）。

## 界面结构

| 区域 | 内容 |
| --- | --- |
| 标题 | 产品名与一句话能力说明 |
| 侧栏 | 数据源（假工具 / 合成数据）、假工具场景、起报时刻、是否默认展开思考链 |
| 输入 | 引导文案、示例按钮、自然语言请求框、「运行」 |
| 输出 | 结论、预测表、复核证据、完整文本报告 |
| 思考链 | 折叠面板：思考 / 决策 / 工具调用；附原始轨迹表 |

思考链由 `agent/behavior_view.py` 的 `build_behavior_chain(out)` 根据一次 `run` 的结果拼出，不改状态机。

## 数据源说明

- **假工具**：不读真实/合成数据，用 `agent.mock_tools` 场景（如 `normal`、`curtailment`）快速演示。
- **合成数据 + 真实工具**：需要先具备 `artifacts/models.pkl` 与 `data/synth/scada.parquet`（通常跑完 `scripts/run_all.sh`）。缺文件时侧栏会提示改用假工具。

## 相关文件

| 路径 | 作用 |
| --- | --- |
| `ui/app.py` | Streamlit 入口 |
| `agent/behavior_view.py` | 把运行结果整理成界面用的思考链步骤 |
| `agent/behavior_log.py` | 行为日志辅助（可选落盘；当前主路径未强制接入状态机） |
| `agent/state_machine.py` | 实际编排 |
| `agent/parse.py` | 请求解析（当前为规则版占位） |

## 注意

- 自然语言解析目前是规则版，不是大模型。
- 不要把本机大 CSV、模型产物或 `agent/logs/` 运行记录提交进仓库。
