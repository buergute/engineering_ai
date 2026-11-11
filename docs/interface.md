# 工具接口约定 v3

改动需三人同意，并同步修改 `agent/mock_tools.py` 和测试。

## 公共字段

所有工具都返回 `common.schema.ToolResult`：

```json
{
  "tool": "forecast_power",
  "status": "ok | empty | error",
  "as_of": "2023-03-04T14:00",
  "data_hash": "...",
  "tool_version": "...",
  "units": {"power": "MW", "wind_speed": "m/s"},
  "evidence_ids": ["F1"],
  "warnings": [],
  "run_id": "...",
  "result": {}
}
```

- 查不到东西返回 `status: "empty"`，不要编造。
- 读到起报时刻之后的数据，抛 `FutureDataError`；校验器也会再查一遍 `latest_time`。
- `evidence_ids` 是报告里可以引用的证据编号；报告引用的每个编号都必须能在某个工具结果里找到。

## 数据访问

工具不直接读文件，统一通过 `common.store.DataStore`。宽表列名：`p_<机组>`（MW）、`w_<机组>`（m/s），时间戳为 10 分钟区间的结束时刻。
换成真实 Kelmarsh 数据时，只需重写 `DataStore.load()`，工具和模型都不用改。
需要数据的工具第一个参数是 `store`，由 `agent.toolbox.real_toolbox(store)` 绑定，所以状态机看到的签名与下表一致。

## 各工具

| 工具 | 负责人 | 状态机看到的签名 | `result` 里的主要内容 |
| --- | --- | --- | --- |
| read_asof | A | `read_asof(turbine, as_of, window_hours=7)` | time, series（本机 + 邻机）, latest_time, missing_count |
| review_context | A | `review_context(data, turbine)` | power_jump, neighbor_sync, turbulence, operation, thresholds；证据 ID V1 |
| check_quality | A | `check_quality(data)` | issues: [{id, turbine, field, type: gap / random_missing / frozen / out_of_range / drift / anemometer_mismatch, detail}] |
| check_power_curve | A | `check_power_curve(data, turbine, wind_source="own" \| "neighbors", exclude_neighbors=())` | wind_speed_used, expected_power, actual_power, residual, neighbors_used, curve_source |
| query_evidence | A | `query_evidence(turbine, as_of, lookback_hours=24)` | events: [{id, type, start, end（未结束为 null）, active_at_as_of, available_at, setpoint_mw}] |
| forecast_power | B | `forecast_power(data, quality, turbine, horizons_minutes, use_fallback, exclude_neighbors, cap_mw)` | forecasts: [{horizon_min, p50, lower, upper, coverage}], model, calibration_group, curtailment_cap_mw |
| validate_result | C | `validate_result(as_of, results, cited_evidence, rated_mw)` | ok, errors |

## 从训练段 / 验证段来的参数

| 文件 | 由谁生成 | 内容 |
| --- | --- | --- |
| `artifacts/reference.json` | `models.train`（train 段） | 每台机的经验功率曲线、两两风速比例 |
| `artifacts/models.pkl` | `models.train`（train 训练、val 早停、calib 校准） | 三种模型 × 三个时距 × 三个分位数，分组 CQR 外扩量 |
| `artifacts/thresholds.json` | `eval.thresholds`（val 段） | 欠发判定阈值，按风速来源（本机 / 两台邻机 / 一台邻机）分开 |


## 复核数据与判定

`read_asof` 本机可返回 `wind_speed_2`、`wind_std`、`wind_max`、`wind_min`、`pitch`、`rpm`；邻机额外字段只返回 `wind_speed_2`。缺失字段不出现。`Toolbox.review_context` 可为 `None`，状态机跳过上下文复核。

复核按数据问题、运行状态变化、风况变化、无法解释顺序检查正面证据：`data_issue_only` 表示数据问题；`operating_change` 表示运行状态变化；`normal` 且 `primary_cause=wind_change` 表示风况变化；`undetermined` 表示无法解释。不可通过排除法推定原因。

`thresholds.json` 的 `review` 包含 `power_jump_mw`（30 分钟功率变化）、`anemometer_diff_ms`（风速计差异）、`turbulence_ti`（湍流强度后备阈值）、`turbulence_curve`（按风速分组的正常标准差分位数）、`pitch_deg`（桨距角）、`rpm_stopped`（停机转速）、`sync_ratio`（邻机变化同步比例）和 `n_clean`（拟合样本数）；均来自验证段或明确的规则参数。
