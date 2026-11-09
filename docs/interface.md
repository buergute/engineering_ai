# 工具接口约定 v1

11/8 三人确认后冻结，之后改动需三人同意，并同步修改 `agent/mock_tools.py` 和测试。

## 公共字段

所有工具都返回 `common.schema.ToolResult`：

```json
{
  "tool": "forecast_power",
  "status": "ok | empty | error",
  "as_of": "2019-07-14T14:00",
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
- 请求起报时刻之后的数据，抛 `FutureDataError`。
- `evidence_ids` 是报告里可以引用的证据编号；报告引用的每个编号都必须能在某个工具结果里找到。

## 各工具

| 工具 | 负责人 | 函数签名 | `result` 里的主要内容 |
| --- | --- | --- | --- |
| read_asof | A | `read_asof(turbine, as_of, window_hours=6, include_neighbors=True)` | series, missing_mask, neighbors |
| check_quality | A | `check_quality(data)` | issues, quality_flags, to_check |
| check_power_curve | A | `check_power_curve(data, turbine, wind_source="own" \| "neighbors")` | wind_speed_used, expected_power, actual_power, residual, curve_source |
| query_evidence | A | `query_evidence(turbine, as_of, lookback_hours=24)` | events（含 available_at）, neighbor_comparison |
| forecast_power | B | `forecast_power(data, quality, turbine, horizons_minutes, use_fallback)` | forecasts: [{horizon_min, p50, lower, upper, coverage}], model |
| validate_result | C | `validate_result(as_of, results, cited_evidence)` | ok, errors |

每个工具的详细说明见对应文件的 docstring，示例输出见 `agent/mock_tools.py`。
