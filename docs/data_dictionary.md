# 数据说明与字段字典（负责人：A，v1 截止 11/9）

## 数据来源

| 项目 | 内容 |
| --- | --- |
| 数据集 | Kelmarsh wind farm SCADA v4 |
| 入口 | https://zenodo.org/records/16807551 |
| 版本 / 下载日期 | v4 / 2026-11-08 |
| 许可证 | CC-BY-4.0 DOI:10.5281/zenodo.16807551 |
| 文件校验和 | Zenodo zip 官方 MD5：2019_3085: 14348a1572b722e1be10e53cd145c39a；2020_3086: e189e02da98f32e74acc72e8559ef0db；2021_4456: 4e0ef3d65f067fd3d8f3d30e79ab01ae；2022_4457: b695d2b21cd5a236b2e690e0a43df5ab；2023_5961: 685d9589347d1866f6f2cfe45b9af4f4；2024_5962: e9e3318f5b393d8596ff1801f8da0414；本地解压后 CSV 的 SHA256 清单：data/checksums_kelmarsh.txt |
| 使用的机组和时段 | 6 台 Senvion MM92（Kelmarsh 1–6 号机组，单机额定 2.05 MW），2019-01-01 至 2024-12-31，10 分钟分辨率 |

备选：SDWPF（https://doi.org/10.6084/m9.figshare.24798654），若 11/7 前 Kelmarsh 接不通则切换。

## 字段字典

| 原始字段名 | 统一后的名字 | 含义 | 单位 | 缺失率 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Power (kW) | power | 有功功率 | kW（入库时 ÷1000 转 MW） | 0.2% | 保留负值，单独统计 |
| Wind speed Sensor 1 (m/s) | wind_speed | 机舱风速（风速计 1） | m/s | 0.4% | 不用组合风速列（两者平均差 0.45 m/s） |
| Wind speed Sensor 2 (m/s) | wind_speed_2 | 机舱风速（风速计 2） | m/s | 0.2% | 用于风速计一致性检测 |
| Wind speed Sensor 1, Standard deviation (m/s) | wind_std | 风速标准差 | m/s | 4.8% | 无 |
| Wind speed Sensor 1, Maximum (m/s) | wind_max | 风速最大值 | m/s | 4.8% | 无 |
| Wind speed Sensor 1, Minimum (m/s) | wind_min | 风速最小值 | m/s | 5.7% | 无 |
| Blade angle (pitch position) A (°) | pitch | 桨距角（A 叶片） | ° | 4.6% | 三支叶片取 A |
| Rotor speed (RPM) | rpm | 转子转速 | RPM | 0.2% | 无 |

（缺失率为 2019 年 T1 实测，其余机组/年份相近）

## 时间约定

- 原始时间戳为 10 分钟区间的**开始**时刻（已用停机事件锚点法实测确认：2019-01-15 12:09:26 停机后，12:10 行功率即为 -3 kW）。入库时统一 +10 分钟偏移为**区间结束**时刻，与系统约定一致（到达该时刻数据才可见）。
- 事件按"系统可见时间"使用；可见时间不明的事件只用于离线复核。
- 时间切分见 `configs/split.yaml`。

## 扩展字段与缺失处理

合成数据使用以下宽表列；真实 Kelmarsh 原始字段名、单位和可用性须由 A 对照实际下载版本核验，当前不能把下表中的含义当作已确认的原始列名。

| 合成列 | Kelmarsh 原始字段名 | 含义 | 单位 | 可用性（缺失率） |
| --- | --- | --- | --- | --- |
| `p_<机组>` | Power (kW) | 有功功率 | 原始 kW，入库 ÷1000 转 MW | 已确认（0.2%） |
| `w_<机组>` | Wind speed Sensor 1 (m/s) | 风速计 1 | m/s | 已确认（0.4%） |
| `w2_<机组>` | Wind speed Sensor 2 (m/s) | 风速计 2 | m/s | 已确认（0.2%） |
| `ws_<机组>` | Wind speed Sensor 1, Standard deviation (m/s) | 风速标准差 | m/s | 已确认（4.8%） |
| `wx_<机组>` | Wind speed Sensor 1, Maximum (m/s) | 风速最大值 | m/s | 已确认（4.8%） |
| `wn_<机组>` | Wind speed Sensor 1, Minimum (m/s) | 风速最小值 | m/s | 已确认（5.7%） |
| `pt_<机组>` | Blade angle (pitch position) A (°) | 桨距角（A 叶片） | ° | 已确认（4.6%） |
| `rpm_<机组>` | Rotor speed (RPM) | 转子转速 | RPM | 已确认（0.2%） |

机组编号统一为 T01–T06（对应 Kelmarsh 1–6 号机组）。7 个字段全部可用，无“可能缺失”情形。

缺哪个字段就去掉依赖它的规则；如果没有其他正面证据，相应情形归入“无法解释”，报告中说明缺失字段。不能仅凭其他规则不成立推断原因。
