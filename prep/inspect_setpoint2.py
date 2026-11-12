# prep/inspect_setpoint2.py —— 破解 setpoint 语义：412 是什么？~1250 真的封顶吗？
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"C:\Users\24117\Desktop\大二\大二上\工程域人工智能\风电智能体\data\processed")
f = BASE / "data_parquet_2019" / "Turbine_Data_Kelmarsh_1_2019-01-01_-_2020-01-01_228.parquet"
d = pd.read_parquet(f, columns=[
    "Date and time", "Power (kW)", "Wind speed Sensor 1 (m/s)",
    "Turbine Power setpoint (kW)", "Rotor speed (RPM)",
    "Lost Production to Curtailment (Noise) (kWh)",
    "Lost Production to Curtailment (Sector Management) (kWh)",
    "Available Capacity for Production (kW)"])
d["hour"] = pd.to_datetime(d["Date and time"]).dt.hour
sp, p, w = d["Turbine Power setpoint (kW)"], d["Power (kW)"], d["Wind speed Sensor 1 (m/s)"]
state = np.where(d["Rotor speed (RPM)"] < 1, "stopped", "producing")

# Q1: setpoint 分档 × 运行状态
d["sp_bin"] = pd.cut(sp, [0, 500, 1000, 1500, 1900, 2100],
                     labels=["<500", "500-1000", "1000-1500", "1500-1900", "1900+"])
print("【Q1】setpoint 分档 × 状态："); print(pd.crosstab(d["sp_bin"], state))
print("setpoint 为 NaN 的行数:", sp.isna().sum())

# Q2: setpoint=412 时机组在干什么
m = d[sp.round(0) == 412]
print(f"\n【Q2】setpoint=412 共 {len(m)} 行 | 功率中位 {m['Power (kW)'].median():.0f} kW | "
      f"转速中位 {m['Rotor speed (RPM)'].median():.1f} | 风速中位 {m['Wind speed Sensor 1 (m/s)'].median():.1f} m/s")

# Q3: 关键测试——大风且 setpoint~1250 时，功率被封住了吗？
hi = d[(w > 11) & (sp > 1000) & (sp < 1500)]
print(f"\n【Q3】风速>11 且 setpoint∈(1000,1500)：{len(hi)} 行")
print(f"  setpoint 中位 {hi['Turbine Power setpoint (kW)'].median():.0f} | 功率中位 {hi['Power (kW)'].median():.0f} | "
      f"功率>setpoint 的比例 {(hi['Power (kW)'] > hi['Turbine Power setpoint (kW)']).mean():.1%}")
lg = lambda c: d[c].diff().clip(lower=0).groupby(d["sp_bin"]).sum().round(0)
print("  各 setpoint 档位的 Curtailment 损失增量（kWh/年）：")
print("  Noise：", lg("Lost Production to Curtailment (Noise) (kWh)").to_dict())
print("  Sector：", lg("Lost Production to Curtailment (Sector Management) (kWh)").to_dict())

# Q4: 对照组——大风且 setpoint=2060
full = d[(w > 11) & (sp >= 2000)]
print(f"\n【Q4】风速>11 且 setpoint≥2000：{len(full)} 行 | 功率中位 {full['Power (kW)'].median():.0f} kW")

# Q5: Available Capacity 列是不是真正的"可用容量"
ac = d["Available Capacity for Production (kW)"]
print(f"\n【Q5】Available Capacity：中位 {ac.median():.0f} | 众数 {ac.round(-2).mode().head(3).tolist()}")
print(pd.crosstab(d["sp_bin"], ac.round(-2)).iloc[:, :5])