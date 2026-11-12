# prep/inspect_setpoint.py —— 验证 setpoint 是否存在昼夜降额模式（夜间噪声模式）
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\24117\Desktop\大二\大二上\工程域人工智能\风电智能体\data\processed")

for year in (2019, 2022, 2023):
    f = next((BASE / f"data_parquet_{year}").glob("Turbine_Data_Kelmarsh_1_*.parquet"))
    d = pd.read_parquet(f, columns=["Date and time", "Turbine Power setpoint (kW)"])
    sp = d["Turbine Power setpoint (kW)"]
    d = d.assign(hour=pd.to_datetime(d["Date and time"]).dt.hour)
    print(f"\n===== {year} T1 =====")
    print(f"setpoint < 2000 kW 的时间占比: {(sp < 2000).mean():.1%}")
    print(f"setpoint 最常见的 8 个值 (kW):")
    print(sp.round(0).value_counts().head(8).to_string())
    print("各小时 setpoint 中位数 (kW):")
    print(d.groupby("hour")["Turbine Power setpoint (kW)"].median().round(0).to_string())