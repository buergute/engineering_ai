# prep/inspect_curtailment.py —— 细查 2022/2023 的限电事件，并用 SCADA 交叉验证一条
import pandas as pd
from pathlib import Path

BASE = Path(r"C:\Users\24117\Desktop\大二\大二上\工程域人工智能\风电智能体\data\processed")

def secs(d):
    try:
        h, m, s = str(d).split(":"); return int(h)*3600 + int(m)*60 + int(s)
    except Exception:
        return None

# ── 第 1 部分：2022/2023 每台机组的 Curtailment 和 Fallback setpoints 明细 ──
for year in (2022, 2023):
    for f in sorted((BASE / f"data_parquet_{year}").glob("Status_*.parquet")):
        tb = int(f.stem.split("_")[2])          # 机组号在文件名第 3 段
        st = pd.read_parquet(f, columns=["Timestamp start", "Timestamp end",
                                         "Duration", "Status", "Code", "Message"])
        cur = st[st["Status"] == "Curtailment"].copy()
        fb  = st[st["Code"] == 8405].copy()
        if not len(cur) and not len(fb):
            continue
        print(f"\n===== {year} T{tb:02d} =====")
        if len(cur):
            cur["dur_min"] = cur["Duration"].map(secs).astype("float") / 60
            print(f"-- Curtailment {len(cur)} 条 | 总时长 {cur['dur_min'].sum():.0f} 分钟 | "
                  f"最早 {cur['Timestamp start'].min()} | 最晚 {cur['Timestamp start'].max()}")
            print(cur[["Timestamp start", "Timestamp end", "dur_min", "Code", "Message"]]
                  .head(8).to_string(index=False))
        if len(fb):
            fb["dur_min"] = fb["Duration"].map(secs).astype("float") / 60
            print(f"-- Fallback setpoints(8405) {len(fb)} 条 | 总时长 {fb['dur_min'].sum():.0f} 分钟")
            print(fb[["Timestamp start", "Timestamp end", "dur_min"]].head(4).to_string(index=False))

# ── 第 2 部分：拿 2022 年 T1 的第一条限电，看 SCADA 里功率/设定值/潜在功率的表现 ──
print("\n\n========== SCADA 交叉验证：2022 T1 第一条限电事件 ==========")
st = pd.read_parquet(BASE / "data_parquet_2022" / "Status_Kelmarsh_1_2022-01-01_-_2023-01-01_228.parquet",
                     columns=["Timestamp start", "Timestamp end", "Status", "Code", "Message"])
cur = st[st["Status"] == "Curtailment"]
e = cur.iloc[0]
print(e[["Timestamp start", "Timestamp end", "Code", "Message"]].to_string(), "\n")

sc = pd.read_parquet(BASE / "data_parquet_2022" / "Turbine_Data_Kelmarsh_1_2022-01-01_-_2023-01-01_228.parquet",
                     columns=["Date and time", "Power (kW)", "Turbine Power setpoint (kW)",
                              "Potential power default PC (kW)",
                              "Lost Production to Curtailment (Total) (kWh)"]).set_index("Date and time")
t0, t1 = pd.Timestamp(e["Timestamp start"]), pd.Timestamp(e["Timestamp end"])
win = sc.loc[t0 - pd.Timedelta(hours=1): t1 + pd.Timedelta(hours=1)]
print(win.iloc[::6].to_string())     # 每小时采一个点