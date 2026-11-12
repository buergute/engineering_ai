# prep/scan_curtailment.py —— 扫描全部 36 个 Status 文件，确认限电事件的来源
import pandas as pd
from pathlib import Path

# parquet 数据在桌面，不在 clone 的仓库里
BASE = Path(r"C:\Users\24117\Desktop\大二\大二上\工程域人工智能\风电智能体\data\processed")

for year in range(2019, 2025):
    for f in sorted((BASE / f"data_parquet_{year}").glob("Status_*.parquet")):
        st = pd.read_parquet(f, columns=["Status", "Code", "Message", "IEC category"])
        tb = f.stem.split("_")[3]
        hit = st[st["Message"].str.contains(
            "curtail|Curtail|derate|Derate|noise|Noise|setpoint|Setpoint", na=False)]
        stop = st[st["Status"] == "Stop"]
        print(f"{year} T{tb}: 总行数={len(st)}, Stop事件={len(stop)}, 限电相关={len(hit)}")
        if len(hit):
            print(hit[["Status", "Code", "Message"]].value_counts().head(5).to_string())