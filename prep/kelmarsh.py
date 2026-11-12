# prep/kelmarsh.py —— Kelmarsh 原始数据 → 系统标准三件套（scada.parquet / events.csv / meta.json）
# 运行：python prep/kelmarsh.py   （在项目根目录下运行）
# 输入：桌面的 6 个年份文件夹里的 parquet（只读，不改动）
# 输出：engineering_ai/data/kelmarsh/ 下的三个文件
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ───────────────────── 配置 ─────────────────────
SRC = Path(r"C:\Users\24117\Desktop\大二\大二上\工程域人工智能\风电智能体\data\processed")
OUT = Path(__file__).resolve().parents[1] / "data" / "kelmarsh"
OUT.mkdir(parents=True, exist_ok=True)

TURBINES = [f"T0{i}" for i in range(1, 7)]        # Kelmarsh 1-6 → T01-T06
RATED_MW = 2.05

# 统一字段 ← Kelmarsh 原始列名
FIELD_MAP = {
    "p":   "Power (kW)",                                    # ÷1000
    "w":   "Wind speed Sensor 1 (m/s)",
    "w2":  "Wind speed Sensor 2 (m/s)",
    "ws":  "Wind speed Sensor 1, Standard deviation (m/s)",
    "wx":  "Wind speed Sensor 1, Maximum (m/s)",
    "wn":  "Wind speed Sensor 1, Minimum (m/s)",
    "pt":  "Blade angle (pitch position) A (°)",
    "rpm": "Rotor speed (RPM)",
}
SETPOINT_COL = "Turbine Power setpoint (kW)"

# 邻机关系（Kelmarsh 是单排布置，按编号相邻；T01/T06 只有一侧）
NEIGHBORS = {
    "T01": ["T02"],
    "T02": ["T01", "T03"],
    "T03": ["T02", "T04"],
    "T04": ["T03", "T05"],
    "T05": ["T04", "T06"],
    "T06": ["T05"],
}

# ───────────────────── 第 1 部分：宽表 scada.parquet ─────────────────────
print("== 生成宽表 scada.parquet")
per_tb = {}   # 机组号 → 该机组各年的数据块列表
for year in range(2019, 2025):
    for tb_no in range(1, 7):
        f = next((SRC / f"data_parquet_{year}").glob(f"Turbine_Data_Kelmarsh_{tb_no}_*.parquet"))
        cols = ["Date and time", *FIELD_MAP.values()]
        d = pd.read_parquet(f, columns=cols)
        d = d.loc[:, ~d.columns.duplicated()]
        # 时间戳：区间开始 → +10min 变区间结束；统一 ns 精度，避免不同文件精度不一致导致无法对齐
        d.index = pd.to_datetime(d["Date and time"]).astype("datetime64[ns]") + pd.Timedelta(minutes=10)
        d = d.drop(columns=["Date and time"]).rename(
            columns={v: f"{k}_T{tb_no:02d}" for k, v in FIELD_MAP.items()})
        # 功率 kW → MW
        d[f"p_T{tb_no:02d}"] = d[f"p_T{tb_no:02d}"] / 1000.0
        per_tb.setdefault(tb_no, []).append(d)
        print(f"  {year} T{tb_no:02d}: {len(d)} 行")

# 关键：同一机组各年的块列名相同（p_T01...），必须先竖向拼接（各年时间不重叠），
# 再把 6 台机组横向拼成宽表。若把 36 个块全部 axis=1 拼接，会产生 6 份重名列。
wide = pd.concat([pd.concat(per_tb[tb_no]).sort_index() for tb_no in sorted(per_tb)], axis=1)
wide = wide[~wide.index.duplicated(keep="first")].sort_index()
wide.to_parquet(OUT / "scada.parquet")
print(f"宽表完成: {wide.shape[0]} 行 × {wide.shape[1]} 列，"
      f"{wide.index.min()} ~ {wide.index.max()}")

# ───────────────────── 第 2 部分：事件表 events.csv ─────────────────────
print("== 生成事件表 events.csv")
def secs(d):
    try:
        h, m, s = str(d).split(":")
        return int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return None

ev_frames = []
for year in range(2019, 2025):
    for tb_no in range(1, 7):
        tb = f"T{tb_no:02d}"
        f = next((SRC / f"data_parquet_{year}").glob(f"Status_Kelmarsh_{tb_no}_*.parquet"))
        st = pd.read_parquet(f, columns=["Timestamp start", "Timestamp end", "Duration",
                                         "Status", "Code", "Message"])
        # 目标事件：Stop + Curtailment + 8405(Fallback setpoints 封顶模式)
        mask = (st["Status"].isin(["Stop", "Curtailment"]) | (st["Code"] == 8405))
        ev = st[mask].copy()
        if not len(ev):
            continue
        ev["type"] = np.where(ev["Status"] == "Stop", "stop", "curtailment")
        ev["start"] = pd.to_datetime(ev["Timestamp start"])
        # end：有就用，没有用 start+Duration，再没有就丢弃
        end_raw = ev["Timestamp end"].astype(str)
        has_end = end_raw != "-"
        ev["end"] = pd.to_datetime(ev["Timestamp end"].where(has_end))
        ev.loc[~has_end, "end"] = (ev.loc[~has_end, "start"]
                                   + pd.to_timedelta(ev.loc[~has_end, "Duration"].map(secs), unit="s"))
        ev = ev.dropna(subset=["end"])
        ev["turbine"] = tb
        ev["detail"] = ev["Message"].fillna("")
        ev_frames.append(ev[["turbine", "type", "start", "end", "detail"]])

events = pd.concat(ev_frames, ignore_index=True)
# 去重：同一机组、同类型、同起止时间的重复行（2022-05-17 那批）
n0 = len(events)
events = events.drop_duplicates(subset=["turbine", "type", "start", "end"]).sort_values("start")
print(f"  去重: {n0} → {len(events)}")

# available_at = start + 1 分钟（日志为实时记录的约定）
events["available_at"] = events["start"] + pd.Timedelta(minutes=1)

# setpoint_mw：curtailment 事件期间 SCADA setpoint 的中位数（MW）；stop 为空

sp_frames = {}   # 机组号 → 该机组各年的 setpoint 序列（必须逐年追加，不能覆盖）
for year in range(2019, 2025):
    for tb_no in range(1, 7):
        f = next((SRC / f"data_parquet_{year}").glob(f"Turbine_Data_Kelmarsh_{tb_no}_*.parquet"))
        d = pd.read_parquet(f, columns=["Date and time", SETPOINT_COL])
        d.index = pd.to_datetime(d["Date and time"]).astype("datetime64[ns]") + pd.Timedelta(minutes=10)
        sp_frames.setdefault(tb_no, []).append(d[SETPOINT_COL].rename(f"T{tb_no:02d}"))
sp_wide = pd.concat([pd.concat(sp_frames[tb_no]).sort_index() for tb_no in sorted(sp_frames)], axis=1)

setpoints = []
for _, e in events.iterrows():
    if e["type"] != "curtailment":
        setpoints.append(None)
        continue
    s = sp_wide.loc[e["start"]:e["end"], e["turbine"]].dropna()
    s_f = s[(s > 500) & (s < 2000)]        # 排除待机 412 和满发 2060 的污染
    if not len(s_f):
        s_f = s[s < 2000]                   # 回退：窗口内没有典型封顶值时，取一切非满发值
    setpoints.append(round(float(s_f.median()) / 1000, 3) if len(s_f) else None)
events["setpoint_mw"] = setpoints

# 事件 id：按全局顺序编号
events = events.reset_index(drop=True)
events.insert(0, "id", [f"E{i:04d}" for i in range(1, len(events) + 1)])
events[["id", "turbine", "type", "start", "end", "available_at", "setpoint_mw", "detail"]] \
    .to_csv(OUT / "events.csv", index=False)
print(f"事件表完成: {len(events)} 条（stop {(events.type=='stop').sum()}，curtailment {(events.type=='curtailment').sum()}）")

# ───────────────────── 第 3 部分：meta.json ─────────────────────
meta = {
    "turbines": TURBINES,
    "rated_mw": RATED_MW,
    "neighbors": NEIGHBORS,
    "dataset": "kelmarsh_v4",
    "source": "https://zenodo.org/records/16807551",
    "timestamp_convention": "interval_end",
    "unit_power": "MW",
    "fields": list(FIELD_MAP.keys()),
}
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print("meta.json 完成")

# ───────────────────── 验收自检 ─────────────────────
print("\n== 验收自检")
sc = pd.read_parquet(OUT / "scada.parquet")
print(f"1. 宽表: {sc.shape}；索引间隔唯一值: {sc.index.to_series().diff().dropna().unique()}")
print(f"2. 功率范围: {sc[[f'p_T{i:02d}' for i in range(1,7)]].min().min():.3f} ~ "
      f"{sc[[f'p_T{i:02d}' for i in range(1,7)]].max().max():.3f} MW（应在 ±0.02 ~ 2.08）")
print(f"3. 第一行时间: {sc.index[0]}（应为 2019-01-01 00:10，区间结束时刻）")
ev = pd.read_csv(OUT / "events.csv", parse_dates=["start", "end", "available_at"])
print(f"4. 事件: {len(ev)} 条；available_at 全部 ≥ start? {(ev.available_at >= ev.start).all()}")
print(f"5. 限电事件 setpoint_mw 分布:\n{ev[ev.type=='curtailment'].setpoint_mw.describe().to_string()}")
tr = ev[(ev.start < "2021-01-01") & (ev.type == "curtailment")]
ca = ev[(ev.start >= "2022-01-01") & (ev.start < "2023-01-01") & (ev.type == "curtailment")]
te = ev[(ev.start >= "2023-01-01") & (ev.type == "curtailment")]
print(f"6. 限电事件分布: train(2019-20) {len(tr)} 条（少量短时 8405 属正常）、"
      f"calib(2022) {len(ca)} 条、test(2023) {len(te)} 条")