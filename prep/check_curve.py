# prep/check_curve.py —— 功率曲线 sanity check：一眼验证整条转换管线是否正确
import json
from pathlib import Path

import pandas as pd

ART = Path(__file__).resolve().parents[1] / "artifacts"
ref = json.loads((ART / "reference.json").read_text(encoding="utf-8"))

print("curve_source:", ref["curve_source"])
for tb in ref["curve"]:
    c = ref["curve"][tb]
    df = pd.DataFrame({"ws": c["ws"], "mw": c["mw"]})
    print(f"\n===== {tb} 功率曲线（隔行采样）=====")
    print(df.iloc[::2].to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    print("最后 4 个 bin（完整，不采样）：")
    print(df.tail(4).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
    first_gen = df[df.mw > 0.05].ws.min()
    reach = df[df.mw >= 0.95 * 2.05].ws.min()
    down = (df.mw.diff().dropna() < 0).mean()
    print(f"  切入(>0.05MW): ~{first_gen} m/s（MM92 预期 ~3）| "
          f"达 95% 额定: ~{reach} m/s（预期 11-12）| "
          f"最大 {df.mw.max():.3f} MW（应≈2.05）| 下降 bin 占比 {down:.0%}（<10% 正常）")

print("\n===== 邻机风速比例（应都在 0.9~1.1 附近）=====")
for tb, r in ref["wind_ratio"].items():
    print(f"  {tb}: {r}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for tb in ref["curve"]:
        ax.plot(ref["curve"][tb]["ws"], ref["curve"][tb]["mw"], label=tb)
    ax.axhline(2.05, ls="--", c="gray", lw=1)
    ax.set(xlabel="wind speed (m/s)", ylabel="power (MW)",
           title="Empirical power curves, train 2019-20")
    ax.legend()
    out = Path(__file__).resolve().parents[1] / "outputs" / "curve_check.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print("\n曲线图已保存:", out)
except Exception as e:
    print("画图跳过:", e)