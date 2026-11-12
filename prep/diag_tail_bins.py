# prep/diag_tail_bins.py —— 看各机组曲线尾部 bin 的样本数和中位数，决定处理方案
import pandas as pd
import numpy as np
from common.config import load_split
from common.store import DataStore

store = DataStore.load()
split = load_split()
tr = store.frame.loc[split["splits"]["train"]["start"]:split["splits"]["train"]["end"]]
ev = store.events
bins = np.arange(0, 26, 0.5)

for t in store.turbines:
    p, w = tr[f"p_{t}"].copy(), tr[f"w_{t}"]
    for _, e in ev[ev.turbine == t].iterrows():     # 和 fit_reference 完全一致的过滤
        p.loc[e.start:e.end] = np.nan
    ok = p.notna() & w.notna() & (p > 0.02) & (w <= 20)
    g = p[ok].groupby(pd.cut(w[ok], bins), observed=True)
    med, cnt = g.median(), g.count()
    tail = pd.DataFrame({"ws": [iv.mid for iv in med.index], "median_mw": med.values,
                         "n样本": cnt.values}).tail(6)
    print(f"\n{t} 尾部 6 个 bin：")
    print(tail.to_string(index=False))