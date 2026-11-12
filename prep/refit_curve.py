# prep/refit_curve.py —— 只重拟合功率曲线和风速比例（不重训模型）
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_split
from common.store import DataStore
from models.train import ART_DIR, fit_reference

ref = fit_reference(DataStore.load(), load_split())
(ART_DIR / "reference.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
print("reference.json 已重写。各机组曲线尾部：")
for tb in ref["curve"]:
    c = ref["curve"][tb]
    print(f"  {tb}: 最后 bin {c['ws'][-1]:5.2f} m/s → {c['mw'][-1]:.3f} MW（共 {len(c['ws'])} 个 bin）")