"""Run B's development evaluation using only SCADA and the pre-test splits."""
import json
import pickle

import pandas as pd

from common.config import ROOT, load_split
from common.store import DataStore
from models.settings import forecast_settings
from models.train import train_forecast
from eval.numeric import evaluate_split, split_errors, write_results


def development_store(split):
    if split["dataset"] != "kelmarsh_v4":
        raise ValueError("此验证入口仅用于 Kelmarsh SCADA")
    cfg = forecast_settings()
    turbines = split["turbines"]
    path = ROOT / "data" / "kelmarsh"
    meta = {"turbines": turbines, "rated_mw": cfg["kelmarsh_rated_mw"],
            "neighbors": cfg["kelmarsh_neighbors"]}
    if (path / "meta.json").exists():
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    columns = [f"{prefix}_{t}" for t in turbines for prefix in ("p", "w")]
    frame = pd.read_parquet(path / "scada.parquet", columns=columns, filters=[
        ("Date and time", "<=", pd.Timestamp(split["splits"]["calib"]["end"]))])
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("SCADA 时间索引必须严格递增")
    frame = frame.asfreq("10min")
    return DataStore(frame, pd.DataFrame(), meta)


def main():
    split = load_split()
    store = development_store(split)
    bundle = train_forecast(store, split)
    dest = ROOT / "outputs" / "forecast_validation"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "development_models.pkl").open("wb") as f:
        pickle.dump(bundle, f)
    df, strata = evaluate_split(store, bundle, split, "val")
    write_results(df, strata, bundle, "val", dest, split_errors(store, bundle, split, df, "val"))


if __name__ == "__main__":
    main()
