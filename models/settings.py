from pathlib import Path

import yaml


def forecast_settings():
    path = Path(__file__).resolve().parents[1] / "configs" / "forecast.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))
