from pathlib import Path
import yaml


def load_config(path: str | Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    return cfg
