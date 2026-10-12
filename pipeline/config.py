"""Load pipeline/config.yaml and resolve the {data_root} templates."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("raw_dir", "subset_dir", "manifest_dir"):
        cfg[key] = Path(cfg[key].format(data_root=cfg["data_root"]))
    cfg["data_root"] = Path(cfg["data_root"])
    return cfg


def local_path(cfg: dict, archive_path: str) -> Path:
    """Map /neodc/esacci/<rest> to <raw_dir>/<rest>."""
    rel = archive_path.split("/neodc/esacci/", 1)[1]
    return cfg["raw_dir"] / rel
