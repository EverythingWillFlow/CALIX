"""Configuration loading utilities."""
from __future__ import annotations

import copy
import os

import yaml

from .env import load_dotenv


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML configuration file and apply dotted-key overrides.

    Parameters
    ----------
    path:
        Path to the YAML file. Defaults to ``configs/default.yaml`` relative
        to the repository root (the parent directory of this package).
    overrides:
        Optional mapping of dotted keys (e.g. ``{"qnetwork.lr": 1e-3}``)
        applied on top of the file contents.
    """
    # credentials from <repo_root>/.env are loaded before anything reads them
    load_dotenv()
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "configs", "default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # kept so a run manifest can hash the exact file that produced it
    if isinstance(cfg, dict):
        cfg["_path"] = os.path.abspath(path)
    if overrides:
        cfg = copy.deepcopy(cfg)
        for key, value in overrides.items():
            node = cfg
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
    return cfg
