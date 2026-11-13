"""Configuration loading and small helpers."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> dict:
    """Load the default YAML config, apply an optional file and dict overrides."""
    cfg = yaml.safe_load((PROJECT_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    if path is not None:
        cfg = deep_merge(cfg, yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["_root"] = str(PROJECT_ROOT)
    return cfg


def cfg_path(cfg: dict, key: str) -> Path:
    """Resolve a configured relative path against the project root."""
    return PROJECT_ROOT / cfg["paths"][key]


def ensure_dirs(cfg: dict) -> None:
    for key in ("processed_dir", "results_dir", "figures_dir", "tables_dir"):
        (PROJECT_ROOT / cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def flatten(cfg: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config into dotted keys (used by the calibration layer)."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        if key.startswith("_"):
            continue
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def nested_get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def nested_set(cfg: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
