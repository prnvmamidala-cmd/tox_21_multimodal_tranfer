"""Load and merge YAML configs. Paths in configs are repository-relative."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (parent of src/ or the cwd when installed)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").is_dir():
            return parent
    return Path.cwd()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must be a mapping.")
    return loaded


def deep_update(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``overlay`` onto a copy of ``base``."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, root: Path | None = None) -> dict[str, Any]:
    """Merge configs/data, models, adult_cv, transfer, and ablations, then overlay ``path``."""
    root = root or repo_root()
    config: dict[str, Any] = {}
    for name in ("data", "models", "adult_cv", "transfer", "ablations"):
        config = deep_update(config, _read_yaml(root / "configs" / f"{name}.yaml"))
    overlay_path = Path(path)
    if not overlay_path.is_absolute():
        overlay_path = root / overlay_path
    config = deep_update(config, _read_yaml(overlay_path))
    config["_config_path"] = str(overlay_path)
    config["_root"] = str(root)
    return config
