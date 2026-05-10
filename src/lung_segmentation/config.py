"""Configuration loading helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the current project root.

    Commands are expected to be run from the repository root. Keeping this explicit makes local
    paths predictable and avoids hidden dependencies on Colab or Google Drive paths.
    """

    return Path.cwd()


def resolve_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a user path relative to the repository root."""

    value = Path(path)
    if value.is_absolute():
        return value
    return (Path(root) if root is not None else project_root()) / value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config."""

    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a mapping: {config_path}")

    config = dict(config)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root())
    return config


def config_get(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read a nested config value using dotted syntax."""

    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current
