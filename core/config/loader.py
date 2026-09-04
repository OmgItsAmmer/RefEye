"""Loads and validates application configuration.

Resolution order (later wins):
    1. config/default.yaml          — versioned defaults
    2. config/local.yaml            — optional, gitignored machine overrides
    3. environment variables        — SOCCER_<SECTION>__<KEY>, deployment values

Environment variables use a double-underscore path separator, e.g.
    SOCCER_RUNTIME__DEVICE=cpu
    SOCCER_VIDEO__LOCAL_FILE__PATH=D:/clips/match.mp4
    SOCCER_LOGGING__LEVEL=DEBUG

Scalar values are parsed as YAML so `true`, `12`, and `1.5` arrive typed.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from core.config.paths import ensure_editable_config, resolve
from core.config.schema import AppSettings
from core.errors.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
LOCAL_CONFIG_PATH = Path("config/local.yaml")
ENV_PREFIX = "SOCCER_"
ENV_PATH_SEPARATOR = "__"


def load_settings(
    config_path: str | Path | None = None,
    local_path: str | Path | None = LOCAL_CONFIG_PATH,
    apply_env: bool = True,
) -> AppSettings:
    # No explicit path: resolve against the application root and, in a
    # packaged build, seed the operator-editable copy beside the executable.
    if config_path is None:
        config_path = ensure_editable_config()
    raw = _read_yaml(config_path, required=True)

    if local_path is not None:
        overrides = _read_yaml(resolve(local_path), required=False)
        if overrides:
            raw = _deep_merge(raw, overrides)

    if apply_env:
        load_dotenv(override=False)
        raw = _deep_merge(raw, _env_overrides())

    try:
        return AppSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid configuration in {config_path}:\n{exc}") from exc


def _read_yaml(path: str | Path, *, required: bool) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        if required:
            raise ConfigurationError(f"Config file not found: {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Could not parse YAML in {path}:\n{exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a mapping, got {type(data).__name__}: {path}")

    return data


def _env_overrides() -> dict[str, Any]:
    """Build a nested dict from SOCCER_-prefixed environment variables."""
    result: dict[str, Any] = {}

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue

        path = env_key[len(ENV_PREFIX) :].lower().split(ENV_PATH_SEPARATOR)
        if not all(path):
            continue

        cursor = result
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ConfigurationError(
                    f"Environment override {env_key} conflicts with an earlier override"
                )
        cursor[path[-1]] = _parse_scalar(env_value)

    return result


def _parse_scalar(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
