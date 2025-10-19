"""Environment configuration helpers for the trading bot."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_environment() -> None:
    """Load environment variables from .env and .env.key files."""
    base_env = Path(".env")
    if base_env.exists():
        load_dotenv(dotenv_path=base_env, override=False)

    secrets_env = Path(".env.key")
    if secrets_env.exists():
        load_dotenv(dotenv_path=secrets_env, override=True)


def _get(name: str, default: str | None) -> str:
    value = os.getenv(name)
    if value is None:
        if default is None:
            raise ValueError(
                f"Environment variable '{name}' is required but was not found."
            )
        return default
    return value


def get_env(name: str, default: str | None = None) -> str:
    return _get(name, default)


def get_int_env(name: str, default: int | None = None) -> int:
    value = get_env(name, None if default is None else str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{name}' must be an integer, got '{value}'."
        ) from exc


def get_float_env(name: str, default: float | None = None) -> float:
    value = get_env(name, None if default is None else str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable '{name}' must be a float, got '{value}'."
        ) from exc


def get_bool_env(name: str, default: bool | None = None) -> bool:
    if default is None:
        default_str = None
    else:
        default_str = "true" if default else "false"

    value = get_env(name, default_str)
    value_normalized = value.strip().lower()
    if value_normalized in {"1", "true", "yes", "on"}:
        return True
    if value_normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Environment variable '{name}' must be a boolean value, got '{value}'."
    )


# Ensure environment variables are loaded as soon as this module is imported.
load_environment()
