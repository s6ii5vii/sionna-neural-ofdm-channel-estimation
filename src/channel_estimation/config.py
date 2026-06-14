"""Configuration loading and lightweight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a YAML experiment configuration."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ConfigError("The configuration root must be a mapping.")

    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate settings shared by the included experiment runners."""
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ConfigError("Missing 'experiment' mapping.")

    snr_values = experiment.get("snr_db")
    if not isinstance(snr_values, list) or not snr_values:
        raise ConfigError("'experiment.snr_db' must be a non-empty list.")
    if not all(isinstance(value, (int, float)) for value in snr_values):
        raise ConfigError("Every SNR value must be numeric.")

    for key in ("num_samples", "num_pilots", "random_seed"):
        value = experiment.get(key)
        if not isinstance(value, int):
            raise ConfigError(f"'experiment.{key}' must be an integer.")
        if key != "random_seed" and value <= 0:
            raise ConfigError(f"'experiment.{key}' must be positive.")

    pilot_density = experiment.get("pilot_density", 1.0)
    if not isinstance(pilot_density, (int, float)) or not 0 < pilot_density <= 1:
        raise ConfigError("'experiment.pilot_density' must be in (0, 1].")

    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ConfigError("Missing 'output' mapping.")
    for key in ("figures_dir", "tables_dir"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise ConfigError(f"'output.{key}' must be a non-empty path.")


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path

    config_path = Path(str(config.get("_config_path", Path.cwd())))
    if config_path.is_file():
        config_path = config_path.parent

    for parent in (config_path, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent / path
    return Path.cwd() / path
