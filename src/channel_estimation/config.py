"""Configuration loading and lightweight validation."""

from __future__ import annotations

import math
import re
from numbers import Real
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
    config["config-path"] = str(config_path.resolve())
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate settings shared by the included experiment runners."""
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ConfigError("Missing 'experiment' mapping.")

    name = experiment.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ConfigError("'experiment.name' must be a lowercase hyphenated slug.")

    snr_values = experiment.get("snr-db")
    if not isinstance(snr_values, list) or not snr_values:
        raise ConfigError("'experiment.snr-db' must be a non-empty list.")
    if not all(_is_finite_number(value) for value in snr_values):
        raise ConfigError("Every SNR value must be a finite number.")

    for key in ("num-samples", "random-seed"):
        value = experiment.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(f"'experiment.{key}' must be an integer.")
        if key != "random-seed" and value <= 0:
            raise ConfigError(f"'experiment.{key}' must be positive.")

    # two experiment shapes are supported: the original flat pilot model and the
    # sionna ofdm resource-grid model. presence of 'num-subcarriers' selects the
    # grid schema.
    if "num-subcarriers" in experiment:
        _validate_grid_experiment(experiment)
    else:
        _validate_flat_experiment(experiment)

    for key in ("dataset-size-target", "model-parameter-target"):
        value = experiment.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            raise ConfigError(f"'experiment.{key}' must be a positive integer.")

    output = config.get("output")
    if not isinstance(output, Mapping):
        raise ConfigError("Missing 'output' mapping.")
    for key in ("figures-dir", "tables-dir"):
        if not isinstance(output.get(key), str) or not output[key].strip():
            raise ConfigError(f"'output.{key}' must be a non-empty path.")

    training = config.get("training")
    if training is not None:
        _validate_training_config(training)

    neural = config.get("neural")
    if neural is not None:
        _validate_neural_config(neural)
    if isinstance(training, Mapping) and isinstance(neural, Mapping):
        if training["hidden-units"] != neural["filters"]:
            raise ConfigError("'training.hidden-units' must match 'neural.filters'.")
        if training.get("num-layers", 3) != neural["num-layers"]:
            raise ConfigError("Training and neural layer counts must match.")
        generation = training.get("dataset-generation")
        if isinstance(generation, Mapping) and generation.get(
            "input-source", "received"
        ) != neural.get("input-source", "received"):
            raise ConfigError("Training and neural input sources must match.")
        for key, default in _SHARED_ARCHITECTURE_DEFAULTS.items():
            if training.get(key, default) != neural.get(key, default):
                raise ConfigError(f"'training.{key}' and 'neural.{key}' must match.")


_SHARED_ARCHITECTURE_DEFAULTS = {"dilations": None, "residual": False, "snr-input": False}


_VALID_CHANNEL_MODELS = ("tdl-a", "tdl-b", "tdl-c", "tdl-d", "tdl-e", "rayleigh")
_VALID_GRID_NEURAL_INPUTS = ("received", "ls-nn", "ls-lin")


def _require_positive_int(experiment: Mapping[str, Any], key: str) -> int:
    value = experiment.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"'experiment.{key}' must be a positive integer.")
    return value


def _validate_flat_experiment(experiment: Mapping[str, Any]) -> None:
    """Validate the original flat pilot-observation experiment schema."""
    _require_positive_int(experiment, "num-pilots")
    pilot_density = experiment.get("pilot-density", 1.0)
    if not _is_finite_number(pilot_density) or not 0 < pilot_density <= 1:
        raise ConfigError("'experiment.pilot-density' must be in (0, 1].")


def _validate_grid_experiment(experiment: Mapping[str, Any]) -> None:
    """Validate the sionna ofdm resource-grid experiment schema."""
    _require_positive_int(experiment, "num-subcarriers")
    num_ofdm_symbols = _require_positive_int(experiment, "num-ofdm-symbols")

    pilot_indices = experiment.get("pilot-ofdm-symbol-indices")
    if not isinstance(pilot_indices, list) or not pilot_indices:
        raise ConfigError(
            "'experiment.pilot-ofdm-symbol-indices' must be a non-empty list."
        )
    if any(
        not isinstance(index, int)
        or isinstance(index, bool)
        or not 0 <= index < num_ofdm_symbols
        for index in pilot_indices
    ):
        raise ConfigError(
            "Every pilot symbol index must be an integer within the grid."
        )

    channel_model = experiment.get("channel-model", "tdl-a")
    if channel_model not in _VALID_CHANNEL_MODELS:
        raise ConfigError(
            f"'experiment.channel-model' must be one of {_VALID_CHANNEL_MODELS}."
        )

    delay_spread = experiment.get("delay-spread-ns", 100.0)
    if not _is_finite_number(delay_spread) or delay_spread <= 0:
        raise ConfigError("'experiment.delay-spread-ns' must be positive.")

    max_doppler = experiment.get("max-doppler-hz", 0.0)
    if not _is_finite_number(max_doppler) or max_doppler < 0:
        raise ConfigError("'experiment.max-doppler-hz' must be non-negative.")

    spacing = experiment.get("pilot-subcarrier-spacing", 1)
    if (
        not isinstance(spacing, int)
        or isinstance(spacing, bool)
        or not 1 <= spacing <= experiment["num-subcarriers"]
    ):
        raise ConfigError(
            "'experiment.pilot-subcarrier-spacing' must be an integer between 1 "
            "and 'num-subcarriers'."
        )

    assumed = experiment.get("lmmse-assumed")
    if assumed is not None:
        _validate_lmmse_assumed(assumed)


def _validate_lmmse_assumed(assumed: object) -> None:
    """Validate the channel statistics assumed by the mismatched LMMSE baseline."""
    if not isinstance(assumed, Mapping) or not assumed:
        raise ConfigError("'experiment.lmmse-assumed' must be a non-empty mapping.")
    unknown = set(assumed) - {"channel-model", "delay-spread-ns", "max-doppler-hz"}
    if unknown:
        raise ConfigError(
            f"Unknown 'experiment.lmmse-assumed' keys: {sorted(unknown)}."
        )
    channel_model = assumed.get("channel-model", "tdl-a")
    if channel_model not in _VALID_CHANNEL_MODELS or channel_model == "rayleigh":
        raise ConfigError(
            "'experiment.lmmse-assumed.channel-model' must be a TDL model."
        )
    delay_spread = assumed.get("delay-spread-ns", 100.0)
    if not _is_finite_number(delay_spread) or delay_spread <= 0:
        raise ConfigError("'experiment.lmmse-assumed.delay-spread-ns' must be positive.")
    max_doppler = assumed.get("max-doppler-hz", 0.0)
    if not _is_finite_number(max_doppler) or max_doppler < 0:
        raise ConfigError(
            "'experiment.lmmse-assumed.max-doppler-hz' must be non-negative."
        )


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    )


def _validate_training_config(training: object) -> None:
    if not isinstance(training, Mapping):
        raise ConfigError("'training' must be a mapping when provided.")

    for key in ("dataset-path", "checkpoint-path"):
        value = training.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"'training.{key}' must be a non-empty path.")

    for key in ("hidden-units", "epochs", "batch-size"):
        value = training.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"'training.{key}' must be a positive integer.")

    dropout_rate = training.get("dropout-rate", 0.0)
    if not _is_finite_number(dropout_rate) or not 0 <= dropout_rate < 1:
        raise ConfigError("'training.dropout-rate' must be in [0, 1).")

    num_layers = training.get("num-layers", 3)
    if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers <= 0:
        raise ConfigError("'training.num-layers' must be a positive integer.")

    model_seed = training.get("model-seed", 42)
    if not isinstance(model_seed, int) or isinstance(model_seed, bool):
        raise ConfigError("'training.model-seed' must be an integer.")

    learning_rate = training.get("learning-rate", 1e-3)
    if not _is_finite_number(learning_rate) or learning_rate <= 0:
        raise ConfigError("'training.learning-rate' must be positive.")

    _validate_architecture_options(training, "training", num_layers)

    generation = training.get("dataset-generation")
    if generation is not None:
        _validate_dataset_generation(generation)
        if training.get("residual") and generation.get(
            "input-source", "received"
        ) == "received":
            raise ConfigError(
                "'training.residual' needs an 'ls-nn' or 'ls-lin' input source."
            )


def _validate_dataset_generation(generation: object) -> None:
    """Validate an optional block that parameterizes grid dataset generation."""
    if not isinstance(generation, Mapping):
        raise ConfigError("'training.dataset-generation' must be a mapping.")

    for key in ("num-samples", "random-seed"):
        value = generation.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ConfigError(
                f"'training.dataset-generation.{key}' must be an integer."
            )
        if key != "random-seed" and value <= 0:
            raise ConfigError(
                f"'training.dataset-generation.{key}' must be positive."
            )

    snr_db = generation.get("snr-db")
    if isinstance(snr_db, list):
        if (
            len(snr_db) != 2
            or not all(_is_finite_number(value) for value in snr_db)
            or snr_db[0] >= snr_db[1]
        ):
            raise ConfigError(
                "'training.dataset-generation.snr-db' range must be [low, high] "
                "with low < high."
            )
    elif not _is_finite_number(snr_db):
        raise ConfigError(
            "'training.dataset-generation.snr-db' must be a finite number or a "
            "[low, high] range."
        )

    batch_size = generation.get("batch-size", 256)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ConfigError(
            "'training.dataset-generation.batch-size' must be a positive integer."
        )

    input_source = generation.get("input-source", "received")
    if input_source not in _VALID_GRID_NEURAL_INPUTS:
        raise ConfigError(
            f"'training.dataset-generation.input-source' must be one of "
            f"{_VALID_GRID_NEURAL_INPUTS}."
        )


def _validate_neural_config(neural: object) -> None:
    """Validate the optional block used to evaluate a trained neural estimator."""
    if not isinstance(neural, Mapping):
        raise ConfigError("'neural' must be a mapping when provided.")

    checkpoint = neural.get("checkpoint-path")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ConfigError("'neural.checkpoint-path' must be a non-empty path.")

    for key in ("filters", "num-layers"):
        value = neural.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"'neural.{key}' must be a positive integer.")

    input_source = neural.get("input-source", "received")
    if input_source not in _VALID_GRID_NEURAL_INPUTS:
        raise ConfigError(
            f"'neural.input-source' must be one of {_VALID_GRID_NEURAL_INPUTS}."
        )

    _validate_architecture_options(neural, "neural", neural["num-layers"])
    if neural.get("residual") and input_source == "received":
        raise ConfigError(
            "'neural.residual' needs an 'ls-nn' or 'ls-lin' input source."
        )


def _validate_architecture_options(
    block: Mapping[str, Any], name: str, num_layers: int
) -> None:
    """Validate the optional grid-CNN extensions shared by training and neural."""
    dilations = block.get("dilations")
    if dilations is not None:
        if (
            not isinstance(dilations, list)
            or len(dilations) != num_layers
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in dilations
            )
        ):
            raise ConfigError(
                f"'{name}.dilations' must list one positive integer per layer."
            )
    for key in ("residual", "snr-input"):
        value = block.get(key)
        if value is not None and not isinstance(value, bool):
            raise ConfigError(f"'{name}.{key}' must be true or false.")


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value)
    if path.is_absolute():
        return path

    config_path = Path(str(config.get("config-path", Path.cwd())))
    if config_path.is_file():
        config_path = config_path.parent

    for parent in (config_path, *config_path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent / path
    return Path.cwd() / path
