"""Lightweight neural estimator model definitions."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def build_lightweight_estimator(
    num_pilots: int,
    *,
    hidden_units: int = 64,
    dropout_rate: float = 0.0,
) -> Any:
    """Build a deliberately small Keras estimator for real/imaginary features.

    This is an experimental baseline skeleton, not a claim of novelty or
    state-of-the-art performance. TensorFlow is imported lazily so classical
    experiments and unit tests do not require the ML stack.
    """
    if (
        not isinstance(num_pilots, int)
        or isinstance(num_pilots, bool)
        or num_pilots <= 0
        or not isinstance(hidden_units, int)
        or isinstance(hidden_units, bool)
        or hidden_units <= 0
    ):
        raise ValueError("num_pilots and hidden_units must be positive integers.")
    if (
        not isinstance(dropout_rate, Real)
        or isinstance(dropout_rate, bool)
        or not math.isfinite(dropout_rate)
        or not 0 <= dropout_rate < 1
    ):
        raise ValueError("dropout_rate must be in [0, 1).")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the neural estimator. "
            "Install the project with the 'ml' extra or use requirements.txt."
        ) from exc

    inputs = keras.Input(shape=(num_pilots, 2), name="received_pilots")
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(hidden_units, activation="relu")(x)
    if dropout_rate:
        x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(num_pilots * 2)(x)
    outputs = keras.layers.Reshape((num_pilots, 2), name="channel_estimate")(outputs)
    return keras.Model(inputs=inputs, outputs=outputs, name="lightweight_estimator")


def build_grid_estimator(
    num_ofdm_symbols: int,
    num_subcarriers: int,
    *,
    filters: int = 16,
    num_layers: int = 3,
    dropout_rate: float = 0.0,
) -> Any:
    """Build a deliberately small convolutional estimator over a resource grid.

    the network maps a received grid ``(num_ofdm_symbols, num_subcarriers, 2)`` to
    a full-grid channel estimate of the same shape. a stack of ``same``-padded
    convolutions with no downsampling keeps the time-frequency resolution while
    letting the model interpolate the channel across non-pilot resource elements.
    ``filters`` is the per-layer width and is the primary knob for meeting a
    parameter-count budget; keep it small to stay lightweight.

    this is an experimental baseline, not a claim of novelty or state-of-the-art
    performance, and it has not been executed in this environment. TensorFlow is
    imported lazily so classical experiments and unit tests do not require the ml
    stack.
    """
    if (
        not isinstance(num_ofdm_symbols, int)
        or isinstance(num_ofdm_symbols, bool)
        or num_ofdm_symbols <= 0
        or not isinstance(num_subcarriers, int)
        or isinstance(num_subcarriers, bool)
        or num_subcarriers <= 0
        or not isinstance(filters, int)
        or isinstance(filters, bool)
        or filters <= 0
        or not isinstance(num_layers, int)
        or isinstance(num_layers, bool)
        or num_layers <= 0
    ):
        raise ValueError(
            "num_ofdm_symbols, num_subcarriers, filters, and num_layers must be "
            "positive integers."
        )
    if (
        not isinstance(dropout_rate, Real)
        or isinstance(dropout_rate, bool)
        or not math.isfinite(dropout_rate)
        or not 0 <= dropout_rate < 1
    ):
        raise ValueError("dropout_rate must be in [0, 1).")

    try:
        from tensorflow import keras
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the neural estimator. "
            "Install the project with the 'ml' extra or use requirements.txt."
        ) from exc

    inputs = keras.Input(
        shape=(num_ofdm_symbols, num_subcarriers, 2), name="received_grid"
    )
    x = inputs
    for index in range(num_layers):
        x = keras.layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            activation="relu",
            name=f"conv-{index}",
        )(x)
        if dropout_rate:
            x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Conv2D(
        2, kernel_size=3, padding="same", name="channel_estimate"
    )(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="grid_estimator")
