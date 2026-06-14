"""Lightweight neural estimator model definitions."""

from __future__ import annotations

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
    if num_pilots <= 0 or hidden_units <= 0:
        raise ValueError("num_pilots and hidden_units must be positive.")
    if not 0 <= dropout_rate < 1:
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
