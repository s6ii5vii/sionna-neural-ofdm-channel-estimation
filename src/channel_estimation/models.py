"""Lightweight neural estimator model definitions (PyTorch).

two estimators are provided:

* ``build_lightweight_estimator`` -- a small multilayer perceptron over the flat
  pilot representation ``(num_pilots, 2)``.
* ``build_grid_estimator`` -- a deliberately small convolutional network that maps
  a received ofdm resource grid ``(num_ofdm_symbols, num_subcarriers, 2)`` to a
  full-grid channel estimate of the same shape.

both accept and return tensors with a trailing real/imaginary axis so they match
the numpy ``.npz`` datasets directly; the channels-first permutation that PyTorch
convolutions expect happens inside the modules. torch is imported lazily so the
classical baseline, metrics, dataset, and config code stay usable without the ml
stack. these modules have not been executed in this environment.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any


def _validate_common(dropout_rate: float) -> None:
    if (
        not isinstance(dropout_rate, Real)
        or isinstance(dropout_rate, bool)
        or not math.isfinite(dropout_rate)
        or not 0 <= dropout_rate < 1
    ):
        raise ValueError("dropout_rate must be in [0, 1).")


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the neural estimator. Install the project "
            "with the 'ml' extra or use requirements.txt."
        ) from exc
    return torch


def build_lightweight_estimator(
    num_pilots: int,
    *,
    hidden_units: int = 64,
    dropout_rate: float = 0.0,
) -> Any:
    """Build a deliberately small MLP estimator for real/imaginary features.

    returns a ``torch.nn.Module`` accepting ``(batch, num_pilots, 2)`` and
    returning the same shape. this is an experimental baseline, not a claim of
    novelty or state-of-the-art performance.
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
    _validate_common(dropout_rate)

    torch = _import_torch()
    nn = torch.nn

    class LightweightEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_pilots = num_pilots
            self.flatten = nn.Flatten()
            self.hidden = nn.Linear(num_pilots * 2, hidden_units)
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate) if dropout_rate else nn.Identity()
            self.output = nn.Linear(hidden_units, num_pilots * 2)

        def forward(self, x: Any) -> Any:  # x: (batch, num_pilots, 2)
            batch = x.shape[0]
            h = self.activation(self.hidden(self.flatten(x)))
            h = self.dropout(h)
            return self.output(h).reshape(batch, self.num_pilots, 2)

    return LightweightEstimator()


def build_grid_estimator(
    num_ofdm_symbols: int,
    num_subcarriers: int,
    *,
    filters: int = 16,
    num_layers: int = 3,
    dropout_rate: float = 0.0,
) -> Any:
    """Build a deliberately small convolutional estimator over a resource grid.

    returns a ``torch.nn.Module`` mapping a received grid
    ``(batch, num_ofdm_symbols, num_subcarriers, 2)`` to a full-grid channel
    estimate of the same shape. a stack of ``same``-padded convolutions with no
    downsampling preserves the time-frequency resolution while letting the model
    interpolate the channel across non-pilot resource elements. ``filters`` is
    the per-layer width and the primary knob for a parameter-count budget.

    this is an experimental baseline, not a claim of novelty or state-of-the-art
    performance, and it has not been executed in this environment.
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
    _validate_common(dropout_rate)

    torch = _import_torch()
    nn = torch.nn

    class GridEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[Any] = []
            in_channels = 2
            for _ in range(num_layers):
                layers.append(
                    nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
                )
                layers.append(nn.ReLU())
                if dropout_rate:
                    layers.append(nn.Dropout(dropout_rate))
                in_channels = filters
            layers.append(nn.Conv2d(in_channels, 2, kernel_size=3, padding=1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:  # x: (batch, symbols, subcarriers, 2)
            x = x.permute(0, 3, 1, 2)  # -> (batch, 2, symbols, subcarriers)
            y = self.net(x)
            return y.permute(0, 2, 3, 1)  # -> (batch, symbols, subcarriers, 2)

    return GridEstimator()


def count_parameters(model: Any) -> int:
    """Return the number of trainable parameters in a torch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
