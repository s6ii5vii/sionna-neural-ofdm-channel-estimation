"""Metrics used by channel-estimation experiments."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike


def bit_error_rate(reference_bits: ArrayLike, estimated_bits: ArrayLike) -> float:
    """Return the fraction of unequal binary values."""
    reference = np.asarray(reference_bits)
    estimated = np.asarray(estimated_bits)
    if reference.shape != estimated.shape:
        raise ValueError("reference_bits and estimated_bits must have equal shapes.")
    if reference.size == 0:
        raise ValueError("Bit arrays must not be empty.")
    if not np.all(np.isin(reference, (0, 1))) or not np.all(
        np.isin(estimated, (0, 1))
    ):
        raise ValueError("BER inputs must contain only 0 and 1.")
    return float(np.mean(reference != estimated))


def normalized_mean_squared_error(
    reference: ArrayLike,
    estimate: ArrayLike,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return mean squared error normalized by reference signal power."""
    reference_array = np.asarray(reference)
    estimate_array = np.asarray(estimate)
    if reference_array.shape != estimate_array.shape:
        raise ValueError("reference and estimate must have equal shapes.")
    if reference_array.size == 0:
        raise ValueError("Metric inputs must not be empty.")

    denominator = float(np.mean(np.abs(reference_array) ** 2))
    if denominator <= epsilon:
        raise ValueError("Reference power is too small to compute NMSE.")
    numerator = float(np.mean(np.abs(reference_array - estimate_array) ** 2))
    return numerator / denominator


def nmse_db(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return NMSE in decibels."""
    value = normalized_mean_squared_error(reference, estimate)
    return float(10.0 * np.log10(value))


def aggregate_by_snr(
    snr_values: Iterable[float],
    metric_values: Iterable[float],
) -> dict[float, dict[str, float]]:
    """Aggregate repeated metric values by SNR using mean and sample spread."""
    grouped: dict[float, list[float]] = {}
    snr_list = list(snr_values)
    value_list = list(metric_values)
    if len(snr_list) != len(value_list):
        raise ValueError("snr_values and metric_values must have equal lengths.")

    for snr, value in zip(snr_list, value_list, strict=True):
        grouped.setdefault(float(snr), []).append(float(value))

    return {
        snr: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "count": float(len(values)),
        }
        for snr, values in sorted(grouped.items())
    }
