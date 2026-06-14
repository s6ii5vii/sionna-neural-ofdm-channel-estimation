"""Small NumPy helpers for the current OFDM experiment model."""

from __future__ import annotations

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def snr_db_to_linear(snr_db: ArrayLike) -> NDArray[np.float64]:
    """Convert SNR values from decibels to linear scale."""
    return np.power(10.0, np.asarray(snr_db, dtype=float) / 10.0)


def noise_power_from_snr(snr_db: float, signal_power: float = 1.0) -> float:
    """Return complex AWGN power for a target SNR."""
    if (
        not isinstance(signal_power, Real)
        or isinstance(signal_power, bool)
        or not np.isfinite(signal_power)
        or signal_power <= 0
    ):
        raise ValueError("signal_power must be positive.")
    return float(signal_power / snr_db_to_linear(snr_db))


def generate_pilot_symbols(
    num_pilots: int,
    *,
    batch_size: int | None = None,
    value: complex = 1.0 + 0.0j,
    dtype: np.dtype = np.complex64,
) -> NDArray[np.complexfloating]:
    """Generate non-zero constant pilots used by the current baseline."""
    if (
        not isinstance(num_pilots, int)
        or isinstance(num_pilots, bool)
        or num_pilots <= 0
    ):
        raise ValueError("num_pilots must be a positive integer.")
    if batch_size is not None and (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer when provided.")
    if not np.isfinite(value) or value == 0:
        raise ValueError("Pilot symbols must be finite and non-zero.")
    shape = (num_pilots,) if batch_size is None else (batch_size, num_pilots)
    return np.full(shape, value, dtype=dtype)


def pilot_indices(num_subcarriers: int, pilot_density: float) -> NDArray[np.int64]:
    """Choose approximately uniform pilot locations for sparse-pilot studies."""
    if (
        not isinstance(num_subcarriers, int)
        or isinstance(num_subcarriers, bool)
        or num_subcarriers <= 0
    ):
        raise ValueError("num_subcarriers must be a positive integer.")
    if (
        not isinstance(pilot_density, Real)
        or isinstance(pilot_density, bool)
        or not np.isfinite(pilot_density)
        or not 0 < pilot_density <= 1
    ):
        raise ValueError("pilot_density must be in (0, 1].")

    count = max(1, int(round(num_subcarriers * pilot_density)))
    return np.unique(np.linspace(0, num_subcarriers - 1, count, dtype=int))


def generate_rayleigh_channel(
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Generate independent, unit-power circular complex Gaussian samples."""
    real = rng.normal(size=shape)
    imag = rng.normal(size=shape)
    return (real + 1j * imag) / np.sqrt(2.0)


def add_complex_awgn(
    signal: ArrayLike,
    snr_db: float,
    rng: np.random.Generator,
    *,
    reference_power: float = 1.0,
) -> NDArray[np.complex128]:
    """Add circular complex Gaussian noise at the requested SNR."""
    signal_array = np.asarray(signal)
    noise_power = noise_power_from_snr(snr_db, reference_power)
    scale = np.sqrt(noise_power / 2.0)
    noise = scale * (
        rng.normal(size=signal_array.shape) + 1j * rng.normal(size=signal_array.shape)
    )
    return signal_array + noise


def simulate_pilot_observations(
    channels: ArrayLike,
    pilots: ArrayLike,
    snr_db: float,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Apply pilot symbols to channels and add unit-reference-power AWGN."""
    channels_array = np.asarray(channels)
    pilots_array = np.asarray(pilots)
    try:
        clean = channels_array * pilots_array
    except ValueError as exc:
        raise ValueError("channels and pilots must be broadcast-compatible.") from exc
    return add_complex_awgn(clean, snr_db, rng)
