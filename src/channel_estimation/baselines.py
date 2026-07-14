"""classical channel-estimation baselines.

three baselines live here:

* ``least_squares_estimate`` -- the scalar ``h_hat = y / x`` estimate used by the
  original pilot-observation model and the fast numpy tests.
* ``lmmse_estimate`` -- linear minimum-mean-squared-error estimation over a set of
  subcarriers given a channel covariance matrix and the noise power. this is
  implemented in numpy so it is fully testable without the ml stack.
* ``grid_ls_estimate`` -- least-squares estimation with interpolation over a full
  sionna ofdm resource grid. this wraps sionna's ``LSChannelEstimator`` and is
  imported lazily; it has not been executed in this environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def least_squares_estimate(
    received_pilots: ArrayLike,
    pilot_symbols: ArrayLike,
) -> NDArray[np.complex128]:
    """Estimate channel coefficients by dividing observations by known pilots.

    For the scalar pilot model ``y = h*x + n``, the LS solution is ``h_hat = y/x``.
    Pilot symbols must be non-zero and broadcast-compatible with the observations.
    """
    received = np.asarray(received_pilots)
    pilots = np.asarray(pilot_symbols)
    try:
        pilots_are_finite = bool(np.all(np.isfinite(pilots)))
    except TypeError as exc:
        raise ValueError("Pilot symbols must be numeric and finite.") from exc
    if not pilots_are_finite:
        raise ValueError("Pilot symbols must be finite.")
    if np.any(pilots == 0):
        raise ValueError("Least-squares estimation requires non-zero pilot symbols.")
    try:
        return np.asarray(received / pilots)
    except ValueError as exc:
        raise ValueError(
            "received_pilots and pilot_symbols must be broadcast-compatible."
        ) from exc


def estimate_channel_covariance(channels: ArrayLike) -> NDArray[np.complex128]:
    """Estimate the subcarrier channel covariance matrix from realizations.

    ``channels`` must be shaped ``(num_samples, num_subcarriers)``. returns the
    hermitian covariance ``R[a, b] = mean(h_a * conj(h_b))`` used by the lmmse
    baseline. this is the explicit, documented covariance assumption that the
    previous placeholder lmmse implementation was waiting on.
    """
    array = np.asarray(channels)
    if not np.iscomplexobj(array):
        raise ValueError("channels must be a complex-valued array.")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("channels must have shape (num_samples, num_subcarriers).")
    num_samples = array.shape[0]
    return np.asarray(array.T @ array.conj()) / num_samples


def lmmse_estimate(
    ls_estimate: ArrayLike,
    channel_covariance: ArrayLike,
    noise_power: float,
) -> NDArray[np.complex128]:
    """Apply frequency-domain LMMSE smoothing to a least-squares estimate.

    given per-subcarrier least-squares estimates ``h_ls = h + n`` with unit-power
    pilots, channel covariance ``R`` and noise power ``sigma^2``, the lmmse
    estimate is ``h_hat = R (R + sigma^2 I)^-1 h_ls``. ``ls_estimate`` is shaped
    ``(num_samples, num_subcarriers)`` and ``channel_covariance`` is the matching
    ``(num_subcarriers, num_subcarriers)`` matrix.
    """
    h_ls = np.asarray(ls_estimate)
    covariance = np.asarray(channel_covariance)
    if not np.isfinite(noise_power) or noise_power < 0:
        raise ValueError("noise_power must be a non-negative finite value.")
    if h_ls.ndim != 2:
        raise ValueError("ls_estimate must have shape (num_samples, num_subcarriers).")
    num_subcarriers = h_ls.shape[1]
    if covariance.shape != (num_subcarriers, num_subcarriers):
        raise ValueError(
            "channel_covariance must be square with the subcarrier dimension."
        )

    regularized = covariance + noise_power * np.eye(num_subcarriers)
    weight = covariance @ np.linalg.inv(regularized)
    return np.asarray(h_ls @ weight.T)


def grid_ls_estimate(
    received_grid: Any,
    noise_power: float,
    resource_grid: Any,
    *,
    interpolation_type: str = "lin",
) -> NDArray[np.complexfloating]:
    """Least-squares estimation with interpolation over a sionna resource grid.

    wraps sionna's ``LSChannelEstimator`` to estimate the channel at every
    resource element from the pilot observations. ``received_grid`` must be the
    sionna-shaped received grid and ``resource_grid`` the matching
    ``ResourceGrid``. ``interpolation_type`` is ``"nn"`` (nearest) or ``"lin"``
    (linear). returns a numpy array; not executed in this environment.
    """
    if interpolation_type not in ("nn", "lin", "lin_time_avg"):
        raise ValueError("interpolation_type must be 'nn', 'lin', or 'lin_time_avg'.")
    try:
        from sionna.phy.ofdm import LSChannelEstimator
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError(
            "sionna is required for grid least-squares estimation. install the "
            "'ml' extra or use requirements.txt."
        ) from exc

    from .sionna_ofdm import to_numpy

    estimator = LSChannelEstimator(
        resource_grid, interpolation_type=interpolation_type
    )
    h_hat, _err_var = estimator(received_grid, noise_power)
    return to_numpy(h_hat)


def grid_lmmse_estimate(
    received_grid: Any,
    noise_power: float,
    resource_grid: Any,
    spec: Any,
    *,
    order: str = "f-t",
) -> NDArray[np.complexfloating]:
    """Estimate a TDL grid with covariance-informed LMMSE interpolation.

    The covariance matrices use the configured TDL model, delay spread,
    mobility, carrier frequency, and OFDM symbol duration. This is a
    model-informed benchmark and should be reported as such.
    """
    if not str(spec.channel_kind).startswith("tdl-"):
        raise ValueError("Grid LMMSE interpolation currently requires a TDL channel.")
    try:
        from sionna.phy.ofdm import (
            LMMSEInterpolator,
            LSChannelEstimator,
            tdl_freq_cov_mat,
            tdl_time_cov_mat,
        )
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError("Sionna is required for grid LMMSE estimation.") from exc

    from .sionna_ofdm import _doppler_to_speed, to_numpy

    model = str(spec.channel_kind).split("-")[-1].upper()
    speed = _doppler_to_speed(spec.max_doppler_hz, spec.carrier_frequency_hz)
    ofdm_symbol_duration = (
        spec.num_subcarriers + spec.cyclic_prefix_length
    ) / (spec.num_subcarriers * spec.subcarrier_spacing_hz)
    covariance_frequency = tdl_freq_cov_mat(
        model,
        spec.subcarrier_spacing_hz,
        spec.num_subcarriers,
        spec.delay_spread_ns * 1e-9,
    )
    covariance_time = tdl_time_cov_mat(
        model,
        speed,
        spec.carrier_frequency_hz,
        ofdm_symbol_duration,
        spec.num_ofdm_symbols,
    )
    interpolator = LMMSEInterpolator(
        resource_grid.pilot_pattern,
        covariance_time,
        covariance_frequency,
        order=order,
    )
    estimator = LSChannelEstimator(resource_grid, interpolator=interpolator)
    h_hat, _err_var = estimator(received_grid, noise_power)
    return to_numpy(h_hat)
