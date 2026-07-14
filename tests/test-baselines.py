import numpy as np
import pytest

from channel_estimation.baselines import (
    estimate_channel_covariance,
    grid_lmmse_estimate,
    least_squares_estimate,
    lmmse_estimate,
)


class _RayleighSpec:
    channel_kind = "rayleigh"


def test_grid_lmmse_rejects_non_tdl_channel():
    with pytest.raises(ValueError, match="requires a TDL channel"):
        grid_lmmse_estimate(None, 0.1, None, _RayleighSpec())


def test_least_squares_recovers_noiseless_channel():
    channel = np.array([1 + 2j, -0.5 + 0.25j])
    pilots = np.array([1 + 0j, -1j])
    received = channel * pilots
    np.testing.assert_allclose(least_squares_estimate(received, pilots), channel)


def test_least_squares_rejects_zero_pilot():
    with pytest.raises(ValueError, match="non-zero"):
        least_squares_estimate([1 + 0j], [0 + 0j])


def test_least_squares_rejects_non_finite_pilot():
    with pytest.raises(ValueError, match="finite"):
        least_squares_estimate([1 + 0j], [np.nan + 0j])


def test_estimate_channel_covariance_is_hermitian():
    rng = np.random.default_rng(0)
    channels = (
        rng.normal(size=(2000, 4)) + 1j * rng.normal(size=(2000, 4))
    ) / np.sqrt(2.0)
    covariance = estimate_channel_covariance(channels)
    assert covariance.shape == (4, 4)
    np.testing.assert_allclose(covariance, covariance.conj().T, atol=1e-12)


def test_estimate_channel_covariance_rejects_wrong_rank():
    with pytest.raises(ValueError, match="num_samples, num_subcarriers"):
        estimate_channel_covariance(np.zeros((3, 3, 3), dtype=complex))


def test_lmmse_returns_ls_estimate_without_noise():
    rng = np.random.default_rng(1)
    channels = (
        rng.normal(size=(1000, 3)) + 1j * rng.normal(size=(1000, 3))
    ) / np.sqrt(2.0)
    covariance = estimate_channel_covariance(channels)
    estimate = lmmse_estimate(channels, covariance, noise_power=0.0)
    np.testing.assert_allclose(estimate, channels, atol=1e-6)


def test_lmmse_reduces_error_versus_ls_under_noise():
    rng = np.random.default_rng(2)
    # correlated channel across subcarriers so smoothing can help.
    base = rng.normal(size=(4000, 1)) + 1j * rng.normal(size=(4000, 1))
    channels = np.tile(base, (1, 6)) / np.sqrt(2.0)
    covariance = estimate_channel_covariance(channels)
    noise_power = 0.5
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=channels.shape) + 1j * rng.normal(size=channels.shape)
    )
    ls_estimate = channels + noise
    lmmse = lmmse_estimate(ls_estimate, covariance, noise_power)
    ls_error = np.mean(np.abs(channels - ls_estimate) ** 2)
    lmmse_error = np.mean(np.abs(channels - lmmse) ** 2)
    assert lmmse_error < ls_error


def test_lmmse_rejects_mismatched_covariance():
    with pytest.raises(ValueError, match="square"):
        lmmse_estimate(np.zeros((5, 4), dtype=complex), np.eye(3), noise_power=1.0)
