import sys
from types import ModuleType, SimpleNamespace

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


@pytest.mark.parametrize(
    ("channel_kind", "max_doppler_hz", "delay_spread_ns", "order"),
    [
        ("tdl-a", 0.0, 100.0, "f-t"),
        ("tdl-c", 70.0, 300.0, "t-f"),
        ("tdl-e", 250.0, 1000.0, "f-t"),
    ],
)
def test_grid_lmmse_uses_single_precision_covariances_across_tdl_profiles(
    monkeypatch, channel_kind, max_doppler_hz, delay_spread_ns, order
):
    calls = {}

    class FakeInterpolator:
        def __init__(self, _pattern, _time, _frequency, *, order):
            calls["order"] = order

    class FakeEstimator:
        def __init__(self, _grid, *, interpolator):
            calls["interpolator"] = interpolator

        def __call__(self, _received, _noise):
            return np.ones((1, 1), dtype=np.complex64), None

    def fake_frequency(model, spacing, subcarriers, delay_spread, **kwargs):
        calls["frequency-model"] = model
        calls["frequency-spacing"] = spacing
        calls["frequency-subcarriers"] = subcarriers
        calls["frequency-delay-spread"] = delay_spread
        calls["frequency-precision"] = kwargs.get("precision")
        return object()

    def fake_time(model, speed, carrier_frequency, symbol_duration, symbols, **kwargs):
        calls["time-model"] = model
        calls["time-speed"] = speed
        calls["time-carrier-frequency"] = carrier_frequency
        calls["time-symbol-duration"] = symbol_duration
        calls["time-symbols"] = symbols
        calls["time-precision"] = kwargs.get("precision")
        return object()

    fake_ofdm = ModuleType("sionna.phy.ofdm")
    fake_ofdm.LMMSEInterpolator = FakeInterpolator
    fake_ofdm.LSChannelEstimator = FakeEstimator
    fake_ofdm.tdl_freq_cov_mat = fake_frequency
    fake_ofdm.tdl_time_cov_mat = fake_time
    monkeypatch.setitem(sys.modules, "sionna.phy.ofdm", fake_ofdm)

    spec = SimpleNamespace(
        channel_kind=channel_kind,
        max_doppler_hz=max_doppler_hz,
        carrier_frequency_hz=3.5e9,
        num_subcarriers=72,
        cyclic_prefix_length=6,
        subcarrier_spacing_hz=30e3,
        delay_spread_ns=delay_spread_ns,
        num_ofdm_symbols=14,
    )
    grid = SimpleNamespace(pilot_pattern=object())

    result = grid_lmmse_estimate(object(), 0.1, grid, spec, order=order)

    assert result.dtype == np.complex64
    assert calls["order"] == order
    assert calls["frequency-model"] == channel_kind.removeprefix("tdl-").upper()
    assert calls["frequency-spacing"] == spec.subcarrier_spacing_hz
    assert calls["frequency-subcarriers"] == spec.num_subcarriers
    assert calls["frequency-delay-spread"] == pytest.approx(delay_spread_ns * 1e-9)
    assert calls["frequency-precision"] == "single"
    assert calls["time-model"] == channel_kind.removeprefix("tdl-").upper()
    assert calls["time-speed"] >= 0.0
    assert calls["time-carrier-frequency"] == spec.carrier_frequency_hz
    assert calls["time-symbol-duration"] == pytest.approx(
        (spec.num_subcarriers + spec.cyclic_prefix_length)
        / (spec.num_subcarriers * spec.subcarrier_spacing_hz)
    )
    assert calls["time-symbols"] == spec.num_ofdm_symbols
    assert calls["time-precision"] == "single"


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
