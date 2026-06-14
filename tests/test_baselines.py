import numpy as np
import pytest

from channel_estimation.baselines import least_squares_estimate, lmmse_estimate


def test_least_squares_recovers_noiseless_channel():
    channel = np.array([1 + 2j, -0.5 + 0.25j])
    pilots = np.array([1 + 0j, -1j])
    received = channel * pilots
    np.testing.assert_allclose(least_squares_estimate(received, pilots), channel)


def test_least_squares_rejects_zero_pilot():
    with pytest.raises(ValueError, match="non-zero"):
        least_squares_estimate([1 + 0j], [0 + 0j])


def test_lmmse_is_explicitly_planned():
    with pytest.raises(NotImplementedError, match="planned"):
        lmmse_estimate()
