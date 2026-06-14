import numpy as np
import pytest

from channel_estimation.metrics import (
    aggregate_by_snr,
    bit_error_rate,
    nmse_db,
    normalized_mean_squared_error,
)


def test_bit_error_rate_known_case():
    assert bit_error_rate([0, 1, 1, 0], [0, 0, 1, 1]) == 0.5


def test_nmse_is_zero_for_exact_estimate():
    reference = np.array([1 + 1j, 2 - 1j])
    assert normalized_mean_squared_error(reference, reference) == 0.0
    assert nmse_db(reference, reference) == float("-inf")


def test_nmse_rejects_zero_power_reference():
    with pytest.raises(ValueError, match="Reference power"):
        normalized_mean_squared_error(np.zeros(2), np.ones(2))


def test_nmse_rejects_invalid_epsilon():
    with pytest.raises(ValueError, match="epsilon"):
        normalized_mean_squared_error(np.ones(2), np.ones(2), epsilon=0)


def test_aggregate_by_snr():
    result = aggregate_by_snr([0, 0, 5], [1.0, 3.0, 0.5])
    assert result[0.0]["mean"] == 2.0
    assert result[0.0]["count"] == 2
