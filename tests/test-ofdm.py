import numpy as np
import pytest

from channel_estimation.ofdm import generate_pilot_symbols, pilot_indices


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_generate_pilots_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        generate_pilot_symbols(8, batch_size=batch_size)


def test_pilot_indices_are_unique_and_cover_requested_density():
    indices = pilot_indices(64, 0.25)
    assert len(indices) == 16
    assert len(np.unique(indices)) == 16
    assert indices[0] == 0
    assert indices[-1] == 63
