import numpy as np
import pytest

from channel_estimation.sionna_ofdm import GridSpec, simulate_grid


def make_spec(**overrides):
    params = dict(
        num_subcarriers=16,
        num_ofdm_symbols=14,
        pilot_ofdm_symbol_indices=(2, 11),
    )
    params.update(overrides)
    return GridSpec(**params)


def test_grid_spec_accepts_valid_values():
    spec = make_spec()
    assert spec.num_subcarriers == 16
    assert spec.pilot_ofdm_symbol_indices == (2, 11)


def test_grid_spec_rejects_pilot_index_outside_grid():
    with pytest.raises(ValueError, match="pilot symbol index"):
        make_spec(pilot_ofdm_symbol_indices=(2, 99))


def test_grid_spec_rejects_empty_pilot_indices():
    with pytest.raises(ValueError, match="must not be empty"):
        make_spec(pilot_ofdm_symbol_indices=())


def test_grid_spec_rejects_unknown_channel_kind():
    with pytest.raises(ValueError, match="channel_kind"):
        make_spec(channel_kind="lte")


def test_grid_spec_rejects_non_positive_delay_spread():
    with pytest.raises(ValueError, match="delay_spread_ns"):
        make_spec(delay_spread_ns=0.0)


# the simulation itself requires the ml stack (tensorflow + sionna) and,
# realistically, a gpu; skip cleanly where those are unavailable so the fast
# numpy suite stays green everywhere.
def test_simulate_grid_shapes():
    pytest.importorskip("sionna")
    pytest.importorskip("tensorflow")
    spec = make_spec()
    y, h_true, no = simulate_grid(spec, snr_db=10.0, batch_size=4, seed=0)
    expected = (4, spec.num_ofdm_symbols, spec.num_subcarriers)
    assert y.shape == expected
    assert h_true.shape == expected
    assert np.iscomplexobj(y) and np.iscomplexobj(h_true)
    assert no > 0
