import pytest

from channel_estimation.models import build_grid_estimator


def test_build_grid_estimator_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        build_grid_estimator(0, 16)


def test_build_grid_estimator_rejects_bad_dropout():
    with pytest.raises(ValueError, match="dropout_rate"):
        build_grid_estimator(14, 16, dropout_rate=1.0)


# constructing the keras model requires tensorflow; skip cleanly without it.
def test_build_grid_estimator_output_shape_and_size():
    pytest.importorskip("tensorflow")
    model = build_grid_estimator(14, 16, filters=16, num_layers=3)
    assert model.input_shape == (None, 14, 16, 2)
    assert model.output_shape == (None, 14, 16, 2)
    # stays comfortably under the lightweight 50k-parameter budget.
    assert model.count_params() < 50_000
