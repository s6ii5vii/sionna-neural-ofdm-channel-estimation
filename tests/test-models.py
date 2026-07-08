import pytest

from channel_estimation.models import build_grid_estimator


def test_build_grid_estimator_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="positive integers"):
        build_grid_estimator(0, 16)


def test_build_grid_estimator_rejects_bad_dropout():
    with pytest.raises(ValueError, match="dropout_rate"):
        build_grid_estimator(14, 16, dropout_rate=1.0)


# constructing the module requires torch; skip cleanly without it.
def test_build_grid_estimator_output_shape_and_size():
    torch = pytest.importorskip("torch")
    from channel_estimation.models import count_parameters

    model = build_grid_estimator(14, 16, filters=16, num_layers=3)
    inputs = torch.zeros((4, 14, 16, 2), dtype=torch.float32)
    outputs = model(inputs)
    assert tuple(outputs.shape) == (4, 14, 16, 2)
    # stays comfortably under the lightweight 50k-parameter budget.
    assert count_parameters(model) < 50_000


def test_build_lightweight_estimator_output_shape():
    torch = pytest.importorskip("torch")
    from channel_estimation.models import build_lightweight_estimator

    model = build_lightweight_estimator(8, hidden_units=32)
    inputs = torch.zeros((5, 8, 2), dtype=torch.float32)
    outputs = model(inputs)
    assert tuple(outputs.shape) == (5, 8, 2)
