"""Tests for the v2 grid experiment: sparse pilots, SNR ranges, mismatched
LMMSE, and the extended CNN options. Everything here runs without Sionna; the
CNN tests skip cleanly when PyTorch is not installed."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from channel_estimation import sionna_ofdm
from channel_estimation.baselines import grid_lmmse_estimate
from channel_estimation.config import ConfigError, load_config, validate_config
from channel_estimation.dataset import (
    batch_snr_schedule,
    generate_grid_dataset,
    split_dataset,
    split_snr_values,
)
from channel_estimation.models import build_grid_estimator
from channel_estimation.sionna_ofdm import (
    GridSpec,
    assumed_channel_spec,
    qpsk_pilot_values,
    sparse_pilot_mask,
)

V2_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "grid-neural-comparison-v2"
    / "config.yaml"
)


# --- sparse pilots ---------------------------------------------------------


def test_sparse_pilot_mask_places_pilots_on_spaced_subcarriers():
    mask = sparse_pilot_mask(14, 16, (2, 11), 4)
    assert mask.shape == (14, 16)
    expected_columns = [0, 4, 8, 12, 15]
    for symbol in (2, 11):
        assert np.flatnonzero(mask[symbol]).tolist() == expected_columns
    assert mask.sum() == 2 * len(expected_columns)
    assert not mask[[0, 1, 3, 10, 13]].any()


def test_sparse_pilot_mask_with_unit_spacing_fills_pilot_symbols():
    mask = sparse_pilot_mask(14, 8, (2,), 1)
    assert mask[2].all()
    assert mask.sum() == 8


def test_qpsk_pilot_values_are_unit_power_and_deterministic():
    first = qpsk_pilot_values(64, seed=3)
    second = qpsk_pilot_values(64, seed=3)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(np.abs(first), 1.0, rtol=1e-6)


def test_grid_spec_rejects_bad_pilot_spacing():
    with pytest.raises(ValueError, match="pilot_subcarrier_spacing"):
        GridSpec(16, 14, (2, 11), pilot_subcarrier_spacing=0)
    with pytest.raises(ValueError, match="pilot_subcarrier_spacing"):
        GridSpec(16, 14, (2, 11), pilot_subcarrier_spacing=17)


def test_grid_spec_reads_pilot_spacing_from_experiment():
    spec = GridSpec.from_experiment(
        {
            "num-subcarriers": 144,
            "num-ofdm-symbols": 14,
            "pilot-ofdm-symbol-indices": [2, 11],
            "pilot-subcarrier-spacing": 4,
        }
    )
    assert spec.pilot_subcarrier_spacing == 4
    default = GridSpec.from_experiment(
        {"num-subcarriers": 72, "num-ofdm-symbols": 14, "pilot-ofdm-symbol-indices": [2]}
    )
    assert default.pilot_subcarrier_spacing == 1


# --- mismatched lmmse ------------------------------------------------------


def test_assumed_channel_spec_keeps_grid_and_replaces_statistics():
    spec = GridSpec(144, 14, (2, 11), "tdl-c", 300.0, 200.0, pilot_subcarrier_spacing=4)
    assumed = assumed_channel_spec(
        spec, {"channel-model": "tdl-a", "delay-spread-ns": 100, "max-doppler-hz": 0}
    )
    assert assumed.channel_kind == "tdl-a"
    assert assumed.delay_spread_ns == 100.0
    assert assumed.max_doppler_hz == 0.0
    assert assumed.num_subcarriers == 144
    assert assumed.pilot_subcarrier_spacing == 4
    assert spec.channel_kind == "tdl-c"


def test_grid_lmmse_uses_assumed_statistics_for_covariances(monkeypatch):
    calls = {}

    class FakeInterpolator:
        def __init__(self, _pattern, _time, _frequency, *, order):
            pass

    class FakeEstimator:
        def __init__(self, _grid, *, interpolator):
            pass

        def __call__(self, _received, _noise):
            return np.ones((1, 1), dtype=np.complex64), None

    def fake_frequency(model, _spacing, _subcarriers, delay_spread, **_kwargs):
        calls["frequency"] = (model, delay_spread)
        return object()

    def fake_time(model, speed, *_args, **_kwargs):
        calls["time"] = (model, speed)
        return object()

    fake_ofdm = ModuleType("sionna.phy.ofdm")
    fake_ofdm.LMMSEInterpolator = FakeInterpolator
    fake_ofdm.LSChannelEstimator = FakeEstimator
    fake_ofdm.tdl_freq_cov_mat = fake_frequency
    fake_ofdm.tdl_time_cov_mat = fake_time
    monkeypatch.setitem(sys.modules, "sionna.phy.ofdm", fake_ofdm)

    true_spec = GridSpec(72, 14, (2, 11), "tdl-c", 300.0, 200.0)
    assumed = assumed_channel_spec(
        true_spec, {"channel-model": "tdl-a", "delay-spread-ns": 100, "max-doppler-hz": 0}
    )
    grid = SimpleNamespace(pilot_pattern=object())
    grid_lmmse_estimate(object(), 0.1, grid, true_spec, assumed_spec=assumed)

    assert calls["frequency"] == ("A", pytest.approx(100e-9))
    assert calls["time"] == ("A", 0.0)


# --- snr ranges ------------------------------------------------------------


def test_batch_snr_schedule_fixed_value():
    np.testing.assert_array_equal(batch_snr_schedule(10.0, 3, 0), [10.0, 10.0, 10.0])


def test_batch_snr_schedule_range_is_reproducible_and_bounded():
    first = batch_snr_schedule([-5.0, 15.0], 200, 42)
    second = batch_snr_schedule([-5.0, 15.0], 200, 42)
    np.testing.assert_array_equal(first, second)
    assert first.min() >= -5.0 and first.max() < 15.0
    # a range should actually cover low and high snr, not cluster at one point.
    assert first.min() < 0.0 and first.max() > 10.0


@pytest.mark.parametrize("bad", [[5.0, 5.0], [10.0, 0.0], [0.0, 1.0, 2.0]])
def test_batch_snr_schedule_rejects_bad_range(bad):
    with pytest.raises(ValueError, match="range"):
        batch_snr_schedule(bad, 3, 0)


def test_split_snr_values_matches_dataset_split_boundaries():
    values = np.arange(20, dtype=np.float32)
    inputs = np.zeros((20, 3, 2), dtype=np.float32)
    split = split_dataset(inputs, inputs)
    snr_split = split_snr_values(values)
    for name in ("train", "val", "test"):
        assert len(snr_split[f"snr_db_{name}"]) == len(split[f"x_{name}"])


def test_generate_grid_dataset_simulates_each_batch_at_its_scheduled_snr(monkeypatch):
    seen = []

    def fake_simulate_grid(spec, snr_db, *, batch_size, seed):
        seen.append((float(snr_db), seed))
        shape = (batch_size, spec.num_ofdm_symbols, spec.num_subcarriers)
        grid = np.full(shape, snr_db, dtype=np.complex64)
        return grid, grid, 0.1

    monkeypatch.setattr(sionna_ofdm, "simulate_grid", fake_simulate_grid)
    spec = GridSpec(8, 4, (1,))
    dataset = generate_grid_dataset(
        spec, num_samples=40, snr_db=[-5.0, 15.0], batch_size=10, random_seed=7
    )

    expected = batch_snr_schedule([-5.0, 15.0], 4, 7)
    assert [snr for snr, _seed in seen] == pytest.approx(expected.tolist())
    assert [seed for _snr, seed in seen] == [7, 8, 9, 10]
    all_snr = np.concatenate(
        [dataset["snr_db_train"], dataset["snr_db_val"], dataset["snr_db_test"]]
    )
    np.testing.assert_allclose(all_snr, np.repeat(expected, 10), rtol=1e-6)
    # each stored sample's snr matches the snr it was simulated at.
    np.testing.assert_allclose(
        dataset["x_train"][:, 0, 0, 0], dataset["snr_db_train"], rtol=1e-6
    )


# --- config ----------------------------------------------------------------


def test_v2_config_is_valid():
    config = load_config(V2_CONFIG)
    experiment = config["experiment"]
    assert experiment["pilot-subcarrier-spacing"] > 1
    assert experiment["max-doppler-hz"] > 0
    assert config["training"]["dataset-generation"]["snr-db"] == [-5, 15]


def _v2_config():
    return load_config(V2_CONFIG)


def test_config_rejects_pilot_spacing_wider_than_grid():
    config = _v2_config()
    config["experiment"]["pilot-subcarrier-spacing"] = 1000
    with pytest.raises(ConfigError, match="pilot-subcarrier-spacing"):
        validate_config(config)


def test_config_rejects_non_tdl_assumed_lmmse_channel():
    config = _v2_config()
    config["experiment"]["lmmse-assumed"]["channel-model"] = "rayleigh"
    with pytest.raises(ConfigError, match="lmmse-assumed.channel-model"):
        validate_config(config)


def test_config_rejects_unknown_assumed_lmmse_key():
    config = _v2_config()
    config["experiment"]["lmmse-assumed"]["speed"] = 3
    with pytest.raises(ConfigError, match="Unknown"):
        validate_config(config)


def test_config_rejects_reversed_snr_range():
    config = _v2_config()
    config["training"]["dataset-generation"]["snr-db"] = [15, -5]
    with pytest.raises(ConfigError, match="low < high"):
        validate_config(config)


def test_config_rejects_dilations_with_wrong_length():
    config = _v2_config()
    config["training"]["dilations"] = [1, 2]
    config["neural"]["dilations"] = [1, 2]
    with pytest.raises(ConfigError, match="dilations"):
        validate_config(config)


def test_config_rejects_training_and_neural_architecture_mismatch():
    config = _v2_config()
    config["neural"]["snr-input"] = False
    with pytest.raises(ConfigError, match="snr-input"):
        validate_config(config)


def test_config_rejects_residual_on_raw_received_grid():
    config = _v2_config()
    config["training"]["dataset-generation"]["input-source"] = "received"
    config["neural"]["input-source"] = "received"
    with pytest.raises(ConfigError, match="residual"):
        validate_config(config)


def test_config_treats_omitted_architecture_options_as_defaults():
    config = _v2_config()
    for block in (config["training"], config["neural"]):
        block.pop("dilations")
        block.pop("snr-input")
        block["residual"] = False
    config["neural"].pop("residual")
    validate_config(config)


# --- cnn -------------------------------------------------------------------


def test_build_grid_estimator_rejects_bad_dilations():
    with pytest.raises(ValueError, match="one value per hidden layer"):
        build_grid_estimator(14, 16, num_layers=3, dilations=(1, 2))
    with pytest.raises(ValueError, match="positive integers"):
        build_grid_estimator(14, 16, num_layers=2, dilations=(1, 0))


def test_extended_grid_estimator_shapes_and_budget():
    torch = pytest.importorskip("torch")
    from channel_estimation.models import count_parameters

    model = build_grid_estimator(
        14,
        144,
        filters=32,
        num_layers=4,
        dilations=(1, 2, 4, 8),
        residual=True,
        snr_input=True,
    )
    inputs = torch.randn((3, 14, 144, 2))
    outputs = model(inputs, torch.tensor([-5.0, 5.0, 15.0]))
    assert tuple(outputs.shape) == (3, 14, 144, 2)
    assert count_parameters(model) < 50_000
    with pytest.raises(ValueError, match="snr_input"):
        model(inputs)


def test_residual_estimator_starts_from_its_input():
    torch = pytest.importorskip("torch")
    model = build_grid_estimator(4, 8, filters=4, num_layers=2, residual=True)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    inputs = torch.randn((2, 4, 8, 2))
    torch.testing.assert_close(model(inputs), inputs)


def test_dilated_estimator_has_wider_receptive_field():
    torch = pytest.importorskip("torch")

    def receptive_width(dilations):
        torch.manual_seed(0)
        model = build_grid_estimator(
            1, 101, filters=4, num_layers=4, dilations=dilations
        )
        inputs = torch.zeros((1, 1, 101, 2), requires_grad=True)
        model(inputs)[0, 0, 50, 0].backward()
        touched = inputs.grad[0, 0, :, :].abs().sum(dim=-1) > 0
        return int(touched.sum())

    assert receptive_width((1, 1, 1, 1)) == 11
    assert receptive_width((1, 2, 4, 8)) == 33


def test_default_grid_estimator_keeps_v1_checkpoint_layout():
    pytest.importorskip("torch")
    model = build_grid_estimator(14, 72, filters=16, num_layers=3)
    assert list(model.state_dict()) == [
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
        "net.6.weight",
        "net.6.bias",
    ]


def test_train_model_with_snr_input_and_residual(tmp_path):
    pytest.importorskip("torch")
    from channel_estimation.dataset import save_npz_dataset
    from channel_estimation.train import train_model

    rng = np.random.default_rng(0)
    targets = rng.standard_normal((40, 4, 8, 2)).astype(np.float32)
    inputs = targets + 0.3 * rng.standard_normal(targets.shape).astype(np.float32)
    dataset = split_dataset(inputs, targets)
    dataset.update(split_snr_values(rng.uniform(-5, 15, size=40).astype(np.float32)))
    dataset_path = save_npz_dataset(tmp_path / "grid.npz", dataset)

    config = {
        "experiment": {"name": "smoke", "model-parameter-target": 50_000},
        "training": {
            "dataset-path": str(dataset_path),
            "checkpoint-path": str(tmp_path / "model.pt"),
            "hidden-units": 4,
            "num-layers": 2,
            "dilations": [1, 2],
            "residual": True,
            "snr-input": True,
            "epochs": 2,
            "batch-size": 8,
            "model-seed": 1,
        },
    }
    checkpoint = train_model(config)
    report = __import__("json").loads(
        checkpoint.with_suffix(".report.json").read_text(encoding="utf-8")
    )
    assert report["training"]["snr-input"] is True
    assert report["training"]["dilations"] == [1, 2]
    assert np.isfinite(report["test-nmse"])


def test_train_model_with_snr_input_needs_snr_arrays(tmp_path):
    pytest.importorskip("torch")
    from channel_estimation.dataset import save_npz_dataset
    from channel_estimation.train import train_model

    data = np.zeros((20, 4, 8, 2), dtype=np.float32)
    dataset_path = save_npz_dataset(tmp_path / "grid.npz", split_dataset(data, data))
    config = {
        "experiment": {"name": "smoke"},
        "training": {
            "dataset-path": str(dataset_path),
            "checkpoint-path": str(tmp_path / "model.pt"),
            "hidden-units": 4,
            "num-layers": 2,
            "snr-input": True,
            "epochs": 1,
            "batch-size": 8,
        },
    }
    with pytest.raises(KeyError, match="snr_db_train"):
        train_model(config)


def test_neural_predict_passes_evaluation_snr():
    torch = pytest.importorskip("torch")
    from channel_estimation.evaluate import _neural_predict

    seen = {}

    class Recorder(torch.nn.Module):
        snr_input = True

        def forward(self, x, snr):
            seen["snr"] = snr.tolist()
            return x

    grid = np.ones((2, 4, 8), dtype=np.complex64)
    out = _neural_predict(torch, Recorder(), torch.device("cpu"), grid, 7.5)
    assert seen["snr"] == [7.5, 7.5]
    np.testing.assert_allclose(out, grid)


def test_sparse_resource_grid_matches_mask():
    pytest.importorskip("sionna")
    from channel_estimation.sionna_ofdm import build_resource_grid, simulate_grid

    spec = GridSpec(16, 14, (2, 11), pilot_subcarrier_spacing=4)
    grid = build_resource_grid(spec)
    mask = np.asarray(sionna_ofdm.to_numpy(grid.pilot_pattern.mask))[0, 0].astype(bool)
    np.testing.assert_array_equal(mask, sparse_pilot_mask(14, 16, (2, 11), 4))
    y, h_true, _no = simulate_grid(spec, snr_db=10.0, batch_size=2, seed=0)
    assert y.shape == h_true.shape == (2, 14, 16)
