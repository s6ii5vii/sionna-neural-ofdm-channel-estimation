import pytest

from channel_estimation.config import ConfigError, validate_config


def valid_config():
    return {
        "experiment": {
            "name": "baseline-ls",
            "snr-db": [0, 5, 10],
            "num-samples": 10,
            "num-pilots": 8,
            "pilot-density": 1.0,
            "random-seed": 42,
        },
        "output": {
            "figures-dir": "results/figures",
            "tables-dir": "results/tables",
        },
    }


def test_validate_config_accepts_hyphenated_settings():
    validate_config(valid_config())


@pytest.mark.parametrize("name", ["baseline_ls", "../baseline", "Baseline-LS"])
def test_validate_config_rejects_unsafe_experiment_name(name):
    config = valid_config()
    config["experiment"]["name"] = name
    with pytest.raises(ConfigError, match="hyphenated slug"):
        validate_config(config)


def test_validate_config_rejects_boolean_integer():
    config = valid_config()
    config["experiment"]["num-samples"] = True
    with pytest.raises(ConfigError, match="integer"):
        validate_config(config)


def valid_grid_config():
    return {
        "experiment": {
            "name": "grid-tdl-v1",
            "snr-db": [0, 5, 10],
            "num-samples": 100,
            "num-subcarriers": 72,
            "num-ofdm-symbols": 14,
            "pilot-ofdm-symbol-indices": [2, 11],
            "channel-model": "tdl-a",
            "delay-spread-ns": 100,
            "max-doppler-hz": 0,
            "random-seed": 42,
        },
        "output": {
            "figures-dir": "results/figures",
            "tables-dir": "results/tables",
        },
    }


def test_validate_config_accepts_grid_schema():
    validate_config(valid_grid_config())


def test_validate_config_rejects_pilot_index_outside_grid():
    config = valid_grid_config()
    config["experiment"]["pilot-ofdm-symbol-indices"] = [2, 14]
    with pytest.raises(ConfigError, match="pilot symbol index"):
        validate_config(config)


def test_validate_config_rejects_unknown_channel_model():
    config = valid_grid_config()
    config["experiment"]["channel-model"] = "lte"
    with pytest.raises(ConfigError, match="channel-model"):
        validate_config(config)


def test_validate_config_rejects_negative_delay_spread():
    config = valid_grid_config()
    config["experiment"]["delay-spread-ns"] = -1
    with pytest.raises(ConfigError, match="delay-spread-ns"):
        validate_config(config)


def test_validate_config_accepts_neural_block():
    config = valid_grid_config()
    config["neural"] = {
        "checkpoint-path": "results/checkpoints/model.pt",
        "filters": 16,
        "num-layers": 3,
        "input-source": "ls-lin",
    }
    validate_config(config)


def test_validate_config_rejects_neural_missing_checkpoint():
    config = valid_grid_config()
    config["neural"] = {"filters": 16, "num-layers": 3}
    with pytest.raises(ConfigError, match="neural.checkpoint-path"):
        validate_config(config)


def test_validate_config_rejects_neural_non_positive_filters():
    config = valid_grid_config()
    config["neural"] = {
        "checkpoint-path": "results/checkpoints/model.pt",
        "filters": 0,
        "num-layers": 3,
    }
    with pytest.raises(ConfigError, match="neural.filters"):
        validate_config(config)


def test_validate_config_rejects_neural_unknown_input_source():
    config = valid_grid_config()
    config["neural"] = {
        "checkpoint-path": "results/checkpoints/model.pt",
        "filters": 16,
        "num-layers": 3,
        "input-source": "pilots-only",
    }
    with pytest.raises(ConfigError, match="neural.input-source"):
        validate_config(config)


def test_validate_config_accepts_dataset_generation_block():
    config = valid_grid_config()
    config["training"] = {
        "dataset-path": "data/foo.npz",
        "checkpoint-path": "results/checkpoints/m.pt",
        "hidden-units": 16,
        "epochs": 10,
        "batch-size": 32,
        "num-layers": 3,
        "dataset-generation": {
            "num-samples": 1000,
            "snr-db": 10.0,
            "random-seed": 42,
            "input-source": "ls-lin",
        },
    }
    validate_config(config)


def test_validate_config_rejects_dataset_generation_bad_snr():
    config = valid_grid_config()
    config["training"] = {
        "dataset-path": "data/foo.npz",
        "checkpoint-path": "results/checkpoints/m.pt",
        "hidden-units": 16,
        "epochs": 10,
        "batch-size": 32,
        "dataset-generation": {
            "num-samples": 1000,
            "snr-db": "not-a-number",
            "random-seed": 42,
        },
    }
    with pytest.raises(ConfigError, match="dataset-generation.snr-db"):
        validate_config(config)


def test_validate_config_rejects_dataset_generation_unknown_input_source():
    config = valid_grid_config()
    config["training"] = {
        "dataset-path": "data/foo.npz",
        "checkpoint-path": "results/checkpoints/m.pt",
        "hidden-units": 16,
        "epochs": 10,
        "batch-size": 32,
        "dataset-generation": {
            "num-samples": 1000,
            "snr-db": 10.0,
            "random-seed": 42,
            "input-source": "pilots-only",
        },
    }
    with pytest.raises(ConfigError, match="dataset-generation.input-source"):
        validate_config(config)
