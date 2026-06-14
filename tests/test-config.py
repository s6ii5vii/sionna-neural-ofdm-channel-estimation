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
