from __future__ import annotations

import pytest

from channel_estimation.sweep import (
    build_sweep_config,
    summarize_estimator_rows,
    summarize_neural_margins,
)


def _row(channel_model, seed, snr_db, estimator, nmse):
    return {
        "channel-model": channel_model,
        "random-seed": seed,
        "snr-db": snr_db,
        "estimator": estimator,
        "nmse": nmse,
        "num-samples": 100,
    }


def test_summarize_estimator_rows_groups_by_channel_snr_and_estimator():
    rows = [
        _row("tdl-a", 42, 0.0, "ls-lin", 0.9),
        _row("tdl-a", 43, 0.0, "ls-lin", 0.7),
        _row("tdl-a", 42, 0.0, "neural-cnn", 0.4),
        _row("tdl-a", 43, 0.0, "neural-cnn", 0.2),
    ]

    summary = summarize_estimator_rows(rows)

    ls_row = next(row for row in summary if row["estimator"] == "ls-lin")
    neural_row = next(row for row in summary if row["estimator"] == "neural-cnn")
    assert ls_row["nmse-mean"] == pytest.approx(0.8)
    assert ls_row["num-runs"] == 2
    assert neural_row["nmse-mean"] == pytest.approx(0.3)


def test_summarize_neural_margins_reports_win_rate_and_improvement():
    rows = [
        _row("tdl-a", 42, 0.0, "ls-lin", 1.0),
        _row("tdl-a", 42, 0.0, "neural-cnn", 0.5),
        _row("tdl-a", 43, 0.0, "ls-lin", 1.0),
        _row("tdl-a", 43, 0.0, "neural-cnn", 1.2),
    ]

    summary = summarize_neural_margins(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["improvement-percent-mean"] == pytest.approx(15.0)
    assert row["neural-win-rate"] == pytest.approx(0.5)
    assert row["num-runs"] == 2


def test_build_sweep_config_isolates_dataset_and_checkpoint_paths():
    base_config = {
        "experiment": {
            "name": "grid-neural-comparison-v1",
            "num-subcarriers": 72,
            "num-ofdm-symbols": 14,
            "pilot-ofdm-symbol-indices": [2, 11],
            "channel-model": "tdl-a",
            "delay-spread-ns": 100,
            "max-doppler-hz": 0,
            "snr-db": [-5, 0],
            "num-samples": 100,
            "random-seed": 42,
        },
        "training": {
            "dataset-path": "data/base.npz",
            "hidden-units": 16,
            "num-layers": 3,
            "epochs": 1,
            "batch-size": 8,
            "checkpoint-path": "results/checkpoints/base.pt",
            "dataset-generation": {
                "num-samples": 100,
                "snr-db": 10.0,
                "random-seed": 42,
                "input-source": "ls-lin",
            },
        },
        "neural": {
            "checkpoint-path": "results/checkpoints/base.pt",
            "filters": 16,
            "num-layers": 3,
            "input-source": "ls-lin",
        },
        "output": {
            "figures-dir": "results/figures",
            "tables-dir": "results/tables",
        },
    }

    config = build_sweep_config(
        base_config,
        channel_model="tdl-b",
        seed=43,
        run_name="grid-neural-comparison-v1-tdl-b-seed-43",
    )

    assert config["experiment"]["channel-model"] == "tdl-b"
    assert config["experiment"]["random-seed"] == 100_043
    assert config["training"]["dataset-generation"]["random-seed"] == 43
    assert config["training"]["model-seed"] == 50_043
    assert config["experiment"]["random-seed"] != config["training"]["model-seed"]
    assert config["training"]["dataset-path"].endswith("tdl-b-seed-43-dataset.npz")
    assert config["neural"]["checkpoint-path"] == config["training"]["checkpoint-path"]
    assert base_config["experiment"]["channel-model"] == "tdl-a"
