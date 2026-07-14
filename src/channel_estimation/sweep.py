"""Validation sweeps for grid neural channel-estimation experiments."""

from __future__ import annotations

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Sequence

from .config import load_config, resolve_path, validate_config
from .evaluate import evaluate_grid_baselines, write_metrics_table
from .plotting import save_nmse_sweep_comparison
from .train import train_model


DEFAULT_CHANNEL_MODELS = ("tdl-a", "tdl-b", "tdl-c")
DEFAULT_SEEDS = (42, 43, 44)


def _safe_token(value: object) -> str:
    return str(value).replace(".", "p").replace("_", "-")


def build_sweep_config(
    base_config: dict[str, Any],
    *,
    channel_model: str,
    seed: int,
    run_name: str,
) -> dict[str, Any]:
    """Return an isolated config for one channel-model and seed run."""
    config = copy.deepcopy(base_config)
    config["experiment"]["name"] = run_name
    config["experiment"]["channel-model"] = channel_model
    config["experiment"]["random-seed"] = seed

    training = config["training"]
    training["dataset-path"] = f"data/{run_name}-dataset.npz"
    training["checkpoint-path"] = f"results/checkpoints/{run_name}.pt"
    generation = training.get("dataset-generation")
    if isinstance(generation, dict):
        generation["random-seed"] = seed

    neural = config.get("neural")
    if isinstance(neural, dict):
        neural["checkpoint-path"] = training["checkpoint-path"]

    validate_config(config)
    return config


def summarize_estimator_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    """Summarize NMSE by channel model, SNR, and estimator across seeds."""
    grouped: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["channel-model"]),
            float(row["snr-db"]),
            str(row["estimator"]),
        )
        grouped[key].append(float(row["nmse"]))

    summary: list[dict[str, float | int | str]] = []
    for (channel_model, snr_db, estimator), values in sorted(grouped.items()):
        summary.append(
            {
                "channel-model": channel_model,
                "snr-db": snr_db,
                "estimator": estimator,
                "nmse-mean": mean(values),
                "nmse-std": stdev(values) if len(values) > 1 else 0.0,
                "num-runs": len(values),
            }
        )
    return summary


def summarize_neural_margins(
    rows: Sequence[dict[str, float | int | str]],
    *,
    baseline: str = "ls-lin",
    neural: str = "neural-cnn",
) -> list[dict[str, float | int | str]]:
    """Summarize how often the neural estimator beats a baseline."""
    grouped: dict[tuple[str, int, float], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["channel-model"]), int(row["random-seed"]), float(row["snr-db"]))
        grouped[key][str(row["estimator"])] = float(row["nmse"])

    margins: dict[tuple[str, float], list[float]] = defaultdict(list)
    wins: dict[tuple[str, float], list[int]] = defaultdict(list)
    for (channel_model, _seed, snr_db), estimators in grouped.items():
        if baseline not in estimators or neural not in estimators:
            continue
        baseline_nmse = estimators[baseline]
        neural_nmse = estimators[neural]
        key = (channel_model, snr_db)
        margins[key].append(100.0 * (baseline_nmse - neural_nmse) / baseline_nmse)
        wins[key].append(1 if neural_nmse < baseline_nmse else 0)

    summary: list[dict[str, float | int | str]] = []
    for (channel_model, snr_db), values in sorted(margins.items()):
        win_values = wins[(channel_model, snr_db)]
        summary.append(
            {
                "channel-model": channel_model,
                "snr-db": snr_db,
                "baseline": baseline,
                "neural": neural,
                "improvement-percent-mean": mean(values),
                "improvement-percent-std": stdev(values) if len(values) > 1 else 0.0,
                "neural-win-rate": mean(win_values),
                "num-runs": len(values),
            }
        )
    return summary


def write_table(rows: Sequence[dict[str, float | int | str]], path: Path) -> Path:
    """Write a sequence of mapping rows to CSV."""
    if not rows:
        raise ValueError("No rows were produced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_grid_validation_sweep(
    config_path: str | Path,
    *,
    channel_models: Iterable[str] = DEFAULT_CHANNEL_MODELS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> dict[str, Path]:
    """Train/evaluate the grid experiment over channel models and seeds."""
    base_config = load_config(config_path)
    base_name = str(base_config["experiment"]["name"])
    output = base_config["output"]

    raw_rows: list[dict[str, float | int | str]] = []
    for channel_model in channel_models:
        for seed in seeds:
            run_name = f"{base_name}-{_safe_token(channel_model)}-seed-{seed}"
            config = build_sweep_config(
                base_config,
                channel_model=channel_model,
                seed=int(seed),
                run_name=run_name,
            )
            print(f"training {run_name}")
            train_model(config)
            print(f"evaluating {run_name}")
            for row in evaluate_grid_baselines(config):
                raw_rows.append(
                    {
                        "channel-model": channel_model,
                        "random-seed": int(seed),
                        **row,
                    }
                )

    table_dir = resolve_path(base_config, output["tables-dir"])
    figure_dir = resolve_path(base_config, output["figures-dir"])
    raw_path = table_dir / f"{base_name}-sweep-v1.csv"
    summary_path = table_dir / f"{base_name}-sweep-summary-v1.csv"
    margins_path = table_dir / f"{base_name}-sweep-margins-v1.csv"
    figure_path = figure_dir / f"{base_name}-sweep-v1-nmse.png"

    estimator_summary = summarize_estimator_rows(raw_rows)
    margin_summary = summarize_neural_margins(raw_rows)
    write_metrics_table(raw_rows, raw_path)
    write_table(estimator_summary, summary_path)
    write_table(margin_summary, margins_path)
    save_nmse_sweep_comparison(estimator_summary, figure_path)

    return {
        "raw-table": raw_path,
        "summary-table": summary_path,
        "margins-table": margins_path,
        "figure": figure_path,
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_seed_list(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(value)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the base experiment config.")
    parser.add_argument(
        "--channel-models",
        default=",".join(DEFAULT_CHANNEL_MODELS),
        help="Comma-separated channel models to sweep.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds to sweep.",
    )
    args = parser.parse_args(argv)

    outputs = run_grid_validation_sweep(
        args.config,
        channel_models=_parse_csv_list(args.channel_models),
        seeds=_parse_seed_list(args.seeds),
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
