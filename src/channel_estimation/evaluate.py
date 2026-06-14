"""Configuration-driven evaluation for implemented baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import least_squares_estimate
from .config import load_config, resolve_path
from .metrics import normalized_mean_squared_error
from .ofdm import (
    generate_pilot_symbols,
    generate_rayleigh_channel,
    pilot_indices,
    simulate_pilot_observations,
)
from .plotting import save_nmse_vs_snr


def evaluate_ls(config: dict[str, Any]) -> list[dict[str, float | int]]:
    """Evaluate LS estimation across configured SNR values."""
    experiment = config["experiment"]
    rng = np.random.default_rng(experiment["random-seed"])
    num_samples = experiment["num-samples"]
    num_subcarriers = experiment["num-pilots"]
    pilot_density = experiment.get("pilot-density", 1.0)
    indices = pilot_indices(num_subcarriers, pilot_density)
    channels = generate_rayleigh_channel((num_samples, len(indices)), rng)
    pilots = generate_pilot_symbols(len(indices), batch_size=num_samples)

    rows = []
    for snr_db in experiment["snr-db"]:
        received = simulate_pilot_observations(channels, pilots, snr_db, rng)
        estimate = least_squares_estimate(received, pilots)
        rows.append(
            {
                "snr-db": float(snr_db),
                "nmse": normalized_mean_squared_error(channels, estimate),
                "num-samples": num_samples,
                "num-pilots": len(indices),
                "pilot-density": float(pilot_density),
            }
        )
    return rows


def write_metrics_table(rows: list[dict[str, float | int]], path: Path) -> Path:
    """Write experiment metrics as CSV."""
    if not rows:
        raise ValueError("No metrics were produced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_evaluation(config_path: str | Path) -> dict[str, Path]:
    """Run the configured LS experiment and persist its table and figure."""
    config = load_config(config_path)
    rows = evaluate_ls(config)
    output = config["output"]
    name = str(config["experiment"]["name"])
    table_path = resolve_path(config, output["tables-dir"]) / f"{name}.csv"
    figure_path = resolve_path(config, output["figures-dir"]) / f"{name}-nmse.png"
    write_metrics_table(rows, table_path)
    save_nmse_vs_snr(
        [row["snr-db"] for row in rows],
        [row["nmse"] for row in rows],
        figure_path,
    )
    return {"table": table_path, "figure": figure_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to an experiment YAML file.")
    args = parser.parse_args(argv)
    outputs = run_evaluation(args.config)
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
