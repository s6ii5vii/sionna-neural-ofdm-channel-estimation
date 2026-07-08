"""Configuration-driven evaluation for implemented baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import grid_ls_estimate, least_squares_estimate
from .config import load_config, resolve_path
from .metrics import normalized_mean_squared_error
from .ofdm import (
    generate_pilot_symbols,
    generate_rayleigh_channel,
    pilot_indices,
    simulate_pilot_observations,
)
from .plotting import save_nmse_comparison, save_nmse_vs_snr


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


def evaluate_grid_baselines(config: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Evaluate Sionna grid LS baselines across SNR on a full resource grid.

    for each configured SNR this simulates a batch on the ofdm resource grid and
    estimates the full-grid channel with least squares plus nearest and linear
    interpolation, reporting NMSE against the true frequency-domain channel.

    requires the ml stack (TensorFlow + Sionna); ``sionna_ofdm`` is imported
    lazily. not executed in this environment.
    """
    from .sionna_ofdm import GridSpec, simulate_grid_tensors

    experiment = config["experiment"]
    spec = GridSpec.from_experiment(experiment)
    num_samples = experiment["num-samples"]
    seed = experiment["random-seed"]

    rows: list[dict[str, float | int | str]] = []
    for snr_db in experiment["snr-db"]:
        y, h_freq, no, resource_grid = simulate_grid_tensors(
            spec, float(snr_db), batch_size=num_samples, seed=seed
        )
        h_freq_np = np.asarray(h_freq)
        for interpolation in ("nn", "lin"):
            h_hat = grid_ls_estimate(
                y, no, resource_grid, interpolation_type=interpolation
            )
            rows.append(
                {
                    "estimator": f"ls-{interpolation}",
                    "snr-db": float(snr_db),
                    "nmse": normalized_mean_squared_error(h_freq_np, np.asarray(h_hat)),
                    "num-samples": int(num_samples),
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
    """Run the configured experiment and persist its table and figure.

    the flat pilot schema runs the NumPy least-squares sweep; the grid schema
    (selected by a ``num-subcarriers`` experiment key) runs the Sionna grid
    baselines and plots one NMSE curve per estimator.
    """
    config = load_config(config_path)
    output = config["output"]
    name = str(config["experiment"]["name"])
    table_path = resolve_path(config, output["tables-dir"]) / f"{name}.csv"
    figure_path = resolve_path(config, output["figures-dir"]) / f"{name}-nmse.png"

    if "num-subcarriers" in config["experiment"]:
        rows = evaluate_grid_baselines(config)
        write_metrics_table(rows, table_path)
        series: dict[str, tuple[list[float], list[float]]] = {}
        for row in rows:
            label = str(row["estimator"])
            snr_list, nmse_list = series.setdefault(label, ([], []))
            snr_list.append(float(row["snr-db"]))
            nmse_list.append(float(row["nmse"]))
        save_nmse_comparison(series, figure_path)
        return {"table": table_path, "figure": figure_path}

    rows = evaluate_ls(config)
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
