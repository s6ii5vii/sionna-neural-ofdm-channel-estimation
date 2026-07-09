"""Configuration-driven evaluation for implemented baselines."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import grid_ls_estimate, least_squares_estimate
from .config import load_config, resolve_path
from .dataset import complex_to_features, features_to_complex
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


def _squeeze_batch_grid(array: np.ndarray) -> np.ndarray:
    """Reduce a sionna grid tensor (already numpy) to (batch, symbols, sc)."""
    return array.reshape((array.shape[0], *array.shape[-2:]))


def _load_neural_estimator(config: dict[str, Any]) -> Any:
    """Load the trained convolutional grid estimator from the checkpoint.

    the ``neural`` block declares the architecture (``filters``, ``num-layers``)
    so the checkpoint state-dict can be loaded into a matching module. torch is
    imported lazily.
    """
    from .models import build_grid_estimator

    neural = config["neural"]
    experiment = config["experiment"]
    checkpoint = resolve_path(config, neural["checkpoint-path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Neural checkpoint not found: {checkpoint}")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - requires ml stack
        raise ImportError(
            "PyTorch is required to evaluate the neural estimator."
        ) from exc

    model = build_grid_estimator(
        int(experiment["num-ofdm-symbols"]),
        int(experiment["num-subcarriers"]),
        filters=int(neural["filters"]),
        num_layers=int(neural["num-layers"]),
    )
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, device, torch


def _neural_predict(
    torch: Any, model: Any, device: Any, received_grid_complex: np.ndarray
) -> np.ndarray:
    """Run the CNN on a batch of squeezed complex observations, return complex."""
    features = complex_to_features(received_grid_complex).astype(np.float32)
    with torch.no_grad():
        predictions = model(torch.as_tensor(features).to(device)).cpu().numpy()
    return features_to_complex(predictions)


def evaluate_grid_baselines(config: dict[str, Any]) -> list[dict[str, float | int | str]]:
    """Evaluate grid channel estimators across SNR on a full resource grid.

    for each configured SNR this simulates a batch on the ofdm resource grid and
    estimates the full-grid channel with least squares plus nearest and linear
    interpolation, reporting NMSE against the true frequency-domain channel. if
    the config includes a ``neural`` block referencing a trained checkpoint, the
    convolutional grid estimator is scored at the same SNRs so all three curves
    land on one plot.

    requires the ml stack (Sionna 2.x, which uses PyTorch); ``sionna_ofdm`` is
    imported lazily. not executed in this environment.
    """
    from .sionna_ofdm import GridSpec, simulate_grid_tensors, to_numpy

    experiment = config["experiment"]
    spec = GridSpec.from_experiment(experiment)
    num_samples = experiment["num-samples"]
    seed = experiment["random-seed"]

    neural_bundle = _load_neural_estimator(config) if "neural" in config else None

    rows: list[dict[str, float | int | str]] = []
    for snr_db in experiment["snr-db"]:
        y, h_freq, no, resource_grid = simulate_grid_tensors(
            spec, float(snr_db), batch_size=num_samples, seed=seed
        )
        h_freq_np = to_numpy(h_freq)
        for interpolation in ("nn", "lin"):
            h_hat = grid_ls_estimate(
                y, no, resource_grid, interpolation_type=interpolation
            )
            rows.append(
                {
                    "estimator": f"ls-{interpolation}",
                    "snr-db": float(snr_db),
                    "nmse": normalized_mean_squared_error(h_freq_np, h_hat),
                    "num-samples": int(num_samples),
                }
            )
        if neural_bundle is not None:
            model, device, torch_module = neural_bundle
            y_squeezed = _squeeze_batch_grid(to_numpy(y))
            h_true_squeezed = _squeeze_batch_grid(h_freq_np)
            h_hat_neural = _neural_predict(torch_module, model, device, y_squeezed)
            rows.append(
                {
                    "estimator": "neural-cnn",
                    "snr-db": float(snr_db),
                    "nmse": normalized_mean_squared_error(
                        h_true_squeezed, h_hat_neural
                    ),
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
