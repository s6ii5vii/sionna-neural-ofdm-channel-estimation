"""Configuration-driven training entry point (PyTorch).

trains either the flat MLP or the convolutional grid estimator, selected
automatically from the rank of the dataset, then writes a report next to the
checkpoint recording test NMSE and lightweight-deployment resource metrics
(parameter count, serialized size, and per-example latency).

torch is imported lazily so the classical utilities remain usable without it.
this module has not been executed in this environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .dataset import (
    DATASET_METADATA_KEY,
    dataset_metadata,
    features_to_complex,
    generate_grid_dataset,
    grid_dataset_metadata,
    load_npz_dataset,
    save_npz_dataset,
)
from .metrics import normalized_mean_squared_error
from .models import build_grid_estimator, build_lightweight_estimator, count_parameters


def _generate_dataset_from_config(config: dict[str, Any], dataset_path: Path) -> None:
    """Generate the grid dataset described by the config to ``dataset_path``.

    only supported for the grid experiment schema (a ``num-subcarriers`` key
    under ``experiment``); the ``training.dataset-generation`` block supplies
    the number of samples, snr, and random seed.
    """
    experiment = config.get("experiment", {})
    if "num-subcarriers" not in experiment:
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Automatic generation is "
            "only supported for the grid experiment schema."
        )
    training = config["training"]
    generation = training.get("dataset-generation")
    if not generation:
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}. Add a "
            "'training.dataset-generation' block to enable automatic generation."
        )

    from .sionna_ofdm import GridSpec

    spec = GridSpec.from_experiment(experiment)
    dataset = generate_grid_dataset(
        spec,
        num_samples=int(generation["num-samples"]),
        snr_db=float(generation["snr-db"]),
        random_seed=int(generation.get("random-seed", 42)),
        input_source=str(generation.get("input-source", "received")),
    )
    dataset[DATASET_METADATA_KEY] = np.asarray(
        json.dumps(grid_dataset_metadata(config), sort_keys=True)
    )
    save_npz_dataset(dataset_path, dataset)
    print(f"generated grid dataset at {dataset_path}")


def _build_model(data: dict[str, Any], training: dict[str, Any]) -> Any:
    """Choose the flat MLP or the grid CNN based on the dataset rank."""
    rank = np.asarray(data["x_train"]).ndim
    dropout_rate = float(training.get("dropout-rate", 0.0))
    if rank == 4:
        _, num_ofdm_symbols, num_subcarriers, _ = data["x_train"].shape
        return build_grid_estimator(
            int(num_ofdm_symbols),
            int(num_subcarriers),
            filters=int(training["hidden-units"]),
            num_layers=int(training.get("num-layers", 3)),
            dropout_rate=dropout_rate,
        )
    num_pilots = int(data["x_train"].shape[1])
    return build_lightweight_estimator(
        num_pilots,
        hidden_units=int(training["hidden-units"]),
        dropout_rate=dropout_rate,
    )


def _loader(
    torch: Any,
    x: Any,
    y: Any,
    batch_size: int,
    shuffle: bool,
    *,
    generator: Any | None = None,
) -> Any:
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(np.asarray(x), dtype=torch.float32),
        torch.as_tensor(np.asarray(y), dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator
    )


def _set_training_seed(torch: Any, seed: int) -> None:
    """Seed model initialization and data loading for a reproducible run."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resource_report(
    torch: Any, model: Any, data: dict[str, Any], checkpoint: Path, device: Any
) -> dict[str, Any]:
    """Measure test NMSE plus lightweight-deployment resource metrics."""
    model.eval()
    x_test = torch.as_tensor(np.asarray(data["x_test"]), dtype=torch.float32)
    with torch.no_grad():
        predictions = model(x_test.to(device)).cpu().numpy()

    test_nmse = normalized_mean_squared_error(
        features_to_complex(np.asarray(data["y_test"])),
        features_to_complex(predictions),
    )

    benchmark_size = min(32, int(x_test.shape[0]))
    benchmark = x_test[:benchmark_size].to(device)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.no_grad():
        for _ in range(5):
            model(benchmark)
        synchronize()
        timings = []
        for _ in range(20):
            start = time.perf_counter()
            model(benchmark)
            synchronize()
            timings.append(time.perf_counter() - start)

    checkpoint = Path(checkpoint)
    size_bytes = checkpoint.stat().st_size if checkpoint.is_file() else 0
    return {
        "test-nmse": float(test_nmse),
        "parameter-count": count_parameters(model),
        "serialized-size-bytes": int(size_bytes),
        "latency-ms-per-example": float(
            1000.0 * float(np.median(timings)) / max(benchmark_size, 1)
        ),
        "latency-benchmark-batch-size": benchmark_size,
        "latency-benchmark-repetitions": len(timings),
        "device": str(device),
        "gpu-name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python-version": platform.python_version(),
        "torch-version": str(torch.__version__),
        "sionna-version": _package_version("sionna"),
        "numpy-version": str(np.__version__),
        "git-revision": _git_revision(),
    }


def train_model(config: dict[str, Any]) -> Path:
    """Train the estimator on an existing split NPZ dataset and report metrics."""
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("The config needs a 'training' mapping for model training.")

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install the 'ml' extra or use "
            "requirements.txt."
        ) from exc

    dataset_path = resolve_path(config, training["dataset-path"])
    expected_metadata = (
        grid_dataset_metadata(config)
        if training.get("dataset-generation") and "num-subcarriers" in config["experiment"]
        else None
    )
    regenerate = not dataset_path.is_file()
    if not regenerate and expected_metadata is not None:
        existing = load_npz_dataset(dataset_path)
        existing_metadata = dataset_metadata(existing)
        regenerate = (
            existing_metadata is None
            or existing_metadata.get("fingerprint") != expected_metadata["fingerprint"]
        )
        if regenerate:
            print(f"dataset metadata changed; regenerating {dataset_path}")
    if regenerate:
        _generate_dataset_from_config(config, dataset_path)
    data = load_npz_dataset(dataset_path)

    model_seed = int(training.get("model-seed", 42))
    _set_training_seed(torch, model_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(data, training).to(device)
    parameter_count = count_parameters(model)
    parameter_target = config["experiment"].get("model-parameter-target")
    if parameter_target is not None and parameter_count > int(parameter_target):
        raise ValueError(
            f"Model has {parameter_count} parameters, exceeding target {parameter_target}."
        )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(training.get("learning-rate", 1e-3))
    )
    loss_fn = torch.nn.MSELoss()

    batch_size = int(training["batch-size"])
    loader_generator = torch.Generator().manual_seed(model_seed)
    train_loader = _loader(
        torch,
        data["x_train"],
        data["y_train"],
        batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    val_loader = _loader(
        torch, data["x_val"], data["y_val"], batch_size, shuffle=False
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    validation_history: list[float] = []
    for epoch in range(int(training["epochs"])):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_loss += loss_fn(model(inputs), targets).item() * inputs.shape[0]
        val_loss /= max(len(val_loader.dataset), 1)
        validation_history.append(float(val_loss))
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        print(f"epoch {epoch + 1}: val_loss={val_loss:.6f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)

    checkpoint = resolve_path(config, training["checkpoint-path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)

    report = _resource_report(torch, model, data, checkpoint, device)
    report["training"] = {
        "hidden-units": int(training["hidden-units"]),
        "num-layers": int(training.get("num-layers", 3)),
        "dropout-rate": float(training.get("dropout-rate", 0.0)),
        "epochs": int(training["epochs"]),
        "batch-size": int(training["batch-size"]),
        "learning-rate": float(training.get("learning-rate", 1e-3)),
        "model-seed": model_seed,
        "best-epoch": best_epoch,
        "best-validation-loss": best_val_loss,
        "validation-loss-history": validation_history,
    }
    report_path = checkpoint.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    checkpoint = train_model(load_config(args.config))
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
