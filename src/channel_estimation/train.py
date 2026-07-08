"""Minimal configuration-driven training entry point."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .dataset import features_to_complex, load_npz_dataset
from .metrics import normalized_mean_squared_error
from .models import build_grid_estimator, build_lightweight_estimator


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


def _resource_report(
    model: Any, x_test: Any, y_test: Any, checkpoint: Path
) -> dict[str, float]:
    """Measure test NMSE plus lightweight-deployment resource metrics."""
    predictions = model.predict(x_test, verbose=0)
    test_nmse = normalized_mean_squared_error(
        features_to_complex(np.asarray(y_test)),
        features_to_complex(np.asarray(predictions)),
    )
    # time a forward pass on the test split for a latency estimate.
    start = time.perf_counter()
    model.predict(x_test, verbose=0)
    elapsed = time.perf_counter() - start
    num_examples = int(np.asarray(x_test).shape[0])
    checkpoint = Path(checkpoint)
    if checkpoint.is_dir():
        size_bytes = sum(f.stat().st_size for f in checkpoint.rglob("*") if f.is_file())
    else:
        size_bytes = checkpoint.stat().st_size
    return {
        "test-nmse": float(test_nmse),
        "parameter-count": int(model.count_params()),
        "serialized-size-bytes": int(size_bytes),
        "latency-ms-per-example": float(1000.0 * elapsed / max(num_examples, 1)),
    }


def train_model(config: dict[str, Any]) -> Path:
    """Train the estimator on an existing split NPZ dataset and report metrics."""
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("The config needs a 'training' mapping for model training.")

    dataset_path = resolve_path(config, training["dataset-path"])
    data = load_npz_dataset(dataset_path)
    model = _build_model(data, training)
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch-size"]),
        verbose=2,
    )

    checkpoint = resolve_path(config, training["checkpoint-path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)

    report = _resource_report(model, data["x_test"], data["y_test"], checkpoint)
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
