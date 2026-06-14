"""Minimal configuration-driven training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path
from .dataset import load_npz_dataset
from .models import build_lightweight_estimator


def train_model(config: dict[str, Any]) -> Path:
    """Train the lightweight estimator on an existing split NPZ dataset."""
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("The config needs a 'training' mapping for model training.")

    dataset_path = resolve_path(config, training["dataset-path"])
    data = load_npz_dataset(dataset_path)
    num_pilots = int(data["x_train"].shape[1])
    model = build_lightweight_estimator(
        num_pilots,
        hidden_units=int(training["hidden-units"]),
        dropout_rate=float(training.get("dropout-rate", 0.0)),
    )
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
