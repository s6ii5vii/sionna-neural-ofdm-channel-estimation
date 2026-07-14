"""Run the grid neural validation sweep across channel models and seeds."""

from pathlib import Path

from channel_estimation.sweep import run_grid_validation_sweep


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    for kind, path in run_grid_validation_sweep(config_path).items():
        print(f"{kind}: {path}")
