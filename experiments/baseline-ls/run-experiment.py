"""Run the reproducible least-squares baseline experiment."""

from pathlib import Path

from channel_estimation.evaluate import run_evaluation


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    for kind, path in run_evaluation(config_path).items():
        print(f"{kind}: {path}")
