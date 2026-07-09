"""Run the head-to-head grid comparison of LS estimators and the CNN.

Assumes the checkpoint referenced by the config already exists. To create it,
first run:

    python -m channel_estimation.train experiments/grid-neural-comparison-v1/config.yaml

which auto-generates the grid dataset if missing and trains the CNN.
"""

from pathlib import Path

from channel_estimation.evaluate import run_evaluation


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    for kind, path in run_evaluation(config_path).items():
        print(f"{kind}: {path}")
