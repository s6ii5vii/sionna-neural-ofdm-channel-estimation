"""Run the implemented LS portion of the first low-resource profile.

Neural training is deliberately separate and must be invoked through
``python -m channel_estimation.train`` after reviewing the training settings.
"""

from pathlib import Path

from channel_estimation.evaluate import run_evaluation


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    for kind, path in run_evaluation(config_path).items():
        print(f"{kind}: {path}")
