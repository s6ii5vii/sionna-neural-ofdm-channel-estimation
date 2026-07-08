"""Run the Sionna OFDM resource-grid LS baseline sweep.

This experiment requires the ml stack (Sionna 2.x, which uses PyTorch) and,
realistically, a GPU. It simulates a full OFDM resource grid over a TDL-A channel
and
compares least-squares estimation with nearest and linear interpolation.
"""

from pathlib import Path

from channel_estimation.evaluate import run_evaluation


if __name__ == "__main__":
    config_path = Path(__file__).with_name("config.yaml")
    for kind, path in run_evaluation(config_path).items():
        print(f"{kind}: {path}")
