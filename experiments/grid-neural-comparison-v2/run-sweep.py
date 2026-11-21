"""Run the grid neural validation sweep across channel models and seeds."""

from pathlib import Path
import sys

from channel_estimation.sweep import main


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].startswith("-"):
        default_config = Path(__file__).with_name("config.yaml")
        sys.argv.insert(1, str(default_config))
    raise SystemExit(main())
