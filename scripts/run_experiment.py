"""Convenience entry point for running experiments.

Usage:
    python scripts/run_experiment.py                          # smoke test
    python scripts/run_experiment.py --config configs/default.yaml
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosade.runner.experiment import run_experiment


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a MOSADE experiment")
    parser.add_argument(
        "--config", "-c",
        default="configs/smoke_test.yaml",
        help="Path to YAML config file (default: configs/smoke_test.yaml)",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory. If omitted, the runner creates a timestamped run.",
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.config, run_dir=args.run_dir)
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
