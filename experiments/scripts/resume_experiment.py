"""Resume an existing experiment directory using the standard runner."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mosade.runner.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume an experiment in an existing results directory")
    parser.add_argument("--config", "-c", required=True, help="Path to the YAML config file")
    parser.add_argument("--run-dir", "-r", required=True, help="Existing results directory to resume")
    parser.add_argument("--workers", "-w", type=int, default=None, help="Optional worker override")
    args = parser.parse_args()

    run_dir = run_experiment(args.config, run_dir=args.run_dir, workers=args.workers)
    print(run_dir)


if __name__ == "__main__":
    main()
