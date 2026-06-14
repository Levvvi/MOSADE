"""Merge multiple suite-level experiment directories into one benchmark directory.

Usage
-----
    python scripts/merge_benchmark_results.py \
        results/<zdt_dir> results/<dtlz3obj_dir> results/<wfg_dir> results/<dascmop_dir> \
        --tag benchmark_merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosade.analysis.merge import merge_results_dirs  # noqa: E402
from mosade.utils.io import make_run_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple suite-level MOSADE results directories.",
    )
    parser.add_argument(
        "results_dirs",
        nargs="+",
        help="Suite-level results directories to merge",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Explicit output directory for the merged benchmark",
    )
    parser.add_argument(
        "--tag",
        default="benchmark_merged",
        help="Tag used when creating a timestamped merged directory (default: benchmark_merged)",
    )
    parser.add_argument(
        "--results-base",
        default="results",
        help="Base directory used with --tag when --output-dir is omitted (default: results)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output-dir if it already exists",
    )
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = make_run_dir(args.results_base, args.tag)

    manifest = merge_results_dirs(
        args.results_dirs,
        output_dir,
        overwrite=args.overwrite,
    )

    print(f"Merged benchmark directory: {output_dir}")
    print(f"Sources merged:            {len(manifest['sources'])}")
    print(f"Problems copied:           {len(manifest['problems'])}")
    if manifest.get("algorithms"):
        print(f"Algorithms:                {', '.join(manifest['algorithms'])}")
    if manifest.get("n_runs") is not None:
        print(f"n_runs:                    {manifest['n_runs']}")
    if manifest.get("seed") is not None:
        print(f"seed:                      {manifest['seed']}")
    print()
    print("Next steps:")
    print(f"  python scripts/plot_results.py {output_dir}")
    print(f"  python scripts/statistical_analysis.py {output_dir}")


if __name__ == "__main__":
    main()
