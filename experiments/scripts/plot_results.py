"""Generate plots from experiment results.

Usage:
    python scripts/plot_results.py results/20260412_174955_smoke_test
    python scripts/plot_results.py results/20260416_000000_benchmark_merged
"""

import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/ for mosade_experiments

import argparse
from mosade_experiments.analysis.plotting import plot_experiment_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots from a suite-level or merged benchmark results directory",
    )
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument(
        "--pf-problems",
        nargs="*",
        default=None,
        help="Optional subset of problem directory names for PF overlays",
    )
    parser.add_argument(
        "--pf-selection",
        default="median_igd",
        help=(
            "Representative-run rule for PF plots: "
            "median_igd, best_igd, best_hv, median_hv, seed=<n>, run_id=<n>"
        ),
    )
    parser.add_argument(
        "--pf-source",
        default="auto",
        choices=["auto", "archive", "final_population"],
        help="Formal PF source preference: archive, final_population, or auto (default)",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=None,
        help="Optional subset of algorithms for PF overlays",
    )
    parser.add_argument(
        "--max-pf-algorithms",
        type=int,
        default=6,
        help="Maximum algorithms per PF page before pagination",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optionally keep only the top-k algorithms for formal PF overlays",
    )
    parser.add_argument(
        "--include-debug-pf",
        action="store_true",
        help="Also export per-algorithm debug PF plots using raw objectives.txt points",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory for plots; defaults to <results_dir>/plots",
    )
    args = parser.parse_args()
    plot_experiment_results(
        args.results_dir,
        pf_problems=args.pf_problems,
        pf_selection=args.pf_selection,
        pf_source=args.pf_source,
        algorithms=args.algorithms,
        max_pf_algorithms=args.max_pf_algorithms,
        top_k=args.top_k,
        pf_debug_all_points=args.include_debug_pf,
        output_dir=args.output_dir,
    )
    output_dir = Path(args.output_dir) if args.output_dir is not None else (Path(args.results_dir) / "plots")
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
