"""Generate sensitivity analysis plots and tables from experiment results.

Reads a sensitivity manifest JSON produced by ``run_sensitivity.py``
(or auto-discovers result directories by tag pattern) and outputs:

- One PNG plot per (parameter × metric) combination
- One Markdown table file per parameter (covering all metrics)
- A combined ``sensitivity_report.md`` summary

Output is written to ``results/sensitivity_analysis/`` by default.

Usage
-----
    # Standard: read a manifest written by run_sensitivity.py
    python scripts/analyze_sensitivity.py \\
        --manifest results/sensitivity_manifest.json

    # Specify custom output directory
    python scripts/analyze_sensitivity.py \\
        --manifest results/sensitivity_manifest.json \\
        --output results/my_sensitivity_analysis

    # Only analyse specific parameters or metrics
    python scripts/analyze_sensitivity.py \\
        --manifest results/sensitivity_manifest.json \\
        --params lp memory_H \\
        --metrics hv igd
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # experiments/ for mosade_experiments

from mosade_experiments.analysis.sensitivity import (  # noqa: E402
    format_sensitivity_table,
    plot_sensitivity,
)

# ---------------------------------------------------------------------------
# Sweep specification (must match generate_sensitivity_configs.py)
# ---------------------------------------------------------------------------

SWEEP_VALUES: dict[str, list[int | float]] = {
    "lp":           [20, 50, 100, 200],
    "memory_H":     [3, 5, 10, 20],
    "T_base_ratio": [0.05, 0.10, 0.20, 0.30],
    "pi_min":       [0.01, 0.05, 0.10, 0.15],
}

DEFAULT_METRICS = ["hv", "igd", "spread"]


def _value_str(v: int | float) -> str:
    """Convert a parameter value to the filesystem-safe string used in filenames."""
    if isinstance(v, float):
        return str(v).replace(".", "p")
    return str(v)


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    """Load the JSON manifest produced by run_sensitivity.py.

    Returns ``{param_name: {value_str: result_dir_str}}``.
    """
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _manifest_to_ordered(
    manifest: dict[str, dict[str, str]],
    params: list[str] | None = None,
) -> dict[str, tuple[list, list[Path]]]:
    """Convert the manifest to ordered (param_values, result_dirs) pairs.

    Returns ``{param_name: (param_values, result_dirs)}``.
    Ordering follows SWEEP_VALUES to ensure correct x-axis order.
    """
    result = {}
    for param_name, value_map in manifest.items():
        if params is not None and param_name not in params:
            continue
        if param_name not in SWEEP_VALUES:
            continue
        # Order by the canonical sweep order, skipping missing values.
        ordered_values = []
        ordered_dirs = []
        for v in SWEEP_VALUES[param_name]:
            vstr = _value_str(v)
            if vstr in value_map:
                ordered_values.append(v)
                ordered_dirs.append(Path(value_map[vstr]))
        result[param_name] = (ordered_values, ordered_dirs)
    return result


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def analyze_sensitivity(
    manifest_path: Path,
    output_dir: Path,
    params: list[str] | None = None,
    metrics: list[str] | None = None,
    algo_name: str = "MOSADE",
) -> None:
    """Run all sensitivity analysis for the given manifest.

    Parameters
    ----------
    manifest_path : Path
        JSON manifest from run_sensitivity.py.
    output_dir : Path
        Root directory for output files.
    params : list of str or None
        Parameters to analyse; all manifest params when None.
    metrics : list of str or None
        Metrics to analyse; defaults to hv, igd, spread.
    algo_name : str
        Algorithm name to look up in each result directory.
    """
    metrics = metrics or DEFAULT_METRICS
    manifest = _load_manifest(manifest_path)
    ordered = _manifest_to_ordered(manifest, params)

    if not ordered:
        print("No parameters to analyse (check manifest and --params filter).")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_sections: list[str] = [
        "# MOSADE Parameter Sensitivity Analysis\n",
        f"Manifest: `{manifest_path}`  \n",
        f"Algorithm: `{algo_name}`  \n",
        f"Metrics: {', '.join(metrics)}\n",
        "---\n",
    ]

    for param_name, (param_values, result_dirs) in ordered.items():
        print(f"\n=== Parameter: {param_name} ===")
        print(f"  Values: {param_values}")
        print(f"  Dirs:   {[str(d) for d in result_dirs]}")

        param_section = [f"\n## Parameter: `{param_name}`\n"]

        for metric in metrics:
            print(f"  Metric: {metric}")

            # Plot
            plot_path = output_dir / f"{param_name}_{metric}.png"
            try:
                plot_sensitivity(
                    result_dirs=result_dirs,
                    param_name=param_name,
                    param_values=param_values,
                    metric=metric,
                    algo_name=algo_name,
                    save_path=plot_path,
                )
                print(f"    Plot saved: {plot_path}")
            except Exception as exc:
                print(f"    WARNING: plot failed ({exc})")

            # Table
            try:
                table_md = format_sensitivity_table(
                    result_dirs=result_dirs,
                    param_name=param_name,
                    param_values=param_values,
                    metric=metric,
                    algo_name=algo_name,
                )
                table_path = output_dir / f"{param_name}_{metric}_table.md"
                table_path.write_text(table_md, encoding="utf-8")
                print(f"    Table saved: {table_path}")

                param_section.append(f"\n### {metric.upper()}\n\n")
                param_section.append(f"![{param_name} {metric}]({plot_path.name})\n\n")
                param_section.append(table_md + "\n")
            except Exception as exc:
                print(f"    WARNING: table failed ({exc})")

        report_sections.extend(param_section)

    # Write combined report
    report_path = output_dir / "sensitivity_report.md"
    report_path.write_text("".join(report_sections), encoding="utf-8")
    print(f"\nCombined report: {report_path}")
    print(f"All outputs in: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate sensitivity analysis plots and tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/analyze_sensitivity.py --manifest results/sensitivity_manifest.json
              python scripts/analyze_sensitivity.py --manifest results/sensitivity_manifest.json --params lp
              python scripts/analyze_sensitivity.py --manifest results/sensitivity_manifest.json --metrics hv igd
        """),
    )
    parser.add_argument(
        "--manifest", "-m",
        required=True,
        help="Path to the sensitivity_manifest.json produced by run_sensitivity.py",
    )
    parser.add_argument(
        "--output", "-o",
        default="results/sensitivity_analysis",
        help="Output directory (default: results/sensitivity_analysis)",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        choices=list(SWEEP_VALUES.keys()),
        default=None,
        help="Parameters to analyse (default: all in manifest)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to analyse (default: hv igd spread)",
    )
    parser.add_argument(
        "--algo",
        default="MOSADE",
        help="Algorithm name to look up in result directories (default: MOSADE)",
    )
    args = parser.parse_args()

    analyze_sensitivity(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output),
        params=args.params,
        metrics=args.metrics,
        algo_name=args.algo,
    )


if __name__ == "__main__":
    main()
