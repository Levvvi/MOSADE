"""Statistical comparison of experiment results.

Loads metrics for all algorithms and problems from a results directory,
runs pairwise Wilcoxon rank-sum tests with Holm-Bonferroni correction,
and prints a markdown comparison table to stdout.  A LaTeX version is
saved to the results directory.

Usage:
    python scripts/compare_results.py results/<experiment_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(results_dir: Path) -> dict[str, dict[str, list[dict]]]:
    """Load per-run metrics organised as {problem: {algo: [run_metrics]}}.

    Supports the multi-algorithm directory layout:
        results_dir / <problem> / <algo> / run_NNN / metrics.json
    """
    data: dict[str, dict[str, list[dict]]] = {}
    for problem_dir in sorted(results_dir.iterdir()):
        if not problem_dir.is_dir():
            continue
        # Skip non-problem directories (e.g. plots/)
        problem_name = problem_dir.name
        if problem_name in {"plots", "__pycache__"}:
            continue
        algos: dict[str, list[dict]] = {}
        for algo_dir in sorted(problem_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            # Check if this is an algo dir (contains run_NNN subdirs)
            runs = sorted(algo_dir.glob("run_*"))
            if not runs:
                continue
            run_metrics = []
            for run_dir in runs:
                mf = run_dir / "metrics.json"
                if mf.exists():
                    with open(mf) as f:
                        run_metrics.append(json.load(f))
            if run_metrics:
                algos[algo_dir.name] = run_metrics
        if algos:
            data[problem_name] = algos
    return data


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_ranksum_pairwise(
    values: dict[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    """Run pairwise two-sided Wilcoxon rank-sum tests.

    Returns {(algo_a, algo_b): p_value} for every unordered pair.
    """
    algos = sorted(values.keys())
    p_values: dict[tuple[str, str], float] = {}
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            va, vb = values[a], values[b]
            if len(va) < 2 or len(vb) < 2:
                p_values[(a, b)] = float("nan")
                continue
            # If all values are identical and aligned in length, p=1.0
            if len(va) == len(vb) and np.allclose(va, vb):
                p_values[(a, b)] = 1.0
                continue
            _, p = stats.ranksums(va, vb)
            p_values[(a, b)] = p
    return p_values


def holm_bonferroni(p_values: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Apply Holm-Bonferroni step-down correction to a set of p-values.

    Returns adjusted p-values (clipped to [0, 1]).
    """
    pairs = list(p_values.keys())
    raw = np.array([p_values[k] for k in pairs])

    # Handle NaN: keep them as NaN in the output
    finite_mask = np.isfinite(raw)
    adjusted = np.full_like(raw, np.nan)

    if not np.any(finite_mask):
        return {k: float("nan") for k in pairs}

    finite_idx = np.where(finite_mask)[0]
    finite_raw = raw[finite_idx]
    m = len(finite_raw)

    # Sort by raw p-value
    order = np.argsort(finite_raw)
    sorted_p = finite_raw[order]

    # Holm step-down: adjusted_p[i] = max(adjusted_p[i-1], (m - i) * p[i])
    adj = np.zeros(m)
    for i in range(m):
        adj[i] = sorted_p[i] * (m - i)
    # Enforce monotonicity
    for i in range(1, m):
        adj[i] = max(adj[i], adj[i - 1])
    adj = np.clip(adj, 0.0, 1.0)

    # Un-sort
    result_finite = np.zeros(m)
    result_finite[order] = adj
    adjusted[finite_idx] = result_finite

    return {k: float(adjusted[i]) for i, k in enumerate(pairs)}


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

METRICS = ["hv", "igd", "igd_plus", "spread"]
METRIC_LABELS = {"hv": "HV", "igd": "IGD", "igd_plus": "IGD+", "spread": "Spread"}
# Higher is better for HV; lower is better for IGD, IGD+, Spread
HIGHER_IS_BETTER = {"hv": True, "igd": False, "igd_plus": False, "spread": False}


def _sig_marker(p_adj: float, threshold: float = 0.05) -> str:
    """Return significance marker for adjusted p-value."""
    if np.isnan(p_adj):
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < threshold:
        return "*"
    return ""


def build_comparison_table(
    data: dict[str, dict[str, list[dict]]],
) -> tuple[list[str], list[list[str]], dict]:
    """Build comparison table data.

    Returns (headers, rows, pairwise_results) where:
    - headers: column names
    - rows: list of row values (strings)
    - pairwise_results: nested dict of statistical test results
    """
    all_algos = sorted({a for p in data.values() for a in p.keys()})
    headers = ["Problem", "Metric"] + [f"{a} (mean +/- std)" for a in all_algos] + ["Sig."]
    rows: list[list[str]] = []
    pairwise_results: dict[str, dict[str, dict]] = {}

    for problem in sorted(data.keys()):
        algos_data = data[problem]
        pairwise_results[problem] = {}

        for metric in METRICS:
            # Collect values per algorithm
            values: dict[str, np.ndarray] = {}
            for algo in all_algos:
                if algo not in algos_data:
                    continue
                vals = [m.get(metric) for m in algos_data[algo] if m.get(metric) is not None]
                vals = [v for v in vals if np.isfinite(v)]
                if vals:
                    values[algo] = np.array(vals)

            if len(values) < 2:
                continue

            # Pairwise Wilcoxon rank-sum
            raw_p = wilcoxon_ranksum_pairwise(values)
            adj_p = holm_bonferroni(raw_p)
            pairwise_results[problem][metric] = {
                "raw_p": {f"{a} vs {b}": v for (a, b), v in raw_p.items()},
                "adj_p": {f"{a} vs {b}": v for (a, b), v in adj_p.items()},
            }

            # Find best algorithm
            hib = HIGHER_IS_BETTER.get(metric, False)
            means = {a: float(np.mean(v)) for a, v in values.items()}
            best_algo = max(means, key=means.get) if hib else min(means, key=means.get)

            # Significance summary: pairs where best is significantly different
            sig_parts = []
            for (a, b), p in adj_p.items():
                marker = _sig_marker(p)
                if marker and (a == best_algo or b == best_algo):
                    other = b if a == best_algo else a
                    sig_parts.append(f"{best_algo}>{other}{marker}" if hib
                                     else f"{best_algo}<{other}{marker}")

            # Build row
            row = [problem, METRIC_LABELS.get(metric, metric)]
            for algo in all_algos:
                if algo in values:
                    m = np.mean(values[algo])
                    s = np.std(values[algo])
                    cell = f"{m:.4f} +/- {s:.4f}"
                    if algo == best_algo:
                        cell = f"**{cell}**"
                    row.append(cell)
                else:
                    row.append("-")
            row.append("; ".join(sig_parts) if sig_parts else "-")
            rows.append(row)

    return headers, rows, pairwise_results


def format_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Format comparison table as markdown."""
    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    lines = [fmt_row(headers)]
    lines.append("| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |")
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def format_latex(
    headers: list[str],
    rows: list[list[str]],
    data: dict[str, dict[str, list[dict]]],
) -> str:
    """Format comparison table as LaTeX."""
    all_algos = sorted({a for p in data.values() for a in p.keys()})
    n_algo = len(all_algos)
    col_spec = "ll" + "c" * n_algo + "l"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Statistical comparison of algorithms (Wilcoxon rank-sum with "
        r"Holm-Bonferroni correction). Best mean is \textbf{bold}. "
        r"Significance: *$p<0.05$, **$p<0.01$, ***$p<0.001$.}",
        r"\label{tab:comparison}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Header row
    header_cells = ["Problem", "Metric"] + all_algos + ["Sig."]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    prev_problem = None
    for row in rows:
        problem = row[0]
        metric = row[1]
        # Add midrule between problems
        if prev_problem is not None and problem != prev_problem:
            lines.append(r"\midrule")
        prev_problem = problem

        # Convert markdown bold to LaTeX bold
        latex_cells = [problem, metric]
        for cell in row[2:]:
            cell = cell.replace("**", "")
            # Re-bold the best: find which cells had ** markers
            latex_cells.append(cell)

        # Re-apply bold to best cells
        latex_row = []
        for i, orig_cell in enumerate(row):
            if orig_cell.startswith("**") and orig_cell.endswith("**"):
                content = orig_cell.strip("*").replace("+/-", r"$\pm$")
                latex_row.append(r"\textbf{" + content + "}")
            else:
                latex_row.append(orig_cell.replace("+/-", r"$\pm$"))

        lines.append(" & ".join(latex_row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statistical comparison of experiment results"
    )
    parser.add_argument("results_dir", help="Path to experiment results directory")
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load data
    data = load_metrics(results_dir)
    if not data:
        print("Error: No metrics found in the results directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded results: {len(data)} problems, "
          f"algorithms: {sorted({a for p in data.values() for a in p.keys()})}\n")

    # Build and print markdown table
    headers, rows, pairwise = build_comparison_table(data)
    md_table = format_markdown(headers, rows)
    print(md_table)
    print()

    # Print pairwise p-values detail
    print("## Pairwise adjusted p-values (Holm-Bonferroni)")
    print()
    for problem in sorted(pairwise.keys()):
        for metric in pairwise[problem]:
            print(f"**{problem} / {METRIC_LABELS.get(metric, metric)}**:")
            for pair, p in pairwise[problem][metric]["adj_p"].items():
                marker = _sig_marker(p)
                print(f"  {pair}: p_adj = {p:.4f} {marker}")
            print()

    # Save LaTeX table
    latex_table = format_latex(headers, rows, data)
    latex_path = results_dir / "comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    # Also save pairwise results as JSON
    pairwise_path = results_dir / "pairwise_tests.json"
    with open(pairwise_path, "w") as f:
        json.dump(pairwise, f, indent=2, default=str)
    print(f"Pairwise test results saved to: {pairwise_path}")


if __name__ == "__main__":
    main()
