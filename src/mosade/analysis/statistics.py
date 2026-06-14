"""Statistical comparison utilities for multi-algorithm benchmarking.

Provides:
- Pairwise Wilcoxon rank-sum tests
- Holm-Bonferroni correction for multiple comparisons
- Median/IQR summary statistics
- Aggregated comparison tables (markdown and LaTeX)

Requires scipy (install via: pip install mosade[analysis]).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _get_scipy_stats():
    """Lazy-import scipy.stats to avoid hard dependency."""
    try:
        from scipy import stats
        return stats
    except ImportError:
        raise ImportError(
            "scipy is required for statistical tests. "
            "Install it with: pip install mosade[analysis]"
        )


def median_iqr(values: np.ndarray | list[float]) -> tuple[float, float]:
    """Return (median, IQR) for finite numeric values."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    q25, q75 = np.percentile(arr, [25, 75])
    return float(np.median(arr)), float(q75 - q25)


# ======================================================================
# Pairwise Wilcoxon rank-sum test
# ======================================================================

def wilcoxon_ranksum(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    alternative: str = "two-sided",
) -> tuple[float, float]:
    """Wilcoxon rank-sum (Mann-Whitney U) test.

    Parameters
    ----------
    a, b : arrays of metric values from two algorithms.
    alternative : 'two-sided', 'less', or 'greater'.

    Returns
    -------
    statistic, p_value
    """
    stats = _get_scipy_stats()
    result = stats.ranksums(a, b, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


# ======================================================================
# Holm-Bonferroni correction
# ======================================================================

def holm_bonferroni_adjust(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Apply Holm-Bonferroni correction and return reject flags + adjusted p.

    Parameters
    ----------
    p_values : list of raw p-values from individual tests.
    alpha : family-wise error rate.

    Returns
    -------
    (reject, p_adjusted)
        ``reject[i]`` is True when the null hypothesis is rejected after Holm
        correction.  ``p_adjusted[i]`` is the monotone Holm-adjusted p-value.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m
    adjusted = [1.0] * m

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        factor = m - rank
        adj = min(1.0, factor * p)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
        if p <= alpha / factor:
            rejected[orig_idx] = True

    # Holm step-down: once we hit a non-rejection, all later hypotheses remain.
    stepdown_reject = [False] * m
    still_rejecting = True
    for rank, (orig_idx, p) in enumerate(indexed):
        factor = m - rank
        if still_rejecting and p <= alpha / factor:
            stepdown_reject[orig_idx] = True
        else:
            still_rejecting = False

    return stepdown_reject, adjusted


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Backward-compatible Holm wrapper returning only reject flags."""
    rejected, _ = holm_bonferroni_adjust(p_values, alpha)
    return rejected


# ======================================================================
# Multi-algorithm comparison table
# ======================================================================

def pairwise_comparison(
    results: dict[str, list[float]],
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compare all algorithm pairs on a single metric using Wilcoxon + Holm.

    Parameters
    ----------
    results : dict mapping algorithm name -> list of metric values across runs.
    alpha : significance level.
    higher_is_better : True for HV, False for IGD.

    Returns
    -------
    dict with keys:
        'medians': dict[name -> median]
        'iqrs': dict[name -> IQR]
        'pairwise': list of dicts with 'a', 'b', 'p_value', 'significant', 'winner'
        'rankings': dict[name -> W/T/L counts string]
    """
    cleaned = {
        name: [float(v) for v in vals if np.isfinite(float(v))]
        for name, vals in results.items()
    }
    cleaned = {name: vals for name, vals in cleaned.items() if vals}
    names = sorted(cleaned.keys())
    n = len(names)

    medians = {}
    iqrs = {}
    for name in names:
        medians[name], iqrs[name] = median_iqr(cleaned[name])

    # Pairwise tests
    pairs = []
    p_values_raw = []
    for i in range(n):
        for j in range(i + 1, n):
            _, p = wilcoxon_ranksum(cleaned[names[i]], cleaned[names[j]])
            pairs.append((names[i], names[j]))
            p_values_raw.append(p)

    # Apply Holm-Bonferroni
    significant, p_adjusted = holm_bonferroni_adjust(p_values_raw, alpha)

    # Build results
    pairwise = []
    wins = {name: 0 for name in names}
    ties = {name: 0 for name in names}
    losses = {name: 0 for name in names}

    for idx, ((a, b), p, sig, p_adj) in enumerate(
        zip(pairs, p_values_raw, significant, p_adjusted)
    ):
        winner = None
        if sig:
            med_a, med_b = medians[a], medians[b]
            if higher_is_better:
                winner = a if med_a > med_b else b
            else:
                winner = a if med_a < med_b else b
            loser = b if winner == a else a
            wins[winner] += 1
            losses[loser] += 1
        else:
            ties[a] += 1
            ties[b] += 1

        pairwise.append({
            "a": a,
            "b": b,
            "p_value": p,
            "p_adjusted": p_adj,
            "significant": sig,
            "winner": winner,
        })

    rankings = {
        name: f"{wins[name]}W / {ties[name]}T / {losses[name]}L"
        for name in names
    }

    return {
        "medians": medians,
        "iqrs": iqrs,
        "pairwise": pairwise,
        "rankings": rankings,
    }


# ======================================================================
# Formatted table output
# ======================================================================

def format_comparison_table(
    all_results: dict[str, dict[str, list[float]]],
    metric_name: str = "HV",
    higher_is_better: bool = True,
    fmt: str = "markdown",
) -> str:
    """Generate a comparison table across problems and algorithms.

    Parameters
    ----------
    all_results : dict mapping problem_name -> {algo_name -> [values]}
    metric_name : name of the metric
    higher_is_better : direction of improvement
    fmt : 'markdown' or 'latex'

    Returns
    -------
    str : formatted table
    """
    problems = sorted(all_results.keys())
    # Collect all algorithm names
    algo_names = set()
    for prob_data in all_results.values():
        algo_names.update(prob_data.keys())
    algo_names = sorted(algo_names)

    if fmt == "latex":
        return _format_latex(all_results, problems, algo_names, metric_name, higher_is_better)
    return _format_markdown(all_results, problems, algo_names, metric_name, higher_is_better)


def _format_markdown(
    all_results: dict, problems: list[str], algos: list[str],
    metric: str, higher_better: bool,
) -> str:
    """Format comparison results as a markdown table."""
    lines = []
    header = f"| Problem | " + " | ".join(algos) + " |"
    sep = "|" + "|".join(["---"] * (len(algos) + 1)) + "|"
    lines.append(f"### {metric} Comparison (median [IQR])")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for prob in problems:
        row = [prob]
        prob_data = all_results[prob]
        medians = {}
        for algo in algos:
            if algo in prob_data and prob_data[algo]:
                vals = np.array(prob_data[algo])
                med = np.median(vals)
                q25, q75 = np.percentile(vals, [25, 75])
                iqr = q75 - q25
                medians[algo] = med
                row.append(f"{med:.4f} [{iqr:.4f}]")
            else:
                row.append("—")

        # Bold the best
        if medians:
            best_algo = max(medians, key=medians.get) if higher_better else min(medians, key=medians.get)
            idx = algos.index(best_algo) + 1
            row[idx] = f"**{row[idx]}**"

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _format_latex(
    all_results: dict, problems: list[str], algos: list[str],
    metric: str, higher_better: bool,
) -> str:
    """Format comparison results as a LaTeX table."""
    n_algo = len(algos)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{metric} comparison (median [IQR])}}",
        r"\begin{tabular}{l" + "c" * n_algo + "}",
        r"\toprule",
        "Problem & " + " & ".join(algos) + r" \\",
        r"\midrule",
    ]

    for prob in problems:
        row_parts = [prob.replace("_", r"\_")]
        prob_data = all_results[prob]
        medians = {}
        cells = []
        for algo in algos:
            if algo in prob_data and prob_data[algo]:
                vals = np.array(prob_data[algo])
                med = np.median(vals)
                q25, q75 = np.percentile(vals, [25, 75])
                iqr = q75 - q25
                medians[algo] = med
                cells.append(f"{med:.4f} [{iqr:.4f}]")
            else:
                cells.append("---")

        # Bold best
        if medians:
            best_algo = max(medians, key=medians.get) if higher_better else min(medians, key=medians.get)
            idx = algos.index(best_algo)
            cells[idx] = r"\textbf{" + cells[idx] + "}"

        lines.append(" & ".join(row_parts + cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ======================================================================
# Load results from experiment directory
# ======================================================================

def load_experiment_metrics(
    results_dir: str | Path,
    metric_key: str = "hv",
) -> dict[str, list[float]]:
    """Load per-run metric values from an experiment results directory.

    Parameters
    ----------
    results_dir : path to a single experiment's results directory
    metric_key : which metric to extract ('hv', 'igd', 'igd_plus', 'spread')

    Returns
    -------
    dict mapping problem_name -> list of metric values (one per run)
    """
    results_dir = Path(results_dir)
    output: dict[str, list[float]] = {}

    for prob_dir in sorted(results_dir.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in ("plots",):
            continue
        run_dirs = sorted([d for d in prob_dir.iterdir()
                           if d.is_dir() and d.name.startswith("run_")])
        if not run_dirs:
            continue

        values = []
        for rd in run_dirs:
            mp = rd / "metrics.json"
            if mp.exists():
                with open(mp) as f:
                    m = json.load(f)
                if metric_key in m:
                    value = float(m[metric_key])
                    if np.isfinite(value):
                        values.append(value)
        if values:
            output[prob_dir.name] = values

    return output
