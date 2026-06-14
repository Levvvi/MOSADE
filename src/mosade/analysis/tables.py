"""Result table generation for MOSADE experiments.

Produces report-ready LaTeX and Markdown tables from experiment result
directories. Five table generators cover common experiment-reporting needs:

* :func:`generate_settings_table` — experiment configuration summary
* :func:`generate_main_results_table` — median [IQR] with significance markers
* :func:`generate_ranking_table` — average Friedman ranks
* :func:`generate_constrained_table` — DASCMOP-specific results with
  feasibility ratios
* :func:`generate_convergence_speed_table` — FE to 80%/95% of final HV

All functions accept ``fmt="latex"`` or ``fmt="markdown"`` (some accept
``fmt="both"`` for paired output).

Requires scipy for statistical tests (``pip install mosade[analysis]``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


# ======================================================================
# Constants
# ======================================================================

#: Metrics where higher is better (used for bold-best and sig-marker logic).
_HIGHER_IS_BETTER: frozenset[str] = frozenset({
    "hv", "feasibility_ratio", "n_solutions",
})

#: Canonical suite ordering for row grouping.
_SUITE_ORDER = ["ZDT", "DTLZ", "WFG", "DASCMOP"]

_DEPRECATED_ALGORITHM_LABELS: dict[str, str] = {
    "MOSADE_fixed_eps": (
        "Deprecated ambiguous epsilon ablation label. Use "
        "MOSADE_fixed_eps_initial or MOSADE_fixed_eps_zero."
    ),
}


def _reject_deprecated_algorithm_label(label: str, context: str) -> None:
    """Raise if *label* is unsafe for reported tables."""
    if label in _DEPRECATED_ALGORITHM_LABELS:
        raise ValueError(
            f"{context} uses deprecated algorithm label {label!r}: "
            f"{_DEPRECATED_ALGORITHM_LABELS[label]}"
        )


# ======================================================================
# Lazy imports
# ======================================================================

def _get_scipy_stats():
    """Lazy-import scipy.stats."""
    try:
        from scipy import stats
        return stats
    except ImportError:
        raise ImportError(
            "scipy is required for statistical tests. "
            "Install it with: pip install mosade[analysis]"
        )


def _load_yaml(path: str | Path) -> dict:
    """Lazy-import yaml and load a YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required for generate_settings_table.")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ======================================================================
# Data loading
# ======================================================================

def _load_all_run_metrics(
    results_dir: str | Path,
) -> dict[str, dict[str, list[dict]]]:
    """Load per-run metrics for all problems and algorithms.

    Handles both multi-algo (``problem/algo/run_XXX/``) and single-algo
    (``problem/run_XXX/``) layouts.

    Parameters
    ----------
    results_dir : str or Path
        Root experiment directory.

    Returns
    -------
    dict
        ``{problem_dir_name: {algo_name: [run_metrics_dict, ...]}}``.
    """
    results_dir = Path(results_dir)
    data: dict[str, dict[str, list[dict]]] = {}

    _skip = {"plots", "tables"}

    for prob_dir in sorted(results_dir.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name in _skip:
            continue
        # Detect layout by inspecting children.
        children = [d for d in prob_dir.iterdir() if d.is_dir()]
        run_dirs = sorted(d for d in children if d.name.startswith("run_"))
        algo_dirs = sorted(
            d for d in children
            if not d.name.startswith("run_") and not d.name.startswith(".")
        )

        if algo_dirs:
            # Multi-algo mode: problem/algo/run_XXX/
            prob_data: dict[str, list[dict]] = {}
            for algo_dir in algo_dirs:
                _reject_deprecated_algorithm_label(
                    algo_dir.name,
                    f"{results_dir / prob_dir.name}",
                )
                runs = _read_run_dirs(algo_dir)
                if runs:
                    prob_data[algo_dir.name] = runs
            if prob_data:
                data[prob_dir.name] = prob_data
        elif run_dirs:
            # Single-algo mode: problem/run_XXX/
            runs = _read_run_dirs(prob_dir)
            if runs:
                data[prob_dir.name] = {"MOSADE": runs}

    return data


def _read_run_dirs(parent: Path) -> list[dict]:
    """Read metrics.json from all run_XXX sub-directories of *parent*."""
    runs: list[dict] = []
    for rd in sorted(parent.iterdir()):
        if rd.is_dir() and rd.name.startswith("run_"):
            mp = rd / "metrics.json"
            if mp.exists():
                with open(mp, encoding="utf-8") as f:
                    runs.append(json.load(f))
    return runs


def _get_metric_values(
    run_metrics: list[dict], key: str, *, drop_nan: bool = True,
) -> list[float]:
    """Extract a numeric metric from a list of run-metric dicts.

    Parameters
    ----------
    run_metrics : list of dict
    key : metric name
    drop_nan : if True, silently drop NaN and None values.
    """
    vals: list[float] = []
    for m in run_metrics:
        v = m.get(key)
        if v is None:
            continue
        fv = float(v)
        if drop_nan and math.isnan(fv):
            continue
        vals.append(fv)
    return vals


def _algo_order(data: dict[str, dict[str, list[dict]]]) -> list[str]:
    """Determine a stable algorithm ordering: MOSADE first, then sorted."""
    names: set[str] = set()
    for prob_data in data.values():
        names.update(prob_data.keys())
    ordered = sorted(names)
    # Promote MOSADE to first position if present.
    if "MOSADE" in ordered:
        ordered.remove("MOSADE")
        ordered.insert(0, "MOSADE")
    return ordered


# ======================================================================
# Problem grouping
# ======================================================================

def _get_suite(name: str) -> str:
    """Infer benchmark suite from a problem directory name."""
    upper = name.upper()
    for suite in _SUITE_ORDER:
        if upper.startswith(suite):
            return suite
    return "Other"


def _group_by_suite(
    names: list[str],
) -> list[tuple[str, list[str]]]:
    """Group problem names by suite, preserving order within each suite.

    Returns a list of (suite_label, [prob_names]) tuples in canonical
    suite order.
    """
    buckets: dict[str, list[str]] = {}
    for n in names:
        s = _get_suite(n)
        buckets.setdefault(s, []).append(n)
    result: list[tuple[str, list[str]]] = []
    for s in _SUITE_ORDER:
        if s in buckets:
            result.append((s, buckets.pop(s)))
    for s in sorted(buckets):
        result.append((s, buckets[s]))
    return result


# ======================================================================
# Statistics helpers
# ======================================================================

def _wilcoxon_and_holm(
    ref_vals: list[float],
    competitors: dict[str, list[float]],
    higher_is_better: bool,
    alpha: float = 0.05,
) -> dict[str, str]:
    """Wilcoxon rank-sum + Holm-Bonferroni for reference vs competitors.

    Parameters
    ----------
    ref_vals : metric values for the reference algorithm.
    competitors : ``{algo_name: [values]}`` for each competitor.
    higher_is_better : direction of optimality.
    alpha : family-wise significance level.

    Returns
    -------
    dict
        ``{algo_name: marker}`` where marker is ``"plus"`` (ref wins),
        ``"minus"`` (competitor wins), or ``"approx"`` (no sig. diff).
    """
    stats = _get_scipy_stats()
    comp_names = list(competitors.keys())
    p_values: list[float] = []
    valid_mask: list[bool] = []

    for name in comp_names:
        c_vals = competitors[name]
        if len(ref_vals) < 3 or len(c_vals) < 3:
            p_values.append(1.0)
            valid_mask.append(False)
        else:
            _, p = stats.ranksums(ref_vals, c_vals)
            p_values.append(float(p))
            valid_mask.append(True)

    # Holm-Bonferroni on the valid p-values.
    m = sum(valid_mask)
    if m > 0:
        valid_ps = [p for p, v in zip(p_values, valid_mask) if v]
        indexed = sorted(enumerate(valid_ps), key=lambda x: x[1])
        holm_rejected = [False] * m
        for rank, (orig_idx, p) in enumerate(indexed):
            if p <= alpha / (m - rank):
                holm_rejected[orig_idx] = True
            else:
                break
        # Map back to full list.
        rejected: list[bool] = []
        vi = 0
        for v in valid_mask:
            if v:
                rejected.append(holm_rejected[vi])
                vi += 1
            else:
                rejected.append(False)
    else:
        rejected = [False] * len(comp_names)

    ref_med = float(np.median(ref_vals)) if ref_vals else float("nan")
    markers: dict[str, str] = {}
    for i, name in enumerate(comp_names):
        if not rejected[i]:
            markers[name] = "approx"
        else:
            c_med = float(np.median(competitors[name]))
            if higher_is_better:
                markers[name] = "plus" if ref_med > c_med else "minus"
            else:
                markers[name] = "plus" if ref_med < c_med else "minus"
    return markers


# ======================================================================
# Formatting helpers
# ======================================================================

def _tex_escape(s: str) -> str:
    """Escape underscores for LaTeX."""
    return s.replace("_", r"\_")


_LATEX_SIG = {"plus": r"$^{+}$", "minus": r"$^{-}$", "approx": r"$^{\approx}$"}
_MD_SIG = {"plus": "\u207a", "minus": "\u207b", "approx": "\u2248"}


def _fmt_cell(
    center: float, spread: float, *, fmt: str, bold: bool = False,
    sig: str | None = None, precision: int = 4,
) -> str:
    """Format a single ``median [IQR]`` cell.

    Parameters
    ----------
    center, spread : metric statistics (median and IQR)
    fmt : ``"latex"`` or ``"markdown"``
    bold : whether to embolden
    sig : significance marker key (``"plus"``, ``"minus"``, ``"approx"``)
    precision : decimal places
    """
    if math.isnan(center):
        return "---" if fmt == "latex" else "\u2014"
    p = f".{precision}f"
    core = f"{center:{p}} [{spread:{p}}]"
    if sig:
        core += _LATEX_SIG[sig] if fmt == "latex" else _MD_SIG[sig]
    if bold:
        core = rf"\textbf{{{core}}}" if fmt == "latex" else f"**{core}**"
    return core


def _fmt_int_cell(
    value: float | None, *, fmt: str, bold: bool = False,
) -> str:
    """Format an integer metric cell (e.g. FE count)."""
    if value is None or math.isnan(value):
        return "---" if fmt == "latex" else "\u2014"
    s = str(int(round(value)))
    if bold:
        s = rf"\textbf{{{s}}}" if fmt == "latex" else f"**{s}**"
    return s


# ======================================================================
# 1. Settings table
# ======================================================================

def generate_settings_table(
    config_path: str | Path,
    fmt: str = "both",
) -> str | tuple[str, str]:
    """Generate an experiment-settings summary table.

    Parameters
    ----------
    config_path : str or Path
        YAML experiment config file.
    fmt : str
        ``"latex"``, ``"markdown"``, or ``"both"``.

    Returns
    -------
    str or (str, str)
        Formatted table(s).  ``"both"`` returns ``(latex, markdown)``.
    """
    cfg = _load_yaml(config_path)

    # Algorithm info -------------------------------------------------------
    if "algorithms" in cfg:
        algo_entries = cfg["algorithms"]
    elif "algorithm" in cfg:
        algo_entries = [{"name": "MOSADE", **cfg["algorithm"]}]
    else:
        algo_entries = [{"name": "MOSADE"}]

    n_runs = cfg.get("n_runs", 5)
    seed = cfg.get("seed", 42)

    # Problem summary ------------------------------------------------------
    problems = cfg.get("problems", [])
    prob_names: list[str] = []
    for p in problems:
        if isinstance(p, dict):
            prob_names.append(p["name"])
        else:
            prob_names.append(str(p))
    suite_counts: dict[str, int] = {}
    for n in prob_names:
        s = _get_suite(n)
        suite_counts[s] = suite_counts.get(s, 0) + 1

    # ----- LaTeX ----------------------------------------------------------
    latex_parts: list[str] = []

    # Algorithm table
    algo_cols = ["Algorithm", "pop\\_size", "max\\_evals", "Params"]
    n_cols = len(algo_cols)
    latex_parts.append(r"\begin{table}[htbp]")
    latex_parts.append(r"\centering")
    latex_parts.append(r"\caption{Experiment settings}")
    latex_parts.append(r"\label{tab:settings}")
    latex_parts.append(r"\begin{tabular}{" + "l" * n_cols + "}")
    latex_parts.append(r"\toprule")
    latex_parts.append(" & ".join(algo_cols) + r" \\")
    latex_parts.append(r"\midrule")

    for ae in algo_entries:
        ae = dict(ae)
        _reject_deprecated_algorithm_label(
            str(ae.get("name", "MOSADE")),
            f"settings table config {config_path}",
        )
        if "type" in ae:
            _reject_deprecated_algorithm_label(
                str(ae["type"]),
                f"settings table config {config_path}",
            )
        name = _tex_escape(ae.pop("name", "MOSADE"))
        pop = ae.pop("pop_size", "---")
        evals = ae.pop("max_evals", "---")
        ae.pop("type", None)
        # Remaining keys are extra params.
        extras = ", ".join(f"{_tex_escape(k)}={v}" for k, v in sorted(ae.items()))
        if not extras:
            extras = "---"
        latex_parts.append(f"{name} & {pop} & {evals} & {extras}" + r" \\")

    latex_parts.append(r"\midrule")
    suite_str = ", ".join(f"{s}: {c}" for s, c in sorted(suite_counts.items()))
    latex_parts.append(
        rf"\multicolumn{{{n_cols}}}{{l}}"
        rf"{{n\_runs = {n_runs}, seed = {seed}, "
        rf"problems = {len(problems)} ({suite_str})}}" + r" \\"
    )
    latex_parts.append(r"\bottomrule")
    latex_parts.append(r"\end{tabular}")
    latex_parts.append(r"\end{table}")
    latex_str = "\n".join(latex_parts)

    # ----- Markdown -------------------------------------------------------
    md_parts: list[str] = ["### Experiment Settings", ""]
    md_parts.append("| Algorithm | pop_size | max_evals | Params |")
    md_parts.append("| --- | --- | --- | --- |")
    for ae in algo_entries:
        ae = dict(ae)
        _reject_deprecated_algorithm_label(
            str(ae.get("name", "MOSADE")),
            f"settings table config {config_path}",
        )
        if "type" in ae:
            _reject_deprecated_algorithm_label(
                str(ae["type"]),
                f"settings table config {config_path}",
            )
        name = ae.pop("name", "MOSADE")
        pop = ae.pop("pop_size", "—")
        evals = ae.pop("max_evals", "—")
        ae.pop("type", None)
        extras = ", ".join(f"{k}={v}" for k, v in sorted(ae.items()))
        if not extras:
            extras = "—"
        md_parts.append(f"| {name} | {pop} | {evals} | {extras} |")
    md_parts.append("")
    md_parts.append(
        f"n_runs = {n_runs}, seed = {seed}, "
        f"problems = {len(problems)} ({suite_str})"
    )
    md_str = "\n".join(md_parts)

    if fmt == "both":
        return latex_str, md_str
    return latex_str if fmt == "latex" else md_str


# ======================================================================
# 2. Main results table (median [IQR], significance, W/T/L)
# ======================================================================

def generate_main_results_table(
    results_dir: str | Path,
    metric: str = "hv",
    reference_algo: str = "MOSADE",
    fmt: str = "latex",
) -> str:
    """Generate the main benchmark results table with significance markers.

    The table reports median and interquartile range (IQR) rather than mean
    and standard deviation, which is more appropriate for stochastic MOEA
    benchmarking.

    Parameters
    ----------
    results_dir : str or Path
        Experiment results directory.
    metric : str
        Metric to report (default ``"hv"``).
    reference_algo : str
        Reference algorithm for pairwise significance tests.
    fmt : str
        ``"latex"`` or ``"markdown"``.

    Returns
    -------
    str
        Formatted table.
    """
    data = _load_all_run_metrics(results_dir)
    algos = _algo_order(data)
    problems = list(data.keys())
    grouped = _group_by_suite(problems)
    higher = metric.lower() in _HIGHER_IS_BETTER

    # Pre-compute stats and significance for every problem.
    rows_info: list[dict[str, Any]] = []
    for prob in problems:
        prob_data = data[prob]
        stats_by_algo: dict[str, tuple[float, float]] = {}
        vals_by_algo: dict[str, list[float]] = {}
        for algo in algos:
            vals = _get_metric_values(prob_data.get(algo, []), metric)
            vals_by_algo[algo] = vals
            if vals:
                med = float(np.median(vals))
                q25, q75 = np.percentile(vals, [25, 75])
                stats_by_algo[algo] = (med, float(q75 - q25))
            else:
                stats_by_algo[algo] = (float("nan"), float("nan"))

        # Significance markers (reference vs each competitor).
        ref_vals = vals_by_algo.get(reference_algo, [])
        competitors = {
            a: v for a, v in vals_by_algo.items()
            if a != reference_algo and v
        }
        if ref_vals and competitors:
            sig_markers = _wilcoxon_and_holm(ref_vals, competitors, higher)
        else:
            sig_markers = {}

        # Determine best algorithm for this problem.
        valid_centers = {
            a: s[0] for a, s in stats_by_algo.items() if not math.isnan(s[0])
        }
        best_algo = (
            max(valid_centers, key=valid_centers.get) if higher
            else min(valid_centers, key=valid_centers.get)
        ) if valid_centers else None

        rows_info.append({
            "prob": prob,
            "stats": stats_by_algo,
            "sig": sig_markers,
            "best": best_algo,
        })

    # W/T/L accumulation.
    wtl: dict[str, list[int]] = {a: [0, 0, 0] for a in algos if a != reference_algo}
    for ri in rows_info:
        for a, marker in ri["sig"].items():
            if marker == "plus":
                wtl[a][0] += 1  # win for reference
            elif marker == "minus":
                wtl[a][2] += 1  # loss for reference
            else:
                wtl[a][1] += 1  # tie

    if fmt == "latex":
        return _main_table_latex(grouped, algos, rows_info, wtl, metric, reference_algo, data)
    return _main_table_md(grouped, algos, rows_info, wtl, metric, reference_algo, data)


def _main_table_latex(
    grouped: list[tuple[str, list[str]]],
    algos: list[str],
    rows_info: list[dict],
    wtl: dict[str, list[int]],
    metric: str,
    ref_algo: str,
    data: dict,
) -> str:
    n_algo = len(algos)
    prob_map = {r["prob"]: r for r in rows_info}
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{metric.upper()} results (median [IQR]). "
        rf"Reference: {_tex_escape(ref_algo)}.}}",
        rf"\label{{tab:{metric}_results}}",
        r"\begin{tabular}{l" + "c" * n_algo + "}",
        r"\toprule",
        "Problem & " + " & ".join(_tex_escape(a) for a in algos) + r" \\",
        r"\midrule",
    ]

    for suite_idx, (suite, probs) in enumerate(grouped):
        if suite_idx > 0:
            lines.append(r"\midrule")
        n_total = n_algo + 1
        lines.append(
            rf"\multicolumn{{{n_total}}}{{l}}"
            rf"{{\textit{{{suite} suite}}}}" + r" \\"
        )
        for prob in probs:
            ri = prob_map[prob]
            cells = [_tex_escape(prob)]
            for algo in algos:
                mean, std = ri["stats"][algo]
                bold = (algo == ri["best"])
                sig = ri["sig"].get(algo) if algo != ref_algo else None
                cells.append(_fmt_cell(mean, std, fmt="latex", bold=bold, sig=sig))
            lines.append(" & ".join(cells) + r" \\")

    # W/T/L row.
    lines.append(r"\midrule")
    wtl_cells = ["W / T / L"]
    for algo in algos:
        if algo == ref_algo:
            wtl_cells.append("---")
        else:
            w, t, l_ = wtl[algo]
            wtl_cells.append(f"{w} / {t} / {l_}")
    lines.append(" & ".join(wtl_cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _main_table_md(
    grouped: list[tuple[str, list[str]]],
    algos: list[str],
    rows_info: list[dict],
    wtl: dict[str, list[int]],
    metric: str,
    ref_algo: str,
    data: dict,
) -> str:
    prob_map = {r["prob"]: r for r in rows_info}
    lines: list[str] = [
        f"### {metric.upper()} Results (mean \u00b1 std, ref: {ref_algo})",
        "",
        "| Problem | " + " | ".join(algos) + " |",
        "| --- | " + " | ".join(["---"] * len(algos)) + " |",
    ]

    for suite_idx, (suite, probs) in enumerate(grouped):
        if suite_idx > 0:
            lines.append(
                "| **" + suite + "** | "
                + " | ".join([""] * len(algos)) + " |"
            )
        for prob in probs:
            ri = prob_map[prob]
            cells = [prob]
            for algo in algos:
                mean, std = ri["stats"][algo]
                bold = (algo == ri["best"])
                sig = ri["sig"].get(algo) if algo != ref_algo else None
                cells.append(
                    _fmt_cell(mean, std, fmt="markdown", bold=bold, sig=sig)
                )
            lines.append("| " + " | ".join(cells) + " |")

    # W/T/L row.
    wtl_cells = ["**W / T / L**"]
    for algo in algos:
        if algo == ref_algo:
            wtl_cells.append("\u2014")
        else:
            w, t, l_ = wtl[algo]
            wtl_cells.append(f"{w} / {t} / {l_}")
    lines.append("| " + " | ".join(wtl_cells) + " |")

    return "\n".join(lines)


# ======================================================================
# 3. Ranking table (Friedman)
# ======================================================================

def generate_ranking_table(
    results_dir: str | Path,
    metrics: list[str] | None = None,
    fmt: str = "latex",
) -> str:
    """Generate a Friedman-rank table across problems.

    Parameters
    ----------
    results_dir : str or Path
        Experiment results directory.
    metrics : list of str
        Metrics to rank on (default ``["hv", "igd"]``).
    fmt : str
        ``"latex"`` or ``"markdown"``.

    Returns
    -------
    str
        Formatted table.
    """
    if metrics is None:
        metrics = ["hv", "igd"]
    stats = _get_scipy_stats()
    data = _load_all_run_metrics(results_dir)
    algos = _algo_order(data)
    problems = list(data.keys())
    n_algo = len(algos)

    # Build rankings per (metric).
    results: dict[str, dict[str, float]] = {}  # metric -> {algo -> avg_rank}
    p_values: dict[str, float] = {}

    for metric in metrics:
        higher = metric.lower() in _HIGHER_IS_BETTER
        rank_matrix: list[list[float]] = []  # one row per problem

        for prob in problems:
            prob_data = data[prob]
            medians: list[float] = []
            for algo in algos:
                vals = _get_metric_values(prob_data.get(algo, []), metric)
                medians.append(float(np.median(vals)) if vals else float("nan"))

            arr = np.array(medians)
            supported_idx = np.where(np.isfinite(arr))[0]
            if len(supported_idx) < 2:
                continue

            supported_vals = arr[supported_idx]
            order_local = np.argsort(-supported_vals if higher else supported_vals)
            ranks = np.full(n_algo, np.nan, dtype=float)
            # Handle ties by averaging ranks.
            i = 0
            while i < len(order_local):
                j = i + 1
                while (
                    j < len(order_local)
                    and supported_vals[order_local[j]] == supported_vals[order_local[i]]
                ):
                    j += 1
                avg_rank = np.mean(np.arange(i, j)) + 1.0  # 1-based
                for k in range(i, j):
                    ranks[supported_idx[order_local[k]]] = avg_rank
                i = j
            rank_matrix.append(ranks.tolist())

        if not rank_matrix:
            results[metric] = {algo: float("nan") for algo in algos}
            p_values[metric] = float("nan")
            continue

        rank_arr = np.array(rank_matrix, dtype=float)  # (n_problems, n_algo)
        avg_ranks = np.nanmean(rank_arr, axis=0)
        results[metric] = {algos[i]: float(avg_ranks[i]) for i in range(n_algo)}

        # Friedman test.
        complete_rows = rank_arr[np.all(np.isfinite(rank_arr), axis=1)]
        if complete_rows.shape[0] >= 3 and n_algo >= 3:
            try:
                columns = [complete_rows[:, i] for i in range(n_algo)]
                _, p = stats.friedmanchisquare(*columns)
                p_values[metric] = float(p)
            except Exception:
                p_values[metric] = float("nan")
        else:
            p_values[metric] = float("nan")

    if fmt == "latex":
        return _ranking_latex(algos, metrics, results, p_values, problems)
    return _ranking_md(algos, metrics, results, p_values, problems)


def _ranking_latex(
    algos: list[str],
    metrics: list[str],
    results: dict[str, dict[str, float]],
    p_values: dict[str, float],
    problems: list[str],
) -> str:
    n_met = len(metrics)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Average Friedman ranks across "
        + str(len(problems)) + r" problems}",
        r"\label{tab:friedman_ranks}",
        r"\begin{tabular}{l" + "c" * n_met + "}",
        r"\toprule",
        "Algorithm & " + " & ".join(m.upper() + " rank" for m in metrics) + r" \\",
        r"\midrule",
    ]

    # Find best (lowest) rank per metric.
    best: dict[str, str] = {}
    for m in metrics:
        finite = {algo: rank for algo, rank in results[m].items() if not math.isnan(rank)}
        best[m] = min(finite, key=finite.get) if finite else ""

    for algo in algos:
        cells = [_tex_escape(algo)]
        for m in metrics:
            val = results[m][algo]
            s = "---" if math.isnan(val) else f"{val:.2f}"
            if algo == best[m] and not math.isnan(val):
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    p_cells = ["Friedman $p$"]
    for m in metrics:
        pv = p_values[m]
        p_cells.append(f"{pv:.2e}" if not math.isnan(pv) else "---")
    lines.append(" & ".join(p_cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _ranking_md(
    algos: list[str],
    metrics: list[str],
    results: dict[str, dict[str, float]],
    p_values: dict[str, float],
    problems: list[str],
) -> str:
    best: dict[str, str] = {}
    for m in metrics:
        finite = {algo: rank for algo, rank in results[m].items() if not math.isnan(rank)}
        best[m] = min(finite, key=finite.get) if finite else ""

    lines: list[str] = [
        f"### Average Friedman Ranks ({len(problems)} problems)",
        "",
        "| Algorithm | " + " | ".join(m.upper() + " rank" for m in metrics) + " |",
        "| --- | " + " | ".join(["---"] * len(metrics)) + " |",
    ]

    for algo in algos:
        cells = [algo]
        for m in metrics:
            val = results[m][algo]
            s = "\u2014" if math.isnan(val) else f"{val:.2f}"
            if algo == best[m] and not math.isnan(val):
                s = f"**{s}**"
            cells.append(s)
        lines.append("| " + " | ".join(cells) + " |")

    p_cells = ["Friedman p"]
    for m in metrics:
        pv = p_values[m]
        p_cells.append(f"{pv:.2e}" if not math.isnan(pv) else "\u2014")
    lines.append("| " + " | ".join(p_cells) + " |")

    return "\n".join(lines)


# ======================================================================
# 4. Constrained results table (DASCMOP only)
# ======================================================================

def generate_constrained_table(
    results_dir: str | Path,
    fmt: str = "latex",
) -> str:
    """Generate a DASCMOP-specific results table with feasibility ratios.

    Parameters
    ----------
    results_dir : str or Path
    fmt : str

    Returns
    -------
    str
    """
    data = _load_all_run_metrics(results_dir)
    algos = _algo_order(data)

    # Filter to DASCMOP problems.
    dascmop = {p: d for p, d in data.items() if p.upper().startswith("DASCMOP")}
    if not dascmop:
        return "% No DASCMOP problems found." if fmt == "latex" else \
            "_No DASCMOP problems found._"

    # Parse difficulty and sort.
    def _difficulty(name: str) -> int:
        m = re.search(r"difficulty(\d+)", name)
        return int(m.group(1)) if m else 0

    probs_sorted = sorted(dascmop.keys(), key=lambda n: (_difficulty(n), n))

    # Group by difficulty.
    diff_groups: list[tuple[int, list[str]]] = []
    current_diff = -1
    for p in probs_sorted:
        d = _difficulty(p)
        if d != current_diff:
            diff_groups.append((d, []))
            current_diff = d
        diff_groups[-1][1].append(p)

    sub_metrics = [("hv", True), ("igd", False), ("feasibility_ratio", True)]

    if fmt == "latex":
        return _constrained_latex(diff_groups, dascmop, algos, sub_metrics)
    return _constrained_md(diff_groups, dascmop, algos, sub_metrics)


def _constrained_latex(
    diff_groups: list[tuple[int, list[str]]],
    data: dict[str, dict[str, list[dict]]],
    algos: list[str],
    sub_metrics: list[tuple[str, bool]],
) -> str:
    # Columns: Problem | for each algo: HV, IGD, FR
    n_sub = len(sub_metrics)
    n_algo = len(algos)

    col_spec = "l" + "c" * (n_algo * n_sub)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Constrained problem results (median, DASCMOP suite)}",
        r"\label{tab:constrained}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    # Multi-row header.
    header1 = [""]
    for algo in algos:
        header1.append(
            rf"\multicolumn{{{n_sub}}}{{c}}{{{_tex_escape(algo)}}}"
        )
    lines.append(" & ".join(header1) + r" \\")

    header2 = ["Problem"]
    for _ in algos:
        for m, _ in sub_metrics:
            header2.append(m.upper())
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    for gi, (diff, probs) in enumerate(diff_groups):
        if gi > 0:
            lines.append(r"\midrule")
        n_total = 1 + n_algo * n_sub
        lines.append(
            rf"\multicolumn{{{n_total}}}{{l}}"
            rf"{{\textit{{Difficulty {diff}}}}}" + r" \\"
        )
        for prob in probs:
            cells = [_tex_escape(prob)]
            for algo in algos:
                runs = data[prob].get(algo, [])
                for mkey, higher in sub_metrics:
                    vals = _get_metric_values(runs, mkey)
                    if vals:
                        med = float(np.median(vals))
                        cells.append(f"{med:.4f}")
                    else:
                        cells.append("---")
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _constrained_md(
    diff_groups: list[tuple[int, list[str]]],
    data: dict[str, dict[str, list[dict]]],
    algos: list[str],
    sub_metrics: list[tuple[str, bool]],
) -> str:
    n_sub = len(sub_metrics)
    n_algo = len(algos)

    # Header.
    cols = ["Problem"]
    for algo in algos:
        for m, _ in sub_metrics:
            cols.append(f"{algo}/{m.upper()}")
    lines: list[str] = [
        "### Constrained Results (DASCMOP, median)",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]

    for gi, (diff, probs) in enumerate(diff_groups):
        lines.append(
            "| **Difficulty " + str(diff) + "** | "
            + " | ".join([""] * (n_algo * n_sub)) + " |"
        )
        for prob in probs:
            cells = [prob]
            for algo in algos:
                runs = data[prob].get(algo, [])
                for mkey, higher in sub_metrics:
                    vals = _get_metric_values(runs, mkey)
                    if vals:
                        med = float(np.median(vals))
                        cells.append(f"{med:.4f}")
                    else:
                        cells.append("\u2014")
            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ======================================================================
# 5. Convergence speed table (FE_80 / FE_95)
# ======================================================================

def generate_convergence_speed_table(
    results_dir: str | Path,
    fmt: str = "latex",
) -> str:
    """Generate a table of median evaluations to 80%/95% of final HV.

    Parameters
    ----------
    results_dir : str or Path
    fmt : str

    Returns
    -------
    str
    """
    data = _load_all_run_metrics(results_dir)
    algos = _algo_order(data)
    problems = list(data.keys())

    # Build median FE per (problem, algo, threshold).
    fe_data: dict[str, dict[str, dict[str, float | None]]] = {}
    any_data = False
    for prob in problems:
        fe_data[prob] = {}
        for algo in algos:
            runs = data[prob].get(algo, [])
            fe80 = _get_metric_values(runs, "fe_80")
            fe95 = _get_metric_values(runs, "fe_95")
            med80 = float(np.median(fe80)) if fe80 else None
            med95 = float(np.median(fe95)) if fe95 else None
            fe_data[prob][algo] = {"fe_80": med80, "fe_95": med95}
            if med80 is not None or med95 is not None:
                any_data = True

    if not any_data:
        return ("% No convergence speed data available (fe_80/fe_95 all None)."
                if fmt == "latex" else
                "_No convergence speed data (no track_interval configured)._")

    # Determine best (smallest) per problem per threshold.
    best80: dict[str, str | None] = {}
    best95: dict[str, str | None] = {}
    for prob in problems:
        vals80 = {a: fe_data[prob][a]["fe_80"] for a in algos
                  if fe_data[prob][a]["fe_80"] is not None}
        vals95 = {a: fe_data[prob][a]["fe_95"] for a in algos
                  if fe_data[prob][a]["fe_95"] is not None}
        best80[prob] = min(vals80, key=vals80.get) if vals80 else None
        best95[prob] = min(vals95, key=vals95.get) if vals95 else None

    if fmt == "latex":
        return _convergence_latex(problems, algos, fe_data, best80, best95)
    return _convergence_md(problems, algos, fe_data, best80, best95)


def _convergence_latex(
    problems: list[str],
    algos: list[str],
    fe_data: dict,
    best80: dict,
    best95: dict,
) -> str:
    n_algo = len(algos)
    col_spec = "l" + "c" * (n_algo * 2)
    lines: list[str] = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Convergence speed: median FE to 80\%/95\% of final HV}",
        r"\label{tab:convergence_speed}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]
    # Two-level header.
    h1 = [""]
    for algo in algos:
        h1.append(rf"\multicolumn{{2}}{{c}}{{{_tex_escape(algo)}}}")
    lines.append(" & ".join(h1) + r" \\")
    h2 = ["Problem"]
    for _ in algos:
        h2.extend([r"FE$_{80}$", r"FE$_{95}$"])
    lines.append(" & ".join(h2) + r" \\")
    lines.append(r"\midrule")

    for prob in problems:
        cells = [_tex_escape(prob)]
        for algo in algos:
            v80 = fe_data[prob][algo]["fe_80"]
            v95 = fe_data[prob][algo]["fe_95"]
            cells.append(_fmt_int_cell(v80, fmt="latex", bold=(algo == best80.get(prob))))
            cells.append(_fmt_int_cell(v95, fmt="latex", bold=(algo == best95.get(prob))))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _convergence_md(
    problems: list[str],
    algos: list[str],
    fe_data: dict,
    best80: dict,
    best95: dict,
) -> str:
    cols = ["Problem"]
    for algo in algos:
        cols.extend([f"{algo}/FE80", f"{algo}/FE95"])
    lines: list[str] = [
        "### Convergence Speed (median FE to 80%/95% of final HV)",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for prob in problems:
        cells = [prob]
        for algo in algos:
            v80 = fe_data[prob][algo]["fe_80"]
            v95 = fe_data[prob][algo]["fe_95"]
            cells.append(_fmt_int_cell(v80, fmt="markdown", bold=(algo == best80.get(prob))))
            cells.append(_fmt_int_cell(v95, fmt="markdown", bold=(algo == best95.get(prob))))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
