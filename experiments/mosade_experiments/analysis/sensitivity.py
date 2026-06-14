"""Parameter sensitivity analysis for MOSADE experiments.

Provides tools for loading, visualising, and tabulating the effect of
individual MOSADE hyper-parameters on solution quality metrics.

Requires matplotlib (``pip install mosade[analysis]``) for plotting.

Typical workflow::

    from mosade_experiments.analysis.sensitivity import plot_sensitivity, format_sensitivity_table

    # result_dirs[i] is the experiment output directory for param_value[i]
    result_dirs = [
        "results/20260414_120000_sens_lp_20",
        "results/20260414_120500_sens_lp_50",
        "results/20260414_121000_sens_lp_100",
        "results/20260414_121500_sens_lp_200",
    ]
    param_values = [20, 50, 100, 200]

    plot_sensitivity(result_dirs, "lp", param_values, metric="hv",
                     save_path="results/sensitivity_analysis/lp_hv.png")
    print(format_sensitivity_table(result_dirs, "lp", param_values, metric="hv"))
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Metrics where *higher* is better (used for bold-best logic in tables).
_HIGHER_IS_BETTER = {"hv", "spread", "feasibility_ratio"}

#: Summary key suffixes to look up in per-algorithm summary.json files.
_MEDIAN_SUFFIX = "_median"
_Q25_SUFFIX = "_mean"   # fallback if percentile keys absent
_IQR_SUFFIX = "_iqr"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_plt():
    """Lazy-import matplotlib to avoid making it a hard dependency."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for sensitivity plots. "
            "Install it with: pip install mosade[analysis]"
        )


def _finish(plt, save_path: str | Path | None, dpi: int = 150) -> None:
    """Save the current figure or display it interactively."""
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def _discover_problems(result_dir: Path, algo_name: str) -> list[str]:
    """Return sorted list of problem directory names that have algo_name results.

    Scans *result_dir* for subdirectories containing
    ``{algo_name}/summary.json``.
    """
    problems = []
    for child in sorted(result_dir.iterdir()):
        if child.is_dir() and (child / algo_name / "summary.json").exists():
            problems.append(child.name)
    return problems


def _load_summary(
    result_dir: Path,
    problem_dir_name: str,
    algo_name: str,
) -> dict[str, Any]:
    """Load ``{result_dir}/{problem_dir_name}/{algo_name}/summary.json``."""
    path = result_dir / problem_dir_name / algo_name / "summary.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_metric(summary: dict, metric: str) -> tuple[float, float, float]:
    """Extract (median, q25, q75) from a summary dict for *metric*.

    Falls back gracefully when IQR or percentile keys are absent.

    Returns
    -------
    median, q25, q75 : float
        All NaN when the metric is not present in *summary*.
    """
    nan = float("nan")
    median = summary.get(f"{metric}{_MEDIAN_SUFFIX}", nan)
    iqr = summary.get(f"{metric}{_IQR_SUFFIX}", nan)
    if not math.isnan(median) and not math.isnan(iqr):
        q25 = median - iqr / 2.0
        q75 = median + iqr / 2.0
    else:
        q25 = q75 = nan
    return float(median), float(q25), float(q75)


# ---------------------------------------------------------------------------
# Public API: data loading
# ---------------------------------------------------------------------------

def load_sensitivity_results(
    result_dirs: list[str | Path],
    param_values: list[Any],
    algo_name: str = "MOSADE",
    metric: str = "hv",
    problems: list[str] | None = None,
) -> dict[str, dict[Any, dict[str, float]]]:
    """Load sensitivity results from a list of experiment directories.

    Parameters
    ----------
    result_dirs : list of str or Path
        One directory per parameter value, in the same order as
        *param_values*.  Each directory is the root of a single experiment
        run (i.e. the directory returned by ``run_experiment()``).
    param_values : list
        Parameter values corresponding to each directory.
    algo_name : str
        Name of the algorithm sub-directory to read (default ``"MOSADE"``).
    metric : str
        Metric name (e.g. ``"hv"``, ``"igd"``, ``"spread"``).
    problems : list of str or None
        Problem directory names to include.  When ``None``, problems are
        discovered automatically from the first result directory.

    Returns
    -------
    dict
        ``{problem_dir_name: {param_value: {"median": float, "q25": float,
        "q75": float}}}``.
    """
    if len(result_dirs) != len(param_values):
        raise ValueError(
            f"result_dirs ({len(result_dirs)}) and param_values "
            f"({len(param_values)}) must have the same length."
        )

    dirs = [Path(d) for d in result_dirs]

    # Discover problem list from the first valid directory.
    if problems is None:
        for d in dirs:
            if d.exists():
                problems = _discover_problems(d, algo_name)
                break
        if not problems:
            raise ValueError(
                "Could not discover problems: no valid result directory found."
            )

    data: dict[str, dict[Any, dict[str, float]]] = {p: {} for p in problems}

    for result_dir, pvalue in zip(dirs, param_values):
        if not result_dir.exists():
            for p in problems:
                data[p][pvalue] = {"median": float("nan"),
                                   "q25": float("nan"),
                                   "q75": float("nan")}
            continue

        for prob_name in problems:
            summary = _load_summary(result_dir, prob_name, algo_name)
            median, q25, q75 = _extract_metric(summary, metric)
            data[prob_name][pvalue] = {
                "median": median,
                "q25": q25,
                "q75": q75,
            }

    return data


# ---------------------------------------------------------------------------
# Public API: plotting
# ---------------------------------------------------------------------------

def plot_sensitivity(
    result_dirs: list[str | Path],
    param_name: str,
    param_values: list[Any],
    metric: str = "hv",
    algo_name: str = "MOSADE",
    problems: list[str] | None = None,
    problem_labels: dict[str, str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple[float, float] | None = None,
    ncols: int = 3,
) -> None:
    """Plot sensitivity of *metric* to *param_name* across problems.

    Produces a grid of subplots — one per problem — each showing a line
    chart (x-axis = parameter value, y-axis = median metric) with shaded
    IQR bands.

    Parameters
    ----------
    result_dirs : list of str or Path
        One directory per parameter value (in same order as *param_values*).
    param_name : str
        Parameter being swept (used for axis/title labels).
    param_values : list
        Parameter values corresponding to each directory.
    metric : str
        Metric to plot (default ``"hv"``).
    algo_name : str
        Algorithm sub-directory name (default ``"MOSADE"``).
    problems : list of str or None
        Problem dir names to include; auto-discovered when None.
    problem_labels : dict or None
        Optional human-readable labels: ``{dir_name: display_name}``.
    save_path : str or Path or None
        Save figure to this path if given; otherwise display interactively.
    figsize : (width, height) or None
        Override figure size.
    ncols : int
        Number of subplot columns (default 3).
    """
    plt = _get_plt()

    data = load_sensitivity_results(
        result_dirs, param_values, algo_name=algo_name, metric=metric,
        problems=problems,
    )
    problem_names = list(data.keys())
    n_problems = len(problem_names)

    if n_problems == 0:
        raise ValueError("No problems found in the provided result directories.")

    nrows = math.ceil(n_problems / ncols)
    if figsize is None:
        figsize = (ncols * 4.5, nrows * 3.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f"Sensitivity of {metric.upper()} to {param_name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    x = list(param_values)
    x_numeric = np.array([float(v) for v in x])

    for idx, prob_name in enumerate(problem_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        prob_data = data[prob_name]
        medians = np.array([prob_data[v]["median"] for v in param_values], dtype=float)
        q25s    = np.array([prob_data[v]["q25"]    for v in param_values], dtype=float)
        q75s    = np.array([prob_data[v]["q75"]    for v in param_values], dtype=float)

        label = (problem_labels or {}).get(prob_name, prob_name)

        ax.plot(x_numeric, medians, marker="o", color="tab:blue",
                linewidth=1.8, markersize=5, zorder=3)
        # Shade IQR band
        valid = ~(np.isnan(q25s) | np.isnan(q75s))
        if valid.any():
            ax.fill_between(
                x_numeric[valid], q25s[valid], q75s[valid],
                alpha=0.20, color="tab:blue", zorder=2,
            )
        # Mark default value with a vertical dashed line if it's in the sweep
        ax.set_title(label, fontsize=10, pad=3)
        ax.set_xlabel(param_name, fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=9)
        ax.set_xticks(x_numeric)
        ax.set_xticklabels([str(v) for v in param_values], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_problems, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    _finish(plt, save_path)


# ---------------------------------------------------------------------------
# Public API: markdown table
# ---------------------------------------------------------------------------

def format_sensitivity_table(
    result_dirs: list[str | Path],
    param_name: str,
    param_values: list[Any],
    metric: str = "hv",
    algo_name: str = "MOSADE",
    problems: list[str] | None = None,
    problem_labels: dict[str, str] | None = None,
    higher_is_better: bool | None = None,
    precision: int = 4,
) -> str:
    """Format a Markdown table of sensitivity results.

    Rows represent problems; columns represent parameter values.
    Each cell shows ``median±IQR/2`` with the best value per row **bolded**.

    Parameters
    ----------
    result_dirs : list of str or Path
        One directory per parameter value.
    param_name : str
        Parameter name (used in column header).
    param_values : list
        Parameter values in column order.
    metric : str
        Metric to report (default ``"hv"``).
    algo_name : str
        Algorithm sub-directory name (default ``"MOSADE"``).
    problems : list of str or None
        Problem dir names; auto-discovered when None.
    problem_labels : dict or None
        Optional ``{dir_name: display_name}`` mapping.
    higher_is_better : bool or None
        Direction of optimality; auto-detected from *metric* when None.
    precision : int
        Decimal places for metric values (default 4).

    Returns
    -------
    str
        A Markdown-formatted table.
    """
    data = load_sensitivity_results(
        result_dirs, param_values, algo_name=algo_name, metric=metric,
        problems=problems,
    )
    problem_names = list(data.keys())

    if higher_is_better is None:
        higher_is_better = metric.lower() in _HIGHER_IS_BETTER

    fmt = f".{precision}f"

    # --- Header ---
    col_headers = [f"{param_name}={v}" for v in param_values]
    header = "| Problem | " + " | ".join(col_headers) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(param_values)) + " |"
    rows = [header, sep]

    for prob_name in problem_names:
        prob_data = data[prob_name]
        medians = [prob_data[v]["median"] for v in param_values]
        iqrs    = [abs(prob_data[v]["q75"] - prob_data[v]["q25"]) for v in param_values]

        # Find best column (ignoring NaN)
        valid_medians = [
            (i, m) for i, m in enumerate(medians) if not math.isnan(m)
        ]
        if valid_medians:
            best_idx = max(valid_medians, key=lambda t: t[1])[0] if higher_is_better \
                       else min(valid_medians, key=lambda t: t[1])[0]
        else:
            best_idx = -1

        # Build cell strings
        cells = []
        for i, (med, iqr) in enumerate(zip(medians, iqrs)):
            if math.isnan(med):
                cell = "—"
            else:
                half_iqr = iqr / 2.0
                cell = f"{med:{fmt}} ±{half_iqr:{fmt}}"
            if i == best_idx:
                cell = f"**{cell}**"
            cells.append(cell)

        label = (problem_labels or {}).get(prob_name, prob_name)
        rows.append("| " + label + " | " + " | ".join(cells) + " |")

    return "\n".join(rows)
