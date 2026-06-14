"""Plotting utilities for MOSADE experiment analysis.

Requires matplotlib (install via: pip install mosade[analysis]).

All functions accept a save_path parameter.  If provided, the figure is
saved to disk and plt.show() is NOT called.  If save_path is None, the
figure is displayed interactively.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import textwrap
from typing import Any

import numpy as np

from mosade.algorithm.selection import nondominated_mask
from mosade.metrics import hypervolume, igd
from mosade.problems import get_problem
from mosade.utils.io import save_json

ALGORITHM_DISPLAY_NAMES: dict[str, str] = {
    "MOEAD": "MOEA/D",
    "MOEAD_DE": "MOEA/D-DE",
    "NSGA2": "NSGA-II",
    "NSGA3": "NSGA-III",
    "SMSEMOA": "SMS-EMOA",
}

ALGORITHM_ORDER: list[str] = [
    "MOSADE",
    "MOSADE_single_S1",
    "MOSADE_no_credit",
    "MOSADE_no_epsilon",
    "NSGA2",
    "MOEAD",
    "MOEAD_DE",
    "NSGA3",
    "SPEA2",
    "SMSEMOA",
]

ALGORITHM_COLORS: dict[str, str] = {
    "MOSADE": "#1f77b4",
    "MOSADE_single_S1": "#6baed6",
    "MOSADE_no_credit": "#17becf",
    "MOSADE_no_epsilon": "#9edae5",
    "NSGA2": "#ff7f0e",
    "MOEAD": "#2ca02c",
    "MOEAD_DE": "#8c564b",
    "NSGA3": "#d62728",
    "SPEA2": "#9467bd",
    "SMSEMOA": "#e377c2",
}

ALGORITHM_MARKERS: dict[str, str] = {
    "MOSADE": "o",
    "MOSADE_single_S1": "X",
    "MOSADE_no_credit": "P",
    "MOSADE_no_epsilon": "*",
    "NSGA2": "s",
    "MOEAD": "^",
    "MOEAD_DE": "D",
    "NSGA3": "v",
    "SPEA2": "h",
    "SMSEMOA": ">",
}

SUITE_ORDER: list[str] = ["ZDT", "DTLZ", "WFG", "DASCMOP"]


def algorithm_display_name(name: str) -> str:
    """Return the human-facing label for an algorithm identifier."""
    return ALGORITHM_DISPLAY_NAMES.get(name, name.replace("_", " "))


def _algorithm_style(name: str) -> dict[str, str]:
    """Return the shared plotting style for an algorithm."""
    return {
        "color": ALGORITHM_COLORS.get(name, "#4c4c4c"),
        "marker": ALGORITHM_MARKERS.get(name, "o"),
    }


def _algorithm_sort_key(name: str) -> tuple[int, str]:
    """Sort algorithms by shared benchmark order, then display label."""
    try:
        index = ALGORITHM_ORDER.index(name)
    except ValueError:
        index = len(ALGORITHM_ORDER)
    return index, algorithm_display_name(name)


def _ordered_algorithm_names(names: list[str] | set[str]) -> list[str]:
    """Return algorithm names in the shared display order."""
    return sorted(names, key=_algorithm_sort_key)


def problem_display_label(problem_name: str) -> str:
    """Return a compact label for plots and heatmaps."""
    diff_match = re.match(r"^(DASCMOP\d+)_difficulty(\d+)$", problem_name)
    if diff_match:
        return f"{diff_match.group(1)}\nd={diff_match.group(2)}"

    obj_match = re.match(r"^(DTLZ\d+)_n_obj(\d+)$", problem_name)
    if obj_match:
        return f"{obj_match.group(1)}\nM={obj_match.group(2)}"

    return problem_name


def _suite_name(problem_name: str) -> str:
    """Infer the benchmark suite name from a problem identifier."""
    for suite in SUITE_ORDER:
        if problem_name.startswith(suite):
            return suite
    match = re.match(r"[A-Za-z]+", problem_name)
    return match.group(0) if match else "Other"


def _problem_sort_key(problem_name: str) -> tuple[int, str]:
    """Sort problems by suite then by original identifier."""
    suite = _suite_name(problem_name)
    try:
        suite_index = SUITE_ORDER.index(suite)
    except ValueError:
        suite_index = len(SUITE_ORDER)
    return suite_index, problem_name


def _ordered_problem_names(names: list[str] | set[str]) -> list[str]:
    """Return problem names in shared suite order."""
    return sorted(names, key=_problem_sort_key)


def _wrap_label(text: str, width: int = 12) -> str:
    """Wrap a label for crowded axis ticks without breaking tokens."""
    if "\n" in text:
        return "\n".join(_wrap_label(part, width) for part in text.splitlines())
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def _metric_display_name(metric_key: str) -> str:
    """Return a normalized metric label for plot titles and axes."""
    return {
        "hv": "HV",
        "igd": "IGD",
        "igd_plus": "IGD+",
        "gd": "GD",
        "spread": "Spread",
    }.get(metric_key.lower(), metric_key.upper())


PF_SELECTION_DEFAULT = "median_igd"
PF_OVERLAY_LIMIT = 6
PF_TOP_N = 5

FORMAL_PF_SOURCES = ("archive", "final_population_feasible_nondominated")


@dataclass(frozen=True)
class FrontCandidate:
    """One candidate objective set that may or may not qualify as a formal PF."""

    source: str
    path: Path
    decisions_path: Path | None
    points: np.ndarray | None
    is_formal: bool
    note: str | None = None


@dataclass(frozen=True)
class PFRunSelection:
    """Deterministic representative-run selection used for PF plots."""

    algorithm: str
    requested_rule: str
    resolved_rule: str
    run_dir: Path
    run_id: int
    seed: int | None
    hv: float | None
    igd: float | None
    pf_source: str | None
    source_path: Path
    raw_source_path: Path | None
    front_points: np.ndarray
    pf_points_file: str
    raw_points_file: str
    metrics_hv: float | None = None
    metrics_igd: float | None = None
    note: str | None = None


@dataclass(frozen=True)
class PFPlotData:
    """Resolved PF points and provenance for a plotted algorithm/run."""

    selection: PFRunSelection
    points: np.ndarray
    source_path: Path
    is_debug: bool


def _run_id_from_dir(run_dir: Path) -> int:
    """Extract the numeric run identifier from a ``run_XXX`` directory name."""
    match = re.search(r"(\d+)$", run_dir.name)
    return int(match.group(1)) if match else -1


def _load_matrix(path: Path) -> np.ndarray | None:
    """Load a text matrix file, normalising 1-row and empty files."""
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return np.empty((0, 0), dtype=float)
    arr = np.loadtxt(str(path))
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return np.atleast_2d(arr)


def _parse_pf_selection_rule(selection: str) -> tuple[str, int | None]:
    """Parse the PF representative-run selection rule."""
    selection = selection.strip()
    if selection in {"median_igd", "best_hv", "best_igd", "median_hv"}:
        return selection, None
    if selection.startswith("seed="):
        return "seed", int(selection.split("=", 1)[1])
    if selection.startswith("run_id="):
        raw = selection.split("=", 1)[1].replace("run_", "")
        return "run_id", int(raw)
    raise ValueError(
        "Unsupported PF selection rule. Use one of: "
        "'median_igd', 'best_igd', 'best_hv', 'median_hv', 'seed=<n>', 'run_id=<n>'."
    )


def _metric_sort_key(value: float | None, higher_is_better: bool) -> float:
    """Return a sortable scalar for metric-based PF run selection."""
    if value is None or np.isnan(value):
        return float("inf")
    return -value if higher_is_better else value


def list_algorithms(problem_dir: Path) -> list[str]:
    """Return discovered algorithms for both multi- and single-algorithm layouts."""
    names = []
    for name in _collect_algo_runs(problem_dir):
        names.append(name or "MOSADE")
    return _ordered_algorithm_names(names)


def _run_dirs(problem_dir: Path, algorithm: str) -> list[Path]:
    """Return run directories for one algorithm under either results layout."""
    algo_runs = _collect_algo_runs(problem_dir)
    if algorithm in algo_runs:
        return algo_runs[algorithm]
    if algorithm == "MOSADE" and "" in algo_runs:
        return algo_runs[""]
    return []


def _formal_front(F: np.ndarray | None, cv: np.ndarray | None) -> np.ndarray | None:
    """Return the feasible nondominated subset of a 2-D objective set."""
    arr = _sanitize_pf_points(F)
    if arr is None:
        return None
    if cv is None:
        feasible = arr
    else:
        cv_arr = np.asarray(cv, dtype=float).reshape(-1)
        if cv_arr.shape[0] != arr.shape[0]:
            return None
        feasible = arr[cv_arr <= 0.0]
    if feasible.size == 0:
        return None
    return feasible[nondominated_mask(feasible)]


def _front_cv(
    problem: Any | None,
    points: np.ndarray | None,
    decisions_path: Path | None,
) -> tuple[np.ndarray | None, str | None]:
    """Return per-point CV when it can be verified from stored decisions."""
    if points is None:
        return None, "missing points"
    if problem is None:
        return None, "problem unavailable"
    if getattr(problem, "n_constr", 0) == 0:
        return np.zeros(points.shape[0], dtype=float), None
    if decisions_path is None or not decisions_path.exists():
        return None, "missing decisions for constrained problem"

    decisions = _load_matrix(decisions_path)
    if decisions is None:
        return None, "missing decisions for constrained problem"
    decisions = np.atleast_2d(np.asarray(decisions, dtype=float))
    if decisions.shape[0] != points.shape[0]:
        return None, "decision/objective row mismatch"

    _, G = problem._evaluate(decisions)
    if G is None or G.size == 0:
        return np.zeros(points.shape[0], dtype=float), None
    return np.sum(np.maximum(0.0, G), axis=1), None


def _load_front_candidates(
    run_dir: Path,
    problem: Any | None = None,
    metrics: dict[str, Any] | None = None,
) -> list[FrontCandidate]:
    """Load all front candidates available for a run directory."""
    metrics = metrics or _load_run_metrics_json(run_dir)
    formal_name = str(metrics.get("pf_file", "pareto_approximation.txt"))
    raw_name = str(metrics.get("pf_debug_file", "objectives.txt"))
    formal_path = run_dir / formal_name
    raw_path = run_dir / raw_name
    formal_decisions = run_dir / "pareto_approximation_decisions.txt"
    raw_decisions = run_dir / "decisions.txt"
    candidates: list[FrontCandidate] = []

    if formal_path.exists():
        formal_points = _sanitize_pf_points(_load_matrix(formal_path))
        source_hint = str(metrics.get("pf_source") or "unknown")
        cv, note = _front_cv(problem, formal_points, formal_decisions if formal_decisions.exists() else None)
        is_formal = source_hint in FORMAL_PF_SOURCES
        if getattr(problem, "n_constr", 0) > 0 and cv is None:
            is_formal = False
            note = "cannot verify constrained feasibility from stored data"
        if is_formal:
            formal_points = _formal_front(formal_points, cv)
        candidates.append(
            FrontCandidate(
                source=source_hint,
                path=formal_path,
                decisions_path=formal_decisions if formal_decisions.exists() else None,
                points=formal_points,
                is_formal=is_formal and formal_points is not None,
                note=note,
            )
        )

    if raw_path.exists():
        raw_points = _sanitize_pf_points(_load_matrix(raw_path))
        candidates.append(
            FrontCandidate(
                source="raw_final_population",
                path=raw_path,
                decisions_path=raw_decisions if raw_decisions.exists() else None,
                points=raw_points,
                is_formal=False,
                note=None,
            )
        )

    return candidates


def _choose_front_source(
    candidates: list[FrontCandidate],
    source_preference: str = "auto",
    allow_debug: bool = False,
) -> FrontCandidate | None:
    """Choose the front source to use for formal or debug PF plotting."""
    if source_preference == "archive":
        preferred = ["archive"]
    elif source_preference == "final_population":
        preferred = ["final_population_feasible_nondominated"]
    else:
        preferred = list(FORMAL_PF_SOURCES)

    for source in preferred:
        for candidate in candidates:
            if (
                candidate.is_formal
                and candidate.source == source
                and candidate.points is not None
                and candidate.points.size > 0
            ):
                return candidate

    if allow_debug:
        for candidate in candidates:
            if (
                candidate.source == "raw_final_population"
                and candidate.points is not None
                and candidate.points.size > 0
            ):
                return candidate
    return None


def _recompute_pf_metrics(
    points: np.ndarray | None,
    ref_point: np.ndarray | None,
    PF: np.ndarray | None,
) -> tuple[float | None, float | None]:
    """Recompute HV/IGD for the exact point set that will be plotted."""
    if points is None or points.size == 0 or ref_point is None:
        return None, None
    hv = float(hypervolume(points, ref_point))
    igd_val = None
    if PF is not None and PF.size > 0:
        igd_val = float(igd(points, PF))
    return hv, igd_val


def select_representative_run(
    algorithm: str,
    run_dirs: list[Path],
    *,
    problem: Any | None = None,
    PF: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
    selection_rule: str = PF_SELECTION_DEFAULT,
    source_preference: str = "auto",
    allow_debug: bool = False,
) -> PFRunSelection | None:
    """Select a representative run from the exact front that will be plotted."""
    requested_mode, value = _parse_pf_selection_rule(selection_rule)
    resolved_mode = requested_mode
    note = None
    if requested_mode in {"median_igd", "best_igd"} and (PF is None or PF.size == 0):
        resolved_mode = "median_hv" if requested_mode == "median_igd" else "best_hv"
        note = f"requested {requested_mode} but no reference PF is available; used {resolved_mode}"

    available: list[PFRunSelection] = []
    for run_dir in sorted(run_dirs, key=lambda item: (_run_id_from_dir(item), item.name)):
        metrics = _load_run_metrics_json(run_dir)
        candidates = _load_front_candidates(run_dir, problem=problem, metrics=metrics)
        chosen = _choose_front_source(
            candidates,
            source_preference=source_preference,
            allow_debug=allow_debug,
        )
        if chosen is None or chosen.points is None or chosen.points.size == 0:
            continue
        hv, igd_val = _recompute_pf_metrics(chosen.points, ref_point, PF)
        metrics_hv = metrics.get("hv")
        metrics_igd = metrics.get("igd")
        selection_note = note or chosen.note
        available.append(
            PFRunSelection(
                algorithm=algorithm,
                requested_rule=selection_rule,
                resolved_rule=resolved_mode,
                run_dir=run_dir,
                run_id=_run_id_from_dir(run_dir),
                seed=int(metrics["seed"]) if metrics.get("seed") is not None else None,
                hv=hv,
                igd=igd_val,
                pf_source=chosen.source,
                source_path=chosen.path,
                raw_source_path=(run_dir / str(metrics.get("pf_debug_file", "objectives.txt"))),
                front_points=chosen.points,
                pf_points_file=chosen.path.name,
                raw_points_file=str(metrics.get("pf_debug_file", "objectives.txt")),
                metrics_hv=None if metrics_hv is None else float(metrics_hv),
                metrics_igd=None if metrics_igd is None else float(metrics_igd),
                note=selection_note,
            )
        )

    if not available:
        return None

    if resolved_mode == "seed":
        return next((item for item in available if item.seed == value), None)
    if resolved_mode == "run_id":
        return next((item for item in available if item.run_id == value), None)

    metric_key = "hv" if resolved_mode in {"best_hv", "median_hv"} else "igd"
    higher_is_better = metric_key == "hv"
    finite = [
        item for item in available
        if getattr(item, metric_key) is not None and not np.isnan(float(getattr(item, metric_key)))
    ]
    if not finite:
        return None

    finite.sort(
        key=lambda item: (
            _metric_sort_key(getattr(item, metric_key), higher_is_better),
            item.seed if item.seed is not None else float("inf"),
            item.run_id,
        )
    )
    if resolved_mode in {"median_igd", "median_hv"}:
        return finite[len(finite) // 2]
    return finite[0]


def _select_pf_run(
    algorithm: str,
    run_dirs: list[Path],
    selection_rule: str = PF_SELECTION_DEFAULT,
    *,
    problem: Any | None = None,
    PF: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
    source_preference: str = "auto",
    allow_debug: bool = False,
) -> PFRunSelection | None:
    """Backward-compatible wrapper around :func:`select_representative_run`."""
    return select_representative_run(
        algorithm,
        run_dirs,
        problem=problem,
        PF=PF,
        ref_point=ref_point,
        selection_rule=selection_rule,
        source_preference=source_preference,
        allow_debug=allow_debug,
    )


def _load_pf_points(
    selection: PFRunSelection,
    debug_all_points: bool = False,
) -> tuple[np.ndarray | None, Path]:
    """Load the formal or debug PF points for a selected run."""
    if debug_all_points:
        primary_path = selection.raw_source_path or (selection.run_dir / selection.raw_points_file)
        return _sanitize_pf_points(_load_matrix(primary_path)), primary_path
    return selection.front_points, selection.source_path


def _pf_selection_summary(selection: PFRunSelection) -> str:
    """Return a compact human-readable summary for a PF run selection."""
    hv_text = "NaN" if selection.hv is None or np.isnan(selection.hv) else f"{selection.hv:.6f}"
    igd_text = "NaN" if selection.igd is None or np.isnan(selection.igd) else f"{selection.igd:.6f}"
    seed_text = "NA" if selection.seed is None else str(selection.seed)
    source = selection.pf_source or "unknown"
    rule_text = (
        selection.resolved_rule
        if selection.resolved_rule == selection.requested_rule
        else f"{selection.requested_rule}->{selection.resolved_rule}"
    )
    note = f" | note={selection.note}" if selection.note else ""
    return (
        f"{algorithm_display_name(selection.algorithm)} | run_{selection.run_id:03d} | "
        f"seed={seed_text} | rule={rule_text} | HV={hv_text} | IGD={igd_text} | "
        f"source={source} | file={selection.source_path.name}{note}"
    )


def _pf_file_stem(
    experiment_name: str,
    problem_name: str,
    mode: str,
    selection_rule: str,
    algorithm_name: str | None = None,
) -> str:
    """Return a traceable filename stem for PF plot outputs."""
    parts = [
        "pf",
        _safe_filename_token(experiment_name),
        _safe_filename_token(problem_name),
        _safe_filename_token(mode),
        _safe_filename_token(selection_rule),
    ]
    if algorithm_name is not None:
        parts.append(_safe_filename_token(algorithm_name))
    return "_".join(part for part in parts if part)


def _pf_page_stem(
    experiment_name: str,
    problem_name: str,
    selection_rule: str,
    page_index: int,
) -> str:
    """Return the formal PF page stem."""
    return "_".join(
        [
            "pf",
            _safe_filename_token(experiment_name),
            _safe_filename_token(problem_name),
            _safe_filename_token(selection_rule),
            f"page{page_index:02d}",
        ]
    )


def _pf_debug_stem(
    experiment_name: str,
    problem_name: str,
    algorithm_name: str,
    run_id: int,
) -> str:
    """Return the per-algorithm debug PF stem."""
    return "_".join(
        [
            "pf_debug",
            _safe_filename_token(experiment_name),
            _safe_filename_token(problem_name),
            _safe_filename_token(algorithm_name),
            f"run{run_id:03d}",
        ]
    )


def _pf_manifest_path(plot_path: Path) -> Path:
    """Return the sidecar manifest path for a PF plot."""
    return plot_path.with_suffix(".json")


def _points_digest(points: np.ndarray) -> str:
    """Return a stable digest for a plotted point set."""
    arr = np.asarray(points, dtype=float)
    payload = f"{arr.shape}|".encode("utf-8") + np.ascontiguousarray(arr).tobytes()
    return hashlib.sha1(payload).hexdigest()


def _sanitize_pf_points(points: np.ndarray | None) -> np.ndarray | None:
    """Return a clean 2-D objective matrix suitable for a 2-D PF plot."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.size == 0:
        return None
    arr = np.atleast_2d(arr)
    if arr.shape[1] != 2:
        return None
    finite_mask = np.isfinite(arr).all(axis=1)
    if not np.any(finite_mask):
        return None
    return arr[finite_mask]


def _resolve_pf_plot_data(
    algorithm: str,
    run_dirs: list[Path],
    selection_rule: str,
    *,
    problem: Any | None = None,
    PF: np.ndarray | None = None,
    ref_point: np.ndarray | None = None,
    source_preference: str = "auto",
    debug_all_points: bool = False,
) -> PFPlotData | None:
    """Resolve the selected PF points for one algorithm."""
    selection = _select_pf_run(
        algorithm,
        run_dirs,
        selection_rule=selection_rule,
        problem=problem,
        PF=PF,
        ref_point=ref_point,
        source_preference=source_preference,
        allow_debug=debug_all_points,
    )
    if selection is None:
        return None
    points, source_path = _load_pf_points(selection, debug_all_points=debug_all_points)
    points = _sanitize_pf_points(points)
    if points is None:
        return None
    return PFPlotData(
        selection=selection,
        points=points,
        source_path=source_path,
        is_debug=debug_all_points,
    )


def _pf_overlay_sort_key(plot_data: PFPlotData, selection_rule: str) -> tuple[float, float, tuple[int, str]]:
    """Return a deterministic sorting key for picking the plotted top-k algorithms."""
    rule_mode, _ = _parse_pf_selection_rule(selection_rule)
    hv = plot_data.selection.hv
    igd = plot_data.selection.igd
    hv_key = float("inf") if hv is None or np.isnan(hv) else -float(hv)
    igd_key = float("inf") if igd is None or np.isnan(igd) else float(igd)
    algo_key = _algorithm_sort_key(plot_data.selection.algorithm)
    if plot_data.selection.resolved_rule in {"median_igd", "best_igd"}:
        return igd_key, hv_key, algo_key
    return hv_key, igd_key, algo_key


def _pf_manifest(
    experiment_name: str,
    problem_name: str,
    plot_mode: str,
    selection_rule: str,
    plot_items: list[PFPlotData],
    plot_path: Path,
    debug_all_points: bool,
    true_pf_path: Path | None,
) -> dict[str, Any]:
    """Build a JSON-serialisable manifest for a PF plot."""
    return {
        "experiment": experiment_name,
        "problem": problem_name,
        "plot_mode": plot_mode,
        "selection_rule": selection_rule,
        "debug_all_points": debug_all_points,
        "plot_path": str(plot_path.name),
        "true_pf_file": None if true_pf_path is None else str(true_pf_path.name),
        "plotted_algorithms": [item.selection.algorithm for item in plot_items],
        "selections": [
            {
                "algorithm": item.selection.algorithm,
                "display_name": algorithm_display_name(item.selection.algorithm),
                "run_id": item.selection.run_id,
                "seed": item.selection.seed,
                "hv": item.selection.hv,
                "igd": item.selection.igd,
                "metrics_hv": item.selection.metrics_hv,
                "metrics_igd": item.selection.metrics_igd,
                "pf_source": item.selection.pf_source,
                "requested_rule": item.selection.requested_rule,
                "resolved_rule": item.selection.resolved_rule,
                "pf_points_file": item.selection.pf_points_file,
                "raw_points_file": item.selection.raw_points_file,
                "selected_points_path": str(item.source_path.name),
                "selected_points_count": int(item.points.shape[0]),
                "selected_points_digest": _points_digest(item.points),
                "debug_all_points": item.is_debug,
                "is_formal": not item.is_debug,
                "note": item.selection.note,
            }
            for item in plot_items
        ],
    }


def _pf_plot_subtitle(
    experiment_name: str,
    problem_name: str,
    selection_rule: str,
    plot_items: list[PFPlotData],
) -> str:
    """Return a traceable multiline subtitle for PF plots."""
    lines = [
        f"experiment={experiment_name} | problem={problem_name} | selection={selection_rule}",
    ]
    lines.extend(_pf_selection_summary(item.selection) for item in plot_items)
    return "\n".join(lines)


def _get_plt():
    """Lazy-import matplotlib to avoid hard dependency."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for saving
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install mosade[analysis]"
        )


def _finish(plt, save_path: str | Path | None, dpi: int = 150) -> None:
    """Save or show the current figure."""
    if save_path is not None:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(target), dpi=dpi, bbox_inches="tight")
        if target.suffix.lower() != ".svg":
            plt.savefig(str(target.with_suffix(".svg")), dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# ======================================================================
# Pareto front plots
# ======================================================================

def _true_pf_segments(PF: np.ndarray | None) -> list[np.ndarray]:
    """Split a 2-D true PF into disconnected segments for dashed-line plotting."""
    arr = _sanitize_pf_points(PF)
    if arr is None:
        return []
    pts = arr[np.argsort(arr[:, 0])]
    if pts.shape[0] <= 2:
        return [pts]
    dx = np.diff(pts[:, 0])
    positive = dx[dx > 0]
    if positive.size == 0:
        return [pts]
    split_threshold = max(float(np.median(positive)) * 8.0, 0.025)
    split_idx = np.where(dx > split_threshold)[0] + 1
    return [segment for segment in np.split(pts, split_idx) if segment.size > 0]


def plot_pf_overlay(
    results_dict: dict[str, np.ndarray],
    PF: np.ndarray | None = None,
    title: str = "Formal PF Overlay",
    subtitle: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot a two-objective PF overlay with consistent visual semantics."""
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7.6, 5.8))

    segments = _true_pf_segments(PF)
    if segments:
        for idx, segment in enumerate(segments):
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                linestyle="--",
                color="0.25",
                linewidth=2.0,
                label="True PF" if idx == 0 else None,
                zorder=1,
            )

    for idx, name in enumerate(_ordered_algorithm_names(results_dict.keys())):
        style = _algorithm_style(name)
        F = _sanitize_pf_points(results_dict[name])
        if F is None:
            continue
        ax.scatter(
            F[:, 0],
            F[:, 1],
            c=style["color"],
            marker=style["marker"],
            s=36,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.7,
            label=algorithm_display_name(name),
            zorder=2 + idx,
        )

    ax.set_xlabel("$f_1$", fontsize=12)
    ax.set_ylabel("$f_2$", fontsize=12)
    ax.set_title(title, fontsize=12.5, pad=12)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9.5, framealpha=0.92, loc="best")
    if subtitle:
        fig.text(0.5, 0.985, subtitle, ha="center", va="top", fontsize=8.1, linespacing=1.18)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _finish(plt, save_path)


def plot_pareto_front(
    F: np.ndarray,
    PF: np.ndarray | None = None,
    title: str = "Pareto Front Approximation",
    subtitle: str | None = None,
    algorithm_name: str = "MOSADE",
    labels: tuple[str, str] = ("$f_1$", "$f_2$"),
    save_path: str | Path | None = None,
) -> None:
    """Compatibility wrapper forwarding to :func:`plot_pf_overlay`."""
    _ = labels
    plot_pf_overlay(
        {algorithm_name: F},
        PF=PF,
        title=title,
        subtitle=subtitle,
        save_path=save_path,
    )


def plot_pareto_fronts_comparison(
    results: dict[str, np.ndarray],
    PF: np.ndarray | None = None,
    title: str = "Pareto Front Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Compatibility wrapper forwarding to :func:`plot_pf_overlay`."""
    plot_pf_overlay(results, PF=PF, title=title, save_path=save_path)


# ======================================================================
# Convergence curves
# ======================================================================

def plot_convergence(
    history: dict[str, list],
    metric_key: str = "best_scal",
    xlabel: str = "Generation",
    ylabel: str | None = None,
    title: str = "Convergence Curve",
    save_path: str | Path | None = None,
) -> None:
    """Plot a metric over generations from a single run's history.

    Parameters
    ----------
    history : dict
        The history dict from MOSADEResult.history.
    metric_key : str
        Key in history to plot on y-axis.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(7, 4))

    gens = history.get("gen", list(range(len(history[metric_key]))))
    vals = history[metric_key]

    ax.plot(gens, vals, color="tab:blue", linewidth=1.2)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel or metric_key, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Strategy probability traces
# ======================================================================

def plot_strategy_probabilities(
    history: dict[str, list],
    strategy_names: list[str] | None = None,
    title: str = "Strategy Selection Probabilities",
    save_path: str | Path | None = None,
) -> None:
    """Plot strategy probabilities over generations.

    Parameters
    ----------
    history : dict
        Must contain 'gen' and 'strategy_probs' keys.
    strategy_names : list of str, optional
        Labels for each strategy.  Defaults to S1..S4.
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))

    gens = history["gen"]
    probs = np.array(history["strategy_probs"])  # (G, K)
    K = probs.shape[1]

    if strategy_names is None:
        strategy_names = [
            "S1: cur-to-pbest/1",
            "S2: rand/1",
            "S3: cur-to-rand/1",
            "S4: rand-to-pbest/2",
        ][:K]

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for k in range(K):
        ax.plot(gens, probs[:, k], color=colors[k % len(colors)],
                linewidth=1.2, label=strategy_names[k])

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Selection Probability", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Box plots for metric comparison
# ======================================================================

def plot_metric_boxplots(
    data: dict[str, list[float]],
    metric_name: str = "HV",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Box plot comparing a metric across multiple algorithms or configs.

    Parameters
    ----------
    data : dict mapping label -> list of metric values across runs.
    metric_name : str
        Name of the metric (for axis label).
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(max(4, len(data) * 1.2), 4.5))

    labels = _ordered_algorithm_names(data.keys())
    values = [data[k] for k in labels]

    bp = ax.boxplot(
        values,
        labels=[_wrap_label(algorithm_display_name(label), width=10) for label in labels],
        patch_artist=True,
        widths=0.5,
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(_algorithm_style(labels[i])["color"])
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title or f"{metric_name} Comparison", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    if len(labels) > 4:
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Convergence curves from experiment results directory
# ======================================================================

def plot_convergence_curves(
    results_dir: str | Path,
    metric_key: str = "hv",
    algorithms: dict[str, str | Path] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot median convergence curves across runs, with a shaded IQR band.

    Reads the ``convergence`` list recorded inside each run's ``history.json``.
    Each entry in that list is ``{"n_evals": int, "hv": float, "igd": float|None}``.

    Parameters
    ----------
    results_dir : path
        Experiment results directory containing problem subdirectories.
        Also used as the single-algorithm source when *algorithms* is None.
    metric_key : {"hv", "igd"}
        Which metric to plot on the y-axis.
    algorithms : dict mapping algorithm label -> results directory, optional
        For overlaying multiple algorithms on each subplot.  When None a
        single algorithm labelled ``"MOSADE"`` is read from *results_dir*.
    save_path : path or None
        File to write the figure to.  None → interactive display.

    Notes
    -----
    If a run's ``history.json`` has no ``"convergence"`` key (e.g. the run
    was executed with ``track_interval=0``), that run is silently skipped.
    Individual curves are interpolated onto a shared evaluation grid before
    computing the median and IQR, so runs with slightly different checkpoint
    counts still align correctly.
    """
    plt = _get_plt()

    if algorithms is None:
        algorithms = {"MOSADE": results_dir}

    # ------------------------------------------------------------------
    # Discover problems from the first algorithm's directory
    # ------------------------------------------------------------------
    first_dir = Path(next(iter(algorithms.values())))
    prob_dirs = sorted(
        d for d in first_dir.iterdir()
        if d.is_dir() and d.name != "plots"
        and any((d / f"run_{i:03d}" / "history.json").exists() for i in range(50))
    )
    if not prob_dirs:
        return

    n_probs = len(prob_dirs)
    ncols = min(n_probs, 3)
    nrows = (n_probs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows),
                              squeeze=False)

    y_label = _metric_display_name(metric_key)

    for prob_idx, prob_dir in enumerate(prob_dirs):
        ax = axes[prob_idx // ncols][prob_idx % ncols]
        prob_name = prob_dir.name
        ax.set_title(problem_display_label(prob_name), fontsize=10)
        ax.set_xlabel("Evaluations", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, alpha=0.3)

        plotted_any = False
        for algo_name in _ordered_algorithm_names(algorithms.keys()):
            algo_root = Path(algorithms[algo_name])
            algo_root = Path(algo_root)
            algo_prob_dir = algo_root / prob_name
            if not algo_prob_dir.is_dir():
                continue

            run_dirs = sorted(
                d for d in algo_prob_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            )

            # Collect convergence sequences from each run
            all_evals: list[np.ndarray] = []
            all_vals: list[np.ndarray] = []
            for rd in run_dirs:
                hist_path = rd / "history.json"
                if not hist_path.exists():
                    continue
                with open(hist_path) as fh:
                    hist = json.load(fh)
                conv = hist.get("convergence")
                if not conv:
                    continue
                evals = np.array([s["n_evals"] for s in conv], dtype=float)
                raw = [s.get(metric_key) for s in conv]
                if any(v is None for v in raw):
                    continue  # metric not recorded (e.g. igd without PF)
                vals = np.array(raw, dtype=float)
                all_evals.append(evals)
                all_vals.append(vals)

            if len(all_vals) < 1:
                continue

            # Build a common evaluation grid (union of all checkpoints)
            x_min = max(ev[0] for ev in all_evals)
            x_max = min(ev[-1] for ev in all_evals)
            if x_min >= x_max:
                x_grid = all_evals[0]
            else:
                n_pts = max(len(ev) for ev in all_evals)
                x_grid = np.linspace(x_min, x_max, n_pts)

            # Interpolate every run onto the common grid
            interp_vals = np.array([
                np.interp(x_grid, ev, vl)
                for ev, vl in zip(all_evals, all_vals)
            ])  # shape (n_runs, n_pts)

            median = np.median(interp_vals, axis=0)
            q25 = np.percentile(interp_vals, 25, axis=0)
            q75 = np.percentile(interp_vals, 75, axis=0)

            style = _algorithm_style(algo_name)
            ax.plot(
                x_grid,
                median,
                color=style["color"],
                marker=style["marker"],
                markersize=3.5,
                markevery=max(len(x_grid) // 10, 1),
                linewidth=1.4,
                label=f"{algorithm_display_name(algo_name)} (n={len(all_vals)})",
            )
            ax.fill_between(x_grid, q25, q75, color=style["color"], alpha=0.18)
            plotted_any = True

        if plotted_any and len(algorithms) > 1:
            ax.legend(fontsize=7, loc="best")

    # Hide unused subplot slots
    for idx in range(n_probs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(f"{y_label} Convergence", fontsize=12, y=1.01)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Multi-algorithm Pareto front overlay (enhanced markers)
# ======================================================================

def plot_multi_algorithm_pf(
    results_dict: dict[str, np.ndarray],
    PF: np.ndarray | None = None,
    title: str = "Multi-Algorithm Pareto Front Comparison",
    subtitle: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Compatibility wrapper forwarding to :func:`plot_pf_overlay`."""
    plot_pf_overlay(
        results_dict,
        PF=PF,
        title=title,
        subtitle=subtitle,
        save_path=save_path,
    )


# ======================================================================
# Multi-algorithm convergence curves with IQR bands
# ======================================================================

def plot_multi_algorithm_convergence(
    convergence_dict: dict[str, list[list[dict]]],
    metric_key: str = "hv",
    title: str = "Convergence",
    save_path: str | Path | None = None,
) -> None:
    """Plot median convergence curves with IQR bands for multiple algorithms.

    Parameters
    ----------
    convergence_dict : dict
        Maps algorithm name to a list of runs.  Each run is a list of
        snapshot dicts produced by the MOSADE convergence tracker, e.g.
        ``[{"n_evals": 100, "hv": 0.5, "igd": 0.03}, ...]``.
    metric_key : str
        Key to read from each snapshot (e.g. ``"hv"`` or ``"igd"``).
    title : str
    save_path : path or None
    """
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    for algo_name in _ordered_algorithm_names(convergence_dict.keys()):
        runs = convergence_dict[algo_name]
        all_evals: list[np.ndarray] = []
        all_vals: list[np.ndarray] = []

        for run_snaps in runs:
            if not run_snaps:
                continue
            evals = np.array([s["n_evals"] for s in run_snaps], dtype=float)
            raw = [s.get(metric_key) for s in run_snaps]
            if any(v is None for v in raw):
                continue
            all_evals.append(evals)
            all_vals.append(np.array(raw, dtype=float))

        if not all_vals:
            continue

        # Interpolate all runs onto a common evaluation grid.
        x_min = max(ev[0] for ev in all_evals)
        x_max = min(ev[-1] for ev in all_evals)
        n_pts = max(len(ev) for ev in all_evals)
        x_grid = (
            np.linspace(x_min, x_max, n_pts)
            if x_min < x_max else all_evals[0]
        )
        interp = np.array([np.interp(x_grid, ev, vl)
                            for ev, vl in zip(all_evals, all_vals)])

        median = np.median(interp, axis=0)
        q25 = np.percentile(interp, 25, axis=0)
        q75 = np.percentile(interp, 75, axis=0)

        style = _algorithm_style(algo_name)
        mark_every = max(len(x_grid) // 10, 1)
        ax.plot(
            x_grid,
            median,
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            markevery=mark_every,
            linewidth=1.8,
            label=f"{algorithm_display_name(algo_name)} (n={len(all_vals)})",
        )
        ax.fill_between(x_grid, q25, q75, color=style["color"], alpha=0.18)

    ax.set_xlabel("Evaluations", fontsize=12)
    ax.set_ylabel(_metric_display_name(metric_key), fontsize=12)
    ax.set_title(title, fontsize=12)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Cross-problem grouped box plots
# ======================================================================

def plot_grouped_boxplots(
    data: dict[str, dict[str, list[float]]],
    metric_name: str = "HV",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side box plots grouped by problem, colored by algorithm.

    Parameters
    ----------
    data : dict
        ``{problem_name: {algo_name: [metric_values_per_run]}}``
    metric_name : str
        Y-axis label.
    title : str, optional
    save_path : path or None
    """
    plt = _get_plt()

    problems = _ordered_problem_names(data.keys())
    algos = _ordered_algorithm_names({a for d in data.values() for a in d})
    n_prob = len(problems)
    n_algo = len(algos)

    if n_prob == 0 or n_algo == 0:
        return

    box_w = 0.6
    group_spacing = n_algo * box_w + 1.0
    fig_w = max(6.0, n_prob * group_spacing * 0.7)

    fig, ax = plt.subplots(figsize=(fig_w, 5))
    group_centers: list[float] = []

    for i, prob in enumerate(problems):
        center = float(i * group_spacing)
        group_centers.append(center)
        for j, algo in enumerate(algos):
            vals = data.get(prob, {}).get(algo)
            if not vals:
                continue
            pos = center + (j - (n_algo - 1) / 2.0) * box_w
            bp = ax.boxplot(
                [vals],
                positions=[pos],
                widths=[box_w * 0.85],
                patch_artist=True,
                manage_ticks=False,
            )
            color = _algorithm_style(algo)["color"]
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.7)
            for elem in ("medians", "whiskers", "caps"):
                for line in bp[elem]:
                    line.set_color("black")
                    line.set_linewidth(0.8)

    # Legend with colored patches
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(
            facecolor=_algorithm_style(algo)["color"],
            alpha=0.7,
            label=algorithm_display_name(algo),
        )
        for algo in algos
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="best")

    ax.set_xticks(group_centers)
    rotate = n_prob > 5
    ax.set_xticklabels(
        [problem_display_label(problem) for problem in problems],
        rotation=30 if rotate else 0,
        ha="right" if rotate else "center",
        fontsize=11,
    )
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title or f"{metric_name} Comparison by Problem", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Suite-level metric grids
# ======================================================================

def plot_suite_metric_grid(
    suite_name: str,
    data: dict[str, dict[str, list[float]]],
    metric_name: str = "HV",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot per-problem metric boxplots for one benchmark suite.

    Each subplot corresponds to a single problem and uses an independent
    y-axis scale so raw metrics remain readable within the suite.
    """
    plt = _get_plt()

    problems = _ordered_problem_names(data.keys())
    algos = _ordered_algorithm_names({algo for problem_data in data.values() for algo in problem_data})
    if not problems or not algos:
        return

    n_probs = len(problems)
    ncols = min(3, n_probs)
    nrows = (n_probs + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(7.5, 4.2 * ncols), max(3.8, 3.8 * nrows)),
        sharey=False,
        squeeze=False,
    )
    tick_labels = [_wrap_label(algorithm_display_name(algo), width=9) for algo in algos]

    for idx, problem in enumerate(problems):
        ax = axes[idx // ncols][idx % ncols]
        problem_data = data[problem]
        plotted = False
        for j, algo in enumerate(algos):
            vals = problem_data.get(algo)
            if not vals:
                continue
            bp = ax.boxplot(
                [vals],
                positions=[j],
                widths=0.55,
                patch_artist=True,
                manage_ticks=False,
            )
            color = _algorithm_style(algo)["color"]
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.74)
            for elem in ("medians", "whiskers", "caps"):
                for line in bp[elem]:
                    line.set_color("black")
                    line.set_linewidth(0.8)
            plotted = True

        ax.set_title(problem_display_label(problem), fontsize=11)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(
            tick_labels,
            rotation=20,
            ha="right",
            fontsize=8.5,
        )
        ax.set_ylabel(metric_name, fontsize=10)
        ax.grid(True, axis="y", alpha=0.28)
        if not plotted:
            ax.set_visible(False)

    for idx in range(n_probs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title or f"{suite_name} {_metric_display_name(metric_name)} by Problem", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _finish(plt, save_path)


# ======================================================================
# Rank heatmap
# ======================================================================

def plot_rank_heatmap(
    rank_table: "dict[str, dict[str, float]] | Any",
    title: str = "Algorithm Rank Heatmap",
    save_path: str | Path | None = None,
) -> None:
    """Heatmap of average algorithm ranks, annotated with rank values.

    Colour scale: green (rank 1, best) → red (highest rank, worst).

    Parameters
    ----------
    rank_table : dict-of-dicts or pandas DataFrame
        ``{problem_name: {algo_name: rank}}``.  Rows = problems,
        columns = algorithms.  A pandas DataFrame is converted automatically.
    title : str
    save_path : path or None
    """
    # Accept pandas DataFrame
    try:
        import pandas as pd
        if isinstance(rank_table, pd.DataFrame):
            rank_table = rank_table.to_dict(orient="index")
    except ImportError:
        pass

    plt = _get_plt()

    problems = _ordered_problem_names(rank_table.keys())
    algos = _ordered_algorithm_names({a for row in rank_table.values() for a in row})
    n_prob, n_algo = len(problems), len(algos)

    if n_prob == 0 or n_algo == 0:
        return

    mat = np.full((n_prob, n_algo), np.nan)
    for i, prob in enumerate(problems):
        for j, algo in enumerate(algos):
            if algo in rank_table[prob]:
                mat[i, j] = rank_table[prob][algo]

    problem_labels = [_wrap_label(problem_display_label(problem), width=12) for problem in problems]
    algo_labels = [_wrap_label(algorithm_display_name(algo), width=12) for algo in algos]
    transpose = n_algo >= 5 or max(len(label) for label in algo_labels) > 10
    display_mat = mat.T if transpose else mat
    x_labels = problem_labels if transpose else algo_labels
    y_labels = algo_labels if transpose else problem_labels
    n_rows, n_cols = display_mat.shape

    max_x_lines = max(label.count("\n") + 1 for label in x_labels)
    max_y_chars = max(len(line) for label in y_labels for line in label.splitlines())
    fig_w = max(7.0, 0.95 * n_cols + 0.18 * max(len(label) for label in x_labels))
    fig_h = max(4.5, 0.62 * n_rows + 0.12 * max_y_chars + 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color="#d9d9d9")
    vmax = float(np.nanmax(display_mat)) if not np.isnan(display_mat).all() else float(n_algo)
    im = ax.imshow(display_mat, cmap=cmap, aspect="auto", vmin=1, vmax=max(vmax, 1.0))

    # Annotate cells
    anno_fontsize = 10 if max(n_rows, n_cols) <= 8 else 9
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(display_mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{display_mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=anno_fontsize,
                    fontweight="bold",
                )
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=9, color="#555555")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(
        x_labels,
        fontsize=10,
        rotation=0 if max_x_lines > 1 else (18 if n_cols > 4 else 0),
        ha="right" if max_x_lines == 1 and n_cols > 4 else "center",
    )
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Problem" if transpose else "Algorithm", fontsize=10)
    ax.set_ylabel("Algorithm" if transpose else "Problem", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.03)
    cbar.set_label("Rank (1 = best)", fontsize=10)
    left = min(0.42, 0.12 + 0.012 * max_y_chars)
    bottom = min(0.34, 0.10 + 0.05 * max_x_lines)
    fig.subplots_adjust(left=left, bottom=bottom, right=0.93, top=0.90)
    _finish(plt, save_path)


# ======================================================================
# Constraint dynamics plot
# ======================================================================

def _safe_filename_token(text: str) -> str:
    """Return a filesystem-safe token for plot filenames."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return token.strip("_") or "unknown"


def _load_results_config(results_dir: Path) -> dict[str, Any]:
    """Load the frozen config.json stored in a results directory."""
    config_path = results_dir / "config.json"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _problem_from_config(config: dict[str, Any], problem_dir_name: str) -> Any | None:
    """Reconstruct a problem instance from a results config and directory name."""
    for entry in config.get("problems", []):
        if isinstance(entry, dict):
            kwargs = {k: v for k, v in entry.items() if k != "name"}
            suffix = "_".join(f"{k}{v}" for k, v in sorted(kwargs.items()))
            dir_name = f"{entry['name']}_{suffix}" if suffix else entry["name"]
            if dir_name == problem_dir_name:
                return get_problem(entry["name"], **kwargs)
        elif str(entry) == problem_dir_name:
            return get_problem(str(entry))
    return None


def _estimate_plot_ref_point(problem: Any | None, PF: np.ndarray | None) -> np.ndarray | None:
    """Estimate a stable 2-D HV reference point for plotting-time recomputation."""
    if problem is None:
        return None
    rng = np.random.default_rng(12345)
    X = rng.uniform(problem.lower, problem.upper, size=(4096, problem.n_var))
    F_rand, _ = problem._evaluate(X)
    maxima = np.max(F_rand, axis=0)
    if PF is not None and PF.size > 0 and PF.ndim == 2 and PF.shape[1] == maxima.shape[0]:
        maxima = np.maximum(maxima, np.max(PF, axis=0))
    return maxima * 1.1 + 1e-6


def _load_run_metrics_json(run_dir: Path) -> dict:
    """Load a run-level metrics.json file when present."""
    mp = run_dir / "metrics.json"
    if not mp.exists():
        return {}
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def _constraint_dynamics_from_history(
    history: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str] | None:
    """Extract real constraint-dynamics series from run history.

    Returns ``None`` for unconstrained histories or when the required
    constrained-series keys were not recorded.
    """
    n_constr = history.get("n_constr")
    if n_constr is not None and int(n_constr) <= 0:
        return None

    cv_key = None
    if isinstance(history.get("mean_cv"), list):
        cv_key = "mean_cv"
    elif isinstance(history.get("median_cv"), list):
        cv_key = "median_cv"
    if cv_key is None:
        return None

    epsilon = history.get("epsilon")
    feas = history.get("feasibility_ratio")
    cv = history.get(cv_key)
    if not isinstance(epsilon, list) or not isinstance(feas, list) or not isinstance(cv, list):
        return None
    if not epsilon or not feas or not cv:
        return None

    gens = history.get("gen")
    if isinstance(gens, list) and gens:
        x = np.asarray(gens, dtype=float)
    else:
        x = np.arange(1, min(len(epsilon), len(feas), len(cv)) + 1, dtype=float)

    n = min(len(x), len(epsilon), len(feas), len(cv))
    if n == 0:
        return None

    return (
        x[:n],
        np.asarray(epsilon[:n], dtype=float),
        np.asarray(feas[:n], dtype=float),
        np.asarray(cv[:n], dtype=float),
        "Mean CV" if cv_key == "mean_cv" else "Median CV",
    )


def plot_epsilon_feasibility(
    history: dict,
    title: str = "Constraint Dynamics",
    save_path: str | Path | None = None,
) -> None:
    """Plot constrained-run epsilon, feasibility ratio, and CV dynamics.

    Unconstrained runs are skipped by default. Only real constrained-history
    series recorded in ``history.json`` are plotted.
    """
    series = _constraint_dynamics_from_history(history)
    if series is None:
        return

    plt = _get_plt()
    gens, epsilon, feas, cv, cv_label = series

    fig, axes = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    fig.suptitle(title, fontsize=12)

    axes[0].plot(gens, epsilon, color="tab:blue", linewidth=1.6)
    axes[0].set_ylabel("Epsilon", fontsize=10)
    axes[0].set_ylim(bottom=-0.02)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gens, feas, color="tab:orange", linewidth=1.6)
    axes[1].set_ylabel("Feasible Ratio", fontsize=10)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(gens, cv, color="tab:red", linewidth=1.6)
    axes[2].set_ylabel(cv_label, fontsize=10)
    axes[2].set_xlabel("Generation", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    _finish(plt, save_path)


# ======================================================================
# Rank table computation helper
# ======================================================================

def compute_rank_table(
    results_dir: str | Path,
    metric_key: str = "hv",
    higher_is_better: bool | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-problem algorithm ranks from experiment summary files.

    For each problem subdirectory that contains a multi-algorithm
    ``summary.json``, algorithms are ranked by their median metric value
    (rank 1 = best).

    Parameters
    ----------
    results_dir : path to the experiment results directory
    metric_key : metric to rank by (e.g. ``"hv"``, ``"igd"``, ``"gd"``)
    higher_is_better : if ``None``, inferred from *metric_key*:
        ``True`` for ``"hv"``; ``False`` for all others.

    Returns
    -------
    dict[problem_name, dict[algo_name, rank]]
        Only problems with ≥ 2 algorithms are included.
    """
    if higher_is_better is None:
        higher_is_better = metric_key == "hv"

    results_dir = Path(results_dir)
    rank_table: dict[str, dict[str, float]] = {}

    for prob_dir in sorted(results_dir.iterdir()):
        if not prob_dir.is_dir() or prob_dir.name == "plots":
            continue
        summary_path = prob_dir / "summary.json"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        # Multi-algo layout: first-level values are themselves dicts.
        first_val = next(iter(summary.values()), None)
        if not isinstance(first_val, dict):
            continue  # single-algorithm layout — nothing to rank

        col = f"{metric_key}_median"
        algo_medians: dict[str, float] = {}
        for algo_name, algo_data in summary.items():
            if not isinstance(algo_data, dict):
                continue
            if algo_data.get("status") == "unsupported":
                continue
            if algo_data.get(f"{metric_key}_n_valid") == 0:
                continue
            if col not in algo_data:
                continue
            val = float(algo_data[col])
            if np.isnan(val):
                continue
            algo_medians[algo_name] = val

        if len(algo_medians) < 2:
            continue

        sorted_algos = sorted(
            algo_medians,
            key=lambda a: algo_medians[a],
            reverse=higher_is_better,
        )
        rank_table[prob_dir.name] = {
            algo: float(rank + 1) for rank, algo in enumerate(sorted_algos)
        }

    return rank_table


# ======================================================================
# Batch plotting from experiment results directory
# ======================================================================

def _plot_convergence_curves_for_problem(
    prob_dir: Path,
    metric_key: str,
    save_path: str | Path | None,
) -> None:
    """Single-problem variant of plot_convergence_curves used by the batch runner."""
    plt = _get_plt()

    run_dirs = sorted(d for d in prob_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run_"))
    all_evals: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    for rd in run_dirs:
        hist_path = rd / "history.json"
        if not hist_path.exists():
            continue
        with open(hist_path) as fh:
            hist = json.load(fh)
        conv = hist.get("convergence")
        if not conv:
            continue
        evals = np.array([s["n_evals"] for s in conv], dtype=float)
        raw = [s.get(metric_key) for s in conv]
        if any(v is None for v in raw):
            continue
        all_evals.append(evals)
        all_vals.append(np.array(raw, dtype=float))

    if not all_vals:
        return

    x_min = max(ev[0] for ev in all_evals)
    x_max = min(ev[-1] for ev in all_evals)
    if x_min >= x_max:
        x_grid = all_evals[0]
    else:
        n_pts = max(len(ev) for ev in all_evals)
        x_grid = np.linspace(x_min, x_max, n_pts)

    interp_vals = np.array([np.interp(x_grid, ev, vl)
                             for ev, vl in zip(all_evals, all_vals)])

    median = np.median(interp_vals, axis=0)
    q25 = np.percentile(interp_vals, 25, axis=0)
    q75 = np.percentile(interp_vals, 75, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_grid, median, color="tab:blue", linewidth=1.4,
            label=f"Median (n={len(all_vals)})")
    ax.fill_between(x_grid, q25, q75, color="tab:blue", alpha=0.2, label="IQR")
    ax.set_xlabel("Evaluations", fontsize=11)
    ax.set_ylabel(_metric_display_name(metric_key), fontsize=11)
    ax.set_title(
        f"{problem_display_label(prob_dir.name)} - {_metric_display_name(metric_key)} Convergence",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _finish(plt, save_path)


def _is_multi_algo(prob_dir: Path) -> bool:
    """Return True if the problem directory uses the multi-algorithm layout.

    Checks whether subdirectories contain ``run_*`` dirs (algo layout)
    rather than being ``run_*`` dirs themselves (single-algo layout).
    """
    for child in prob_dir.iterdir():
        if child.is_dir() and not child.name.startswith("run_"):
            # Check if this child contains run_ dirs (i.e. it's an algo dir)
            if any(d.name.startswith("run_") for d in child.iterdir() if d.is_dir()):
                return True
    return False


def _collect_algo_runs(prob_dir: Path) -> dict[str, list[Path]]:
    """Collect run directories per algorithm for both layouts.

    Returns {algo_name: [run_dirs]} where algo_name is the algorithm name
    (or "" for single-algorithm layout).
    """
    if _is_multi_algo(prob_dir):
        result: dict[str, list[Path]] = {}
        for algo_dir in sorted(prob_dir.iterdir()):
            if not algo_dir.is_dir() or algo_dir.name.startswith("run_"):
                continue
            runs = sorted(d for d in algo_dir.iterdir()
                          if d.is_dir() and d.name.startswith("run_"))
            if runs:
                result[algo_dir.name] = runs
        return result
    else:
        runs = sorted(d for d in prob_dir.iterdir()
                      if d.is_dir() and d.name.startswith("run_"))
        if runs:
            return {"": runs}
        return {}


def _load_run_metrics(run_dirs: list[Path]) -> tuple[list[float], list[float]]:
    """Load HV and IGD values from a list of run directories."""
    hvs, igds = [], []
    for rd in run_dirs:
        mp = rd / "metrics.json"
        if mp.exists():
            with open(mp) as f:
                m = json.load(f)
            hvs.append(m.get("hv", float("nan")))
            igds.append(m.get("igd", float("nan")))
    return hvs, igds


def _median_metric_run(run_dirs: list[Path], metric_key: str = "hv") -> Path | None:
    """Return the run directory at the median finite metric value."""
    ranked_runs: list[tuple[float, Path]] = []
    for rd in run_dirs:
        mp = rd / "metrics.json"
        if not mp.exists():
            continue
        with open(mp) as f:
            metrics = json.load(f)
        value = metrics.get(metric_key)
        if value is None:
            continue
        value = float(value)
        if np.isnan(value):
            continue
        ranked_runs.append((value, rd))

    if not ranked_runs:
        return None

    ranked_runs.sort(key=lambda item: item[0])
    return ranked_runs[len(ranked_runs) // 2][1]


def plot_experiment_results(
    results_dir: str | Path,
    save: bool = True,
    pf_selection: str = PF_SELECTION_DEFAULT,
    pf_problems: list[str] | None = None,
    pf_source: str = "auto",
    algorithms: list[str] | None = None,
    max_pf_algorithms: int = PF_OVERLAY_LIMIT,
    top_k: int | None = None,
    pf_debug_all_points: bool = False,
    output_dir: str | Path | None = None,
) -> None:
    """Generate all standard plots from an experiment results directory.

    Supports both single-algorithm (problem/run_000/) and multi-algorithm
    (problem/algo/run_000/) directory layouts.

    Creates a ``plots/`` tree inside *results_dir* with:

    - ``plots/problems`` for per-problem PF, convergence, boxplots, and
      MOSADE-specific traces.
    - ``plots/suites`` for suite-level HV/IGD grids where each subplot is one
      problem with an independent y-axis scale.
    - ``plots/summary`` for benchmark-wide overview plots such as rank heatmaps.

    Parameters
    ----------
    results_dir : path to experiment results directory
    save : if True, save plots to the ``plots/`` subdirectory;
           if False, display interactively.
    pf_selection : representative-run selection rule for PF plots
    pf_problems : optional subset of problem directory names for PF plots
    pf_source : formal PF source preference: ``auto``, ``archive``, or ``final_population``
    algorithms : optional subset of algorithms for PF plots
    max_pf_algorithms : maximum algorithms per PF page before pagination
    top_k : optional performance-ranked cap applied before pagination
    pf_debug_all_points : if True, also export raw-point debug PF plots
    output_dir : optional explicit plots directory
    """
    results_dir = Path(results_dir)
    config = _load_results_config(results_dir)
    experiment_name = results_dir.name
    plots_dir = Path(output_dir) if output_dir is not None else (results_dir / "plots")
    summary_dir = plots_dir / "summary"
    suites_dir = plots_dir / "suites"
    problems_dir = plots_dir / "problems"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find problem subdirectories
    prob_names = _ordered_problem_names(
        d.name for d in results_dir.iterdir()
        if d.is_dir() and d.name != "plots" and (d / "summary.json").exists()
    )
    prob_dirs = [results_dir / prob_name for prob_name in prob_names]
    pf_problem_set = set(pf_problems or [])

    # Accumulators for suite/summary plots
    all_hv_data: dict[str, dict[str, list[float]]] = {}   # {prob: {algo: [vals]}}
    all_igd_data: dict[str, dict[str, list[float]]] = {}
    is_multi_experiment = False

    for prob_dir in prob_dirs:
        prob_name = prob_dir.name

        problem = _problem_from_config(config, prob_name)
        pf_path = prob_dir / "pareto_front.txt"
        PF = _sanitize_pf_points(_load_matrix(pf_path))
        ref_point = _estimate_plot_ref_point(problem, PF)

        algo_runs = _collect_algo_runs(prob_dir)
        if not algo_runs:
            continue
        if algorithms:
            wanted_algorithms = set(algorithms)
            algo_runs = {
                key: value
                for key, value in algo_runs.items()
                if (key or "MOSADE") in wanted_algorithms
            }
            if not algo_runs:
                continue

        multi = len(algo_runs) > 1 or "" not in algo_runs
        if multi:
            is_multi_experiment = True

        # ---- Pareto front plot ----
        pf_items: list[PFPlotData] = []
        is_pf_overlay_problem = problem is None or getattr(problem, "n_obj", 2) == 2
        if is_pf_overlay_problem and (not pf_problem_set or prob_name in pf_problem_set):
            for algo_name, run_dirs in algo_runs.items():
                label = algo_name or "MOSADE"
                plot_data = _resolve_pf_plot_data(
                    label,
                    run_dirs,
                    selection_rule=pf_selection,
                    problem=problem,
                    PF=PF,
                    ref_point=ref_point,
                    source_preference=pf_source,
                    debug_all_points=False,
                )
                if plot_data is not None:
                    pf_items.append(plot_data)

        if pf_items:
            ranked_items = sorted(
                pf_items,
                key=lambda item: _pf_overlay_sort_key(item, pf_selection),
            )
            if top_k is not None and top_k > 0:
                ranked_items = ranked_items[:top_k]

            selected_items = sorted(
                ranked_items,
                key=lambda item: _algorithm_sort_key(item.selection.algorithm),
            )
            page_size = max(1, int(max_pf_algorithms))
            selection_token = pf_selection
            if pf_source != "auto":
                selection_token = f"{selection_token}_{pf_source}"
            if top_k is not None and top_k > 0:
                selection_token = f"{selection_token}_top{top_k}"

            pages = [
                selected_items[idx: idx + page_size]
                for idx in range(0, len(selected_items), page_size)
            ]
            for page_idx, page_items in enumerate(pages, start=1):
                title = f"{problem_display_label(prob_name)} - Formal PF Overlay"
                if len(pages) > 1:
                    title += f" (Page {page_idx}/{len(pages)})"
                subtitle = _pf_plot_subtitle(experiment_name, prob_name, selection_token, page_items)
                sp = (
                    problems_dir / f"{_pf_page_stem(experiment_name, prob_name, selection_token, page_idx)}.png"
                    if save else None
                )
                plot_pf_overlay(
                    {item.selection.algorithm: item.points for item in page_items},
                    PF,
                    title=title,
                    subtitle=subtitle,
                    save_path=sp,
                )
                if sp is not None:
                    save_json(
                        _pf_manifest_path(sp),
                        _pf_manifest(
                            experiment_name=experiment_name,
                            problem_name=prob_name,
                            plot_mode="formal_page",
                            selection_rule=selection_token,
                            plot_items=page_items,
                            plot_path=sp,
                            debug_all_points=False,
                            true_pf_path=pf_path if pf_path.exists() else None,
                        ),
                    )

            if pf_debug_all_points:
                for item in selected_items:
                    debug_points, debug_source = _load_pf_points(
                        item.selection,
                        debug_all_points=True,
                    )
                    debug_points = _sanitize_pf_points(debug_points)
                    if debug_points is None:
                        continue
                    debug_item = PFPlotData(
                        selection=item.selection,
                        points=debug_points,
                        source_path=debug_source,
                        is_debug=True,
                    )
                    debug_subtitle = _pf_plot_subtitle(
                        experiment_name,
                        prob_name,
                        f"{selection_token} | debug=all_points",
                        [debug_item],
                    )
                    debug_path = (
                        problems_dir / (
                            f"{_pf_debug_stem(experiment_name, prob_name, item.selection.algorithm, item.selection.run_id)}.png"
                        )
                        if save
                        else None
                    )
                    plot_pf_overlay(
                        {item.selection.algorithm: debug_points},
                        PF,
                        title=f"{problem_display_label(prob_name)} - Debug All Points",
                        subtitle=debug_subtitle,
                        save_path=debug_path,
                    )
                    if debug_path is not None:
                        save_json(
                            _pf_manifest_path(debug_path),
                            _pf_manifest(
                                experiment_name=experiment_name,
                                problem_name=prob_name,
                                plot_mode="debug_raw_points",
                                selection_rule=selection_token,
                                plot_items=[debug_item],
                                plot_path=debug_path,
                                debug_all_points=True,
                                true_pf_path=pf_path if pf_path.exists() else None,
                            ),
                        )

        # ---- Strategy probability trace and constraint dynamics (MOSADE only) ----
        for algo_name, run_dirs in algo_runs.items():
            label = algo_name or "MOSADE"
            if label != "MOSADE":
                continue
            median_run = _median_metric_run(run_dirs, metric_key="hv")
            if median_run is None:
                continue
            hist_path = median_run / "history.json"
            if not hist_path.exists():
                continue
            with open(hist_path) as fh:
                hist = json.load(fh)
            if "strategy_probs" in hist:
                sp = problems_dir / f"{_safe_filename_token(prob_name)}_strategy_probs.png" if save else None
                plot_strategy_probabilities(
                    hist,
                    title=f"{problem_display_label(prob_name)} - Strategy Probabilities",
                    save_path=sp,
                )
            metrics = _load_run_metrics_json(median_run)
            seed = metrics.get("seed")
            selection = f"median_hv_seed{seed}" if seed is not None else "median_hv"
            sp = (
                problems_dir / (
                    "constraint_dynamics_"
                    f"{_safe_filename_token(prob_name)}_"
                    f"{_safe_filename_token(label)}_"
                    f"{_safe_filename_token(selection)}.png"
                )
                if save
                else None
            )
            plot_epsilon_feasibility(
                hist,
                title=(
                    f"{problem_display_label(prob_name)} | "
                    f"{algorithm_display_name(label)} | {selection}"
                ),
                save_path=sp,
            )

        # ---- Per-problem box plots and metric accumulation ----
        hv_data: dict[str, list[float]] = {}
        igd_data: dict[str, list[float]] = {}
        for algo_name, run_dirs in algo_runs.items():
            label = algo_name or "MOSADE"
            hvs, igds = _load_run_metrics(run_dirs)
            valid_hvs = [h for h in hvs if not np.isnan(h)]
            valid_igds = [g for g in igds if not np.isnan(g)]
            if valid_hvs:
                hv_data[label] = valid_hvs
            if valid_igds:
                igd_data[label] = valid_igds

        if hv_data and any(len(v) >= 3 for v in hv_data.values()):
            sp = problems_dir / f"{_safe_filename_token(prob_name)}_hv_boxplot.png" if save else None
            plot_metric_boxplots(
                hv_data, "HV", title=f"{problem_display_label(prob_name)} - HV Comparison", save_path=sp
            )
        if igd_data and any(len(v) >= 3 for v in igd_data.values()):
            sp = problems_dir / f"{_safe_filename_token(prob_name)}_igd_boxplot.png" if save else None
            plot_metric_boxplots(
                igd_data, "IGD", title=f"{problem_display_label(prob_name)} - IGD Comparison", save_path=sp
            )

        if hv_data:
            all_hv_data[prob_name] = hv_data
        if igd_data:
            all_igd_data[prob_name] = igd_data

        # ---- Convergence curves ----
        if not multi:
            # Single-algo: per-metric curves from the problem directory
            for metric in ("hv", "igd"):
                sp = problems_dir / f"{_safe_filename_token(prob_name)}_convergence_{metric}.png" if save else None
                run_parent = next(iter(algo_runs.values()))[0].parent
                _plot_convergence_curves_for_problem(run_parent, metric, sp)
        else:
            # Multi-algo: one overlay per metric using the new convergence function
            for metric in ("hv", "igd"):
                conv_data: dict[str, list[list[dict]]] = {}
                for algo_name, run_dirs in algo_runs.items():
                    label = algo_name or "MOSADE"
                    run_convs: list[list[dict]] = []
                    for rd in run_dirs:
                        hp = rd / "history.json"
                        if not hp.exists():
                            continue
                        with open(hp) as fh:
                            hist = json.load(fh)
                        conv = hist.get("convergence")
                        if conv:
                            run_convs.append(conv)
                    if run_convs:
                        conv_data[label] = run_convs
                if conv_data:
                    sp = (problems_dir / f"{_safe_filename_token(prob_name)}_convergence_{metric}.png"
                          if save else None)
                    plot_multi_algorithm_convergence(
                        conv_data, metric_key=metric,
                        title=f"{problem_display_label(prob_name)} - {_metric_display_name(metric)} Convergence",
                        save_path=sp,
                    )

    # ---- Suite-level metric grids ----
    for metric, all_data in (("hv", all_hv_data), ("igd", all_igd_data)):
        if not all_data:
            continue
        suite_groups: dict[str, dict[str, dict[str, list[float]]]] = {}
        for problem_name, problem_data in all_data.items():
            suite_groups.setdefault(_suite_name(problem_name), {})[problem_name] = problem_data
        for suite, suite_data in suite_groups.items():
            sp = suites_dir / f"{suite}_{metric}_grid.png" if save else None
            plot_suite_metric_grid(
                suite,
                suite_data,
                metric_name=_metric_display_name(metric),
                title=f"{suite} {_metric_display_name(metric)} by Problem",
                save_path=sp,
            )

    # ---- Cross-problem benchmark overview plots ----
    if is_multi_experiment and len(all_hv_data) > 1:
        for metric in ("hv", "igd"):
            rt = compute_rank_table(results_dir, metric_key=metric)
            if rt:
                sp = (summary_dir / f"rank_heatmap_{metric}.png" if save else None)
                plot_rank_heatmap(
                    rt,
                    title=f"{_metric_display_name(metric)} Rank Heatmap",
                    save_path=sp,
                )
