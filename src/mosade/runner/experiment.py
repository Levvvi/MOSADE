"""Config-driven experiment runner.

Reads a YAML config, runs the specified algorithm(s) on each problem for
multiple seeds, computes metrics, and saves structured results.

New in this version
-------------------
- Optional run-level process parallelism via ``parallel_workers`` in YAML.
- Safe Windows-compatible process spawning.
- Resume support remains intact: completed runs are skipped.

Typical usage::

    python scripts/run_experiment.py --config configs/benchmark_zdt.yaml

And inside the YAML::

    parallel_workers: 8

"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mosade.algorithm import ALGORITHM_REGISTRY, MOSADE
from mosade.problems import PROBLEM_REGISTRY
from mosade.algorithm.selection import nondominated_mask
from mosade.metrics import gd, hypervolume, igd, igd_plus, spread
from mosade.problems import get_problem
from mosade.utils.io import load_config, make_run_dir, save_json, save_objectives
from mosade.utils.logging import setup_logger
from mosade.utils.seeding import seed_sequence

log = logging.getLogger("mosade.runner")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _formal_pf_from_result(problem: Any, result: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the final feasible nondominated approximation set for PF plots."""
    F = np.asarray(result.F, dtype=float)
    X = np.asarray(result.X, dtype=float)

    if F.size == 0 or X.size == 0:
        return (
            np.empty((0, problem.n_var), dtype=float),
            np.empty((0, problem.n_obj), dtype=float),
        )

    F = np.atleast_2d(F)
    X = np.atleast_2d(X)
    if F.shape[0] != X.shape[0]:
        n = min(F.shape[0], X.shape[0])
        F = F[:n]
        X = X[:n]

    finite_mask = np.isfinite(F).all(axis=1) & np.isfinite(X).all(axis=1)
    if not np.any(finite_mask):
        return (
            np.empty((0, problem.n_var), dtype=float),
            np.empty((0, problem.n_obj), dtype=float),
        )

    F = F[finite_mask]
    X = X[finite_mask]

    if problem.n_constr > 0:
        _, G = problem._evaluate(X)
        if G is None or G.size == 0:
            feasible_mask = np.ones(F.shape[0], dtype=bool)
        else:
            cv = np.sum(np.maximum(0.0, G), axis=1)
            feasible_mask = cv <= 0.0
    else:
        feasible_mask = np.ones(F.shape[0], dtype=bool)

    if not np.any(feasible_mask):
        return (
            np.empty((0, problem.n_var), dtype=float),
            np.empty((0, problem.n_obj), dtype=float),
        )

    F_feas = F[feasible_mask]
    X_feas = X[feasible_mask]
    nd_mask = nondominated_mask(F_feas)
    return X_feas[nd_mask], F_feas[nd_mask]


def _parse_algo_configs(cfg: dict) -> tuple[list[tuple[str, dict]], bool]:
    """Normalise algorithm config into a list of (name, params) pairs."""
    if "algorithms" in cfg:
        entries = cfg["algorithms"]
        result: list[tuple[str, dict]] = []
        for entry in entries:
            entry = dict(entry)
            name = entry.pop("name")
            result.append((name, entry))
        return result, True
    params = dict(cfg.get("algorithm", {}))
    return [("MOSADE", params)], False


# ---------------------------------------------------------------------------
# Convergence threshold helper
# ---------------------------------------------------------------------------


def _fe_to_threshold(
    convergence_history: list[dict],
    final_hv: float,
    fraction: float,
) -> int | None:
    """Return the eval count at which *fraction* of *final_hv* was first reached."""
    if not np.isfinite(final_hv) or final_hv <= 0.0:
        return None
    threshold = fraction * final_hv
    for entry in convergence_history:
        hv = entry.get("hv")
        if hv is None or not np.isfinite(float(hv)):
            continue
        if float(hv) >= threshold:
            return entry["n_evals"]
    return None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _dominating_reference_point(maxima: np.ndarray) -> np.ndarray:
    """Return a minimisation HV reference point that safely dominates *maxima*."""
    maxima = np.asarray(maxima, dtype=float)
    margin = np.maximum(1e-6, 0.1 * np.maximum(np.abs(maxima), 1.0))
    return maxima + margin


def _final_cv_stats(problem: Any, X: np.ndarray) -> dict[str, Any]:
    """Return final-set feasibility diagnostics for a decision matrix *X*."""
    if X.size == 0:
        return {
            "feasibility_ratio": float("nan") if problem.n_constr > 0 else 1.0,
            "best_cv": float("nan") if problem.n_constr > 0 else 0.0,
            "median_cv": float("nan") if problem.n_constr > 0 else 0.0,
            "n_feasible_final": 0,
        }
    if problem.n_constr <= 0:
        return {
            "feasibility_ratio": 1.0,
            "best_cv": 0.0,
            "median_cv": 0.0,
            "n_feasible_final": int(X.shape[0]),
        }

    _, G = problem._evaluate(np.asarray(X, dtype=float))
    if G is None or G.size == 0:
        cv = np.zeros(X.shape[0], dtype=float)
    else:
        cv = np.sum(np.maximum(0.0, G), axis=1)
    return {
        "feasibility_ratio": float(np.mean(cv <= 0.0)),
        "best_cv": float(np.min(cv)),
        "median_cv": float(np.median(cv)),
        "n_feasible_final": int(np.sum(cv <= 0.0)),
    }


def _downsample_front(F: np.ndarray, max_points: int = 2_000) -> np.ndarray:
    """Return an evenly downsampled copy of *F* when it is very large."""
    if F.shape[0] <= max_points:
        return F
    idx = np.linspace(0, F.shape[0] - 1, max_points, dtype=int)
    return F[idx]


def _fallback_pareto_front(problem: Any) -> np.ndarray | None:
    """Try to obtain a reference front from pymoo for known benchmark suites."""
    try:
        name = problem.__class__.__name__

        if name.startswith("DTLZ"):
            from pymoo.problems.many import dtlz as pymoo_dtlz

            cls = getattr(pymoo_dtlz, name)
            kwargs: dict[str, Any] = {"n_var": problem.n_var, "n_obj": problem.n_obj}
            if hasattr(problem, "_alpha"):
                kwargs["alpha"] = getattr(problem, "_alpha")
            pf = cls(**kwargs).pareto_front()
            return _downsample_front(np.asarray(pf, dtype=float))

        if name.startswith("WFG"):
            from pymoo.problems.many import wfg as pymoo_wfg

            cls = getattr(pymoo_wfg, name)
            pf = cls(n_var=problem.n_var, n_obj=problem.n_obj).pareto_front()
            return _downsample_front(np.asarray(pf, dtype=float))

        if name.startswith("DASCMOP"):
            from pymoo.problems.multi import dascmop as pymoo_dascmop

            cls = getattr(pymoo_dascmop, name)
            difficulty = getattr(problem, "_difficulty", 8)
            pf = cls(difficulty).pareto_front()
            return _downsample_front(np.asarray(pf, dtype=float))
    except Exception as exc:
        log.warning(
            "Failed to obtain fallback Pareto front for %s: %s",
            problem.__class__.__name__,
            exc,
        )

    return None


def _get_reference_front(problem: Any) -> np.ndarray | None:
    """Return an analytical or fallback reference front for *problem*."""
    pf = problem.pareto_front()
    if pf is not None:
        return np.asarray(pf, dtype=float)
    return _fallback_pareto_front(problem)


def _estimate_reference_point(
    problem: Any,
    pf: np.ndarray | None,
    n_samples: int = 4_096,
) -> np.ndarray:
    """Estimate a shared HV reference point that dominates typical outputs."""
    rng = np.random.default_rng(12345)
    X = rng.uniform(problem.lower, problem.upper, size=(n_samples, problem.n_var))
    F_rand, _ = problem._evaluate(X)
    maxima = np.max(F_rand, axis=0)
    if pf is not None and pf.size > 0:
        maxima = np.maximum(maxima, np.max(pf, axis=0))
    return _dominating_reference_point(maxima)


# ---------------------------------------------------------------------------
# Single-run execution helpers
# ---------------------------------------------------------------------------


def _worker_init() -> None:
    """Clamp BLAS/OpenMP threads per worker to avoid oversubscription."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _current_commit_hash() -> str:
    """Return the current Git commit hash, or an explicit unavailable marker."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "NOT_GIT_REPOSITORY"
    return completed.stdout.strip() or "UNKNOWN_COMMIT"


def _read_source_snapshot_sha256() -> str:
    """Return the current source snapshot manifest digest when available."""
    sha_path = Path("audit") / "source_snapshot_manifest.sha256"
    if not sha_path.exists():
        return "MISSING_SOURCE_SNAPSHOT_MANIFEST_SHA256"
    text = sha_path.read_text(encoding="utf-8", errors="replace").strip()
    return text.split()[0] if text else "EMPTY_SOURCE_SNAPSHOT_MANIFEST_SHA256"


def _package_version(name: str) -> str:
    """Return an installed package version or a clear unavailable marker."""
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _environment_payload(parallel_workers: int) -> dict[str, Any]:
    """Build an experiment-level environment record."""
    packages = {
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "pandas": _package_version("pandas"),
        "pymoo": _package_version("pymoo"),
        "pytest": _package_version("pytest"),
    }
    return {
        "python_version": platform.python_version(),
        "package_versions": packages,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_info_if_available": platform.uname()._asdict(),
        "parallel_workers": int(parallel_workers),
        "blas_thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "UNSET"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "UNSET"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "UNSET"),
        },
    }


def _problem_label(prob_entry: Any) -> str:
    """Return the runner's stable directory label for a problem config entry."""
    if isinstance(prob_entry, dict):
        prob_cfg = {k: v for k, v in prob_entry.items() if k != "name"}
        param_suffix = "_".join(f"{k}{v}" for k, v in sorted(prob_cfg.items()))
        return f"{prob_entry['name']}_{param_suffix}" if param_suffix else prob_entry["name"]
    return str(prob_entry)


def _write_experiment_provenance(
    run_dir: Path,
    config_path: str | Path,
    cfg: dict[str, Any],
    algo_configs: list[tuple[str, dict]],
    problems: list[Any],
    seeds: list[int],
    parallel_workers: int,
) -> None:
    """Write manifest, environment, registry snapshots, run matrix, and command line."""
    env = _environment_payload(parallel_workers)
    save_json(run_dir / "environment.json", env)

    save_json(
        run_dir / "algorithm_registry_snapshot.json",
        {
            name: {
                "callable_module": getattr(factory, "__module__", "NA"),
                "callable_name": getattr(factory, "__name__", factory.__class__.__name__),
                "presets": getattr(factory, "_PRESETS", {}),
            }
            for name, factory in sorted(ALGORITHM_REGISTRY.items())
        },
    )
    save_json(
        run_dir / "problem_registry_snapshot.json",
        {
            name: {"module": cls.__module__, "class": cls.__name__}
            for name, cls in sorted(PROBLEM_REGISTRY.items())
        },
    )

    budgets = sorted({
        int(params.get("max_evals"))
        for _, params in algo_configs
        if params.get("max_evals") is not None
    })
    pop_sizes = sorted({
        int(params.get("pop_size"))
        for _, params in algo_configs
        if params.get("pop_size") is not None
    })
    source_sha = _read_source_snapshot_sha256()
    manifest = {
        "experiment_name": cfg.get("tag", "experiment"),
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "working_directory": str(Path.cwd()),
        "git_commit_hash_or_NOT_GIT_REPOSITORY": _current_commit_hash(),
        "source_snapshot_sha256": source_sha,
        "python_version": env["python_version"],
        "package_versions": env["package_versions"],
        "platform": env["platform"],
        "cpu_info_if_available": env["cpu_info_if_available"],
        "blas_thread_env": env["blas_thread_env"],
        "parallel_workers": int(parallel_workers),
        "seed_list": [int(s) for s in seeds],
        "problem_list": [_problem_label(p) for p in problems],
        "algorithm_list": [name for name, _ in algo_configs],
        "budget": budgets[0] if len(budgets) == 1 else budgets,
        "population_size": pop_sizes[0] if len(pop_sizes) == 1 else pop_sizes,
        "termination": "max_evals",
        "output_schema_version": "mosade-a1-a7-2026-05-05",
    }
    save_json(run_dir / "manifest.json", manifest)
    (run_dir / "command_line.txt").write_text(
        " ".join(sys.argv) + f"\nconfig={config_path}\nworkers={parallel_workers}\n",
        encoding="utf-8",
    )

    with open(run_dir / "run_matrix.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "problem",
                "algorithm",
                "algorithm_type",
                "seed",
                "run_index",
                "budget",
                "population_size",
            ],
        )
        writer.writeheader()
        for prob_entry in problems:
            problem = _problem_label(prob_entry)
            for algo_name, raw_params in algo_configs:
                params = dict(raw_params)
                algo_type = params.pop("type", algo_name)
                for idx, seed in enumerate(seeds):
                    writer.writerow({
                        "problem": problem,
                        "algorithm": algo_name,
                        "algorithm_type": algo_type,
                        "seed": int(seed),
                        "run_index": idx,
                        "budget": params.get("max_evals", ""),
                        "population_size": params.get("pop_size", ""),
                    })

    with open(run_dir / "failures_unsupported.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["problem", "algorithm", "seed", "status", "status_reason"],
        )
        writer.writeheader()


def _write_failure_unsupported_record(run_dir: Path) -> None:
    """Scan run metrics and rewrite the experiment-level failure/unsupported record."""
    rows: list[dict[str, Any]] = []
    for metrics_path in run_dir.glob("*/*/run_*/metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            rows.append({
                "problem": metrics_path.parents[2].name,
                "algorithm": metrics_path.parents[1].name,
                "seed": "",
                "status": "metrics_json_decode_failed",
                "status_reason": str(metrics_path),
            })
            continue
        status = str(metrics.get("status", "ok"))
        if status != "ok":
            rows.append({
                "problem": metrics_path.parents[2].name,
                "algorithm": metrics_path.parents[1].name,
                "seed": metrics.get("seed", ""),
                "status": status,
                "status_reason": metrics.get("status_reason", ""),
            })
    with open(run_dir / "failures_unsupported.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["problem", "algorithm", "seed", "status", "status_reason"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_run_metrics(
    problem: Any,
    result: Any,
    pf: np.ndarray | None,
    ref_point: np.ndarray,
    elapsed: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build the persisted metrics dict for a single completed run."""
    pf_X, pf_F = _formal_pf_from_result(problem, result)
    metadata = getattr(result, "metadata", {})
    pf_source = (
        metadata.get("pf_source", "final_population_feasible_nondominated")
        if isinstance(metadata, dict)
        else "final_population_feasible_nondominated"
    )

    final_cv = _final_cv_stats(problem, np.asarray(result.X, dtype=float))
    approx_F = np.asarray(pf_F, dtype=float)
    raw_F = np.asarray(result.F, dtype=float)

    m: dict[str, Any] = {
        "seed": int(getattr(result, "seed", -1)),
        "n_evals": result.n_evals,
        "time_s": elapsed,
        "status": result.status,
        "status_reason": result.message,
        "pf_source": pf_source,
        "pf_file": "pareto_approximation.txt",
        "pf_debug_file": "objectives.txt",
        "pf_n_points": int(approx_F.shape[0]),
        "n_raw_solutions": int(raw_F.shape[0]) if raw_F.ndim == 2 else int(np.atleast_2d(raw_F).shape[0]),
        "ref_point": np.asarray(ref_point, dtype=float).tolist(),
        **final_cv,
    }
    if isinstance(metadata, dict):
        for key in (
            "eps_mode",
            "adaptive_eps",
            "disable_epsilon",
            "epsilon_0",
            "epsilon_initial",
            "epsilon_T_c",
            "epsilon_final",
            "epsilon_min",
            "epsilon_max",
            "epsilon_num_updates",
            "epsilon_history_sha256",
            "constraint_handling_mode",
            "memory_scope",
            "memory_num_pools",
            "memory_update_count",
            "memory_success_count_by_strategy",
            "memory_sampling_count_by_strategy",
            "memory_history_digest",
            "restart_enabled",
            "restart_count",
            "restart_generation_or_eval",
            "selection_mode",
            "success_criterion",
            "telemetry_schema_version",
            "deprecated_variant_label",
            "adapt_fcr",
            "adapt_by_success_rate",
            "disable_credit",
            "fixed_F",
            "fixed_CR",
            "effective_pop_size",
        ):
            if key in metadata:
                m[key] = metadata[key]

    nan = float("nan")
    if result.status != "ok":
        for key in ("hv", "igd", "igd_plus", "gd", "spread"):
            m[key] = nan
        m["fe_80"] = None
        m["fe_95"] = None
        m["n_solutions"] = 0
        return m, pf_X, pf_F

    if approx_F.shape[0] > 0:
        m["hv"] = hypervolume(approx_F, ref_point)
        if pf is not None:
            m["igd"] = igd(approx_F, pf)
            m["igd_plus"] = igd_plus(approx_F, pf)
            m["gd"] = gd(approx_F, pf)
            m["spread"] = spread(approx_F, pf)
        else:
            m["igd"] = nan
            m["igd_plus"] = nan
            m["gd"] = nan
            m["spread"] = spread(approx_F)
    else:
        m["hv"] = 0.0
        m["igd"] = nan
        m["igd_plus"] = nan
        m["gd"] = nan
        m["spread"] = nan

    convergence = result.history.get("convergence", [])
    m["fe_80"] = _fe_to_threshold(convergence, m["hv"], 0.80)
    m["fe_95"] = _fe_to_threshold(convergence, m["hv"], 0.95)
    m["n_solutions"] = int(approx_F.shape[0])
    return m, pf_X, pf_F


def _build_run_metadata(
    problem_name: str,
    prob_cfg: dict[str, Any],
    problem: Any,
    algo_type: str,
    algo_params: dict[str, Any],
    algo: Any,
    result: Any,
    ref_point: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Build the audit metadata persisted beside each run."""
    result_metadata = getattr(result, "metadata", {})
    if not isinstance(result_metadata, dict):
        result_metadata = {}
    return {
        "problem": problem_name,
        "problem_params": prob_cfg,
        "n_var": int(problem.n_var),
        "n_obj": int(problem.n_obj),
        "n_constr": int(problem.n_constr),
        "algorithm_type": algo_type,
        "algorithm_variant": algo_type,
        "algorithm_params": algo_params,
        "epsilon_mode": result_metadata.get("eps_mode", "NA"),
        "adaptive_eps": result_metadata.get("adaptive_eps", "NA"),
        "epsilon_initial": result_metadata.get("epsilon_initial", "NA"),
        "epsilon_0": result_metadata.get("epsilon_0", "NA"),
        "epsilon_T_c": result_metadata.get("epsilon_T_c", "NA"),
        "epsilon_final": result_metadata.get("epsilon_final", "NA"),
        "epsilon_min": result_metadata.get("epsilon_min", "NA"),
        "epsilon_max": result_metadata.get("epsilon_max", "NA"),
        "epsilon_num_updates": result_metadata.get("epsilon_num_updates", "NA"),
        "epsilon_history_sha256": result_metadata.get("epsilon_history_sha256", "NA"),
        "constraint_handling_mode": result_metadata.get("constraint_handling_mode", "NA"),
        "memory_scope": result_metadata.get("memory_scope", "NA"),
        "memory_num_pools": result_metadata.get("memory_num_pools", "NA"),
        "memory_update_count": result_metadata.get("memory_update_count", "NA"),
        "memory_success_count_by_strategy": result_metadata.get(
            "memory_success_count_by_strategy", "NA"
        ),
        "memory_sampling_count_by_strategy": result_metadata.get(
            "memory_sampling_count_by_strategy", "NA"
        ),
        "memory_history_digest": result_metadata.get("memory_history_digest", "NA"),
        "restart_enabled": result_metadata.get("restart_enabled", "NA"),
        "restart_count": result_metadata.get("restart_count", "NA"),
        "restart_generation_or_eval": result_metadata.get("restart_generation_or_eval", "NA"),
        "selection_mode": result_metadata.get("selection_mode", "NA"),
        "success_criterion": result_metadata.get("success_criterion", "NA"),
        "telemetry_schema_version": result_metadata.get("telemetry_schema_version", "NA"),
        "deprecated_variant_label": result_metadata.get("deprecated_variant_label"),
        "adapt_fcr": result_metadata.get("adapt_fcr", "NA"),
        "adapt_by_success_rate": result_metadata.get("adapt_by_success_rate", "NA"),
        "disable_credit": result_metadata.get("disable_credit", "NA"),
        "fixed_F": result_metadata.get("fixed_F", "NA"),
        "fixed_CR": result_metadata.get("fixed_CR", "NA"),
        "seed": int(seed),
        "budget": int(getattr(algo, "max_evals", result.n_evals)),
        "pop_size_requested": int(getattr(algo, "pop_size", -1)),
        "effective_pop_size": result_metadata.get("effective_pop_size", "NA"),
        "reference_point": np.asarray(ref_point, dtype=float).tolist(),
        "metric_implementation": {
            "hv": "mosade.metrics.hypervolume.hypervolume",
            "igd": "mosade.metrics.igd.igd",
            "igd_plus": "mosade.metrics.igd.igd_plus",
            "gd": "mosade.metrics.igd.gd",
            "spread": "mosade.metrics.spread.spread",
        },
        "commit_hash": _current_commit_hash(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "blas_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "UNSET"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "UNSET"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "UNSET"),
        },
    }


def _run_one_seed(
    problem_name: str,
    prob_cfg: dict[str, Any],
    pf: np.ndarray | None,
    ref_point: np.ndarray,
    algo_type: str,
    algo_params: dict[str, Any],
    run_subdir: str | Path,
    seed: int,
) -> dict[str, Any]:
    """Execute one run and persist its outputs. Safe for process workers."""
    run_subdir = Path(run_subdir)
    metrics_path = run_subdir / "metrics.json"
    objectives_path = run_subdir / "objectives.txt"
    decisions_path = run_subdir / "decisions.txt"
    history_path = run_subdir / "history.json"
    metadata_path = run_subdir / "run_metadata.json"
    if (
        metrics_path.exists()
        and objectives_path.exists()
        and decisions_path.exists()
        and history_path.exists()
        and metadata_path.exists()
    ):
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    problem = get_problem(problem_name, **prob_cfg)
    algo_class = ALGORITHM_REGISTRY.get(algo_type, MOSADE)
    algo = algo_class(seed=seed, **algo_params)
    t0 = time.perf_counter()
    result = algo.run(problem, pf=pf, ref_point=ref_point)
    elapsed = time.perf_counter() - t0

    run_subdir.mkdir(parents=True, exist_ok=True)
    save_objectives(run_subdir / "objectives.txt", result.F)
    save_objectives(run_subdir / "decisions.txt", result.X)
    save_json(run_subdir / "history.json", result.history)
    epsilon_history = [float(v) for v in result.history.get("epsilon", [])]
    epsilon_payload = {
        "epsilon_mode": getattr(result, "metadata", {}).get("eps_mode", "NA"),
        "epsilon_initial": getattr(result, "metadata", {}).get("epsilon_initial", "NA"),
        "epsilon_history": epsilon_history,
    }
    epsilon_payload["epsilon_history_sha256"] = hashlib.sha256(
        json.dumps(
            epsilon_history,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    save_json(run_subdir / "epsilon_history.json", epsilon_payload)
    save_json(
        metadata_path,
        _build_run_metadata(
            problem_name=problem_name,
            prob_cfg=prob_cfg,
            problem=problem,
            algo_type=algo_type,
            algo_params=algo_params,
            algo=algo,
            result=result,
            ref_point=ref_point,
            seed=seed,
        ),
    )

    m, pf_X, pf_F = _build_run_metrics(problem, result, pf, ref_point, elapsed)
    m["seed"] = int(seed)
    save_objectives(run_subdir / "pareto_approximation.txt", pf_F)
    save_objectives(run_subdir / "pareto_approximation_decisions.txt", pf_X)
    save_json(run_subdir / "metrics.json", m)
    return m


# ---------------------------------------------------------------------------
# Per-run execution helper
# ---------------------------------------------------------------------------


def _unsupported_reason(
    algo_class: type,
    algo_params: dict[str, Any],
    seed: int,
    problem: Any,
) -> str | None:
    """Return an algorithm/problem unsupported reason when available."""
    algo = algo_class(seed=seed, **algo_params)
    checker = getattr(algo, "unsupported_reason", None)
    if callable(checker):
        return checker(problem)
    return None


def _write_unsupported_runs(
    problem: Any,
    run_parent_dir: Path,
    seeds: list[int],
    n_runs: int,
    reason: str,
) -> list[dict[str, Any]]:
    """Write explicit unsupported placeholders for an algorithm/problem pair."""
    nan = float("nan")
    run_metrics: list[dict[str, Any]] = []

    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        run_subdir = run_parent_dir / f"run_{run_idx:03d}"
        run_subdir.mkdir(parents=True, exist_ok=True)
        save_objectives(run_subdir / "objectives.txt", np.empty((0, problem.n_obj)))
        save_objectives(run_subdir / "decisions.txt", np.empty((0, problem.n_var)))
        save_objectives(run_subdir / "pareto_approximation.txt", np.empty((0, problem.n_obj)))
        save_objectives(
            run_subdir / "pareto_approximation_decisions.txt",
            np.empty((0, problem.n_var)),
        )
        save_json(
            run_subdir / "history.json",
            {"status": "unsupported", "status_reason": reason},
        )
        save_json(
            run_subdir / "epsilon_history.json",
            {
                "epsilon_mode": "unsupported",
                "epsilon_initial": None,
                "epsilon_history": [],
                "epsilon_history_sha256": hashlib.sha256(b"[]").hexdigest(),
            },
        )
        save_json(
            run_subdir / "run_metadata.json",
            {
                "status": "unsupported",
                "status_reason": reason,
                "seed": int(seed),
                "problem": problem.__class__.__name__,
                "commit_hash": _current_commit_hash(),
                "python_version": platform.python_version(),
            },
        )

        metrics = {
            "seed": seed,
            "n_evals": 0,
            "time_s": 0.0,
            "status": "unsupported",
            "status_reason": reason,
            "hv": nan,
            "igd": nan,
            "igd_plus": nan,
            "gd": nan,
            "spread": nan,
            "feasibility_ratio": nan,
            "best_cv": nan,
            "median_cv": nan,
            "n_feasible_final": nan,
            "fe_80": None,
            "fe_95": None,
            "n_solutions": nan,
            "n_raw_solutions": nan,
            "pf_source": "unsupported",
            "pf_file": "pareto_approximation.txt",
            "pf_debug_file": "objectives.txt",
            "pf_n_points": nan,
        }
        save_json(run_subdir / "metrics.json", metrics)
        run_metrics.append(metrics)

    return run_metrics


def _run_seeds(
    problem_name: str,
    prob_cfg: dict[str, Any],
    problem: Any,
    pf: np.ndarray | None,
    ref_point: np.ndarray,
    algo_class: type,
    algo_type: str,
    algo_params: dict,
    run_parent_dir: Path,
    seeds: list[int],
    n_runs: int,
    logger: logging.Logger,
    algo_name: str,
    problem_dir_name: str,
    parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    """Execute *n_runs* independent seeds and persist results."""
    unsupported_reason = _unsupported_reason(algo_class, algo_params, seeds[0], problem)
    if unsupported_reason is not None:
        logger.warning(
            "%s unsupported on constrained problem %s; skipping this algorithm/problem pair",
            algo_name,
            problem_dir_name,
        )
        return _write_unsupported_runs(
            problem=problem,
            run_parent_dir=run_parent_dir,
            seeds=seeds,
            n_runs=n_runs,
            reason=unsupported_reason,
        )

    run_metrics: list[dict[str, Any] | None] = [None] * n_runs
    pending: list[tuple[int, int, Path]] = []
    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        run_subdir = run_parent_dir / f"run_{run_idx:03d}"
        metrics_path = run_subdir / "metrics.json"
        objectives_path = run_subdir / "objectives.txt"
        decisions_path = run_subdir / "decisions.txt"
        history_path = run_subdir / "history.json"
        metadata_path = run_subdir / "run_metadata.json"
        if (
            metrics_path.exists()
            and objectives_path.exists()
            and decisions_path.exists()
            and history_path.exists()
            and metadata_path.exists()
        ):
            logger.info(
                "    Run %d/%d  (seed=%d) already complete; skipping",
                run_idx + 1,
                n_runs,
                seed,
            )
            with open(metrics_path, encoding="utf-8") as f:
                run_metrics[run_idx] = json.load(f)
        else:
            pending.append((run_idx, seed, run_subdir))

    if not pending:
        return [m for m in run_metrics if m is not None]

    workers = max(1, min(int(parallel_workers), len(pending)))
    if workers <= 1:
        for run_idx, seed, run_subdir in pending:
            logger.info("    Run %d/%d  (seed=%d)", run_idx + 1, n_runs, seed)
            m = _run_one_seed(
                problem_name=problem_name,
                prob_cfg=prob_cfg,
                pf=pf,
                ref_point=ref_point,
                algo_type=algo_type,
                algo_params=algo_params,
                run_subdir=run_subdir,
                seed=seed,
            )
            run_metrics[run_idx] = m
            logger.info(
                "      status=%s  HV=%.6f  IGD=%.6f  time=%.2fs  n_sol=%s  feas=%.3f",
                m.get("status", "ok"),
                m.get("hv", float("nan")),
                m.get("igd", float("nan")),
                float(m.get("time_s", float("nan"))),
                m.get("n_solutions", 0),
                m.get("feasibility_ratio", float("nan")),
            )
        return [m for m in run_metrics if m is not None]

    logger.info("    Parallel execution: %d worker processes", workers)
    ctx = mp.get_context("spawn")
    future_to_meta: dict[cf.Future, tuple[int, int]] = {}
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_worker_init) as ex:
        for run_idx, seed, run_subdir in pending:
            logger.info("    Queue run %d/%d  (seed=%d)", run_idx + 1, n_runs, seed)
            fut = ex.submit(
                _run_one_seed,
                problem_name,
                prob_cfg,
                pf,
                ref_point,
                algo_type,
                algo_params,
                str(run_subdir),
                seed,
            )
            future_to_meta[fut] = (run_idx, seed)

        for fut in cf.as_completed(future_to_meta):
            run_idx, seed = future_to_meta[fut]
            try:
                m = fut.result()
            except Exception:
                logger.exception(
                    "Run %d/%d  (seed=%d) failed in worker process",
                    run_idx + 1,
                    n_runs,
                    seed,
                )
                raise
            run_metrics[run_idx] = m
            logger.info(
                "      done run %d/%d  (seed=%d)  status=%s  HV=%.6f  IGD=%.6f  time=%.2fs  n_sol=%s  feas=%.3f",
                run_idx + 1,
                n_runs,
                seed,
                m.get("status", "ok"),
                m.get("hv", float("nan")),
                m.get("igd", float("nan")),
                float(m.get("time_s", float("nan"))),
                m.get("n_solutions", 0),
                m.get("feasibility_ratio", float("nan")),
            )

    return [m for m in run_metrics if m is not None]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_experiment(
    config_path: str | Path,
    run_dir: str | Path | None = None,
    workers: int | None = None,
) -> Path:
    """Execute a full experiment from a YAML config file."""
    cfg = load_config(config_path)
    results_base = Path(cfg.get("results_dir", "results"))
    tag = cfg.get("tag", "experiment")
    if run_dir is None:
        run_dir = make_run_dir(results_base, tag)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("mosade", log_file=run_dir / "experiment.log")
    logger.info("Experiment config: %s", config_path)
    logger.info("Results directory: %s", run_dir)

    save_json(run_dir / "config.json", cfg)

    problems = cfg.get("problems", ["ZDT1"])
    configured_n_runs = cfg.get("n_runs")
    n_runs = configured_n_runs if configured_n_runs is not None else 5
    base_seed = cfg.get("seed", 42)
    parallel_workers = int(workers if workers is not None else cfg.get("parallel_workers", 1))

    algo_configs, is_multi = _parse_algo_configs(cfg)
    if "seeds" in cfg:
        seeds = [int(seed) for seed in cfg["seeds"]]
        if configured_n_runs is not None and int(configured_n_runs) != len(seeds):
            raise ValueError(
                f"n_runs={configured_n_runs} does not match len(seeds)={len(seeds)}"
            )
        n_runs = len(seeds)
    else:
        seeds = seed_sequence(base_seed, int(n_runs))
    _write_experiment_provenance(
        run_dir=run_dir,
        config_path=config_path,
        cfg=cfg,
        algo_configs=algo_configs,
        problems=problems,
        seeds=seeds,
        parallel_workers=parallel_workers,
    )

    logger.info("Independent runs per algorithm/problem: %d", n_runs)
    logger.info("Configured parallel workers: %d", parallel_workers)
    if is_multi:
        algo_names = [name for name, _ in algo_configs]
        logger.info("Multi-algorithm mode: %s", algo_names)
    else:
        logger.info("Single-algorithm mode")

    summary: dict[str, Any] = {}

    for prob_entry in problems:
        prob_cfg: dict[str, Any] = {}
        if isinstance(prob_entry, dict):
            prob_cfg = {k: v for k, v in prob_entry.items() if k != "name"}
            prob_name = prob_entry["name"]
            param_suffix = "_".join(f"{k}{v}" for k, v in sorted(prob_cfg.items()))
            dir_name = f"{prob_name}_{param_suffix}" if param_suffix else prob_name
        else:
            prob_name = str(prob_entry)
            dir_name = prob_name

        logger.info("=== Problem: %s ===", dir_name)
        prob_dir = run_dir / dir_name
        prob_dir.mkdir(parents=True, exist_ok=True)

        problem = get_problem(prob_name, **prob_cfg)
        pf = _get_reference_front(problem)
        ref_point = _estimate_reference_point(problem, pf)
        save_objectives(prob_dir / "reference_point.txt", np.atleast_2d(ref_point))
        if pf is not None:
            save_objectives(prob_dir / "pareto_front.txt", pf)

        if is_multi:
            prob_summary: dict[str, Any] = {}
            for algo_name, algo_params in algo_configs:
                algo_params = dict(algo_params)
                algo_type = algo_params.pop("type", algo_name)
                algo_class = ALGORITHM_REGISTRY.get(algo_type, MOSADE)
                logger.info("  Algorithm: %s (%s)", algo_name, algo_type)
                algo_dir = prob_dir / algo_name
                algo_dir.mkdir(parents=True, exist_ok=True)

                run_metrics = _run_seeds(
                    problem_name=prob_name,
                    prob_cfg=prob_cfg,
                    problem=problem,
                    pf=pf,
                    ref_point=ref_point,
                    algo_class=algo_class,
                    algo_type=algo_type,
                    algo_params=algo_params,
                    run_parent_dir=algo_dir,
                    seeds=seeds,
                    n_runs=n_runs,
                    logger=logger,
                    algo_name=algo_name,
                    problem_dir_name=dir_name,
                    parallel_workers=parallel_workers,
                )

                agg = _aggregate_metrics(run_metrics)
                save_json(algo_dir / "summary.json", agg)
                prob_summary[algo_name] = agg
                logger.info(
                    "  [%s] HV=%.6f±%.6f  IGD=%.6f±%.6f",
                    algo_name,
                    agg.get("hv_mean", 0),
                    agg.get("hv_std", 0),
                    agg.get("igd_mean", float("nan")),
                    agg.get("igd_std", float("nan")),
                )

            save_json(prob_dir / "summary.json", prob_summary)
            summary[dir_name] = prob_summary
        else:
            _, algo_params = algo_configs[0]
            algo_params = dict(algo_params)
            algo_type = algo_params.pop("type", "MOSADE")
            algo_class = ALGORITHM_REGISTRY.get(algo_type, MOSADE)
            run_metrics = _run_seeds(
                problem_name=prob_name,
                prob_cfg=prob_cfg,
                problem=problem,
                pf=pf,
                ref_point=ref_point,
                algo_class=algo_class,
                algo_type=algo_type,
                algo_params=algo_params,
                run_parent_dir=prob_dir,
                seeds=seeds,
                n_runs=n_runs,
                logger=logger,
                algo_name=algo_type,
                problem_dir_name=dir_name,
                parallel_workers=parallel_workers,
            )
            agg = _aggregate_metrics(run_metrics)
            save_json(prob_dir / "summary.json", agg)
            summary[dir_name] = agg
            logger.info(
                "  Summary: HV=%.6f±%.6f  IGD=%.6f±%.6f",
                agg.get("hv_mean", 0),
                agg.get("hv_std", 0),
                agg.get("igd_mean", float("nan")),
                agg.get("igd_std", float("nan")),
            )

    save_json(run_dir / "summary.json", summary)
    _write_failure_unsupported_record(run_dir)
    logger.info("Experiment complete.  Results in %s", run_dir)
    return run_dir


def _aggregate_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute mean, std, median, and IQR for each numeric metric across runs."""
    agg: dict[str, Any] = {}
    statuses = [str(r.get("status", "ok")) for r in runs]
    reasons = sorted({str(r["status_reason"]) for r in runs if r.get("status_reason")})
    if statuses and all(status == "unsupported" for status in statuses):
        agg["status"] = "unsupported"
    elif any(status != "ok" for status in statuses):
        agg["status"] = "partial"
    else:
        agg["status"] = "ok"
    agg["n_ok"] = sum(status == "ok" for status in statuses)
    agg["n_unsupported"] = sum(status == "unsupported" for status in statuses)
    if reasons:
        agg["status_reason"] = reasons[0] if len(reasons) == 1 else reasons

    keys = [k for k in runs[0] if k != "seed" and isinstance(runs[0][k], (int, float))]
    import warnings as _warnings

    for k in keys:
        vals = np.array([r[k] for r in runs if k in r and isinstance(r[k], (int, float))], dtype=float)
        if len(vals) == 0:
            continue
        n_valid = int(np.sum(~np.isnan(vals)))
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            agg[f"{k}_mean"] = float(np.nanmean(vals))
            agg[f"{k}_std"] = float(np.nanstd(vals))
            agg[f"{k}_median"] = float(np.nanmedian(vals))
            if n_valid >= 2:
                q25, q75 = np.nanpercentile(vals, [25, 75])
            else:
                q25 = q75 = float("nan")
        agg[f"{k}_iqr"] = float(q75 - q25)
        if n_valid < len(runs):
            agg[f"{k}_n_valid"] = n_valid
    agg["n_runs"] = len(runs)
    return agg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a MOSADE experiment")
    parser.add_argument(
        "--config", "-c",
        default="configs/smoke_test.yaml",
        help="Path to YAML config file (default: configs/smoke_test.yaml)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Override YAML parallel_workers with this many process workers",
    )
    args = parser.parse_args()
    run_dir = run_experiment(args.config, workers=args.workers)
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
