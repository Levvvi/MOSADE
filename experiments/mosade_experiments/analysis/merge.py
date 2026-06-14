"""Helpers for merging multiple suite-level result directories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from mosade.utils.io import load_config, save_json

_SKIP_TOP_LEVEL_DIRS = frozenset({"plots", "tables", "__pycache__"})


def _iter_problem_dirs(results_dir: Path) -> list[Path]:
    """Return problem directories from an experiment results directory."""
    problem_dirs: list[Path] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or child.name in _SKIP_TOP_LEVEL_DIRS:
            continue
        if (child / "summary.json").exists():
            problem_dirs.append(child)
    return problem_dirs


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dict."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _algorithm_names_from_config(cfg: dict[str, Any]) -> list[str] | None:
    """Extract the configured algorithm names in order, if present."""
    if "algorithms" in cfg:
        return [str(entry["name"]) for entry in cfg["algorithms"]]
    if "algorithm" in cfg:
        return ["MOSADE"]
    return None


def _config_summary(results_dir: Path) -> dict[str, Any]:
    """Return the frozen-config summary for one results directory."""
    config_path = results_dir / "config.json"
    cfg = load_config(config_path) if config_path.exists() else {}
    return {
        "config_path": str(config_path) if config_path.exists() else None,
        "tag": cfg.get("tag"),
        "seed": cfg.get("seed"),
        "n_runs": cfg.get("n_runs"),
        "algorithms": _algorithm_names_from_config(cfg),
    }


def _validate_compatibility(configs: list[dict[str, Any]]) -> tuple[int | None, int | None, list[str] | None]:
    """Validate cross-suite seed / n_runs / algorithm-order compatibility."""
    seeds = {cfg["seed"] for cfg in configs if cfg["seed"] is not None}
    n_runs = {cfg["n_runs"] for cfg in configs if cfg["n_runs"] is not None}
    algo_lists = {tuple(cfg["algorithms"]) for cfg in configs if cfg["algorithms"] is not None}

    if len(seeds) > 1:
        raise ValueError(f"Mismatched seeds across suite results: {sorted(seeds)}")
    if len(n_runs) > 1:
        raise ValueError(f"Mismatched n_runs across suite results: {sorted(n_runs)}")
    if len(algo_lists) > 1:
        raise ValueError("Algorithm lists differ across suite results; cannot build a unified benchmark merge")

    seed = next(iter(seeds)) if seeds else None
    runs = next(iter(n_runs)) if n_runs else None
    algos = list(next(iter(algo_lists))) if algo_lists else None
    return seed, runs, algos


def merge_results_dirs(
    source_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Merge multiple suite-level result directories into one benchmark directory.

    The merged directory keeps the normal per-problem structure so downstream
    plotting/statistics/table scripts can run unchanged.
    """
    sources = [Path(path).resolve() for path in source_dirs]
    if not sources:
        raise ValueError("At least one source results directory is required")
    for src in sources:
        if not src.exists():
            raise FileNotFoundError(f"Results directory not found: {src}")

    output = Path(output_dir).resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    configs = [_config_summary(src) for src in sources]
    seed, n_runs, algorithms = _validate_compatibility(configs)

    top_summary: dict[str, Any] = {}
    merged_problems: list[str] = []
    source_entries: list[dict[str, Any]] = []

    for src, cfg in zip(sources, configs):
        problem_dirs = _iter_problem_dirs(src)
        source_problem_names = [prob_dir.name for prob_dir in problem_dirs]

        for prob_dir in problem_dirs:
            dst_prob_dir = output / prob_dir.name
            if dst_prob_dir.exists():
                raise ValueError(
                    f"Duplicate problem '{prob_dir.name}' encountered while merging {src} into {output}"
                )
            shutil.copytree(prob_dir, dst_prob_dir)
            merged_problems.append(prob_dir.name)
            summary_path = prob_dir / "summary.json"
            if summary_path.exists():
                top_summary[prob_dir.name] = _load_json(summary_path)

        source_entries.append(
            {
                "results_dir": str(src),
                "tag": cfg["tag"],
                "seed": cfg["seed"],
                "n_runs": cfg["n_runs"],
                "algorithms": cfg["algorithms"],
                "problems": source_problem_names,
            }
        )

    manifest = {
        "merged_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output),
        "seed": seed,
        "n_runs": n_runs,
        "algorithms": algorithms,
        "problems": sorted(merged_problems),
        "sources": source_entries,
        "notes": [
            "Merged benchmark directory assembled from suite-level results.",
            "Per-suite budgets may differ; inspect the source suite configs for settings.",
            "This directory intentionally omits config.json so settings tables are not inferred from mixed budgets.",
        ],
    }

    save_json(output / "merge_manifest.json", manifest)
    save_json(output / "summary.json", top_summary)
    return manifest
