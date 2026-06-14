"""Build A1-A7 statistical CSV outputs from available result directories."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from rebuild_tables_from_raw_sources import DIRECTION, METRICS, _scan_result_dir, _stats, _summary


ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "tables"
AUDIT = ROOT / "audit"

GROUPS = {
    "a1_epsilon_modes": {
        "dirs": ["results/a1_ablation_epsilon_modes"],
        "metrics": "a1_epsilon_modes_metrics.csv",
        "summary": "a1_epsilon_modes_summary.csv",
        "stats": "a1_epsilon_modes_pairwise_stats.csv",
        "effects": "a1_epsilon_modes_effect_sizes.csv",
    },
    "a3_memory_ablation": {
        "dirs": ["results/a3_ablation_memory"],
        "metrics": "a3_memory_ablation_metrics.csv",
        "summary": "a3_memory_ablation_summary.csv",
        "stats": "a3_memory_ablation_pairwise_stats.csv",
        "effects": "a3_memory_ablation_effect_sizes.csv",
    },
    "a4_restart_ablation": {
        "dirs": ["results/a4_ablation_restart"],
        "metrics": "a4_restart_ablation_metrics.csv",
        "summary": "a4_restart_ablation_summary.csv",
        "stats": "a4_restart_ablation_pairwise_stats.csv",
        "effects": "a4_restart_ablation_effect_sizes.csv",
    },
    "a5_domselect_ablation": {
        "dirs": ["results/a5_ablation_domselect"],
        "metrics": "a5_domselect_ablation_metrics.csv",
        "summary": "a5_domselect_ablation_summary.csv",
        "stats": "a5_domselect_ablation_pairwise_stats.csv",
        "effects": "a5_domselect_ablation_effect_sizes.csv",
    },
    "e3_design_constants": {
        "dirs": ["results/e3_sensitivity_design_constants"],
        "metrics": "e3_design_constants_by_run.csv",
        "summary": "e3_design_constants_summary.csv",
        "stats": "e3_design_constants_pairwise_stats.csv",
        "effects": "e3_design_constants_effect_sizes.csv",
    },
}


BY_RUN_FIELDS = [
    "result_dir",
    "problem",
    "algorithm",
    "seed",
    "run_dir",
    "status",
    "status_reason",
    "n_evals",
    "n_solutions",
    "n_feasible_final",
    "feasibility_ratio",
    "pf_source",
    *METRICS,
]
SUMMARY_FIELDS = [
    "problem",
    "algorithm",
    "metric",
    "direction",
    "n_runs",
    "median",
    "iqr",
    "mean",
    "std",
    "min",
    "max",
    "n_unsupported",
    "n_failed",
]
STATS_FIELDS = [
    "problem",
    "metric",
    "reference_algorithm",
    "comparison_algorithm",
    "direction",
    "paired_or_unpaired",
    "test_name",
    "statistic",
    "p_value",
    "p_adjust_method",
    "p_adjusted",
    "significant_alpha_0p05",
    "effect_size_name",
    "effect_size",
    "n_reference",
    "n_comparison",
    "unsupported_policy",
    "missing_policy",
    "seed_pairing_policy",
]


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _holm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finite = [
        (idx, float(row["p_value"]))
        for idx, row in enumerate(rows)
        if _is_finite(row.get("p_value"))
    ]
    adjusted = {idx: float("nan") for idx, _ in finite}
    m = len(finite)
    running_max = 0.0
    for rank, (idx, p) in enumerate(sorted(finite, key=lambda item: item[1])):
        adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    for idx, row in enumerate(rows):
        row["p_adjust_method"] = "Holm"
        row["p_adjusted"] = adjusted.get(idx, "")
        row["significant_alpha_0p05"] = (
            bool(_is_finite(row["p_adjusted"]) and float(row["p_adjusted"]) < 0.05)
        )
    return rows


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _group_rows(dirs: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in dirs:
        rows.extend(_scan_result_dir(raw))
    return rows


def _effect_rows(stats_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "problem": row["problem"],
            "metric": row["metric"],
            "reference_algorithm": row["reference_algorithm"],
            "comparison_algorithm": row["comparison_algorithm"],
            "direction": row["direction"],
            "effect_size_name": row["effect_size_name"],
            "effect_size": row["effect_size"],
            "n_reference": row["n_reference"],
            "n_comparison": row["n_comparison"],
        }
        for row in stats_rows
    ]


def _runtime_tables() -> None:
    rows = []
    for raw, workers in [
        ("results/a7_runtime_workers1", 1),
        ("results/a7_runtime_workers6", 6),
    ]:
        for row in _scan_result_dir(raw):
            row["workers"] = workers
            row["wall_time_seconds"] = row.get("time_s", "")
            row["success_status"] = row.get("status", "ok")
            row["failure_reason"] = row.get("status_reason", "")
            rows.append(row)
    runtime_fields = [
        "result_dir",
        "problem",
        "algorithm",
        "seed",
        "workers",
        "wall_time_seconds",
        "n_evals",
        "success_status",
        "failure_reason",
    ]
    _write_csv(TABLES / "a7_runtime_by_run.csv", rows, runtime_fields)

    summary_rows = []
    groups = sorted({(r["problem"], r["algorithm"], r["workers"]) for r in rows})
    for problem, algorithm, workers in groups:
        vals = [
            float(r["wall_time_seconds"])
            for r in rows
            if (r["problem"], r["algorithm"], r["workers"]) == (problem, algorithm, workers)
            and _is_finite(r.get("wall_time_seconds"))
            and r.get("success_status") == "ok"
        ]
        arr = np.asarray(vals, dtype=float)
        q25, q75 = (np.percentile(arr, [25, 75]) if arr.size else (np.nan, np.nan))
        n_evals = [
            float(r["n_evals"])
            for r in rows
            if (r["problem"], r["algorithm"], r["workers"]) == (problem, algorithm, workers)
            and _is_finite(r.get("n_evals"))
        ]
        summary_rows.append({
            "problem": problem,
            "algorithm": algorithm,
            "workers": workers,
            "n_runs": int(arr.size),
            "median_time": float(np.median(arr)) if arr.size else "",
            "iqr_time": float(q75 - q25) if arr.size else "",
            "mean_time": float(np.mean(arr)) if arr.size else "",
            "std_time": float(np.std(arr, ddof=1)) if arr.size > 1 else "",
            "min_time": float(np.min(arr)) if arr.size else "",
            "max_time": float(np.max(arr)) if arr.size else "",
            "median_n_evals": float(np.median(n_evals)) if n_evals else "",
            "success_rate": float(arr.size / max(1, len([
                r for r in rows
                if (r["problem"], r["algorithm"], r["workers"]) == (problem, algorithm, workers)
            ]))),
        })
    _write_csv(
        TABLES / "a7_runtime_summary.csv",
        summary_rows,
        [
            "problem",
            "algorithm",
            "workers",
            "n_runs",
            "median_time",
            "iqr_time",
            "mean_time",
            "std_time",
            "min_time",
            "max_time",
            "median_n_evals",
            "success_rate",
        ],
    )

    speed_rows = []
    by_key = {(r["problem"], r["algorithm"], r["workers"]): r for r in summary_rows}
    for problem, algorithm in sorted({(r["problem"], r["algorithm"]) for r in summary_rows}):
        w1 = by_key.get((problem, algorithm, 1))
        w6 = by_key.get((problem, algorithm, 6))
        t1 = float(w1["median_time"]) if w1 and _is_finite(w1.get("median_time")) else math.nan
        t6 = float(w6["median_time"]) if w6 and _is_finite(w6.get("median_time")) else math.nan
        speedup = t1 / t6 if math.isfinite(t1) and math.isfinite(t6) and t6 > 0 else math.nan
        speed_rows.append({
            "problem": problem,
            "algorithm": algorithm,
            "median_time_workers1": t1 if math.isfinite(t1) else "",
            "median_time_workers6": t6 if math.isfinite(t6) else "",
            "speedup_median_w1_over_w6": speedup if math.isfinite(speedup) else "",
            "efficiency_vs_ideal_6x": speedup / 6.0 if math.isfinite(speedup) else "",
            "n_runs_workers1": w1["n_runs"] if w1 else 0,
            "n_runs_workers6": w6["n_runs"] if w6 else 0,
            "status": "ok" if w1 and w6 and w1["n_runs"] and w6["n_runs"] else "missing",
        })
    _write_csv(
        TABLES / "a7_runtime_speedup_workers6_vs_workers1.csv",
        speed_rows,
        [
            "problem",
            "algorithm",
            "median_time_workers1",
            "median_time_workers6",
            "speedup_median_w1_over_w6",
            "efficiency_vs_ideal_6x",
            "n_runs_workers1",
            "n_runs_workers6",
            "status",
        ],
    )


def _a6_crosscheck() -> None:
    rows = []
    for group, raw in [
        ("A1", "results/a1_ablation_epsilon_modes"),
        ("A3", "results/a3_ablation_memory"),
        ("A4", "results/a4_ablation_restart"),
        ("A5", "results/a5_ablation_domselect"),
        ("E3", "results/e3_sensitivity_design_constants"),
    ]:
        scanned = [r for r in _scan_result_dir(raw) if r["problem"] == "DTLZ2_n_obj3"]
        for algorithm in sorted({r["algorithm"] for r in scanned}):
            seeds = sorted({str(r["seed"]) for r in scanned if r["algorithm"] == algorithm})
            rows.append({
                "ablation_group": group,
                "problem": "DTLZ2_n_obj3",
                "algorithm_or_variant": algorithm,
                "seed_count_expected": 11 if group == "E3" else 31,
                "seed_count_completed": len(seeds),
                "missing_seeds": "",
                "hv_ref_point": "see result problem reference_point.txt",
                "pf_source": "problem.pareto_front or pymoo fallback",
                "status": "ok" if len(seeds) == (11 if group == "E3" else 31) else "missing_or_incomplete",
            })
        if not scanned:
            rows.append({
                "ablation_group": group,
                "problem": "DTLZ2_n_obj3",
                "algorithm_or_variant": "",
                "seed_count_expected": 11 if group == "E3" else 31,
                "seed_count_completed": 0,
                "missing_seeds": "all",
                "hv_ref_point": "",
                "pf_source": "",
                "status": "missing",
            })
    _write_csv(
        TABLES / "a6_dtlz2_ablation_crosscheck.csv",
        rows,
        [
            "ablation_group",
            "problem",
            "algorithm_or_variant",
            "seed_count_expected",
            "seed_count_completed",
            "missing_seeds",
            "hv_ref_point",
            "pf_source",
            "status",
        ],
    )


def main() -> None:
    TABLES.mkdir(exist_ok=True)
    AUDIT.mkdir(exist_ok=True)
    manifest_rows = []
    for group, spec in GROUPS.items():
        rows = _group_rows(spec["dirs"])
        summary = _summary(rows)
        stats_rows = _holm(_stats(rows))
        effects = _effect_rows(stats_rows)
        _write_csv(TABLES / spec["metrics"], rows, BY_RUN_FIELDS)
        _write_csv(TABLES / spec["summary"], summary, SUMMARY_FIELDS)
        _write_csv(TABLES / spec["stats"], stats_rows, STATS_FIELDS)
        _write_csv(
            TABLES / spec["effects"],
            effects,
            [
                "problem",
                "metric",
                "reference_algorithm",
                "comparison_algorithm",
                "direction",
                "effect_size_name",
                "effect_size",
                "n_reference",
                "n_comparison",
            ],
        )
        manifest_rows.append({
            "table_group": group,
            "metric_direction": json_directions(),
            "paired_or_unpaired": "paired by seed when complete, otherwise unpaired",
            "test_name": "Wilcoxon signed-rank or Mann-Whitney U",
            "p_adjust_method": "Holm",
            "alpha": 0.05,
            "effect_size_name": "Cliff's delta",
            "unsupported_policy": "excluded_as_NA",
            "missing_policy": "excluded_and_counted",
            "seed_pairing_policy": "same_seed_when_complete",
        })

    _runtime_tables()
    _a6_crosscheck()
    _write_csv(
        TABLES / "statistical_test_manifest.csv",
        manifest_rows,
        [
            "table_group",
            "metric_direction",
            "paired_or_unpaired",
            "test_name",
            "p_adjust_method",
            "alpha",
            "effect_size_name",
            "unsupported_policy",
            "missing_policy",
            "seed_pairing_policy",
        ],
    )
    (AUDIT / "statistical_methods_summary_cn.md").write_text(
        "\n".join(
            [
                "# Statistical methods summary",
                "",
                "- 同一 problem / seed 的完整配对使用 Wilcoxon signed-rank test。",
                "- 不完整配对使用 Mann-Whitney U，并在表中记录为 unpaired。",
                "- 多重比较使用 Holm correction。",
                "- 效应量使用 Cliff's delta，方向为 comparison minus reference。",
                "- unsupported、missing、failed run 保留 NA，不填 0，不进入统计。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def json_directions() -> str:
    return ";".join(f"{metric}:{direction}" for metric, direction in DIRECTION.items())


if __name__ == "__main__":
    main()
