"""Build E3 clean statistics, warning diagnostics, and sensitivity indices."""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from global_without_a7_common import (
    AUDIT,
    DIRECTION,
    METRICS,
    TABLES,
    e3_child_runs,
    e3_formal_source_sha,
    e3_long_metric_rows,
    ensure_dirs,
    finite_float,
    provenance_fields,
    read_csv,
    values_by_seed,
    write_csv,
    write_text,
)


PAIRWISE_FIELDS = [
    "constant_name",
    "problem",
    "metric",
    "direction",
    "reference_level",
    "comparison_level",
    "n_ref",
    "n_cmp",
    "n_pairs",
    "n_nonzero_differences",
    "n_nan_ref",
    "n_nan_cmp",
    "n_inf_ref",
    "n_inf_cmp",
    "n_unique_ref",
    "n_unique_cmp",
    "test_name",
    "paired",
    "test_executed",
    "p_value_raw",
    "p_value_holm",
    "holm_included",
    "effect_size_name",
    "effect_size",
    "effect_direction",
    "status",
    "reason_class",
    "report_status",
    "interpretation",
    "a1_a5_formal_experiment_source_snapshot_sha256",
    "a1_a5_postprocessing_source_snapshot_sha256",
    "e3_formal_experiment_source_snapshot_sha256",
    "global_postprocessing_source_snapshot_sha256",
]

WARNING_FIELDS = [
    "source_script",
    "table",
    "experiment_group",
    "problem",
    "metric",
    "reference_algorithm_or_variant",
    "comparison_algorithm_or_variant",
    "n_ref",
    "n_cmp",
    "n_pairs",
    "n_nonzero_differences",
    "test_name",
    "warning_type",
    "reason_class",
    "action_taken",
    "report_status",
]


def _raw_metric_counts(
    rows: list[dict[str, Any]],
    constant: str,
    level: str,
    problem: str,
    metric: str,
) -> dict[str, int]:
    subset = [
        row for row in rows
        if row["constant_name"] == constant
        and row["constant_level"] == level
        and row["problem"] == problem
        and row["metric"] == metric
    ]
    counts = {"finite": 0, "nan": 0, "inf": 0}
    for row in subset:
        try:
            value = float(row.get("value", ""))
        except (TypeError, ValueError):
            counts["nan"] += 1
            continue
        if math.isnan(value):
            counts["nan"] += 1
        elif math.isinf(value):
            counts["inf"] += 1
        else:
            counts["finite"] += 1
    return counts


def _cliffs_delta(comparison: np.ndarray, reference: np.ndarray) -> float:
    if comparison.size == 0 or reference.size == 0:
        return float("nan")
    gt = sum(float(x > y) for x in comparison for y in reference)
    lt = sum(float(x < y) for x in comparison for y in reference)
    return float((gt - lt) / (comparison.size * reference.size))


def _effect_direction(effect_size: float, direction: str) -> str:
    if not math.isfinite(effect_size) or abs(effect_size) < 1e-12:
        return "no_effect"
    comparison_better = effect_size > 0 if direction == "higher" else effect_size < 0
    return "comparison_better" if comparison_better else "reference_better"


def _report_status(status: str, p_holm: float | str) -> str:
    if status == "test_ok":
        return "usable_statistical" if p_holm not in ("", None) else "usable_pending_holm"
    if status == "all_zero_differences_no_effect":
        return "usable_no_effect"
    if status in {"insufficient_sample", "insufficient_pairs", "nan_filtered_insufficient_sample"}:
        return "not_usable_statistical_failure"
    return "requires_manual_review"


def _interpretation(status: str, effect_direction: str, p_holm: float | str) -> str:
    if status == "all_zero_differences_no_effect":
        return "No non-zero paired differences; not an executed Wilcoxon test."
    if status != "test_ok":
        return f"Statistical test not usable: {status}."
    try:
        significant = float(p_holm) < 0.05
    except (TypeError, ValueError):
        significant = False
    if significant and effect_direction == "reference_better":
        return "Holm-significant; default level is better for this metric."
    if significant and effect_direction == "comparison_better":
        return "Holm-significant; comparison level is better for this metric."
    return "No Holm-significant difference; use descriptive sensitivity wording."


def _compare_pair(
    rows: list[dict[str, Any]],
    constant: str,
    problem: str,
    metric: str,
    comparison_level: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    direction = DIRECTION[metric]
    ref_counts = _raw_metric_counts(rows, constant, "default", problem, metric)
    cmp_counts = _raw_metric_counts(rows, constant, comparison_level, problem, metric)
    ref_by_seed = values_by_seed(rows, constant, "default", problem, metric)
    cmp_by_seed = values_by_seed(rows, constant, comparison_level, problem, metric)
    common = sorted(set(ref_by_seed) & set(cmp_by_seed))
    ref_vals = np.asarray([ref_by_seed[s] for s in common], dtype=float)
    cmp_vals = np.asarray([cmp_by_seed[s] for s in common], dtype=float)
    all_ref = np.asarray(list(ref_by_seed.values()), dtype=float)
    all_cmp = np.asarray(list(cmp_by_seed.values()), dtype=float)
    diff = cmp_vals - ref_vals
    nonzero = int(np.sum(np.abs(diff) > 1e-12))
    paired = len(common) >= 2 and len(common) == min(len(ref_by_seed), len(cmp_by_seed))
    test_name = "wilcoxon_signed_rank"
    status = "test_ok"
    reason_class = "none"
    test_executed = False
    holm_included = False
    p_raw: float | str = ""
    p_holm: float | str = ""
    warning_type = ""
    action = ""

    if ref_counts["nan"] or cmp_counts["nan"] or ref_counts["inf"] or cmp_counts["inf"]:
        reason_class = "nan_or_inf_values"
    if len(ref_by_seed) < 2 or len(cmp_by_seed) < 2:
        status = "insufficient_sample"
        reason_class = "insufficient_sample"
        warning_type = "SmallSampleWarning_prevented"
        action = "Skipped scipy test before call; sample size below 2."
    elif not paired:
        test_name = "mann_whitney_u"
        if len(all_ref) < 2 or len(all_cmp) < 2:
            status = "insufficient_sample"
            reason_class = "insufficient_sample"
            warning_type = "SmallSampleWarning_prevented"
            action = "Skipped Mann-Whitney U before call; sample size below 2."
        else:
            reason_class = "mismatched_seed_pairs"
            test_executed = True
            holm_included = True
    elif len(common) < 2:
        status = "insufficient_pairs"
        reason_class = "mismatched_seed_pairs"
        warning_type = "SmallSampleWarning_prevented"
        action = "Skipped Wilcoxon; fewer than two paired seeds."
    elif nonzero == 0:
        status = "all_zero_differences_no_effect"
        reason_class = "all_zero_differences"
        warning_type = "Wilcoxon_RuntimeWarning_prevented"
        action = "Skipped Wilcoxon; direct no-effect classification."
        p_raw = 1.0
        p_holm = 1.0
        test_executed = False
        holm_included = False
    elif nonzero < 2:
        status = "insufficient_sample"
        reason_class = "insufficient_sample"
        warning_type = "SmallSampleWarning_prevented"
        action = "Skipped Wilcoxon; fewer than two non-zero paired differences."
    else:
        test_executed = True
        holm_included = True

    if test_executed:
        try:
            from scipy import stats

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                if test_name == "wilcoxon_signed_rank":
                    result = stats.wilcoxon(ref_vals, cmp_vals, zero_method="wilcox")
                else:
                    result = stats.mannwhitneyu(all_ref, all_cmp, alternative="two-sided")
            p_raw = float(result.pvalue)
            if caught:
                warning_type = "|".join(type(w.message).__name__ for w in caught)
                action = "Warning captured and row retained for manual review."
                status = "warning_captured"
                reason_class = "unknown_requires_manual_review"
                holm_included = False
                p_holm = ""
        except Exception as exc:  # noqa: BLE001
            status = "statistical_exception"
            reason_class = "unknown_requires_manual_review"
            warning_type = exc.__class__.__name__
            action = f"Statistical call failed: {exc}"
            holm_included = False
            p_raw = ""
            p_holm = ""

    effect = _cliffs_delta(all_cmp, all_ref) if len(all_cmp) and len(all_ref) else float("nan")
    if status == "all_zero_differences_no_effect":
        effect = 0.0
    effect_direction = _effect_direction(effect, direction)
    row = {
        "constant_name": constant,
        "problem": problem,
        "metric": metric,
        "direction": direction,
        "reference_level": "default",
        "comparison_level": comparison_level,
        "n_ref": len(ref_by_seed),
        "n_cmp": len(cmp_by_seed),
        "n_pairs": len(common),
        "n_nonzero_differences": nonzero,
        "n_nan_ref": ref_counts["nan"],
        "n_nan_cmp": cmp_counts["nan"],
        "n_inf_ref": ref_counts["inf"],
        "n_inf_cmp": cmp_counts["inf"],
        "n_unique_ref": len(set(ref_by_seed.values())),
        "n_unique_cmp": len(set(cmp_by_seed.values())),
        "test_name": test_name,
        "paired": paired,
        "test_executed": test_executed,
        "p_value_raw": p_raw,
        "p_value_holm": p_holm,
        "holm_included": holm_included,
        "effect_size_name": "Cliff's delta",
        "effect_size": "" if not math.isfinite(effect) else effect,
        "effect_direction": effect_direction,
        "status": status,
        "reason_class": reason_class,
        "report_status": _report_status(status, p_holm),
        "interpretation": _interpretation(status, effect_direction, p_holm),
        **provenance_fields(),
    }
    diagnostic = None
    if warning_type:
        diagnostic = {
            "source_script": "scripts/build_e3_clean_stats.py",
            "table": "tables/e3_design_constants_pairwise_stats_clean.csv",
            "experiment_group": "E3",
            "problem": problem,
            "metric": metric,
            "reference_algorithm_or_variant": f"{constant}:default",
            "comparison_algorithm_or_variant": f"{constant}:{comparison_level}",
            "n_ref": len(ref_by_seed),
            "n_cmp": len(cmp_by_seed),
            "n_pairs": len(common),
            "n_nonzero_differences": nonzero,
            "test_name": test_name,
            "warning_type": warning_type,
            "reason_class": reason_class,
            "action_taken": action,
            "report_status": row["report_status"],
        }
    return row, diagnostic


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    families: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("holm_included") is True:
            families.setdefault((row["constant_name"], row["problem"], row["metric"]), []).append(row)
    for family_rows in families.values():
        sortable = []
        for row in family_rows:
            try:
                sortable.append((float(row["p_value_raw"]), row))
            except (TypeError, ValueError):
                row["holm_included"] = False
        sortable.sort(key=lambda x: x[0])
        m = len(sortable)
        running = 0.0
        for rank, (p_raw, row) in enumerate(sortable, start=1):
            adjusted = min(1.0, p_raw * (m - rank + 1))
            running = max(running, adjusted)
            row["p_value_holm"] = running
            row["report_status"] = _report_status(row["status"], running)
            row["interpretation"] = _interpretation(
                row["status"], row["effect_direction"], running
            )


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = {}
    value_lookup: dict[tuple[str, str], Any] = {}
    for row in rows:
        val = finite_float(row.get("value"))
        if val is None or row.get("status") != "ok":
            continue
        key = (row["constant_name"], row["constant_level"], row["problem"], row["metric"])
        grouped.setdefault(key, []).append(val)
        value_lookup[(row["constant_name"], row["constant_level"])] = row["constant_value"]
    medians = {key: float(np.median(vals)) for key, vals in grouped.items()}
    ranks: dict[tuple[str, str, str, str], int] = {}
    for constant in sorted({key[0] for key in grouped}):
        for problem in sorted({key[2] for key in grouped if key[0] == constant}):
            for metric in METRICS:
                keys = [key for key in grouped if key[0] == constant and key[2] == problem and key[3] == metric]
                reverse = DIRECTION[metric] == "higher"
                keys.sort(key=lambda k: medians[k], reverse=reverse)
                for rank, key in enumerate(keys, start=1):
                    ranks[key] = rank
    out: list[dict[str, Any]] = []
    for key, vals in sorted(grouped.items()):
        constant, level, problem, metric = key
        arr = np.asarray(vals, dtype=float)
        q1, q3 = np.percentile(arr, [25, 75])
        out.append({
            "constant_name": constant,
            "constant_level": level,
            "constant_value": value_lookup.get((constant, level), ""),
            "problem": problem,
            "metric": metric,
            "direction": DIRECTION[metric],
            "n_runs": len(arr),
            "median": float(np.median(arr)),
            "iqr": float(q3 - q1),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "rank": ranks.get(key, ""),
            "default_level_flag": level == "default",
            "best_flag": ranks.get(key) == 1,
            "status": "ok",
            **provenance_fields(),
        })
    return out


def _pairwise_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    constants = sorted({r["constant_name"] for r in rows})
    problems = sorted({r["problem"] for r in rows})
    pairwise: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for constant in constants:
        levels = sorted(
            {r["constant_level"] for r in rows if r["constant_name"] == constant},
            key=lambda x: {"low": 0, "default": 1, "high": 2}.get(x, 99),
        )
        for level in levels:
            if level == "default":
                continue
            for problem in problems:
                for metric in METRICS:
                    row, diagnostic = _compare_pair(rows, constant, problem, metric, level)
                    pairwise.append(row)
                    if diagnostic:
                        diagnostics.append(diagnostic)
    _holm_adjust(pairwise)
    return pairwise, diagnostics


def _rank_stability(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in summary:
        grouped.setdefault((row["constant_name"], row["problem"], row["metric"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (constant, problem, metric), rows in sorted(grouped.items()):
        ranks = [int(r["rank"]) for r in rows if str(r.get("rank", "")).isdigit()]
        default_rank = next((r["rank"] for r in rows if r["constant_level"] == "default"), "")
        best_level = next((r["constant_level"] for r in rows if str(r.get("rank")) == "1"), "")
        out.append({
            "constant_name": constant,
            "problem": problem,
            "metric": metric,
            "direction": DIRECTION[metric],
            "default_rank": default_rank,
            "best_level": best_level,
            "rank_min": min(ranks) if ranks else "",
            "rank_max": max(ranks) if ranks else "",
            "rank_range": (max(ranks) - min(ranks)) if ranks else "",
            **provenance_fields(),
        })
    return out


def _sensitivity_index(
    summary: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_by_key = {
        (r["constant_name"], r["constant_level"], r["problem"], r["metric"]): r for r in summary
    }
    pairs_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in pairwise:
        pairs_by_key.setdefault((row["constant_name"], row["problem"], row["metric"]), []).append(row)
    out: list[dict[str, Any]] = []
    for key, rows in sorted(pairs_by_key.items()):
        constant, problem, metric = key
        default = summary_by_key.get((constant, "default", problem, metric))
        if not default:
            continue
        default_median = finite_float(default.get("median"))
        if default_median is None:
            continue
        deltas = []
        rel_deltas = []
        better = 0
        worse = 0
        sig = 0
        max_effect = 0.0
        for pair in rows:
            level = pair["comparison_level"]
            comp = summary_by_key.get((constant, level, problem, metric))
            comp_median = finite_float(comp.get("median") if comp else None)
            if comp_median is not None:
                delta = comp_median - default_median
                if DIRECTION[metric] == "lower":
                    delta = -delta
                deltas.append(delta)
                rel_deltas.append(abs(delta) / max(abs(default_median), 1e-12))
                if delta < -1e-12:
                    worse += 1
                elif delta > 1e-12:
                    better += 1
            effect = finite_float(pair.get("effect_size"))
            if effect is not None:
                max_effect = max(max_effect, abs(effect))
            try:
                if pair.get("holm_included") in {True, "True"} and float(pair["p_value_holm"]) < 0.05:
                    sig += 1
            except (TypeError, ValueError):
                pass
        default_rank = default.get("rank")
        max_abs_delta = max((abs(x) for x in deltas), default=0.0)
        max_rel = max(rel_deltas, default=0.0)
        if any(r["report_status"] == "requires_manual_review" for r in rows):
            robustness = "inconclusive"
        elif sig == 0 and max_effect <= 0.147:
            robustness = "robust_small_effects"
        elif sig == 0 and str(default_rank) == "1":
            robustness = "robust_default_not_worse"
        elif sig > 0 and str(default_rank) == "1":
            robustness = "sensitive_default_best"
        elif sig > 0 and str(default_rank) != "1":
            robustness = "sensitive_default_not_best"
        else:
            robustness = "mixed"
        out.append({
            "constant_name": constant,
            "problem": problem,
            "metric": metric,
            "direction": DIRECTION[metric],
            "default_median": default_median,
            "default_rank": default_rank,
            "max_abs_median_delta_vs_default": max_abs_delta,
            "max_relative_median_delta_vs_default": max_rel,
            "max_abs_effect_size_vs_default": max_effect,
            "n_levels_better_than_default": better,
            "n_levels_worse_than_default": worse,
            "n_significant_after_holm": sig,
            "robustness_class": robustness,
            **provenance_fields(),
        })
    return out


def _pending_a7_diagnostics() -> list[dict[str, Any]]:
    out = []
    for name in ("results/a7_runtime_workers1", "results/a7_runtime_workers6"):
        out.append({
            "source_script": "scripts/build_e3_clean_stats.py",
            "table": "not_entered_e3_or_global_stats",
            "experiment_group": "A7",
            "problem": "",
            "metric": "",
            "reference_algorithm_or_variant": name,
            "comparison_algorithm_or_variant": "",
            "n_ref": 0,
            "n_cmp": 0,
            "n_pairs": 0,
            "n_nonzero_differences": 0,
            "test_name": "",
            "warning_type": "SmallSampleWarning_prevented",
            "reason_class": "pending_experiment_empty",
            "action_taken": f"Excluded {name}; runtime worker=1/6 is pending.",
            "report_status": "not_usable_pending_completion",
        })
    return out


def main() -> None:
    ensure_dirs()
    rows = e3_long_metric_rows()
    write_csv(
        TABLES / "e3_design_constants_by_run_clean.csv",
        rows,
        [
            "experiment_group",
            "constant_name",
            "constant_level",
            "constant_value",
            "problem",
            "seed",
            "metric",
            "value",
            "direction",
            "n_evals",
            "status",
            "source_result_dir",
            "source_metrics_json",
            "source_snapshot_sha256",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "e3_formal_experiment_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    summary = _summary_rows(rows)
    write_csv(
        TABLES / "e3_design_constants_summary_clean.csv",
        summary,
        [
            "constant_name",
            "constant_level",
            "constant_value",
            "problem",
            "metric",
            "direction",
            "n_runs",
            "median",
            "iqr",
            "mean",
            "std",
            "min",
            "max",
            "rank",
            "default_level_flag",
            "best_flag",
            "status",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "e3_formal_experiment_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    pairwise, diagnostics = _pairwise_rows(rows)
    write_csv(TABLES / "e3_design_constants_pairwise_stats_clean.csv", pairwise, PAIRWISE_FIELDS)
    write_csv(
        TABLES / "e3_design_constants_effect_sizes_clean.csv",
        [
            {
                "constant_name": r["constant_name"],
                "problem": r["problem"],
                "metric": r["metric"],
                "direction": r["direction"],
                "reference_level": r["reference_level"],
                "comparison_level": r["comparison_level"],
                "effect_size_name": r["effect_size_name"],
                "effect_size": r["effect_size"],
                "effect_direction": r["effect_direction"],
                "status": r["status"],
                "report_status": r["report_status"],
                **provenance_fields(),
            }
            for r in pairwise
        ],
        [
            "constant_name",
            "problem",
            "metric",
            "direction",
            "reference_level",
            "comparison_level",
            "effect_size_name",
            "effect_size",
            "effect_direction",
            "status",
            "report_status",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "e3_formal_experiment_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    rank_rows = _rank_stability(summary)
    write_csv(
        TABLES / "e3_design_constants_rank_stability_clean.csv",
        rank_rows,
        [
            "constant_name",
            "problem",
            "metric",
            "direction",
            "default_rank",
            "best_level",
            "rank_min",
            "rank_max",
            "rank_range",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "e3_formal_experiment_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    sensitivity = _sensitivity_index(summary, pairwise)
    write_csv(
        TABLES / "e3_design_constants_sensitivity_index_clean.csv",
        sensitivity,
        [
            "constant_name",
            "problem",
            "metric",
            "direction",
            "default_median",
            "default_rank",
            "max_abs_median_delta_vs_default",
            "max_relative_median_delta_vs_default",
            "max_abs_effect_size_vs_default",
            "n_levels_better_than_default",
            "n_levels_worse_than_default",
            "n_significant_after_holm",
            "robustness_class",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "e3_formal_experiment_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    write_csv(
        TABLES / "e3_statistical_test_manifest_clean.csv",
        [{
            "experiment_scope": "e3_completed",
            "metric_direction": ";".join(f"{k}:{v}" for k, v in DIRECTION.items()),
            "paired_or_unpaired": "paired by seed when complete; Mann-Whitney U fallback if not paired",
            "test_name": "Wilcoxon signed-rank; Mann-Whitney U fallback",
            "p_adjust_method": "Holm",
            "holm_family": "constant_name + problem + metric",
            "all_zero_policy": "test_executed=false; holm_included=false; p_value_raw=1.0; p_value_holm=1.0",
            "seed_count": 11,
            "e3_formal_experiment_source_snapshot_sha256": e3_formal_source_sha(),
            **provenance_fields(),
        }],
        [
            "experiment_scope",
            "metric_direction",
            "paired_or_unpaired",
            "test_name",
            "p_adjust_method",
            "holm_family",
            "all_zero_policy",
            "seed_count",
            "e3_formal_experiment_source_snapshot_sha256",
            "a1_a5_formal_experiment_source_snapshot_sha256",
            "a1_a5_postprocessing_source_snapshot_sha256",
            "global_postprocessing_source_snapshot_sha256",
        ],
    )
    diagnostics = diagnostics + _pending_a7_diagnostics()
    write_csv(AUDIT / "e3_statistical_warning_diagnostics.csv", diagnostics, WARNING_FIELDS)
    reason_counts: dict[str, int] = {}
    for row in diagnostics:
        reason_counts[str(row["reason_class"])] = reason_counts.get(str(row["reason_class"]), 0) + 1
    write_text(
        AUDIT / "e3_statistical_warning_diagnostics_cn.md",
        "\n".join(
            [
                "# E3 statistical warning diagnostics",
                "",
                f"- diagnostics rows: {len(diagnostics)}",
                "- reason classes: "
                + "; ".join(f"{k}={v}" for k, v in sorted(reason_counts.items())),
                "- Any all-zero paired differences are direct no-effect classifications, not Wilcoxon tests.",
                "- A7 pending runtime directories are excluded from E3/global statistical inputs.",
            ]
        ),
    )


if __name__ == "__main__":
    main()
