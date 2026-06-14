"""Build A7 runtime audit tables after excluding SMSEMOA from formal comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKERS1_DIR = ROOT / "results" / "a7_runtime_workers1"
WORKERS6_DIR = ROOT / "results" / "a7_runtime_workers6"
CFG_WORKERS1 = ROOT / "configs" / "formal" / "a7_runtime_workers1_no_smsemoa.yaml"
CFG_WORKERS6 = ROOT / "configs" / "formal" / "a7_runtime_workers6_no_smsemoa.yaml"

FORMAL_ALGORITHMS = ["MOSADE", "NSGA2", "MOEAD", "NSGA3", "SPEA2", "MOEAD_DE"]
FORMAL_PROBLEMS = ["ZDT3", "WFG9", "DTLZ2_n_obj3", "DASCMOP7_difficulty13"]
SMSEMOA_EXPLANATION = (
    "SMSEMOA was excluded from the formal A7 runtime table because preliminary "
    "DTLZ2_n_obj3 workers=1 runs required approximately 1.6-2.0 hours per seed "
    "with BLAS/OpenMP thread caps set to 1. Completed SMSEMOA rows are retained "
    "as runtime-prohibitive diagnostic evidence only and are excluded from the "
    "formal median [IQR] runtime comparison."
)


@dataclass(frozen=True)
class RuntimeCell:
    worker_setting: int
    problem: str
    algorithm: str
    seed: int
    run_index: int
    status: str
    runtime_seconds: float | None
    hv: float | None
    igd: float | None
    igd_plus: float | None
    gd: float | None
    spread: float | None
    feasibility_ratio: float | None
    n_evals: int | None
    budget_expected: int | None
    metrics_json_found: bool
    history_json_found: bool
    objectives_found: bool
    decisions_found: bool
    source_metrics_json: str
    notes: str


def ensure_dirs() -> None:
    for dirname in ["audit", "tables", "summaries", "validation"]:
        (ROOT / dirname).mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return data


def problem_dir_name(problem_entry: Any) -> str:
    if isinstance(problem_entry, dict):
        cfg = {k: v for k, v in problem_entry.items() if k != "name"}
        name = str(problem_entry["name"])
        suffix = "_".join(f"{k}{v}" for k, v in sorted(cfg.items()))
        return f"{name}_{suffix}" if suffix else name
    return str(problem_entry)


def seed_sequence(base_seed: int, count: int) -> list[int]:
    ss = np.random.SeedSequence(base_seed)
    return [int(child.generate_state(1)[0]) for child in ss.spawn(count)]


def expected_from_config(config_path: Path) -> tuple[list[str], list[str], list[int], dict[str, dict[str, int]]]:
    cfg = read_yaml(config_path)
    problems = [problem_dir_name(p) for p in cfg.get("problems", [])]
    algorithms = [str(a["name"]) for a in cfg.get("algorithms", [])]
    seeds = seed_sequence(int(cfg.get("seed", 42)), int(cfg.get("n_runs", 31)))
    budgets: dict[str, dict[str, int]] = {}
    for algo in cfg.get("algorithms", []):
        name = str(algo["name"])
        budgets[name] = {
            "budget": int(algo.get("max_evals", 0)),
            "population_size": int(algo.get("pop_size", 0)),
        }
    return problems, algorithms, seeds, budgets


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def read_runtime_cell(
    run_dir: Path,
    worker_setting: int,
    problem: str,
    algorithm: str,
    seed: int,
    run_index: int,
    budget_expected: int | None,
) -> RuntimeCell:
    run_subdir = run_dir / problem / algorithm / f"run_{run_index:03d}"
    metrics_path = run_subdir / "metrics.json"
    history_path = run_subdir / "history.json"
    objectives_path = run_subdir / "objectives.txt"
    decisions_path = run_subdir / "decisions.txt"
    metrics = read_json(metrics_path)
    if metrics is None:
        return RuntimeCell(
            worker_setting=worker_setting,
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            run_index=run_index,
            status="missing",
            runtime_seconds=None,
            hv=None,
            igd=None,
            igd_plus=None,
            gd=None,
            spread=None,
            feasibility_ratio=None,
            n_evals=None,
            budget_expected=budget_expected,
            metrics_json_found=False,
            history_json_found=history_path.exists(),
            objectives_found=objectives_path.exists(),
            decisions_found=decisions_path.exists(),
            source_metrics_json=rel(metrics_path),
            notes="metrics.json not found",
        )

    status = str(metrics.get("status", "ok"))
    notes = str(metrics.get("status_reason") or "")
    n_evals_raw = metrics.get("n_evals")
    n_evals = int(n_evals_raw) if isinstance(n_evals_raw, (int, float)) and not math.isnan(n_evals_raw) else None
    return RuntimeCell(
        worker_setting=worker_setting,
        problem=problem,
        algorithm=algorithm,
        seed=int(metrics.get("seed", seed)),
        run_index=run_index,
        status=status,
        runtime_seconds=finite_float(metrics.get("time_s")),
        hv=finite_float(metrics.get("hv")),
        igd=finite_float(metrics.get("igd")),
        igd_plus=finite_float(metrics.get("igd_plus")),
        gd=finite_float(metrics.get("gd")),
        spread=finite_float(metrics.get("spread")),
        feasibility_ratio=finite_float(metrics.get("feasibility_ratio")),
        n_evals=n_evals,
        budget_expected=budget_expected,
        metrics_json_found=True,
        history_json_found=history_path.exists(),
        objectives_found=objectives_path.exists(),
        decisions_found=decisions_path.exists(),
        source_metrics_json=rel(metrics_path),
        notes=notes,
    )


def no_smsemoa_resume_start(run_dir: Path, config_path: Path) -> float | None:
    """Return local timestamp for the latest no-SMSEMOA resume start, if logged."""
    log_path = run_dir / "experiment.log"
    if not log_path.exists():
        return None
    marker_candidates = {
        f"Experiment config: {config_path.as_posix()}",
        f"Experiment config: {config_path}",
        f"Experiment config: {rel(config_path)}",
        config_path.name,
    }
    latest: float | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not any(marker in line for marker in marker_candidates):
                continue
            raw_ts = line.split("|", 1)[0].strip()
            try:
                dt = datetime.strptime(raw_ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            latest = dt.timestamp()
    return latest


def runtime_cells(run_dir: Path, config_path: Path, worker_setting: int) -> list[RuntimeCell]:
    problems, algorithms, seeds, budgets = expected_from_config(config_path)
    rows: list[RuntimeCell] = []
    for problem in problems:
        if problem not in FORMAL_PROBLEMS:
            continue
        for algorithm in algorithms:
            if algorithm not in FORMAL_ALGORITHMS:
                continue
            budget_expected = budgets.get(algorithm, {}).get("budget")
            for run_index, seed in enumerate(seeds):
                rows.append(
                    read_runtime_cell(
                        run_dir=run_dir,
                        worker_setting=worker_setting,
                        problem=problem,
                        algorithm=algorithm,
                        seed=seed,
                        run_index=run_index,
                        budget_expected=budget_expected,
                    )
                )
    return rows


def cell_to_row(cell: RuntimeCell) -> dict[str, Any]:
    formal_included = cell.status == "ok" and cell.runtime_seconds is not None
    if cell.status == "unsupported":
        excluded_reason = "unsupported_combination"
    elif cell.status != "ok":
        excluded_reason = cell.status
    elif cell.runtime_seconds is None:
        excluded_reason = "missing_runtime_seconds"
    else:
        excluded_reason = ""
    return {
        "worker_setting": cell.worker_setting,
        "problem": cell.problem,
        "algorithm": cell.algorithm,
        "seed": cell.seed,
        "run_index": cell.run_index,
        "status": cell.status,
        "runtime_seconds": cell.runtime_seconds,
        "HV": cell.hv,
        "IGD": cell.igd,
        "IGD_PLUS": cell.igd_plus,
        "GD": cell.gd,
        "spread": cell.spread,
        "feasibility_ratio": cell.feasibility_ratio,
        "n_evals": cell.n_evals,
        "budget_expected": cell.budget_expected,
        "metrics_json_found": cell.metrics_json_found,
        "history_json_found": cell.history_json_found,
        "objectives_found": cell.objectives_found,
        "decisions_found": cell.decisions_found,
        "formal_runtime_included": formal_included,
        "excluded_reason": excluded_reason,
        "source_metrics_json": cell.source_metrics_json,
        "notes": cell.notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_workers1_inventory() -> None:
    rows = []
    resume_start = no_smsemoa_resume_start(WORKERS1_DIR, CFG_WORKERS1)
    for cell in runtime_cells(WORKERS1_DIR, CFG_WORKERS1, 1):
        metrics_path = ROOT / cell.source_metrics_json
        generated_during_resume = False
        if resume_start is not None and metrics_path.exists():
            generated_during_resume = metrics_path.stat().st_mtime >= resume_start
        if generated_during_resume:
            pre_resume_status = "missing_before_no_smsemoa_resume"
        elif cell.status == "missing":
            pre_resume_status = "missing_before_no_smsemoa_resume"
        else:
            pre_resume_status = cell.status
        row = cell_to_row(cell)
        row.update(
            {
                "expected": True,
                "completed_or_explicit_unsupported": cell.status in {"ok", "unsupported"},
                "inventory_scope": "pre_resume_or_current_workers1_no_smsemoa_target_matrix",
                "status_before_no_smsemoa_resume": pre_resume_status,
                "generated_during_no_smsemoa_resume": generated_during_resume,
            }
        )
        rows.append(row)
    write_csv(ROOT / "audit" / "a7_workers1_completion_inventory.csv", rows)


def smsemoa_diagnostic_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for worker_setting, run_dir in [(1, WORKERS1_DIR), (6, WORKERS6_DIR)]:
        smsemoa_dir = run_dir / "DTLZ2_n_obj3" / "SMSEMOA"
        if not smsemoa_dir.exists():
            continue
        for run_subdir in sorted(smsemoa_dir.glob("run_*")):
            try:
                run_index = int(run_subdir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                run_index = -1
            metrics = read_json(run_subdir / "metrics.json")
            if metrics is None:
                rows.append(
                    {
                        "problem": "DTLZ2_n_obj3",
                        "algorithm": "SMSEMOA",
                        "worker_setting": worker_setting,
                        "seed": "",
                        "run_index": run_index,
                        "status": "incomplete_no_metrics",
                        "runtime_seconds": "",
                        "HV": "",
                        "IGD": "",
                        "feasibility_ratio": "",
                        "report_status": "partial_diagnostic_only",
                        "formal_runtime_table_status": "excluded_from_formal_runtime_table",
                        "source_metrics_json": rel(run_subdir / "metrics.json"),
                        "notes": "partial_diagnostic_only;excluded_from_formal_runtime_table",
                    }
                )
                continue
            rows.append(
                {
                    "problem": "DTLZ2_n_obj3",
                    "algorithm": "SMSEMOA",
                    "worker_setting": worker_setting,
                    "seed": metrics.get("seed", ""),
                    "run_index": run_index,
                    "status": metrics.get("status", "ok"),
                    "runtime_seconds": metrics.get("time_s", ""),
                    "HV": metrics.get("hv", ""),
                    "IGD": metrics.get("igd", ""),
                    "feasibility_ratio": metrics.get("feasibility_ratio", ""),
                    "report_status": "partial_diagnostic_only",
                    "formal_runtime_table_status": "excluded_from_formal_runtime_table",
                    "source_metrics_json": rel(run_subdir / "metrics.json"),
                    "notes": "partial_diagnostic_only;excluded_from_formal_runtime_table",
                }
            )
    return rows


def write_smsemoa_diagnostic() -> None:
    write_csv(
        ROOT / "tables" / "runtime_smsemoa_dtlz2_partial_diagnostic.csv",
        smsemoa_diagnostic_rows(),
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summary_for(cells: list[RuntimeCell]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[int, str, str], list[RuntimeCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.worker_setting, cell.problem, cell.algorithm), []).append(cell)
    for (worker, problem, algorithm), group in sorted(grouped.items()):
        ok_values = [
            float(c.runtime_seconds)
            for c in group
            if c.status == "ok" and c.runtime_seconds is not None
        ]
        expected = len(group)
        ok_runs = len(ok_values)
        unsupported = sum(1 for c in group if c.status == "unsupported")
        missing = sum(1 for c in group if c.status == "missing")
        failed = sum(1 for c in group if c.status not in {"ok", "unsupported", "missing"})
        q1 = percentile(ok_values, 25)
        med = percentile(ok_values, 50)
        q3 = percentile(ok_values, 75)
        iqr = None if q1 is None or q3 is None else q3 - q1
        if med is None or iqr is None:
            median_iqr = "NA"
        else:
            median_iqr = f"{med:.2f} [{iqr:.2f}]"
        if ok_runs == expected:
            status = "complete"
        elif ok_runs + unsupported == expected and unsupported:
            status = "complete_with_unsupported"
        elif ok_runs == 0 and unsupported == expected:
            status = "unsupported"
        elif missing or failed:
            status = "incomplete"
        else:
            status = "partial"
        rows.append(
            {
                "worker_setting": worker,
                "problem": problem,
                "algorithm": algorithm,
                "expected_runs": expected,
                "completed_ok_runs": ok_runs,
                "unsupported_runs": unsupported,
                "missing_runs": missing,
                "failed_runs": failed,
                "median_seconds": med,
                "q1_seconds": q1,
                "q3_seconds": q3,
                "iqr_seconds": iqr,
                "median_iqr_seconds": median_iqr,
                "min_seconds": min(ok_values) if ok_values else None,
                "max_seconds": max(ok_values) if ok_values else None,
                "status": status,
            }
        )
    return rows


def write_formal_runtime_tables() -> None:
    cells1 = runtime_cells(WORKERS1_DIR, CFG_WORKERS1, 1)
    cells6 = runtime_cells(WORKERS6_DIR, CFG_WORKERS6, 6)
    rows1 = [cell_to_row(c) for c in cells1]
    rows6 = [cell_to_row(c) for c in cells6]
    write_csv(ROOT / "tables" / "runtime_workers_1_no_smsemoa.csv", rows1)
    write_csv(ROOT / "tables" / "runtime_workers_6_no_smsemoa.csv", rows6)

    summary = summary_for(cells1 + cells6)
    by_key = {(r["worker_setting"], r["problem"], r["algorithm"]): r for r in summary}
    table_rows: list[dict[str, Any]] = []
    for problem in FORMAL_PROBLEMS:
        for algorithm in FORMAL_ALGORITHMS:
            r1 = by_key.get((1, problem, algorithm), {})
            r6 = by_key.get((6, problem, algorithm), {})
            med1 = r1.get("median_seconds")
            med6 = r6.get("median_seconds")
            speedup = None
            if isinstance(med1, (int, float)) and isinstance(med6, (int, float)) and med6 > 0:
                speedup = float(med1) / float(med6)
            status_parts = []
            for worker, row in [(1, r1), (6, r6)]:
                row_status = row.get("status", "missing")
                status_parts.append(f"workers{worker}:{row_status}")
            table_rows.append(
                {
                    "problem": problem,
                    "algorithm": algorithm,
                    "workers1_median_iqr_seconds": r1.get("median_iqr_seconds", "NA"),
                    "workers1_median_seconds": r1.get("median_seconds", ""),
                    "workers1_iqr_seconds": r1.get("iqr_seconds", ""),
                    "workers1_n_runs": r1.get("completed_ok_runs", 0),
                    "workers1_unsupported_runs": r1.get("unsupported_runs", 0),
                    "workers1_status": r1.get("status", "missing"),
                    "workers6_median_iqr_seconds": r6.get("median_iqr_seconds", "NA"),
                    "workers6_median_seconds": r6.get("median_seconds", ""),
                    "workers6_iqr_seconds": r6.get("iqr_seconds", ""),
                    "workers6_n_runs": r6.get("completed_ok_runs", 0),
                    "workers6_unsupported_runs": r6.get("unsupported_runs", 0),
                    "workers6_status": r6.get("status", "missing"),
                    "speedup_workers1_over_workers6": speedup,
                    "formal_table_status": ";".join(status_parts),
                    "notes": "formal filtered protocol; legacy parallel runtime table not used",
                }
            )
    write_csv(ROOT / "tables" / "table_4_12_replacement.csv", table_rows)


def env_value(path: Path, *keys: str) -> Any:
    data = read_json(path)
    if data is None:
        return ""
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(key, "")
    return cur


def write_runtime_environment() -> None:
    lines = [
        "# A7 runtime 环境�?SMSEMOA 排除说明",
        "",
        "## 1. 正式 runtime 范围",
        "",
        "正式 A7 runtime 比较使用 no-SMSEMOA 矩阵：MOSADE, NSGA2, MOEAD, NSGA3, SPEA2, MOEAD_DE；问题为 ZDT3, WFG9, DTLZ2_n_obj3, DASCMOP7_difficulty13；worker 设置�?1 �?6�?,
        "",
        "�?workers=8 runtime 表不进入本次正式 worker=1/6 对比�?,
        "",
        "## 2. 线程环境",
        "",
        "| 变量 | 要求�?|",
        "|---|---:|",
        "| OMP_NUM_THREADS | 1 |",
        "| MKL_NUM_THREADS | 1 |",
        "| OPENBLAS_NUM_THREADS | 1 |",
        "| NUMEXPR_NUM_THREADS | 1 |",
        "",
        "## 3. SMSEMOA 处理",
        "",
        SMSEMOA_EXPLANATION,
        "",
        "SMSEMOA 诊断数据文件：`tables/runtime_smsemoa_dtlz2_partial_diagnostic.csv`�?,
    ]
    (ROOT / "audit" / "runtime_environment_cn.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def validation_rows() -> list[dict[str, Any]]:
    cells = runtime_cells(WORKERS1_DIR, CFG_WORKERS1, 1) + runtime_cells(WORKERS6_DIR, CFG_WORKERS6, 6)
    grouped: dict[tuple[int, str, str], list[RuntimeCell]] = {}
    for cell in cells:
        grouped.setdefault((cell.worker_setting, cell.problem, cell.algorithm), []).append(cell)
    rows: list[dict[str, Any]] = []
    for worker in [1, 6]:
        for problem in FORMAL_PROBLEMS:
            for algorithm in FORMAL_ALGORITHMS:
                group = grouped.get((worker, problem, algorithm), [])
                expected = len(group)
                ok_runs = sum(1 for c in group if c.status == "ok")
                unsupported = sum(1 for c in group if c.status == "unsupported")
                missing = sum(1 for c in group if c.status == "missing")
                failed = sum(1 for c in group if c.status not in {"ok", "unsupported", "missing"})
                pass_check = expected > 0 and ok_runs + unsupported == expected and missing == 0 and failed == 0
                rows.append(
                    {
                        "check_type": "formal_cell_completion",
                        "worker_setting": worker,
                        "problem": problem,
                        "algorithm": algorithm,
                        "expected_runs": expected,
                        "completed_ok_runs": ok_runs,
                        "unsupported_runs": unsupported,
                        "missing_runs": missing,
                        "failed_runs": failed,
                        "check_pass": pass_check,
                        "notes": "unsupported placeholders count as explicit completion but are excluded from median runtime",
                    }
                )

    table_path = ROOT / "tables" / "table_4_12_replacement.csv"
    table_text = table_path.read_text(encoding="utf-8") if table_path.exists() else ""
    diag_rows = smsemoa_diagnostic_rows()
    rows.extend(
        [
            {
                "check_type": "smsemoa_absent_from_formal_table",
                "worker_setting": "",
                "problem": "",
                "algorithm": "SMSEMOA",
                "expected_runs": "",
                "completed_ok_runs": "",
                "unsupported_runs": "",
                "missing_runs": "",
                "failed_runs": "",
                "check_pass": "SMSEMOA" not in table_text,
                "notes": "SMSEMOA must not appear in tables/table_4_12_replacement.csv",
            },
            {
                "check_type": "smsemoa_diagnostic_present",
                "worker_setting": 1,
                "problem": "DTLZ2_n_obj3",
                "algorithm": "SMSEMOA",
                "expected_runs": "",
                "completed_ok_runs": sum(1 for r in diag_rows if r.get("status") == "ok"),
                "unsupported_runs": "",
                "missing_runs": sum(1 for r in diag_rows if r.get("status") != "ok"),
                "failed_runs": "",
                "check_pass": len(diag_rows) > 0
                and all(r.get("formal_runtime_table_status") == "excluded_from_formal_runtime_table" for r in diag_rows),
                "notes": "Partial SMSEMOA rows are diagnostic only",
            },
            {
                "check_type": "workers8_not_used",
                "worker_setting": "",
                "problem": "",
                "algorithm": "",
                "expected_runs": "",
                "completed_ok_runs": "",
                "unsupported_runs": "",
                "missing_runs": "",
                "failed_runs": "",
                "check_pass": "workers8" not in table_text and ",8," not in table_text,
                "notes": "Legacy workers=8 table is not used in the replacement table",
            },
        ]
    )
    return rows


def write_validation_and_summary() -> None:
    rows = validation_rows()
    write_csv(ROOT / "validation" / "a7_runtime_completion_check.csv", rows)
    failed = [r for r in rows if str(r.get("check_pass")).lower() not in {"true", "1"}]
    diag = smsemoa_diagnostic_rows()
    ok_diag = [r for r in diag if r.get("status") == "ok"]
    runtimes = [finite_float(r.get("runtime_seconds")) for r in ok_diag]
    runtimes = [r for r in runtimes if r is not None]
    if runtimes:
        diag_range = f"{min(runtimes) / 3600:.2f}-{max(runtimes) / 3600:.2f} 小时/seed"
    else:
        diag_range = "�?completed SMSEMOA runtime 可用"

    summary = [
        "# A7 runtime no-SMSEMOA 执行摘要",
        "",
        "## 1. 正式比较范围",
        "",
        "正式 A7 runtime 表使�?worker=1 �?worker=6，在 ZDT3、WFG9、DTLZ2_n_obj3、DASCMOP7_difficulty13 上比�?MOSADE、NSGA2、MOEAD、NSGA3、SPEA2、MOEAD_DE�?,
        "",
        "SMSEMOA 不进入正�?runtime 表，�?workers=8 表不用于 worker=1/6 审稿回应�?,
        "",
        "## 2. SMSEMOA 诊断",
        "",
        f"已保�?SMSEMOA DTLZ2_n_obj3 partial diagnostic rows: {len(diag)}；其�?completed rows: {len(ok_diag)}；runtime 范围�?{diag_range}�?,
        "",
        SMSEMOA_EXPLANATION,
        "",
        "## 3. 输出文件",
        "",
        "- `audit/a7_workers1_completion_inventory.csv`",
        "- `tables/runtime_smsemoa_dtlz2_partial_diagnostic.csv`",
        "- `tables/runtime_workers_1_no_smsemoa.csv`",
        "- `tables/runtime_workers_6_no_smsemoa.csv`",
        "- `tables/table_4_12_replacement.csv`",
        "- `validation/a7_runtime_completion_check.csv`",
        "",
        "## 4. 完整性验�?,
        "",
    ]
    if failed:
        summary.append(f"仍有 {len(failed)} �?validation check 未通过。请先查�?`validation/a7_runtime_completion_check.csv`�?)
    else:
        summary.append("所�?formal non-SMSEMOA A7 cells �?worker=1 �?worker=6 均为 completed �?explicit unsupported；formal median [IQR] runtime 表已排除 unsupported、missing、failed �?SMSEMOA�?)
    (ROOT / "summaries" / "a7_runtime_summary_cn.md").write_text(
        "\n".join(summary),
        encoding="utf-8-sig",
    )


def run_inventory_only() -> None:
    ensure_dirs()
    write_workers1_inventory()
    write_smsemoa_diagnostic()


def run_final_outputs(skip_inventory: bool) -> None:
    ensure_dirs()
    if not skip_inventory:
        write_workers1_inventory()
    write_smsemoa_diagnostic()
    write_formal_runtime_tables()
    write_runtime_environment()
    write_validation_and_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true", help="Only write workers=1 inventory and SMSEMOA diagnostic")
    parser.add_argument("--skip-inventory", action="store_true", help="Do not overwrite audit/a7_workers1_completion_inventory.csv")
    args = parser.parse_args()
    if args.inventory_only:
        run_inventory_only()
    else:
        run_final_outputs(skip_inventory=args.skip_inventory)


if __name__ == "__main__":
    main()
