"""Build report-ready figures from A1-A7 CSV and history sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    FIGURES.mkdir(exist_ok=True)
    with (FIGURES / "figure_source_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["figure", "source_csv_or_json", "status", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _first_a1_epsilon_histories() -> tuple[str, dict[str, list[float]]]:
    result_root = ROOT / "results" / "a1_ablation_epsilon_modes"
    if not result_root.exists():
        return "", {}
    for problem_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        histories: dict[str, list[float]] = {}
        for algo_dir in sorted(p for p in problem_dir.iterdir() if p.is_dir()):
            eps_path = algo_dir / "run_000" / "epsilon_history.json"
            if not eps_path.exists():
                continue
            payload = json.loads(eps_path.read_text(encoding="utf-8"))
            hist = [float(v) for v in payload.get("epsilon_history", [])]
            if hist:
                histories[algo_dir.name] = hist
        if histories:
            return problem_dir.name, histories
    return "", {}


def _plot_a1(manifest: list[dict[str, Any]]) -> None:
    problem, histories = _first_a1_epsilon_histories()
    fig_path = FIGURES / f"a1_epsilon_trajectory_{problem}_seed0.png"
    if not histories:
        manifest.append({
            "figure": "a1_epsilon_trajectory_<problem>_<seed>.png",
            "source_csv_or_json": "results/a1_ablation_epsilon_modes/*/epsilon_history.json",
            "status": "missing_data",
            "notes": "formal A1 result directory not available or no epsilon history found",
        })
        return
    plt.figure(figsize=(6.4, 4.0))
    for label, hist in histories.items():
        plt.plot(np.arange(1, len(hist) + 1), hist, label=label)
    plt.xlabel("Generation")
    plt.ylabel("Epsilon")
    plt.title(f"Epsilon Trajectories ({problem}, seed 0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    manifest.append({
        "figure": fig_path.name,
        "source_csv_or_json": "results/a1_ablation_epsilon_modes/*/epsilon_history.json",
        "status": "created",
        "notes": "",
    })


def _plot_effects(manifest: list[dict[str, Any]]) -> None:
    rows = []
    for path in TABLES.glob("a*_effect_sizes.csv"):
        for row in _read_csv(path):
            try:
                effect = float(row["effect_size"])
            except (TypeError, ValueError):
                continue
            rows.append((path.stem, row["comparison_algorithm"], row["metric"], effect))
    fig_path = FIGURES / "ablation_effect_sizes_summary.png"
    if not rows:
        manifest.append({
            "figure": fig_path.name,
            "source_csv_or_json": "tables/a*_effect_sizes.csv",
            "status": "missing_data",
            "notes": "no finite effect sizes found",
        })
        return
    labels = [f"{group}:{algo}:{metric}" for group, algo, metric, _ in rows[:40]]
    vals = [value for *_, value in rows[:40]]
    plt.figure(figsize=(9.0, 5.0))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=7)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.ylabel("Cliff's Delta")
    plt.title("Ablation Effect Sizes Summary")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    manifest.append({
        "figure": fig_path.name,
        "source_csv_or_json": "tables/a*_effect_sizes.csv",
        "status": "created",
        "notes": "",
    })


def _plot_runtime(manifest: list[dict[str, Any]]) -> None:
    rows = _read_csv(TABLES / "a7_runtime_summary.csv")
    data = [r for r in rows if r.get("median_time")]
    fig_path = FIGURES / "a7_runtime_workers1_vs_workers6.png"
    if not data:
        manifest.append({
            "figure": fig_path.name,
            "source_csv_or_json": "tables/a7_runtime_summary.csv",
            "status": "missing_data",
            "notes": "runtime summary has no finite medians",
        })
        return
    keys = sorted({(r["problem"], r["algorithm"]) for r in data})[:30]
    x = np.arange(len(keys))
    w1 = []
    w6 = []
    for problem, algorithm in keys:
        by_worker = {
            int(r["workers"]): float(r["median_time"])
            for r in data
            if r["problem"] == problem and r["algorithm"] == algorithm and r["median_time"]
        }
        w1.append(by_worker.get(1, np.nan))
        w6.append(by_worker.get(6, np.nan))
    plt.figure(figsize=(10.0, 5.0))
    plt.bar(x - 0.2, w1, width=0.4, label="workers=1")
    plt.bar(x + 0.2, w6, width=0.4, label="workers=6")
    plt.xticks(x, [f"{p}\n{a}" for p, a in keys], rotation=60, ha="right", fontsize=7)
    plt.ylabel("Median Wall Time (s)")
    plt.title("Runtime Comparison: workers=1 vs workers=6")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    manifest.append({
        "figure": fig_path.name,
        "source_csv_or_json": "tables/a7_runtime_summary.csv",
        "status": "created",
        "notes": "",
    })


def _plot_cre(manifest: list[dict[str, Any]]) -> None:
    rows = _read_csv(TABLES / "rebuilt_cre_rankings.csv")
    hv_rows = [r for r in rows if r.get("metric") == "hv" and r.get("rank")]
    fig_path = FIGURES / "cre_rankings_summary.png"
    if not hv_rows:
        manifest.append({
            "figure": fig_path.name,
            "source_csv_or_json": "tables/rebuilt_cre_rankings.csv",
            "status": "missing_data",
            "notes": "no CRE HV ranking rows",
        })
        return
    labels = [f"{r['problem']}:{r['algorithm']}" for r in hv_rows[:40]]
    ranks = [float(r["rank"]) for r in hv_rows[:40]]
    plt.figure(figsize=(9.0, 5.0))
    plt.bar(range(len(ranks)), ranks)
    plt.gca().invert_yaxis()
    plt.xticks(range(len(ranks)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("HV Rank (lower is better)")
    plt.title("CRE External Validation Rankings")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    manifest.append({
        "figure": fig_path.name,
        "source_csv_or_json": "tables/rebuilt_cre_rankings.csv",
        "status": "created",
        "notes": "",
    })


def _plot_e3(manifest: list[dict[str, Any]]) -> None:
    rows = _read_csv(TABLES / "e3_design_constants_summary.csv")
    hv = [r for r in rows if r.get("metric") == "hv" and r.get("median")]
    fig_path = FIGURES / "e3_design_constants_sensitivity.png"
    if not hv:
        manifest.append({
            "figure": fig_path.name,
            "source_csv_or_json": "tables/e3_design_constants_summary.csv",
            "status": "missing_data",
            "notes": "no finite E3 HV summary rows",
        })
        return
    labels = [f"{r['problem']}:{r['algorithm']}" for r in hv[:40]]
    med = [float(r["median"]) for r in hv[:40]]
    plt.figure(figsize=(10.0, 5.0))
    plt.bar(range(len(med)), med)
    plt.xticks(range(len(med)), labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Median HV")
    plt.title("Design Constants Sensitivity")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    manifest.append({
        "figure": fig_path.name,
        "source_csv_or_json": "tables/e3_design_constants_summary.csv",
        "status": "created",
        "notes": "",
    })


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    manifest: list[dict[str, Any]] = []
    _plot_a1(manifest)
    _plot_effects(manifest)
    _plot_runtime(manifest)
    _plot_cre(manifest)
    _plot_e3(manifest)
    _write_manifest(manifest)


if __name__ == "__main__":
    main()
