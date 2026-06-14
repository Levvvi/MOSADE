from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mosade.metrics.hypervolume import hypervolume
from mosade.algorithm.selection import nondominated_mask


RESULTS_ROOT = REPO_ROOT / "results"
FORMAL_PATTERNS = {
    "cre_2obj": "*benchmark_realworld_cre_2obj*",
    "cre_3obj": "*benchmark_realworld_cre_3obj*",
    "moon": "*benchmark_realworld_moon_optional*",
}
ALGO_ORDER = ["MOSADE", "NSGA2", "MOEAD", "NSGA3"]
ALGO_COLORS = {
    "MOSADE": "#1b9e77",
    "NSGA2": "#d95f02",
    "MOEAD": "#7570b3",
    "NSGA3": "#e7298a",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _latex_table(df: pd.DataFrame) -> str:
    cols = "l" * len(df.columns)
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\hline"]
    lines.append(" & ".join(_latex_escape(col) for col in df.columns) + r" \\")
    lines.append(r"\hline")
    for row in df.to_numpy():
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def _latest_matching_dir(root: Path, pattern: str) -> Path | None:
    matches = sorted([p for p in root.glob(pattern) if p.is_dir()])
    return matches[-1] if matches else None


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, 0), dtype=float)
    arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return np.empty((0, 0), dtype=float)
    return np.atleast_2d(arr)


def _algorithm_dirs(problem_dir: Path) -> list[tuple[str, Path]]:
    algo_dirs = [d for d in sorted(problem_dir.iterdir()) if d.is_dir() and not d.name.startswith("run_")]
    if algo_dirs:
        return [(d.name, d) for d in algo_dirs]
    return [("MOSADE", problem_dir)]


def load_problem_runs(results_dir: Path) -> dict[str, dict[str, list[Path]]]:
    data: dict[str, dict[str, list[Path]]] = {}
    for problem_dir in sorted(results_dir.iterdir()):
        if not problem_dir.is_dir() or problem_dir.name in {"plots", "tables"}:
            continue
        algos: dict[str, list[Path]] = {}
        for algo_name, algo_dir in _algorithm_dirs(problem_dir):
            run_dirs = sorted([d for d in algo_dir.iterdir() if d.is_dir() and d.name.startswith("run_")])
            if run_dirs:
                algos[algo_name] = run_dirs
        if algos:
            data[problem_dir.name] = algos
    return data


def load_feasible_nd_set(run_dir: Path) -> np.ndarray:
    pf_path = run_dir / "pareto_approximation.txt"
    if pf_path.exists():
        F = _load_matrix(pf_path)
        if F.size > 0:
            return F
    obj_path = run_dir / "objectives.txt"
    F = _load_matrix(obj_path)
    if F.size == 0:
        return np.empty((0, 0), dtype=float)
    mask = nondominated_mask(F)
    return F[mask]


def build_problem_scaler(problem_runs: dict[str, list[Path]]) -> tuple[np.ndarray | None, np.ndarray | None]:
    fronts: list[np.ndarray] = []
    for run_dirs in problem_runs.values():
        for run_dir in run_dirs:
            F = load_feasible_nd_set(run_dir)
            if F.size > 0:
                fronts.append(F)
    if not fronts:
        return None, None
    union = np.vstack(fronts)
    return np.min(union, axis=0), np.max(union, axis=0)


def normalize_F(F: np.ndarray, zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    if F.size == 0:
        return np.empty((0, zmin.shape[0]), dtype=float)
    span = np.where(np.abs(zmax - zmin) > 1e-12, zmax - zmin, 1.0)
    return (F - zmin) / span


def hv_reference(n_obj: int) -> np.ndarray:
    return np.full(n_obj, 1.1, dtype=float)


def _run_seed(run_dir: Path) -> int | None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        return None
    history = _load_json(history_path)
    seed = history.get("seed")
    if isinstance(seed, int):
        return seed
    return None


def compute_realworld_metrics(
    problem_name: str,
    algo_name: str,
    run_dir: Path,
    zmin: np.ndarray | None,
    zmax: np.ndarray | None,
) -> dict:
    metrics = _load_json(run_dir / "metrics.json")
    F = load_feasible_nd_set(run_dir)
    if zmin is None or zmax is None or F.size == 0:
        hv_norm = 0.0
    else:
        F_norm = normalize_F(F, zmin, zmax)
        hv_norm = hypervolume(F_norm, hv_reference(F_norm.shape[1]))

    return {
        "problem": problem_name,
        "algorithm": algo_name,
        "run_id": run_dir.name,
        "seed": _run_seed(run_dir),
        "hv_norm": float(hv_norm),
        "feasibility_ratio": float(metrics.get("feasibility_ratio", float("nan"))),
        "best_cv": float(metrics.get("best_cv", float("nan"))),
        "median_cv": float(metrics.get("median_cv", float("nan"))),
        "runtime_sec": float(metrics.get("time_s", float("nan"))),
        "n_solutions": int(metrics.get("n_solutions", 0)),
        "run_dir": str(run_dir),
    }


def representative_run_idx(values: list[float]) -> int:
    finite = [(idx, val) for idx, val in enumerate(values) if np.isfinite(val)]
    if not finite:
        return 0
    finite.sort(key=lambda item: (item[1], item[0]))
    return finite[len(finite) // 2][0]


def _problem_dimension(problem_rows: pd.DataFrame) -> int | None:
    for run_dir_str in problem_rows["run_dir"]:
        F = load_feasible_nd_set(Path(run_dir_str))
        if F.size > 0:
            return F.shape[1]
    return None


def plot_pf_overlay(problem_name: str, problem_rows: pd.DataFrame, output_path: Path) -> bool:
    n_obj = _problem_dimension(problem_rows)
    if n_obj not in {2, 3}:
        return False

    selected_rows = []
    for algo_name, algo_rows in problem_rows.groupby("algorithm", sort=False):
        algo_rows = algo_rows.sort_values("run_id")
        idx = representative_run_idx(algo_rows["hv_norm"].tolist())
        selected_rows.append(algo_rows.iloc[idx])

    if n_obj == 2:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        any_points = False
        for row in selected_rows:
            F = load_feasible_nd_set(Path(row["run_dir"]))
            if F.size == 0:
                continue
            any_points = True
            ax.scatter(
                F[:, 0],
                F[:, 1],
                s=28,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.5,
                color=ALGO_COLORS.get(row["algorithm"], None),
                label=f"{row['algorithm']} (median-HV)",
            )
        if not any_points:
            plt.close(fig)
            return False
        ax.set_title(f"{problem_name} representative feasible fronts")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.2)
    else:
        fig = plt.figure(figsize=(7.0, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        any_points = False
        for row in selected_rows:
            F = load_feasible_nd_set(Path(row["run_dir"]))
            if F.size == 0:
                continue
            any_points = True
            ax.scatter(
                F[:, 0],
                F[:, 1],
                F[:, 2],
                s=22,
                alpha=0.85,
                color=ALGO_COLORS.get(row["algorithm"], None),
                label=f"{row['algorithm']} (median-HV)",
            )
        if not any_points:
            plt.close(fig)
            return False
        ax.set_title(f"{problem_name} representative feasible fronts")
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_zlabel("f3")
        ax.legend(frameon=False, fontsize=8, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_hv_boxplot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    problems = sorted(metrics_df["problem"].unique())
    fig, axes = plt.subplots(1, len(problems), figsize=(4.8 * len(problems), 4.8), squeeze=False)
    for ax, problem in zip(axes[0], problems):
        subset = metrics_df[metrics_df["problem"] == problem]
        present_algos = [algo for algo in ALGO_ORDER if algo in set(subset["algorithm"])]
        data = [subset.loc[subset["algorithm"] == algo, "hv_norm"].to_numpy(dtype=float) for algo in present_algos]
        bp = ax.boxplot(data, tick_labels=present_algos, patch_artist=True)
        for patch, algo in zip(bp["boxes"], present_algos):
            patch.set_facecolor(ALGO_COLORS.get(algo, "#cccccc"))
            patch.set_alpha(0.75)
        ax.set_title(problem)
        ax.set_ylabel("Normalized HV")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_feasibility_bar(metrics_df: pd.DataFrame, output_path: Path) -> None:
    problems = sorted(metrics_df["problem"].unique())
    fig, axes = plt.subplots(1, len(problems), figsize=(4.8 * len(problems), 4.6), squeeze=False)
    for ax, problem in zip(axes[0], problems):
        subset = metrics_df[metrics_df["problem"] == problem]
        rows = []
        for algo in ALGO_ORDER:
            vals = subset.loc[subset["algorithm"] == algo, "feasibility_ratio"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append((algo, float(np.nanmean(vals))))
        if rows:
            ax.bar(
                [r[0] for r in rows],
                [r[1] for r in rows],
                color=[ALGO_COLORS.get(r[0], "#cccccc") for r in rows],
                alpha=0.85,
            )
        ax.set_ylim(0.0, 1.05)
        ax.set_title(problem)
        ax.set_ylabel("Mean feasibility ratio")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _median_iqr(values: pd.Series) -> str:
    arr = values.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return "NA"
    median = float(np.median(arr))
    q1, q3 = np.percentile(arr, [25, 75])
    return f"{median:.4f} [{(q3 - q1):.4f}]"


def write_main_tables(metrics_df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path, Path, Path]:
    summary = (
        metrics_df.groupby(["problem", "algorithm"], as_index=False)
        .agg(
            hv_norm=("hv_norm", _median_iqr),
            feasibility_ratio=("feasibility_ratio", _median_iqr),
            best_cv=("best_cv", _median_iqr),
            median_cv=("median_cv", _median_iqr),
            runtime_sec=("runtime_sec", _median_iqr),
            n_solutions=("n_solutions", _median_iqr),
        )
    )
    summary["algorithm"] = pd.Categorical(summary["algorithm"], categories=ALGO_ORDER, ordered=True)
    summary = summary.sort_values(["problem", "algorithm"])

    main_md = output_dir / "realworld_main_table.md"
    main_tex = output_dir / "realworld_main_table.tex"
    main_md.write_text(_markdown_table(summary), encoding="utf-8")
    main_tex.write_text(_latex_table(summary), encoding="utf-8")

    rank_rows = []
    for problem, group in metrics_df.groupby("problem"):
        algo_medians = (
            group.groupby("algorithm")["hv_norm"]
            .median()
            .sort_values(ascending=False)
        )
        ranks = algo_medians.rank(method="min", ascending=False)
        for algo, rank in ranks.items():
            rank_rows.append({"problem": problem, "algorithm": algo, "hv_rank": float(rank)})
    rank_df = pd.DataFrame(rank_rows)
    rank_summary = (
        rank_df.groupby("algorithm", as_index=False)["hv_rank"]
        .mean()
        .sort_values("hv_rank")
    )
    rank_tex = output_dir / "realworld_rank_summary.tex"
    rank_tex.write_text(_latex_table(rank_summary), encoding="utf-8")

    stats_csv = output_dir / "realworld_stats.csv"
    merged_stats = summary.merge(rank_summary, on="algorithm", how="left")
    merged_stats.to_csv(stats_csv, index=False)
    return main_tex, main_md, rank_tex, stats_csv


def _write_manifest(
    output_dir: Path,
    input_dirs: list[Path],
    generated_files: list[Path],
    metrics_df: pd.DataFrame,
) -> Path:
    lines = [
        "# Real-World Manifest",
        "",
        "## Input result directories",
    ]
    for path in input_dirs:
        lines.append(f"- {path}")
    lines.extend([
        "",
        "## Problems processed",
    ])
    for problem in sorted(metrics_df["problem"].unique()):
        lines.append(f"- {problem}")
    lines.extend([
        "",
        "## Generated outputs",
    ])
    for path in generated_files:
        lines.append(f"- {path}")
    manifest = output_dir / "realworld_manifest.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _discover_default_dirs(results_root: Path) -> list[Path]:
    dirs: list[Path] = []
    for key in ("cre_2obj", "cre_3obj"):
        path = _latest_matching_dir(results_root, FORMAL_PATTERNS[key])
        if path is not None:
            dirs.append(path)
    moon_dir = _latest_matching_dir(results_root, FORMAL_PATTERNS["moon"])
    if moon_dir is not None:
        dirs.append(moon_dir)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process real-world engineering experiments")
    parser.add_argument("results_dirs", nargs="*", help="Result directories to merge and analyse")
    args = parser.parse_args()

    input_dirs = [Path(p) for p in args.results_dirs] if args.results_dirs else _discover_default_dirs(RESULTS_ROOT)
    if not input_dirs:
        raise SystemExit("No real-world result directories found")

    run_data: dict[str, dict[str, list[Path]]] = {}
    for result_dir in input_dirs:
        data = load_problem_runs(result_dir)
        for problem_name, algos in data.items():
            if problem_name not in run_data:
                run_data[problem_name] = {}
            for algo_name, run_dirs in algos.items():
                run_data[problem_name].setdefault(algo_name, []).extend(run_dirs)

    rows: list[dict] = []
    for problem_name, problem_runs in run_data.items():
        zmin, zmax = build_problem_scaler(problem_runs)
        for algo_name, run_dirs in problem_runs.items():
            for run_dir in run_dirs:
                rows.append(compute_realworld_metrics(problem_name, algo_name, run_dir, zmin, zmax))

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        raise SystemExit("No real-world run metrics found")

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"{timestamp}_realworld_postprocess"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = output_dir / "realworld_metrics_long.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    generated_files: list[Path] = [metrics_csv]

    main_tex, main_md, rank_tex, stats_csv = write_main_tables(metrics_df, output_dir)
    generated_files.extend([main_tex, main_md, rank_tex, stats_csv])

    hv_plot = output_dir / "realworld_hv_boxplot.pdf"
    plot_hv_boxplot(metrics_df, hv_plot)
    generated_files.append(hv_plot)

    feas_plot = output_dir / "realworld_feasibility_bar.pdf"
    plot_feasibility_bar(metrics_df, feas_plot)
    generated_files.append(feas_plot)

    for problem_name in ("CRE21", "CRE31", "CRE32", "MOON"):
        if problem_name not in set(metrics_df["problem"]):
            continue
        pf_path = output_dir / f"pf_{problem_name}.pdf"
        if plot_pf_overlay(problem_name, metrics_df[metrics_df["problem"] == problem_name], pf_path):
            generated_files.append(pf_path)

    manifest = _write_manifest(output_dir, input_dirs, generated_files, metrics_df)
    generated_files.append(manifest)

    print(f"Real-world postprocess complete: {output_dir}")


if __name__ == "__main__":
    main()
