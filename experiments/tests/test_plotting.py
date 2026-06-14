"""Tests for mosade_experiments.analysis.plotting — new functions added in this session.

All tests use the Agg (non-interactive) backend, so they run without a display.
Each test that exercises a plotting function checks that the output file is
actually written to disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_results_config(
    root: Path,
    problems: list[str | dict],
    algorithms: list[dict] | None = None,
) -> None:
    """Write a minimal frozen results config.json for plotting tests."""
    config: dict[str, object] = {
        "tag": root.name,
        "results_dir": str(root),
        "seed": 42,
        "n_runs": 1,
        "problems": problems,
    }
    if algorithms:
        config["algorithms"] = algorithms
    else:
        config["algorithm"] = {"type": "MOSADE", "pop_size": 20, "max_evals": 200}
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")


# ---------------------------------------------------------------------------
# Import smoke test - all public names must be importable
# ---------------------------------------------------------------------------

def test_imports():
    from mosade_experiments.analysis.plotting import (
        algorithm_display_name,
        compute_rank_table,
        plot_epsilon_feasibility,
        plot_experiment_results,
        plot_grouped_boxplots,
        plot_multi_algorithm_convergence,
        plot_multi_algorithm_pf,
        plot_rank_heatmap,
        plot_suite_metric_grid,
        problem_display_label,
    )
    assert callable(algorithm_display_name)
    assert callable(plot_multi_algorithm_pf)
    assert callable(plot_multi_algorithm_convergence)
    assert callable(plot_grouped_boxplots)
    assert callable(plot_rank_heatmap)
    assert callable(plot_epsilon_feasibility)
    assert callable(plot_experiment_results)
    assert callable(compute_rank_table)
    assert callable(plot_suite_metric_grid)
    assert callable(problem_display_label)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_algo_pf_data():
    """Two 2-D approximation sets and a reference PF."""
    pf = np.column_stack([np.linspace(0, 1, 20), np.linspace(1, 0, 20)])
    F1 = pf + np.random.default_rng(0).normal(0, 0.05, pf.shape)
    F2 = pf + np.random.default_rng(1).normal(0, 0.08, pf.shape)
    return {"MOSADE": F1, "NSGA2": F2}, pf


@pytest.fixture()
def convergence_dict():
    """Two algorithms, two runs each, with hv snapshots."""
    def _make_run(seed, n=5):
        rng = np.random.default_rng(seed)
        evals = [100 * (i + 1) for i in range(n)]
        hv = sorted(rng.uniform(0.3, 0.9, n).tolist())
        return [{"n_evals": e, "hv": h, "igd": 0.1} for e, h in zip(evals, hv)]

    return {
        "MOSADE": [_make_run(0), _make_run(1)],
        "NSGA2": [_make_run(2), _make_run(3)],
    }


@pytest.fixture()
def grouped_box_data():
    rng = np.random.default_rng(42)
    return {
        "ZDT1": {
            "MOSADE": rng.uniform(0.75, 0.90, 10).tolist(),
            "NSGA2": rng.uniform(0.70, 0.85, 10).tolist(),
            "MOEAD": rng.uniform(0.72, 0.88, 10).tolist(),
        },
        "ZDT2": {
            "MOSADE": rng.uniform(0.65, 0.80, 10).tolist(),
            "NSGA2": rng.uniform(0.60, 0.75, 10).tolist(),
            "MOEAD": rng.uniform(0.62, 0.78, 10).tolist(),
        },
    }


@pytest.fixture()
def rank_table_dict():
    return {
        "ZDT1": {"MOSADE": 1.0, "NSGA2": 2.0, "MOEAD": 3.0},
        "ZDT2": {"MOSADE": 1.5, "NSGA2": 2.5, "MOEAD": 2.0},
        "ZDT3": {"MOSADE": 2.0, "NSGA2": 1.0, "MOEAD": 3.0},
    }


@pytest.fixture()
def mosade_history():
    n = 20
    return {
        "n_constr": 11,
        "gen": list(range(1, n + 1)),
        "n_evals": [i * 500 for i in range(1, n + 1)],
        "epsilon": [max(0.0, 0.5 - i * 0.025) for i in range(n)],
        "feasibility_ratio": [min(1.0, i * 0.05) for i in range(n)],
        "mean_cv": [max(0.0, 2.0 - i * 0.08) for i in range(n)],
        "strategy_probs": [[0.25, 0.25, 0.25, 0.25]] * n,
    }


@pytest.fixture()
def unconstrained_history():
    return {
        "n_constr": 0,
        "gen": [1, 2, 3],
        "epsilon": [0.0, 0.0, 0.0],
        "feasibility_ratio": [1.0, 1.0, 1.0],
        "mean_cv": [0.0, 0.0, 0.0],
    }


# ---------------------------------------------------------------------------
# display helpers
# ---------------------------------------------------------------------------

def test_display_helpers():
    from mosade_experiments.analysis.plotting import algorithm_display_name, problem_display_label

    assert algorithm_display_name("MOEAD") == "MOEA/D"
    assert algorithm_display_name("MOEAD_DE") == "MOEA/D-DE"
    assert algorithm_display_name("NSGA2") == "NSGA-II"
    assert algorithm_display_name("NSGA3") == "NSGA-III"
    assert algorithm_display_name("SMSEMOA") == "SMS-EMOA"
    assert problem_display_label("DASCMOP7_difficulty7") == "DASCMOP7\nd=7"
    assert problem_display_label("DTLZ2_n_obj3") == "DTLZ2\nM=3"


# ---------------------------------------------------------------------------
# plot_multi_algorithm_pf
# ---------------------------------------------------------------------------

class TestPlotMultiAlgorithmPF:
    def test_creates_file(self, two_algo_pf_data, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_pf
        results_dict, pf = two_algo_pf_data
        out = tmp_path / "pf_multi.png"
        plot_multi_algorithm_pf(results_dict, pf, title="Test", save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_no_pf(self, two_algo_pf_data, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_pf
        results_dict, _ = two_algo_pf_data
        out = tmp_path / "pf_no_ref.png"
        plot_multi_algorithm_pf(results_dict, PF=None, save_path=out)
        assert out.exists()

    def test_single_algorithm(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_pf
        F = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]])
        out = tmp_path / "pf_single.png"
        plot_multi_algorithm_pf({"MOSADE": F}, save_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_multi_algorithm_convergence
# ---------------------------------------------------------------------------

class TestPlotMultiAlgorithmConvergence:
    def test_creates_file(self, convergence_dict, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_convergence
        out = tmp_path / "convergence.png"
        plot_multi_algorithm_convergence(
            convergence_dict, metric_key="hv", save_path=out
        )
        assert out.exists() and out.stat().st_size > 0

    def test_empty_runs_skipped(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_convergence
        data = {"MOSADE": [[], []]}  # all empty runs
        out = tmp_path / "convergence_empty.png"
        # Should not raise; may produce an empty figure
        plot_multi_algorithm_convergence(data, save_path=out)

    def test_missing_metric_key_skipped(self, convergence_dict, tmp_path):
        from mosade_experiments.analysis.plotting import plot_multi_algorithm_convergence
        out = tmp_path / "convergence_missing.png"
        # "xyz" key is absent from all snapshots → runs are skipped silently
        plot_multi_algorithm_convergence(
            convergence_dict, metric_key="xyz", save_path=out
        )


# ---------------------------------------------------------------------------
# plot_grouped_boxplots
# ---------------------------------------------------------------------------

class TestPlotGroupedBoxplots:
    def test_creates_file(self, grouped_box_data, tmp_path):
        from mosade_experiments.analysis.plotting import plot_grouped_boxplots
        out = tmp_path / "grouped.png"
        plot_grouped_boxplots(grouped_box_data, metric_name="HV", save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_missing_algo_in_some_problems(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_grouped_boxplots
        data = {
            "ZDT1": {"MOSADE": [0.8, 0.82], "NSGA2": [0.75, 0.77]},
            "ZDT2": {"MOSADE": [0.7, 0.71]},  # NSGA2 absent
        }
        out = tmp_path / "grouped_partial.png"
        plot_grouped_boxplots(data, save_path=out)
        assert out.exists()

    def test_many_problems_rotates_labels(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_grouped_boxplots
        rng = np.random.default_rng(0)
        data = {f"P{i}": {"A": rng.uniform(0, 1, 5).tolist()} for i in range(8)}
        out = tmp_path / "grouped_many.png"
        plot_grouped_boxplots(data, save_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_suite_metric_grid
# ---------------------------------------------------------------------------

class TestPlotSuiteMetricGrid:
    def test_creates_png_and_svg(self, grouped_box_data, tmp_path):
        from mosade_experiments.analysis.plotting import plot_suite_metric_grid

        out = tmp_path / "ZDT_hv_grid.png"
        plot_suite_metric_grid("ZDT", grouped_box_data, metric_name="HV", save_path=out)

        assert out.exists() and out.stat().st_size > 0
        assert out.with_suffix(".svg").exists() and out.with_suffix(".svg").stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_rank_heatmap
# ---------------------------------------------------------------------------

class TestPlotRankHeatmap:
    def test_creates_file(self, rank_table_dict, tmp_path):
        from mosade_experiments.analysis.plotting import plot_rank_heatmap
        out = tmp_path / "heatmap.png"
        plot_rank_heatmap(rank_table_dict, title="Rank Test", save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_two_algo_two_problem(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_rank_heatmap
        rt = {"ZDT1": {"A": 1.0, "B": 2.0}, "ZDT2": {"A": 2.0, "B": 1.0}}
        out = tmp_path / "heatmap_small.png"
        plot_rank_heatmap(rt, save_path=out)
        assert out.exists()

    def test_pandas_dataframe_accepted(self, rank_table_dict, tmp_path):
        pd = pytest.importorskip("pandas")
        from mosade_experiments.analysis.plotting import plot_rank_heatmap
        df = pd.DataFrame(rank_table_dict).T  # rows=problems, cols=algos
        out = tmp_path / "heatmap_df.png"
        plot_rank_heatmap(df, save_path=out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_epsilon_feasibility
# ---------------------------------------------------------------------------

class TestPlotEpsilonFeasibility:
    def test_creates_file(self, mosade_history, tmp_path):
        from mosade_experiments.analysis.plotting import plot_epsilon_feasibility
        out = tmp_path / "eps_feas.png"
        plot_epsilon_feasibility(mosade_history, save_path=out)
        assert out.exists() and out.stat().st_size > 0

    def test_unconstrained_history_skipped(self, unconstrained_history, tmp_path):
        from mosade_experiments.analysis.plotting import plot_epsilon_feasibility
        out = tmp_path / "unconstrained.png"
        plot_epsilon_feasibility(unconstrained_history, save_path=out)
        assert not out.exists()

    def test_missing_cv_series_returns_early(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_epsilon_feasibility
        history = {
            "n_constr": 7,
            "gen": [1, 2, 3],
            "epsilon": [0.5, 0.3, 0.0],
            "feasibility_ratio": [0.1, 0.5, 0.9],
        }
        out = tmp_path / "missing_cv.png"
        plot_epsilon_feasibility(history, save_path=out)
        assert not out.exists()


# ---------------------------------------------------------------------------
# plot_experiment_results constraint dynamics integration
# ---------------------------------------------------------------------------

class TestPlotExperimentConstraintDynamics:
    def _write_single_algo_problem(
        self,
        root: Path,
        problem_name: str,
        history: dict,
        seed: int = 42,
    ) -> None:
        if problem_name.startswith("DASCMOP") and "_difficulty" in problem_name:
            base, difficulty = problem_name.split("_difficulty", 1)
            problems = [{"name": base, "difficulty": int(difficulty)}]
        else:
            problems = [problem_name]
        prob_dir = root / problem_name
        run_dir = prob_dir / "run_000"
        run_dir.mkdir(parents=True)
        _write_results_config(root, problems)

        (prob_dir / "summary.json").write_text(json.dumps({"hv_mean": 0.8}), encoding="utf-8")
        np.savetxt(prob_dir / "pareto_front.txt", np.array([[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]]))
        (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "hv": 0.8,
                    "igd": 0.2,
                    "pf_source": "archive",
                    "pf_file": "pareto_approximation.txt",
                    "pf_debug_file": "objectives.txt",
                }
            ),
            encoding="utf-8",
        )
        raw = np.array([[0.1, 0.9], [0.4, 0.6], [0.8, 0.2]])
        np.savetxt(run_dir / "objectives.txt", raw)
        np.savetxt(run_dir / "pareto_approximation.txt", raw[:2])

    def test_unconstrained_problem_skips_constraint_dynamics(self, tmp_path, unconstrained_history):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        self._write_single_algo_problem(tmp_path, "ZDT1", unconstrained_history)
        plot_experiment_results(tmp_path)

        plots_dir = tmp_path / "plots" / "problems"
        assert not list(plots_dir.glob("constraint_dynamics_*"))
        assert not list((tmp_path / "plots").rglob("*epsilon_feasibility*"))

    def test_constrained_problem_generates_constraint_dynamics(self, tmp_path, mosade_history):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        self._write_single_algo_problem(tmp_path, "DASCMOP7_difficulty7", mosade_history, seed=7)
        plot_experiment_results(tmp_path)

        plots_dir = tmp_path / "plots" / "problems"
        expected = plots_dir / "constraint_dynamics_DASCMOP7_difficulty7_MOSADE_median_hv_seed7.png"
        assert expected.exists() and expected.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot_experiment_results summary/suite layout
# ---------------------------------------------------------------------------

class TestPlotExperimentSummaryLayout:
    def _write_multi_algo_problem(
        self,
        root: Path,
        problem_name: str,
        algo_metrics: dict[str, tuple[float, float]],
    ) -> None:
        prob_dir = root / problem_name
        prob_dir.mkdir(parents=True)
        np.savetxt(prob_dir / "pareto_front.txt", np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]))

        summary = {}
        for seed, (algo, (hv, igd)) in enumerate(algo_metrics.items(), start=1):
            run_dir = prob_dir / algo / "run_000"
            run_dir.mkdir(parents=True)
            summary[algo] = {"hv_median": hv, "igd_median": igd, "status": "ok"}
            history = {
                "gen": [1, 2, 3],
                "convergence": [
                    {"n_evals": 100, "hv": hv * 0.7, "igd": igd * 1.2},
                    {"n_evals": 200, "hv": hv * 0.85, "igd": igd * 1.1},
                    {"n_evals": 300, "hv": hv, "igd": igd},
                ],
            }
            if algo == "MOSADE":
                history["strategy_probs"] = [[0.25, 0.25, 0.25, 0.25]] * 3
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "hv": hv,
                        "igd": igd,
                        "pf_source": "archive",
                        "pf_file": "pareto_approximation.txt",
                        "pf_debug_file": "objectives.txt",
                    }
                ),
                encoding="utf-8",
            )
            raw = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1], [0.7, 0.8]])
            np.savetxt(run_dir / "objectives.txt", raw)
            np.savetxt(run_dir / "pareto_approximation.txt", raw[:3])

        (prob_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    def test_multi_algo_outputs_summary_and_suite_dirs(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        metrics_a = {"MOSADE": (0.92, 0.08), "NSGA2": (0.84, 0.12), "MOEAD": (0.80, 0.15)}
        metrics_b = {"MOSADE": (0.89, 0.09), "NSGA2": (0.82, 0.14), "MOEAD": (0.79, 0.16)}
        self._write_multi_algo_problem(tmp_path, "ZDT1", metrics_a)
        self._write_multi_algo_problem(tmp_path, "ZDT3", metrics_b)
        _write_results_config(
            tmp_path,
            ["ZDT1", "ZDT3"],
            algorithms=[
                {"name": "MOSADE", "type": "MOSADE", "pop_size": 20, "max_evals": 200},
                {"name": "NSGA2", "type": "NSGA2", "pop_size": 20, "max_evals": 200},
                {"name": "MOEAD", "type": "MOEAD", "pop_size": 20, "max_evals": 200},
            ],
        )

        plot_experiment_results(tmp_path)

        plots_dir = tmp_path / "plots"
        assert not list(plots_dir.rglob("all_problems_hv_grouped*"))
        assert not list(plots_dir.rglob("all_problems_igd_grouped*"))
        assert (plots_dir / "summary" / "rank_heatmap_hv.png").exists()
        assert (plots_dir / "summary" / "rank_heatmap_hv.svg").exists()
        assert (plots_dir / "summary" / "rank_heatmap_igd.png").exists()
        assert (plots_dir / "suites" / "ZDT_hv_grid.png").exists()
        assert (plots_dir / "suites" / "ZDT_igd_grid.png").exists()
        pf_plot = (
            plots_dir
            / "problems"
            / f"pf_{tmp_path.name}_ZDT1_median_igd_page01.png"
        )
        assert pf_plot.exists()
        assert pf_plot.with_suffix(".json").exists()
        assert (plots_dir / "problems" / "ZDT1_convergence_hv.png").exists()


# ---------------------------------------------------------------------------
# plot_experiment_results PF selection and provenance
# ---------------------------------------------------------------------------

class TestPlotExperimentPFSelection:
    def _write_pf_run(
        self,
        run_dir: Path,
        seed: int,
        hv: float,
        igd: float,
        raw_points: np.ndarray,
        formal_points: np.ndarray,
        pf_source: str = "archive",
    ) -> None:
        run_dir.mkdir(parents=True)
        (run_dir / "history.json").write_text(json.dumps({"gen": [1, 2, 3]}), encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "hv": hv,
                    "igd": igd,
                    "pf_source": pf_source,
                    "pf_file": "pareto_approximation.txt",
                    "pf_debug_file": "objectives.txt",
                }
            ),
            encoding="utf-8",
        )
        np.savetxt(run_dir / "objectives.txt", raw_points)
        np.savetxt(run_dir / "pareto_approximation.txt", formal_points)
        np.savetxt(run_dir / "decisions.txt", raw_points)
        np.savetxt(run_dir / "pareto_approximation_decisions.txt", formal_points)

    def test_pf_selection_is_deterministic_across_repeated_generation(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        prob_dir = tmp_path / "ZDT3"
        prob_dir.mkdir()
        _write_results_config(tmp_path, ["ZDT3"])
        (prob_dir / "summary.json").write_text(json.dumps({"hv_mean": 0.8}), encoding="utf-8")
        np.savetxt(prob_dir / "pareto_front.txt", np.array([[0.05, 0.95], [0.2, 0.85], [0.4, 0.7]]))

        self._write_pf_run(
            prob_dir / "run_000",
            seed=11,
            hv=0.75,
            igd=0.30,
            raw_points=np.array([[0.1, 0.9], [0.3, 0.8], [0.9, 0.6]]),
            formal_points=np.array([[0.1, 0.9], [0.3, 0.8]]),
        )
        self._write_pf_run(
            prob_dir / "run_001",
            seed=12,
            hv=0.80,
            igd=0.20,
            raw_points=np.array([[0.2, 0.85], [0.4, 0.7], [0.95, 0.7]]),
            formal_points=np.array([[0.2, 0.85], [0.4, 0.7]]),
        )
        self._write_pf_run(
            prob_dir / "run_002",
            seed=13,
            hv=0.82,
            igd=0.10,
            raw_points=np.array([[0.05, 0.95], [0.25, 0.82], [0.8, 0.5]]),
            formal_points=np.array([[0.05, 0.95], [0.25, 0.82]]),
        )

        plot_experiment_results(tmp_path, pf_selection="median_igd")
        manifest_path = (
            tmp_path
            / "plots"
            / "problems"
            / f"pf_{tmp_path.name}_ZDT3_median_igd_page01.json"
        )
        first = json.loads(manifest_path.read_text(encoding="utf-8"))

        plot_experiment_results(tmp_path, pf_selection="median_igd")
        second = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert first == second
        assert first["selections"][0]["run_id"] == 2
        assert first["selections"][0]["seed"] == 13

    def test_formal_pf_plot_uses_pareto_approximation_not_raw_objectives(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        prob_dir = tmp_path / "ZDT3"
        prob_dir.mkdir()
        _write_results_config(tmp_path, ["ZDT3"])
        (prob_dir / "summary.json").write_text(json.dumps({"hv_mean": 0.8}), encoding="utf-8")
        np.savetxt(prob_dir / "pareto_front.txt", np.array([[0.1, 0.9], [0.3, 0.8], [0.5, 0.7]]))

        raw_points = np.array([[0.1, 0.9], [0.3, 0.8], [0.6, 0.9], [0.9, 0.7]])
        formal_points = np.array([[0.1, 0.9], [0.3, 0.8]])
        self._write_pf_run(
            prob_dir / "run_000",
            seed=21,
            hv=0.81,
            igd=0.12,
            raw_points=raw_points,
            formal_points=formal_points,
        )

        plot_experiment_results(tmp_path, pf_selection="median_igd")
        manifest_path = (
            tmp_path
            / "plots"
            / "problems"
            / f"pf_{tmp_path.name}_ZDT3_median_igd_page01.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert manifest["plot_mode"] == "formal_page"
        assert manifest["selections"][0]["selected_points_path"] == "pareto_approximation.txt"
        assert manifest["selections"][0]["selected_points_count"] == 2

    def test_more_than_max_algorithms_paginate_pf_overlay(self, tmp_path):
        from mosade_experiments.analysis.plotting import plot_experiment_results

        prob_dir = tmp_path / "ZDT3"
        prob_dir.mkdir()
        np.savetxt(prob_dir / "pareto_front.txt", np.array([[0.1, 0.9], [0.4, 0.7], [0.8, 0.3]]))
        (prob_dir / "summary.json").write_text(
            json.dumps(
                {
                    "MOSADE": {"hv_median": 0.92, "igd_median": 0.05},
                    "NSGA2": {"hv_median": 0.90, "igd_median": 0.08},
                    "MOEAD": {"hv_median": 0.88, "igd_median": 0.10},
                    "NSGA3": {"hv_median": 0.86, "igd_median": 0.15},
                    "SPEA2": {"hv_median": 0.84, "igd_median": 0.20},
                    "SMSEMOA": {"hv_median": 0.83, "igd_median": 0.22},
                    "MOEAD_DE": {"hv_median": 0.82, "igd_median": 0.24},
                }
            ),
            encoding="utf-8",
        )

        algo_specs = {
            "MOSADE": (0.92, 0.05),
            "NSGA2": (0.90, 0.08),
            "MOEAD": (0.88, 0.10),
            "NSGA3": (0.86, 0.15),
            "SPEA2": (0.84, 0.20),
            "SMSEMOA": (0.83, 0.22),
            "MOEAD_DE": (0.82, 0.24),
        }
        for seed, (algo, (hv, igd)) in enumerate(algo_specs.items(), start=1):
            self._write_pf_run(
                prob_dir / algo / "run_000",
                seed=seed,
                hv=hv,
                igd=igd,
                raw_points=np.array([[0.1, 0.9], [0.4, 0.7], [0.9, 0.3]]),
                formal_points=np.array([[0.1, 0.9], [0.4, 0.7]]),
            )

        _write_results_config(
            tmp_path,
            ["ZDT3"],
            algorithms=[{"name": name, "type": name, "pop_size": 20, "max_evals": 200} for name in algo_specs],
        )

        plot_experiment_results(tmp_path, pf_selection="median_igd", max_pf_algorithms=3)
        manifest_path_1 = (
            tmp_path
            / "plots"
            / "problems"
            / f"pf_{tmp_path.name}_ZDT3_median_igd_page01.json"
        )
        manifest_path_2 = (
            tmp_path
            / "plots"
            / "problems"
            / f"pf_{tmp_path.name}_ZDT3_median_igd_page02.json"
        )
        manifest_1 = json.loads(manifest_path_1.read_text(encoding="utf-8"))
        manifest_2 = json.loads(manifest_path_2.read_text(encoding="utf-8"))

        assert manifest_1["plot_mode"] == "formal_page"
        assert manifest_2["plot_mode"] == "formal_page"
        assert len(manifest_1["selections"]) == 3
        assert len(manifest_2["selections"]) == 3


# ---------------------------------------------------------------------------
# compute_rank_table
# ---------------------------------------------------------------------------

class TestComputeRankTable:
    def _make_summary(self, tmp_path: Path, prob: str, algo_medians: dict) -> None:
        """Write a multi-algo summary.json for a mock problem directory."""
        prob_dir = tmp_path / prob
        prob_dir.mkdir()
        summary = {
            algo: {"hv_median": med, "igd_median": 1.0 / (med + 1e-6)}
            for algo, med in algo_medians.items()
        }
        (prob_dir / "summary.json").write_text(json.dumps(summary))

    def test_basic_ranking(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table
        # MOSADE has higher HV (better) on both problems
        self._make_summary(tmp_path, "ZDT1", {"MOSADE": 0.9, "NSGA2": 0.8, "MOEAD": 0.85})
        self._make_summary(tmp_path, "ZDT2", {"MOSADE": 0.88, "NSGA2": 0.75, "MOEAD": 0.80})

        rt = compute_rank_table(tmp_path, metric_key="hv", higher_is_better=True)
        assert "ZDT1" in rt and "ZDT2" in rt
        assert rt["ZDT1"]["MOSADE"] == 1.0  # highest HV → rank 1
        assert rt["ZDT1"]["NSGA2"] == 3.0   # lowest HV → rank 3

    def test_igd_ranking_inverted(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table
        # Write explicit igd_median values: A=0.05 (better), B=0.10 (worse)
        prob_dir = tmp_path / "ZDT1"
        prob_dir.mkdir()
        summary = {
            "A": {"igd_median": 0.05},
            "B": {"igd_median": 0.10},
        }
        (prob_dir / "summary.json").write_text(json.dumps(summary))
        rt = compute_rank_table(tmp_path, metric_key="igd", higher_is_better=False)
        assert rt["ZDT1"]["A"] == 1.0  # lower IGD is better
        assert rt["ZDT1"]["B"] == 2.0

    def test_single_algo_skipped(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table
        # Single-algo summary: top-level values are floats, not dicts
        prob_dir = tmp_path / "ZDT1"
        prob_dir.mkdir()
        (prob_dir / "summary.json").write_text(
            json.dumps({"hv_median": 0.9, "hv_mean": 0.89})
        )
        rt = compute_rank_table(tmp_path, metric_key="hv")
        assert "ZDT1" not in rt  # single-algo → excluded

    def test_plots_dir_ignored(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table
        (tmp_path / "plots").mkdir()
        rt = compute_rank_table(tmp_path)
        assert "plots" not in rt

    def test_higher_is_better_auto_detection(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table
        self._make_summary(tmp_path, "ZDT1", {"A": 0.9, "B": 0.7})
        rt_hv = compute_rank_table(tmp_path, metric_key="hv")   # higher=True
        rt_igd = compute_rank_table(tmp_path, metric_key="igd")  # higher=False (auto)
        # For HV: A (0.9) is better → rank 1
        assert rt_hv["ZDT1"]["A"] == 1.0
        # For IGD: summary has igd_median = 1/(0.9+eps) ≈ 1.11 for A,
        #          1/(0.7+eps) ≈ 1.43 for B — lower igd → A is still rank 1
        assert rt_igd["ZDT1"]["A"] == 1.0

    def test_unsupported_entries_are_skipped(self, tmp_path):
        from mosade_experiments.analysis.plotting import compute_rank_table, plot_rank_heatmap

        prob_dir = tmp_path / "DASCMOP7_difficulty7"
        prob_dir.mkdir()
        (prob_dir / "summary.json").write_text(
            json.dumps(
                {
                    "MOSADE": {"hv_median": 0.9, "status": "ok"},
                    "NSGA2": {"hv_median": 0.8, "status": "ok"},
                    "MOEAD_DE": {
                        "hv_median": float("nan"),
                        "hv_n_valid": 0,
                        "status": "unsupported",
                    },
                }
            ),
            encoding="utf-8",
        )

        rt = compute_rank_table(tmp_path, metric_key="hv")
        assert rt["DASCMOP7_difficulty7"]["MOSADE"] == 1.0
        assert "MOEAD_DE" not in rt["DASCMOP7_difficulty7"]

        out = tmp_path / "heatmap_unsupported.png"
        plot_rank_heatmap(rt, save_path=out)
        assert out.exists() and out.stat().st_size > 0
