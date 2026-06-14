"""Tests for merging suite-level benchmark result directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestMergeResults:
    def _write_problem(
        self,
        root: Path,
        problem_name: str,
        algo_metrics: dict[str, tuple[float, float]],
    ) -> None:
        prob_dir = root / problem_name
        prob_dir.mkdir(parents=True)

        problem_summary = {}
        true_pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        np.savetxt(prob_dir / "pareto_front.txt", true_pf)

        for seed, (algo, (hv, igd)) in enumerate(algo_metrics.items(), start=1):
            run_dir = prob_dir / algo / "run_000"
            run_dir.mkdir(parents=True)
            problem_summary[algo] = {
                "hv_median": hv,
                "igd_median": igd,
                "hv_n_valid": 1,
                "igd_n_valid": 1,
                "status": "ok",
            }
            history = {
                "gen": [1, 2, 3],
                "convergence": [
                    {"n_evals": 100, "hv": hv * 0.8, "igd": igd * 1.2},
                    {"n_evals": 200, "hv": hv, "igd": igd},
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
            raw = np.array([[0.1, 0.9], [0.4, 0.6], [0.7, 0.8]])
            np.savetxt(run_dir / "objectives.txt", raw)
            np.savetxt(run_dir / "pareto_approximation.txt", raw[:2])
            np.savetxt(run_dir / "decisions.txt", np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
            np.savetxt(run_dir / "pareto_approximation_decisions.txt", np.array([[0.1, 0.2], [0.3, 0.4]]))

        (prob_dir / "summary.json").write_text(json.dumps(problem_summary), encoding="utf-8")

    def _write_source_result(
        self,
        root: Path,
        tag: str,
        problems: dict[str, dict[str, tuple[float, float]]],
    ) -> Path:
        result_dir = root / tag
        result_dir.mkdir(parents=True)
        config = {
            "tag": tag,
            "results_dir": "results",
            "seed": 42,
            "n_runs": 31,
            "algorithms": [
                {"name": "MOSADE", "pop_size": 100, "max_evals": 300000},
                {"name": "NSGA2", "pop_size": 100, "max_evals": 300000},
            ],
        }
        (result_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        for problem_name, algo_metrics in problems.items():
            self._write_problem(result_dir, problem_name, algo_metrics)
        return result_dir

    def test_merge_results_dirs_support_tables_and_plots(self, tmp_path):
        from mosade_experiments.analysis.merge import merge_results_dirs
        from mosade_experiments.analysis.plotting import plot_experiment_results
        from mosade_experiments.analysis.tables import generate_main_results_table, generate_ranking_table

        src_zdt = self._write_source_result(
            tmp_path,
            "benchmark_zdt_result",
            {"ZDT1": {"MOSADE": (0.91, 0.08), "NSGA2": (0.84, 0.12)}},
        )
        src_wfg = self._write_source_result(
            tmp_path,
            "benchmark_wfg_result",
            {"WFG1": {"MOSADE": (0.73, 0.21), "NSGA2": (0.68, 0.25)}},
        )

        merged_dir = tmp_path / "benchmark_merged"
        manifest = merge_results_dirs([src_zdt, src_wfg], merged_dir)

        assert merged_dir.exists()
        assert (merged_dir / "merge_manifest.json").exists()
        assert (merged_dir / "summary.json").exists()
        assert manifest["algorithms"] == ["MOSADE", "NSGA2"]
        assert manifest["problems"] == ["WFG1", "ZDT1"]

        ranking_md = generate_ranking_table(merged_dir, fmt="markdown")
        assert "Average Friedman Ranks" in ranking_md
        assert "MOSADE" in ranking_md and "NSGA2" in ranking_md

        main_md = generate_main_results_table(merged_dir, metric="hv", fmt="markdown")
        assert "ZDT1" in main_md and "WFG1" in main_md

        plot_experiment_results(merged_dir, pf_selection="median_igd")
        assert (merged_dir / "plots" / "summary" / "rank_heatmap_hv.png").exists()
        assert (merged_dir / "plots" / "suites" / "ZDT_hv_grid.png").exists()
        assert (merged_dir / "plots" / "suites" / "WFG_hv_grid.png").exists()

    def test_merge_rejects_duplicate_problem_names(self, tmp_path):
        from mosade_experiments.analysis.merge import merge_results_dirs

        src_a = self._write_source_result(
            tmp_path,
            "suite_a",
            {"ZDT1": {"MOSADE": (0.91, 0.08), "NSGA2": (0.84, 0.12)}},
        )
        src_b = self._write_source_result(
            tmp_path,
            "suite_b",
            {"ZDT1": {"MOSADE": (0.92, 0.07), "NSGA2": (0.85, 0.11)}},
        )

        with pytest.raises(ValueError, match="Duplicate problem 'ZDT1'"):
            merge_results_dirs([src_a, src_b], tmp_path / "merged_duplicate")
