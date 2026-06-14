from __future__ import annotations

import json
import math

import numpy as np

from mosade.algorithm.pymoo_wrapper import PymooAlgorithm
from mosade.problems import get_problem
from mosade.runner.experiment import run_experiment


def test_moead_de_constrained_returns_explicit_unsupported() -> None:
    problem = get_problem("DASCMOP7", difficulty=7)
    algo = PymooAlgorithm("MOEAD_DE", pop_size=20, max_evals=100, seed=0)

    result = algo.run(problem)

    assert result.status == "unsupported"
    assert result.message is not None
    assert "constrained" in result.message.lower()
    assert result.n_evals == 0
    assert result.F.shape == (0, problem.n_obj)
    assert result.X.shape == (0, problem.n_var)


def test_runner_marks_constrained_moead_de_as_unsupported(tmp_path) -> None:
    config_path = tmp_path / "unsupported.yaml"
    config_path.write_text(
        "\n".join(
            [
                'tag: "unsupported_test"',
                'results_dir: "results"',
                "seed: 42",
                "n_runs: 1",
                "",
                "algorithms:",
                "  - name: MOEAD",
                "    pop_size: 20",
                "    max_evals: 100",
                "  - name: MOEAD_DE",
                "    pop_size: 20",
                "    max_evals: 100",
                "",
                "problems:",
                "  - {name: DASCMOP7, difficulty: 7}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    run_dir = tmp_path / "results_dir"
    result_dir = run_experiment(config_path, run_dir=run_dir)
    problem_dir = result_dir / "DASCMOP7_difficulty7"

    with open(problem_dir / "MOEAD_DE" / "run_000" / "metrics.json", encoding="utf-8") as f:
        moead_de_metrics = json.load(f)
    with open(problem_dir / "MOEAD" / "run_000" / "metrics.json", encoding="utf-8") as f:
        moead_metrics = json.load(f)
    with open(problem_dir / "MOEAD_DE" / "summary.json", encoding="utf-8") as f:
        moead_de_summary = json.load(f)
    log_text = (result_dir / "experiment.log").read_text(encoding="utf-8")

    assert moead_de_metrics["status"] == "unsupported"
    assert math.isnan(float(moead_de_metrics["hv"]))
    assert math.isnan(float(moead_de_metrics["n_solutions"]))
    assert moead_de_summary["status"] == "unsupported"
    assert moead_de_summary["n_unsupported"] == 1
    assert moead_de_summary["hv_n_valid"] == 0
    assert "MOEAD_DE unsupported on constrained problem DASCMOP7_difficulty7" in log_text

    assert moead_metrics["status"] == "ok"
    assert np.isfinite(float(moead_metrics["hv"]))
