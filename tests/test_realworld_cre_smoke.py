from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from mosade.algorithm.mosade import MOSADE
from mosade.problems import get_problem
from mosade.runner.experiment import run_experiment


def test_mosade_smoke_runs_on_cre21_and_cre31():
    for name, pop_size in [("CRE21", 20), ("CRE31", 30)]:
        problem = get_problem(name)
        result = MOSADE(pop_size=pop_size, max_evals=500, seed=42, track_interval=0).run(problem)
        assert result.F.shape[1] == problem.n_obj
        assert result.X.shape[1] == problem.n_var
        assert result.F.shape[0] > 0
        assert np.isfinite(result.F).all()


def test_runner_writes_metrics_for_realworld_smoke(tmp_path: Path):
    results_dir = tmp_path / "results"
    cfg = {
        "tag": "realworld_cre_test_smoke",
        "results_dir": str(results_dir),
        "seed": 42,
        "parallel_workers": 1,
        "n_runs": 1,
        "problems": [{"name": "CRE21"}, {"name": "CRE31"}],
        "algorithms": [
            {
                "name": "MOSADE",
                "pop_size": 30,
                "max_evals": 500,
                "track_interval": 0,
            }
        ],
    }
    cfg_path = tmp_path / "realworld_cre_test_smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    run_dir = run_experiment(cfg_path)
    assert (run_dir / "summary.json").exists()

    for problem_name in ("CRE21", "CRE31"):
        run_subdir = run_dir / problem_name / "run_000"
        if not run_subdir.exists():
            run_subdir = run_dir / problem_name / "MOSADE" / "run_000"
        assert run_subdir.exists()
        assert (run_subdir / "metrics.json").exists()
        assert (run_subdir / "history.json").exists()
        assert (run_subdir / "pareto_approximation.txt").exists()
