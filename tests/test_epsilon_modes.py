from __future__ import annotations

import json

import numpy as np
import pytest

from mosade.algorithm.mosade import MOSADE
from mosade_experiments.analysis.tables import generate_main_results_table, generate_settings_table
from mosade.problems.base import Problem
from mosade.runner.experiment import run_experiment


class AlwaysInfeasibleToyProblem(Problem):
    """Small constrained problem with a strictly positive initial epsilon."""

    def __init__(self) -> None:
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=1,
            lower=np.zeros(2),
            upper=np.ones(2),
        )

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        f1 = X[:, 0]
        f2 = 1.0 - X[:, 0] + X[:, 1]
        g = 1.0 + 0.25 * X[:, 0]
        return np.column_stack([f1, f2]), g.reshape(-1, 1)


def _run_eps_mode(mode: str):
    problem = AlwaysInfeasibleToyProblem()
    algo = MOSADE(pop_size=20, max_evals=80, seed=11, eps_mode=mode)
    return algo.run(problem)


def test_eps_modes_produce_defined_trajectories() -> None:
    adaptive = _run_eps_mode("adaptive")
    fixed_initial = _run_eps_mode("fixed_initial")
    fixed_zero = _run_eps_mode("zero")

    eps_adaptive = np.asarray(adaptive.history["epsilon"], dtype=float)
    eps_initial = np.asarray(fixed_initial.history["epsilon"], dtype=float)
    eps_zero = np.asarray(fixed_zero.history["epsilon"], dtype=float)

    assert adaptive.metadata["eps_mode"] == "adaptive"
    assert fixed_initial.metadata["eps_mode"] == "fixed_initial"
    assert fixed_zero.metadata["eps_mode"] == "zero"

    assert adaptive.metadata["epsilon_0"] > 0.0
    assert np.all(np.diff(eps_adaptive) <= 1e-12)
    assert eps_adaptive[0] > eps_adaptive[-1]
    assert np.allclose(eps_initial, fixed_initial.metadata["epsilon_0"])
    assert np.allclose(eps_zero, 0.0)
    assert not np.allclose(eps_adaptive, eps_initial)
    assert not np.allclose(eps_initial, eps_zero)


def test_disable_epsilon_compatibility_maps_to_fixed_zero() -> None:
    algo = MOSADE(pop_size=20, max_evals=80, seed=11, disable_epsilon=True)
    assert algo.eps_mode == "zero"
    assert algo.disable_epsilon is True


def test_deprecated_fixed_eps_label_is_rejected_by_tables(tmp_path) -> None:
    run_dir = tmp_path / "results"
    for algo in ("MOSADE", "MOSADE_fixed_eps"):
        rd = run_dir / "ZDT1" / algo / "run_000"
        rd.mkdir(parents=True)
        (rd / "metrics.json").write_text(
            json.dumps({"seed": 1, "status": "ok", "hv": 1.0, "igd": 0.1}),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="MOSADE_fixed_eps"):
        generate_main_results_table(run_dir, metric="hv", fmt="markdown")


def test_deprecated_fixed_eps_config_is_rejected_by_settings_table(tmp_path) -> None:
    config = tmp_path / "old_fixed_eps.yaml"
    config.write_text(
        "\n".join(
            [
                'tag: "old"',
                "algorithms:",
                "  - name: MOSADE_fixed_eps",
                "    type: MOSADE_fixed_eps",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="MOSADE_fixed_eps"):
        generate_settings_table(config, fmt="markdown")


def test_runner_writes_epsilon_run_metadata(tmp_path) -> None:
    config = tmp_path / "eps_run.yaml"
    run_dir = tmp_path / "run"
    config.write_text(
        "\n".join(
            [
                'tag: "eps_run"',
                f'results_dir: "{tmp_path.as_posix()}"',
                "seed: 7",
                "n_runs: 1",
                "algorithms:",
                "  - name: MOSADE_eps_zero",
                "    type: MOSADE_eps_zero",
                "    pop_size: 20",
                "    max_evals: 40",
                "problems:",
                "  - ZDT1",
            ]
        ),
        encoding="utf-8",
    )

    run_experiment(config, run_dir=run_dir, workers=1)
    metadata_path = run_dir / "ZDT1" / "MOSADE_eps_zero" / "run_000" / "run_metadata.json"
    metrics_path = run_dir / "ZDT1" / "MOSADE_eps_zero" / "run_000" / "metrics.json"

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metadata["algorithm_type"] == "MOSADE_eps_zero"
    assert metadata["epsilon_mode"] == "zero"
    assert metadata["seed"] != "NA"
    assert metrics["eps_mode"] == "zero"
    assert "epsilon_history_sha256" in metrics
