from __future__ import annotations

from mosade.algorithm.mosade import MOSADE
from mosade.problems import get_problem


def test_mosade_history_contains_strategy_telemetry() -> None:
    problem = get_problem("ZDT1")
    algo = MOSADE(pop_size=40, max_evals=400, seed=42, track_interval=40)
    result = algo.run(problem)

    history = result.history
    required = [
        "strategy_probs",
        "strategy_use_counts",
        "strategy_success_counts",
        "strategy_credit_total",
        "strategy_credit_dg",
        "strategy_credit_dom",
        "memory_F_mean",
        "memory_CR_mean",
        "delta",
        "T",
        "div_ratio",
        "archive_size",
        "restart",
        "restart_count",
    ]
    for key in required:
        assert key in history
        assert len(history[key]) == len(history["gen"])

    assert result.metadata["effective_pop_size"] > 0
