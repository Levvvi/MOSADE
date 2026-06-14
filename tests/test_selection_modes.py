from __future__ import annotations

import numpy as np

from mosade.algorithm.selection import constrained_dominates, select_dominance_survival


def test_constrained_dominance_rules() -> None:
    feasible_a = np.array([1.0, 1.0])
    feasible_b = np.array([2.0, 2.0])
    infeasible_good = np.array([0.1, 0.1])
    infeasible_bad = np.array([0.0, 0.0])

    assert constrained_dominates(feasible_a, 0.0, infeasible_good, 0.5)
    assert constrained_dominates(feasible_a, 0.0, feasible_b, 0.0)
    assert constrained_dominates(infeasible_good, 0.2, infeasible_bad, 0.8)
    assert not constrained_dominates(infeasible_bad, 0.8, feasible_a, 0.0)


def test_dominance_survival_keeps_requested_capacity() -> None:
    F = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [0.5, 2.5],
        ]
    )
    CV = np.array([0.0, 0.0, 0.0, 0.1, 1.0])

    selected, meta = select_dominance_survival(F, CV, n_select=3)

    assert selected.shape == (3,)
    assert set(selected).issubset(set(range(F.shape[0])))
    assert 3 not in selected
    assert 4 not in selected
    assert meta["truncation_method"] in {"none", "crowding_distance", "fallback_cv_sum"}
