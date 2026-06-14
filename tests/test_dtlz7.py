"""Tests for the DTLZ7 benchmark implementation."""

from __future__ import annotations

import numpy as np

from mosade.problems.dtlz import DTLZ7


def _expected_last_objective(prefix: np.ndarray, g: float, n_obj: int) -> float:
    terms = (prefix / (1.0 + g)) * (1.0 + np.sin(3.0 * np.pi * prefix))
    h = n_obj - np.sum(terms)
    return float((1.0 + g) * h)


def _dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


class TestDTLZ7:
    def test_default_dimensions_use_standard_k_20(self):
        p = DTLZ7(n_obj=3)

        assert p.n_obj == 3
        assert p.n_var == 22
        assert p.n_constr == 0
        np.testing.assert_allclose(p.lower, np.zeros(p.n_var))
        np.testing.assert_allclose(p.upper, np.ones(p.n_var))

    def test_evaluate_shape_constraints_and_eval_counter(self):
        p = DTLZ7(n_obj=3)
        X = np.random.default_rng(7).random((6, p.n_var))

        F, CV = p.evaluate(X)

        assert F.shape == (6, 3)
        assert CV.shape == (6,)
        assert np.all(CV == 0.0)
        assert p.n_evals == 6

        p.evaluate(X[:2])
        assert p.n_evals == 8

    def test_objectives_match_formula_on_pareto_set(self):
        p = DTLZ7(n_var=5, n_obj=3)
        X = np.array([[0.2, 0.4, 0.0, 0.0, 0.0]])

        F, CV = p.evaluate(X)

        expected = np.array(
            [
                0.2,
                0.4,
                _expected_last_objective(np.array([0.2, 0.4]), g=1.0, n_obj=3),
            ]
        )
        np.testing.assert_allclose(F[0], expected, atol=1e-12)
        np.testing.assert_allclose(CV, np.zeros(1))

    def test_distance_variables_raise_g_and_last_objective(self):
        p = DTLZ7(n_var=5, n_obj=3)
        X = np.array(
            [
                [0.2, 0.4, 0.0, 0.0, 0.0],
                [0.2, 0.4, 1.0, 1.0, 1.0],
            ]
        )

        F, _ = p.evaluate(X)

        expected_g1 = _expected_last_objective(np.array([0.2, 0.4]), g=1.0, n_obj=3)
        expected_g10 = _expected_last_objective(np.array([0.2, 0.4]), g=10.0, n_obj=3)
        np.testing.assert_allclose(F[:, :2], X[:, :2], atol=0.0)
        np.testing.assert_allclose(F[:, 2], [expected_g1, expected_g10], atol=1e-12)
        assert F[1, 2] > F[0, 2]

    def test_custom_objective_count_preserves_prefix_objectives(self):
        p = DTLZ7(n_var=8, n_obj=4)
        X = np.array([[0.1, 0.2, 0.3, 0.0, 0.25, 0.5, 0.75, 1.0]])

        F, CV = p.evaluate(X)

        g = 1.0 + 9.0 * np.sum(X[0, 3:]) / 5.0
        expected = np.array(
            [
                0.1,
                0.2,
                0.3,
                _expected_last_objective(np.array([0.1, 0.2, 0.3]), g=g, n_obj=4),
            ]
        )
        assert F.shape == (1, 4)
        np.testing.assert_allclose(F[0], expected, atol=1e-12)
        np.testing.assert_allclose(CV, np.zeros(1))

    def test_pareto_front_is_deterministic_and_limited_to_requested_points(self):
        p = DTLZ7(n_obj=3)

        pf1 = p.pareto_front(40)
        pf2 = p.pareto_front(40)

        assert pf1 is not None
        assert pf1.shape == (40, 3)
        np.testing.assert_allclose(pf1, pf2, atol=0.0)
        assert np.all(np.isfinite(pf1))
        assert np.all(pf1 >= 0.0)

    def test_pareto_front_uses_disconnected_intervals_and_is_nondominated(self):
        p = DTLZ7(n_obj=3)

        pf = p.pareto_front(40)

        assert pf is not None
        prefix = pf[:, :2]
        in_low_interval = (prefix >= 0.0) & (prefix <= 0.251412)
        in_high_interval = (prefix >= 0.631627) & (prefix <= 0.859401)
        assert np.all(in_low_interval | in_high_interval)

        expected_last = np.array(
            [_expected_last_objective(row, g=1.0, n_obj=p.n_obj) for row in prefix]
        )
        np.testing.assert_allclose(pf[:, -1], expected_last, atol=1e-12)

        for i, f_i in enumerate(pf):
            for j, f_j in enumerate(pf):
                if i != j:
                    assert not _dominates(f_i, f_j)

    def test_pareto_front_returns_none_for_less_than_two_objectives(self):
        p = DTLZ7(n_obj=1)

        assert p.pareto_front(10) is None
