"""Tests for benchmark problem implementations."""

import numpy as np
import pytest

from mosade.problems import ZDT1, ZDT3, ZDT4, DTLZ1, DTLZ2, DTLZ3, DTLZ4


class TestZDT1:
    def test_output_shapes(self):
        p = ZDT1(n_var=30)
        X = np.random.default_rng(0).random((10, 30))
        F, CV = p.evaluate(X)
        assert F.shape == (10, 2)
        assert CV.shape == (10,)
        assert np.all(CV == 0.0)

    def test_pareto_front_shape(self):
        p = ZDT1()
        pf = p.pareto_front(100)
        assert pf.shape == (100, 2)
        assert np.all(pf[:, 0] >= 0) and np.all(pf[:, 0] <= 1)

    def test_pf_point_is_optimal(self):
        """A point on the true PF should have g=1."""
        p = ZDT1(n_var=5)
        x = np.zeros(5)
        x[0] = 0.25  # f1 = 0.25; g = 1; f2 = 1 - sqrt(0.25) = 0.5
        F, _ = p.evaluate(x.reshape(1, -1))
        assert F[0, 0] == pytest.approx(0.25)
        assert F[0, 1] == pytest.approx(0.5)

    def test_eval_counter(self):
        p = ZDT1(n_var=5)
        X = np.random.default_rng(1).random((7, 5))
        p.evaluate(X)
        assert p.n_evals == 7
        p.evaluate(X[:3])
        assert p.n_evals == 10
        p.reset_eval_counter()
        assert p.n_evals == 0


class TestZDT3:
    def test_disconnected_pf(self):
        p = ZDT3()
        pf = p.pareto_front(500)
        # Should have gaps — the f1 range should not be continuous
        assert pf.shape[0] > 100


class TestZDT4:
    def test_bounds(self):
        p = ZDT4(n_var=10)
        assert p.lower[0] == 0.0
        assert p.upper[0] == 1.0
        assert p.lower[1] == -5.0
        assert p.upper[1] == 5.0


class TestDTLZ:
    def test_dtlz1_shape(self):
        p = DTLZ1(n_obj=3)
        X = np.random.default_rng(0).random((5, p.n_var))
        F, CV = p.evaluate(X)
        assert F.shape == (5, 3)
        assert CV.shape == (5,)

    def test_dtlz2_shape(self):
        p = DTLZ2(n_obj=3)
        X = np.random.default_rng(0).random((5, p.n_var))
        F, CV = p.evaluate(X)
        assert F.shape == (5, 3)

    def test_dtlz2_pf_on_unit_sphere(self):
        """On the PF (xm=0.5), objective vectors should lie on the unit sphere."""
        p = DTLZ2(n_obj=3)
        X = np.full((10, p.n_var), 0.5)
        X[:, 0] = np.linspace(0, 1, 10)
        X[:, 1] = np.linspace(0, 1, 10)
        F, _ = p.evaluate(X)
        norms = np.sqrt(np.sum(F ** 2, axis=1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestDTLZ3:
    def test_shape(self):
        p = DTLZ3(n_obj=3)
        X = np.random.default_rng(0).random((5, p.n_var))
        F, CV = p.evaluate(X)
        assert F.shape == (5, 3)
        assert CV.shape == (5,)
        assert np.all(CV == 0.0)

    def test_default_n_var(self):
        p = DTLZ3(n_obj=3)
        assert p.n_var == 12  # n_obj + 9

    def test_pf_on_unit_sphere(self):
        """On the PF (xm=0.5 so g=0), objectives lie on the unit sphere."""
        p = DTLZ3(n_obj=3)
        X = np.full((10, p.n_var), 0.5)
        X[:, 0] = np.linspace(0.01, 0.99, 10)
        X[:, 1] = np.linspace(0.01, 0.99, 10)
        F, _ = p.evaluate(X)
        norms = np.sqrt(np.sum(F ** 2, axis=1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_many_local_fronts(self):
        """Away from xm=0.5, g >> 0, so objectives scale well beyond the PF."""
        p = DTLZ3(n_obj=3)
        X = np.full((1, p.n_var), 0.0)  # xm far from 0.5
        F, _ = p.evaluate(X)
        norm = np.sqrt(np.sum(F[0] ** 2))
        assert norm > 100, "g should be large when xm != 0.5"

    def test_2obj(self):
        p = DTLZ3(n_obj=2)
        X = np.random.default_rng(1).random((8, p.n_var))
        F, _ = p.evaluate(X)
        assert F.shape == (8, 2)


class TestDTLZ4:
    def test_shape(self):
        p = DTLZ4(n_obj=3)
        X = np.random.default_rng(0).random((5, p.n_var))
        F, CV = p.evaluate(X)
        assert F.shape == (5, 3)
        assert CV.shape == (5,)
        assert np.all(CV == 0.0)

    def test_default_n_var(self):
        p = DTLZ4(n_obj=3)
        assert p.n_var == 12  # n_obj + 9

    def test_pf_on_unit_sphere(self):
        """On the PF (xm=0.5), objectives still lie on the unit sphere."""
        p = DTLZ4(n_obj=3)
        X = np.full((10, p.n_var), 0.5)
        X[:, 0] = np.linspace(0.01, 0.99, 10)
        X[:, 1] = np.linspace(0.01, 0.99, 10)
        F, _ = p.evaluate(X)
        norms = np.sqrt(np.sum(F ** 2, axis=1))
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_alpha1_matches_dtlz2(self):
        """With alpha=1, DTLZ4 should produce identical results to DTLZ2."""
        p2 = DTLZ2(n_obj=3)
        p4 = DTLZ4(n_obj=3, alpha=1.0)
        rng = np.random.default_rng(42)
        X = rng.random((20, p2.n_var))
        F2, _ = p2.evaluate(X)
        F4, _ = p4.evaluate(X)
        np.testing.assert_allclose(F4, F2, atol=1e-12)

    def test_bias_concentrates_near_axis(self):
        """With high alpha, uniform x_i maps mostly near 0, biasing toward f_M."""
        p = DTLZ4(n_obj=2, alpha=100.0)
        rng = np.random.default_rng(0)
        X = np.full((200, p.n_var), 0.5)  # xm=0.5 → g=0 → on PF
        X[:, 0] = rng.random(200)  # uniform x1
        F, _ = p.evaluate(X)
        # x1^100 is near 0 for most x1 in [0,1), so cos(x1^100 * pi/2) ≈ 1
        # meaning f1 ≈ 1, f2 ≈ 0 for most solutions
        assert np.mean(F[:, 0] > 0.9) > 0.8, "Most f1 should be near 1 with high alpha"


class TestDTLZAnalyticalFronts:
    """Covers the DTLZ1 and DTLZ2 analytical pareto_front paths."""

    def test_dtlz1_pf_2obj_is_linear_simplex(self):
        pf = DTLZ1(n_obj=2).pareto_front(100)
        assert pf.shape == (100, 2)
        np.testing.assert_allclose(pf.sum(axis=1), 0.5, atol=1e-9)

    def test_dtlz1_pf_3obj_on_scaled_simplex(self):
        pf = DTLZ1(n_obj=3).pareto_front(200)
        assert pf.shape[1] == 3
        np.testing.assert_allclose(pf.sum(axis=1), 0.5, atol=1e-9)
        assert np.all(pf >= -1e-12)

    def test_dtlz2_pf_2obj_on_unit_circle(self):
        pf = DTLZ2(n_obj=2).pareto_front(100)
        assert pf.shape == (100, 2)
        np.testing.assert_allclose(np.sqrt((pf**2).sum(axis=1)), 1.0, atol=1e-10)

    def test_dtlz2_pf_3obj_returns_none(self):
        assert DTLZ2(n_obj=3).pareto_front() is None
