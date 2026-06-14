"""Tests for performance metrics."""

import numpy as np
import pytest

from mosade.metrics import gd, hypervolume, igd, igd_plus, spread


class TestHypervolume:
    def test_single_point(self):
        F = np.array([[1.0, 1.0]])
        ref = np.array([2.0, 2.0])
        assert hypervolume(F, ref) == pytest.approx(1.0)

    def test_two_points(self):
        F = np.array([[1.0, 2.0], [2.0, 1.0]])
        ref = np.array([3.0, 3.0])
        # sweep line: sort by f1: (1,2), (2,1)
        #   (1,2): width=3-1=2, height=3-2=1 -> 2
        #   (2,1): width=3-2=1, height=2-1=1 -> 1  total=3
        assert hypervolume(F, ref) == pytest.approx(3.0)

    def test_empty(self):
        F = np.empty((0, 2))
        ref = np.array([1.0, 1.0])
        assert hypervolume(F, ref) == 0.0

    def test_dominated_by_ref(self):
        F = np.array([[5.0, 5.0]])
        ref = np.array([3.0, 3.0])
        assert hypervolume(F, ref) == 0.0


class TestHypervolume3D:
    """3-D exact slicing algorithm tests with hand-computed reference values."""

    def test_single_point_3d(self):
        # HV = (r1-f1)*(r2-f2)*(r3-f3) = 1*1*1 = 1
        F = np.array([[1.0, 1.0, 1.0]])
        ref = np.array([2.0, 2.0, 2.0])
        assert hypervolume(F, ref) == pytest.approx(1.0)

    def test_two_points_same_f3(self):
        # Both points at f3=1, ref_f3=2 → slab height=1, 2-D HV of {(1,2),(2,1)} w/ ref (3,3)=3
        # Total = 1 * 3 = 3
        F = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 1.0]])
        ref = np.array([3.0, 3.0, 2.0])
        assert hypervolume(F, ref) == pytest.approx(3.0)

    def test_two_points_different_f3(self):
        # p1=(1,1,1), p2=(2,2,2), ref=(3,3,3)
        # Sort by f3: p1 at f3=1, p2 at f3=2.
        # Slab 1: height=2-1=1, 2D HV of {(1,1)} w/ ref(3,3) = 2*2=4  → 1*4=4
        # Slab 2: height=3-2=1, 2D HV of {(1,1),(2,2)} w/ ref(3,3):
        #   sort by f1: (1,1),(2,2); (1,1): w=3-1=2,h=3-1=2→4; (2,2): w=3-2=1,h=1-2<0 skip → 4
        # Total = 4 + 1*4 = 8
        F = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        ref = np.array([3.0, 3.0, 3.0])
        assert hypervolume(F, ref) == pytest.approx(8.0)

    def test_three_axis_points(self):
        # Points at (1,0,0), (0,1,0), (0,0,1) — each dominates one axis
        # ref = (2,2,2)
        # Sort by f3: (1,0,0) f3=0, (0,1,0) f3=0, (0,0,1) f3=1
        # Slab 1 [f3=0 → f3=0]: height=0 → skip (duplicate f3 values; next distinct is f3=1)
        # Processing order (sorted f3): (1,0,0) f3=0, (0,1,0) f3=0, (0,0,1) f3=1
        #   i=0: next_f3=0, h=0 → skip
        #   i=1: next_f3=1, h=1, 2D HV of {(1,0),(0,1)} w/ ref(2,2)
        #         sort f1: (0,1),(1,0); (0,1): w=2,h=2-1=1→2; (1,0): w=1,h=1-0=1→1; total=3
        #         contribution = 1*3 = 3
        #   i=2: next_f3=2(ref), h=1, 2D HV of {(1,0),(0,1),(0,0)} w/ ref(2,2)
        #         (0,0) dominates (0,1) in f2, so effective set includes (0,0).
        #         sort f1: (0,0),(0,1),(1,0) — (0,0): w=2,h=2-0=2→4; (0,1) skipped (f2>0);
        #         (1,0): w=1,h=0-0=0 skip; total=4
        #         contribution = 1*4 = 4
        # Total = 3 + 4 = 7
        F = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ref = np.array([2.0, 2.0, 2.0])
        assert hypervolume(F, ref) == pytest.approx(7.0)

    def test_3d_exact_vs_monte_carlo(self):
        """Exact 3-D result should closely match a high-sample-count MC estimate."""
        rng = np.random.default_rng(0)
        F = rng.uniform(0, 1, size=(20, 3))
        ref = np.array([1.5, 1.5, 1.5])
        exact = hypervolume(F, ref)

        # MC with a large sample count for comparison
        from mosade.metrics.hypervolume import _hv_mc
        mc = _hv_mc(F, ref, seed=42)
        # Allow 2% relative tolerance given 500k samples
        assert abs(exact - mc) / (exact + 1e-12) < 0.02

    def test_3d_all_filtered(self):
        """Points outside ref give 0."""
        F = np.array([[5.0, 5.0, 5.0]])
        ref = np.array([2.0, 2.0, 2.0])
        assert hypervolume(F, ref) == 0.0

    def test_seed_reproducibility(self):
        """Monte Carlo with the same seed should give the same result."""
        rng = np.random.default_rng(7)
        F = rng.uniform(0, 1, size=(10, 4))
        ref = np.full(4, 2.0)
        v1 = hypervolume(F, ref, seed=123)
        v2 = hypervolume(F, ref, seed=123)
        assert v1 == v2


class TestIGD:
    def test_perfect_match(self):
        pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert igd(pf, pf) == pytest.approx(0.0, abs=1e-12)

    def test_worse_set_has_higher_igd(self):
        pf = np.array([[0.0, 1.0], [1.0, 0.0]])
        good = np.array([[0.1, 0.9], [0.9, 0.1]])
        bad = np.array([[0.5, 0.5]])
        assert igd(good, pf) < igd(bad, pf)


class TestIGDPlus:
    def test_dominated_point_zero_contribution(self):
        # A solution that dominates a PF point should contribute 0
        pf = np.array([[1.0, 1.0]])
        F = np.array([[0.5, 0.5]])  # dominates pf point
        assert igd_plus(F, pf) == pytest.approx(0.0, abs=1e-12)


class TestSpread:
    def test_uniform_front(self):
        # Perfectly uniform: spread should be low
        F = np.column_stack([np.linspace(0, 1, 50), np.linspace(1, 0, 50)])
        s = spread(F)
        assert s < 0.1  # should be very small for uniform spacing


class TestGD:
    def test_perfect_match(self):
        """GD is 0 when F equals PF."""
        pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        assert gd(pf, pf) == pytest.approx(0.0, abs=1e-12)

    def test_single_point_known_distance(self):
        """GD with one F point at known distance from a one-point PF."""
        # F = [[3, 4]], PF = [[0, 0]]: distance = 5, GD = sqrt(25) / 1 = 5
        F = np.array([[3.0, 4.0]])
        pf = np.array([[0.0, 0.0]])
        assert gd(F, pf) == pytest.approx(5.0)

    def test_two_points_formula(self):
        """GD with two F points, each at a known distance from a single PF point."""
        # F = [[3,4],[0,1]], PF = [[0,0]]
        # distances: sqrt(9+16)=5, sqrt(0+1)=1
        # GD = sqrt(25 + 1) / 2 = sqrt(26) / 2
        F = np.array([[3.0, 4.0], [0.0, 1.0]])
        pf = np.array([[0.0, 0.0]])
        expected = np.sqrt(26.0) / 2.0
        assert gd(F, pf) == pytest.approx(expected)

    def test_empty_F_returns_inf(self):
        pf = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert gd(np.empty((0, 2)), pf) == float("inf")

    def test_worse_approx_has_higher_gd(self):
        """A set farther from the front should have higher GD."""
        pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        close = np.array([[0.05, 0.95], [0.5, 0.5], [0.95, 0.05]])
        far = np.array([[0.8, 0.8]])  # dominated point, far from PF
        assert gd(close, pf) < gd(far, pf)

    def test_gd_differs_from_igd(self):
        """GD and IGD measure complementary things and need not be equal."""
        pf = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        F = np.array([[0.1, 0.9], [0.9, 0.1]])  # good approx but missing middle
        assert gd(F, pf) != pytest.approx(igd(F, pf))
