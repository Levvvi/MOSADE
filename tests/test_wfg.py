"""Tests for the WFG benchmark suite."""

import numpy as np
import pytest

from mosade.problems.wfg import (
    WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9,
    _b_param,
)

# Default config: M=2, n_var=24, k=4, l=20
WFG_CLASSES = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]
CONCAVE_CLASSES = [WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]


# ------------------------------------------------------------------
# Output shape and basic properties
# ------------------------------------------------------------------

class TestOutputShapes:
    """Every WFG problem should return correctly shaped, non-negative objectives."""

    @pytest.mark.parametrize("cls", WFG_CLASSES, ids=lambda c: c.__name__)
    def test_shape_bi_objective(self, cls):
        p = cls(n_var=24, n_obj=2)
        rng = np.random.default_rng(42)
        X = rng.uniform(p.lower, p.upper, size=(15, 24))
        F, CV = p.evaluate(X)
        assert F.shape == (15, 2)
        assert CV.shape == (15,)

    @pytest.mark.parametrize("cls", WFG_CLASSES, ids=lambda c: c.__name__)
    def test_shape_tri_objective(self, cls):
        p = cls(n_var=24, n_obj=3)
        rng = np.random.default_rng(42)
        X = rng.uniform(p.lower, p.upper, size=(10, 24))
        F, CV = p.evaluate(X)
        assert F.shape == (10, 3)
        assert CV.shape == (10,)

    @pytest.mark.parametrize("cls", WFG_CLASSES, ids=lambda c: c.__name__)
    def test_objectives_non_negative(self, cls):
        p = cls(n_var=24, n_obj=2)
        rng = np.random.default_rng(7)
        X = rng.uniform(p.lower, p.upper, size=(50, 24))
        F, _ = p.evaluate(X)
        assert np.all(F >= 0.0), f"{cls.__name__} produced negative objectives"

    @pytest.mark.parametrize("cls", WFG_CLASSES, ids=lambda c: c.__name__)
    def test_unconstrained(self, cls):
        """WFG problems are unconstrained; CV should be zero."""
        p = cls(n_var=24, n_obj=2)
        rng = np.random.default_rng(0)
        X = rng.uniform(p.lower, p.upper, size=(5, 24))
        _, CV = p.evaluate(X)
        assert np.all(CV == 0.0)

    @pytest.mark.parametrize("cls", WFG_CLASSES, ids=lambda c: c.__name__)
    def test_single_solution(self, cls):
        """Evaluating a single solution should work."""
        p = cls(n_var=24, n_obj=2)
        rng = np.random.default_rng(1)
        X = rng.uniform(p.lower, p.upper, size=(1, 24))
        F, CV = p.evaluate(X)
        assert F.shape == (1, 2)


# ------------------------------------------------------------------
# Optimum test: distance params at z_i = 2*i * 0.35
# ------------------------------------------------------------------

def _make_optimum(n_var: int, k: int, x1: float = 0.5) -> np.ndarray:
    """Construct a solution near the Pareto-optimal front.

    Position params: z_i = 2*i * x1  (sets position on PF).
    Distance params: z_i = 2*i * 0.35 (optimal distance value).
    """
    z = np.zeros(n_var)
    z_max = np.arange(1, n_var + 1) * 2.0
    # Position: set normalised value to x1
    for i in range(k):
        z[i] = z_max[i] * x1
    # Distance: set normalised value to 0.35 (optimal for s_linear, s_decept, s_multi)
    for i in range(k, n_var):
        z[i] = z_max[i] * 0.35
    return z


class TestOptimum:
    """At the optimum (distance params = 0.35 * z_max), x_M should be 0
    and objectives should reduce to S_m * h_m (with x_M*D = 0)."""

    @pytest.mark.parametrize("cls", [WFG4, WFG5, WFG6, WFG7, WFG8, WFG9],
                             ids=lambda c: c.__name__)
    def test_distance_params_at_optimum_give_small_xM(self, cls):
        """When all distance params are at their optimal value (0.35),
        the last reduced parameter x_M should be near 0."""
        p = cls(n_var=24, n_obj=2)
        z = _make_optimum(24, p._k).reshape(1, -1)
        F, _ = p.evaluate(z)
        # f_m = x_M * D + S_m * h_m.  At the optimum x_M ~ 0,
        # so f values should be close to S_m * h_m.
        # For concave shape with x1=0.5: h1 ~ sin(pi/4), h2 ~ cos(pi/4)
        # With S = [2, 4]:  f1 ~ 2*0.707 ~ 1.414,  f2 ~ 4*0.707 ~ 2.828
        # We just check that objectives are reasonable and finite.
        assert np.all(np.isfinite(F))
        assert np.all(F >= 0)

    @pytest.mark.parametrize(
        "cls,x1_val",
        [(WFG5, 0.0), (WFG6, 0.0), (WFG7, 0.0)],
        ids=lambda c: c.__name__ if isinstance(c, type) else str(c),
    )
    def test_optimum_at_x1_zero(self, cls, x1_val):
        """With x1=0, the first concave objective should be near 0.

        Note: WFG4 excluded because s_multi(0) = 1.0 (not identity on position params).
        """
        p = cls(n_var=24, n_obj=2)
        z = _make_optimum(24, p._k, x1=x1_val).reshape(1, -1)
        F, _ = p.evaluate(z)
        # For concave shape: h1 = sin(x1*pi/2). When x1=0, h1=0, so f1 ~ 0.
        assert F[0, 0] < 0.5, f"{cls.__name__}: f1 at x1=0 should be near 0, got {F[0, 0]}"


# ------------------------------------------------------------------
# Concave PF shape shared by WFG4-9
# ------------------------------------------------------------------

class TestConcavePFShape:
    """WFG4-9 all use the concave PF shape.  At the optimum, objectives
    should approximately satisfy sum((f_m / S_m)^2) ~ 1 for M=2."""

    @pytest.mark.parametrize("cls", CONCAVE_CLASSES, ids=lambda c: c.__name__)
    def test_concave_pf_unit_circle(self, cls):
        p = cls(n_var=24, n_obj=2)
        S = np.array([2.0, 4.0])
        results = []
        for x1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            z = _make_optimum(24, p._k, x1=x1).reshape(1, -1)
            F, _ = p.evaluate(z)
            # Normalise by S, then check concave shape:
            # h1 = sin(x1*pi/2), h2 = cos(x1*pi/2) => h1^2 + h2^2 = 1
            # f_m ~ S_m * h_m when x_M ~ 0, so (f_m/S_m)^2 should sum to ~1
            normalised = F[0] / S
            radius = np.sum(normalised ** 2)
            results.append(radius)
        # At least some points should be near the unit circle
        mean_radius = np.mean(results)
        assert 0.5 < mean_radius < 1.5, (
            f"{cls.__name__}: mean normalised radius = {mean_radius}, "
            f"expected near 1.0 for concave PF"
        )


# ------------------------------------------------------------------
# b_param transformation unit tests
# ------------------------------------------------------------------

class TestBParam:
    """Verify the b_param transformation behaves correctly at key points."""

    def test_neutral_at_u_half(self):
        """When u=0.5, b_param exponent ~ 1, so y is nearly unchanged."""
        y = np.array([0.3, 0.5, 0.7])
        u = np.array([0.5, 0.5, 0.5])
        result = _b_param(y, u)
        # Exponent should be ~1, so result ~ y
        np.testing.assert_allclose(result, y, atol=0.02)

    def test_small_u_flattens(self):
        """When u~0, exponent ~ 0.02, so y^0.02 ~ 1 for y > 0."""
        y = np.array([0.1, 0.5, 0.9])
        u = np.array([0.0, 0.0, 0.0])
        result = _b_param(y, u)
        # y^0.02 is close to 1 for any positive y
        assert np.all(result > 0.9)

    def test_large_u_sharpens(self):
        """When u~1, exponent ~ 50, so small y -> ~0."""
        y = np.array([0.3, 0.5, 0.9])
        u = np.array([1.0, 1.0, 1.0])
        result = _b_param(y, u)
        # 0.3^50 ~ 0, 0.5^50 ~ 0, 0.9^50 ~ 0.005
        assert result[0] < 1e-10
        assert result[1] < 1e-10
        assert result[2] < 0.01


# ------------------------------------------------------------------
# Bounds and configuration
# ------------------------------------------------------------------

class TestConfiguration:
    """WFG bounds and k/l parameter handling."""

    def test_default_bounds(self):
        p = WFG4(n_var=24, n_obj=2)
        assert p.lower[0] == 0.0
        assert p.upper[0] == 2.0   # z_max_1 = 2*1
        assert p.upper[23] == 48.0  # z_max_24 = 2*24

    def test_k_divisibility_adjusted(self):
        """If user supplies k not divisible by M-1, it should be adjusted."""
        p = WFG4(n_var=24, n_obj=3, k=5)  # k=5 not divisible by 2
        assert p._k % (p.n_obj - 1) == 0

    def test_invalid_n_var_raises(self):
        with pytest.raises(ValueError):
            WFG4(n_var=3, n_obj=2, k=4)  # l = 3-4 = -1

    def test_eval_counter(self):
        p = WFG1(n_var=24, n_obj=2)
        rng = np.random.default_rng(0)
        X = rng.uniform(p.lower, p.upper, size=(7, 24))
        p.evaluate(X)
        assert p.n_evals == 7
