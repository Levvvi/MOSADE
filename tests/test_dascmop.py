"""Tests for the DASCMOP benchmark suite."""

import numpy as np
import pytest

from mosade.problems.dascmop import (
    DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
    DASCMOP7, DASCMOP8, DASCMOP9, DIFFICULTY_SETTINGS,
)

BIOBJ_CLASSES = [DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6]
TRIOBJ_CLASSES = [DASCMOP7, DASCMOP8, DASCMOP9]
ALL_CLASSES = BIOBJ_CLASSES + TRIOBJ_CLASSES


# ------------------------------------------------------------------
# Output shapes
# ------------------------------------------------------------------

class TestOutputShapes:
    """F and CV shapes should match n_obj and n_solutions."""

    @pytest.mark.parametrize("cls", BIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_biobj_shape(self, cls):
        p = cls(n_var=30, difficulty=8)
        rng = np.random.default_rng(42)
        X = rng.random((15, 30))
        F, CV = p.evaluate(X)
        assert F.shape == (15, 2)
        assert CV.shape == (15,)

    @pytest.mark.parametrize("cls", TRIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_triobj_shape(self, cls):
        p = cls(n_var=30, difficulty=8)
        rng = np.random.default_rng(42)
        X = rng.random((15, 30))
        F, CV = p.evaluate(X)
        assert F.shape == (15, 3)
        assert CV.shape == (15,)

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_single_solution(self, cls):
        p = cls(n_var=30, difficulty=4)
        X = np.random.default_rng(0).random((1, 30))
        F, CV = p.evaluate(X)
        assert F.shape[0] == 1
        assert CV.shape == (1,)


# ------------------------------------------------------------------
# CV >= 0
# ------------------------------------------------------------------

class TestCVNonNegative:
    """Constraint violation must always be non-negative."""

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("difficulty", [1, 8, 12], ids=lambda d: f"diff{d}")
    def test_cv_non_negative(self, cls, difficulty):
        p = cls(n_var=30, difficulty=difficulty)
        rng = np.random.default_rng(7)
        X = rng.random((200, 30))
        _, CV = p.evaluate(X)
        assert np.all(CV >= 0.0), f"{cls.__name__} at difficulty {difficulty}: CV < 0 found"


# ------------------------------------------------------------------
# n_constr
# ------------------------------------------------------------------

class TestNConstr:
    """Verify correct constraint counts per the reference implementation."""

    @pytest.mark.parametrize("cls", BIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_biobj_n_constr(self, cls):
        p = cls(n_var=30, difficulty=1)
        assert p.n_constr == 11, f"{cls.__name__}: expected 11 constraints, got {p.n_constr}"

    @pytest.mark.parametrize("cls", TRIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_triobj_n_constr(self, cls):
        p = cls(n_var=30, difficulty=1)
        assert p.n_constr == 7, f"{cls.__name__}: expected 7 constraints, got {p.n_constr}"


# ------------------------------------------------------------------
# n_obj
# ------------------------------------------------------------------

class TestNObj:
    """Verify objective counts match the reference."""

    @pytest.mark.parametrize("cls", BIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_biobj(self, cls):
        assert cls(n_var=30, difficulty=1).n_obj == 2

    @pytest.mark.parametrize("cls", TRIOBJ_CLASSES, ids=lambda c: c.__name__)
    def test_triobj(self, cls):
        assert cls(n_var=30, difficulty=1).n_obj == 3


# ------------------------------------------------------------------
# Feasibility ratio decreases with increasing eta
# ------------------------------------------------------------------

class TestFeasibilityMonotonicity:
    """As eta increases (levels 1 -> 5 -> 9), feasibility should decrease.

    These levels isolate the eta parameter (zeta=gamma=0), so the ONLY
    active constraint is the sinusoidal band on x_0 (and x_1 for DASCMOP7-9).

    Expected feasibility fractions:
      DASCMOP1-6: ~2/3, ~1/2, ~1/3  (sin >= -0.5, 0, 0.5)
      DASCMOP7-9: ~4/9, ~1/4, ~1/9  (two independent sinusoidal constraints)
    """

    N_SAMPLES = 5000

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_eta_monotonicity(self, cls):
        rng = np.random.default_rng(123)
        ratios = []
        for level in [1, 5, 9]:
            p = cls(n_var=30, difficulty=level)
            X = rng.random((self.N_SAMPLES, 30))
            _, CV = p.evaluate(X)
            ratios.append(float(np.sum(CV == 0.0) / self.N_SAMPLES))

        assert ratios[0] > ratios[1] > ratios[2], (
            f"{cls.__name__}: feasibility not monotonically decreasing with eta. "
            f"Levels 1,5,9 gave ratios {ratios}"
        )

    @pytest.mark.parametrize("cls", ALL_CLASSES, ids=lambda c: c.__name__)
    def test_level1_has_feasible_solutions(self, cls):
        """At the easiest eta-only level, a decent fraction should be feasible."""
        p = cls(n_var=30, difficulty=1)
        rng = np.random.default_rng(99)
        X = rng.random((5000, 30))
        _, CV = p.evaluate(X)
        ratio = float(np.sum(CV == 0.0) / 5000)
        assert ratio > 0.3, (
            f"{cls.__name__}: too few feasible at difficulty 1 ({ratio:.3f})"
        )


# ------------------------------------------------------------------
# Difficulty table validation
# ------------------------------------------------------------------

class TestDifficultySettings:
    """Verify the difficulty table matches the pymoo reference."""

    def test_table_has_16_entries(self):
        assert len(DIFFICULTY_SETTINGS) == 16

    def test_all_values_in_range(self):
        for level, (eta, zeta, gamma) in DIFFICULTY_SETTINGS.items():
            assert 0.0 <= eta <= 1.0, f"Level {level}: eta={eta}"
            assert 0.0 <= zeta <= 1.0, f"Level {level}: zeta={zeta}"
            assert 0.0 <= gamma <= 1.0, f"Level {level}: gamma={gamma}"

    def test_level1_is_eta_only(self):
        eta, zeta, gamma = DIFFICULTY_SETTINGS[1]
        assert eta == 0.25 and zeta == 0.0 and gamma == 0.0

    def test_level8_is_balanced(self):
        eta, zeta, gamma = DIFFICULTY_SETTINGS[8]
        assert eta == 0.5 and zeta == 0.5 and gamma == 0.5

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError):
            DASCMOP1(difficulty=0)
        with pytest.raises(ValueError):
            DASCMOP1(difficulty=17)


# ------------------------------------------------------------------
# Bounds and eval counter
# ------------------------------------------------------------------

class TestMisc:

    def test_bounds(self):
        p = DASCMOP1(n_var=30, difficulty=1)
        assert np.all(p.lower == 0.0)
        assert np.all(p.upper == 1.0)

    def test_eval_counter(self):
        p = DASCMOP4(n_var=30, difficulty=4)
        rng = np.random.default_rng(0)
        X = rng.random((7, 30))
        p.evaluate(X)
        assert p.n_evals == 7
        p.evaluate(X[:3])
        assert p.n_evals == 10
