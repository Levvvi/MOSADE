"""End-to-end smoke tests.

Covers the full pipeline for both MOSADE and NSGA-II on ZDT1 with a tiny
budget.  Correctness of results is not checked — only that the pipelines
run without errors and produce plausible output.
"""

import numpy as np
import pytest

from mosade.algorithm import ALGORITHM_REGISTRY
from mosade.algorithm.moead import MOEAD
from mosade.algorithm.mosade import MOSADE, MOSADEResult
from mosade.algorithm.nsga2 import NSGA2
from mosade.metrics import hypervolume, igd
from mosade.problems import ZDT1


class TestSmoke:
    def test_mosade_runs_and_returns_result(self):
        problem = ZDT1(n_var=10)
        algo = MOSADE(pop_size=20, max_evals=2000, seed=42)
        result = algo.run(problem)

        assert result.F.shape[1] == 2
        assert result.X.shape[1] == 10
        assert result.F.shape[0] > 0
        assert result.n_evals >= 2000  # may slightly exceed due to restart

    def test_objectives_are_finite(self):
        problem = ZDT1(n_var=10)
        algo = MOSADE(pop_size=20, max_evals=1000, seed=99)
        result = algo.run(problem)
        assert np.all(np.isfinite(result.F))

    def test_hv_is_positive(self):
        problem = ZDT1(n_var=10)
        algo = MOSADE(pop_size=20, max_evals=2000, seed=7)
        result = algo.run(problem)

        pf = problem.pareto_front()
        ref = pf.max(axis=0) * 1.1
        hv = hypervolume(result.F, ref)
        assert hv > 0.0

    def test_igd_is_finite(self):
        problem = ZDT1(n_var=10)
        algo = MOSADE(pop_size=20, max_evals=2000, seed=7)
        result = algo.run(problem)

        pf = problem.pareto_front()
        val = igd(result.F, pf)
        assert np.isfinite(val)
        assert val > 0.0

    def test_history_is_populated(self):
        problem = ZDT1(n_var=10)
        algo = MOSADE(pop_size=20, max_evals=1000, seed=0)
        result = algo.run(problem)
        assert len(result.history["gen"]) > 0
        assert len(result.history["strategy_probs"]) > 0

    def test_deterministic_with_same_seed(self):
        problem = ZDT1(n_var=10)
        r1 = MOSADE(pop_size=20, max_evals=1000, seed=55).run(problem)
        problem.reset_eval_counter()
        r2 = MOSADE(pop_size=20, max_evals=1000, seed=55).run(problem)
        np.testing.assert_array_equal(r1.F, r2.F)


# ---------------------------------------------------------------------------
# NSGA-II smoke tests
# ---------------------------------------------------------------------------


class TestNSGA2Smoke:
    def test_nsga2_runs_and_returns_result(self):
        problem = ZDT1(n_var=10)
        algo = NSGA2(pop_size=20, max_evals=2000, seed=42)
        result = algo.run(problem)

        assert result.F.shape[1] == 2
        assert result.X.shape[1] == 10
        assert result.F.shape[0] > 0
        # NSGA-II budget is exhausted inside the loop; n_evals may slightly
        # exceed max_evals by at most one offspring batch (pop_size).
        assert result.n_evals >= 2000

    def test_objectives_finite(self):
        problem = ZDT1(n_var=10)
        algo = NSGA2(pop_size=20, max_evals=1000, seed=99)
        result = algo.run(problem)
        assert np.all(np.isfinite(result.F))

    def test_hv_positive(self):
        problem = ZDT1(n_var=10)
        algo = NSGA2(pop_size=20, max_evals=2000, seed=7)
        result = algo.run(problem)

        pf = problem.pareto_front()
        ref = pf.max(axis=0) * 1.1
        hv = hypervolume(result.F, ref)
        assert hv > 0.0

    def test_deterministic_with_same_seed(self):
        problem = ZDT1(n_var=10)
        r1 = NSGA2(pop_size=20, max_evals=1000, seed=55).run(problem)
        problem.reset_eval_counter()
        r2 = NSGA2(pop_size=20, max_evals=1000, seed=55).run(problem)
        np.testing.assert_array_equal(r1.F, r2.F)

    def test_output_is_nondominated(self):
        """NSGA-II explicitly returns only the nondominated front."""
        from mosade.algorithm.selection import nondominated_mask

        problem = ZDT1(n_var=10)
        algo = NSGA2(pop_size=20, max_evals=1000, seed=42)
        result = algo.run(problem)
        nd = nondominated_mask(result.F)
        assert nd.all(), "NSGA-II output must be nondominated"

    def test_returns_mosade_result(self):
        """NSGA-II returns MOSADEResult for full API compatibility."""
        problem = ZDT1(n_var=10)
        algo = NSGA2(pop_size=20, max_evals=500, seed=0)
        result = algo.run(problem)
        assert isinstance(result, MOSADEResult)
        assert "gen" in result.history
        assert "n_evals" in result.history

    def test_odd_pop_size_rounded_to_even(self):
        """An odd pop_size is silently rounded up so SBX pairing always works."""
        algo = NSGA2(pop_size=21, max_evals=500, seed=1)
        assert algo.pop_size == 22

    def test_pop_size_is_even(self):
        for n in [10, 20, 50, 100]:
            assert NSGA2(pop_size=n).pop_size % 2 == 0


# ---------------------------------------------------------------------------
# MOEA/D smoke tests
# ---------------------------------------------------------------------------


class TestMOEADSmoke:
    def test_moead_runs_and_returns_result(self):
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=2000, seed=42)
        result = algo.run(problem)

        assert result.F.shape[1] == 2
        assert result.X.shape[1] == 10
        assert result.F.shape[0] > 0
        # Budget is checked inside the inner loop; n_evals may be slightly
        # below max_evals when the budget runs out mid-generation.
        assert result.n_evals <= 2000 + 25  # at most one extra subproblem eval

    def test_objectives_finite(self):
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=1000, seed=99)
        result = algo.run(problem)
        assert np.all(np.isfinite(result.F))

    def test_hv_positive(self):
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=2000, seed=7)
        result = algo.run(problem)

        pf = problem.pareto_front()
        ref = pf.max(axis=0) * 1.1
        hv = hypervolume(result.F, ref)
        assert hv > 0.0

    def test_deterministic_with_same_seed(self):
        problem = ZDT1(n_var=10)
        r1 = MOEAD(pop_size=20, max_evals=1000, seed=55).run(problem)
        problem.reset_eval_counter()
        r2 = MOEAD(pop_size=20, max_evals=1000, seed=55).run(problem)
        np.testing.assert_array_equal(r1.F, r2.F)

    def test_output_is_nondominated(self):
        """MOEA/D explicitly returns only the nondominated front."""
        from mosade.algorithm.selection import nondominated_mask

        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=1000, seed=42)
        result = algo.run(problem)
        nd = nondominated_mask(result.F)
        assert nd.all(), "MOEA/D output must be nondominated"

    def test_returns_mosade_result(self):
        """MOEA/D returns MOSADEResult for full API compatibility."""
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=500, seed=0)
        result = algo.run(problem)
        assert isinstance(result, MOSADEResult)
        assert "gen" in result.history
        assert "n_evals" in result.history

    def test_neighborhood_mating(self):
        """delta=1.0 forces all mating from the neighbourhood (no crash)."""
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=500, seed=3, delta=1.0)
        result = algo.run(problem)
        assert result.F.shape[0] > 0

    def test_global_mating(self):
        """delta=0.0 forces all mating from the full population (no crash)."""
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=500, seed=3, delta=0.0)
        result = algo.run(problem)
        assert result.F.shape[0] > 0

    def test_t_ratio_extreme_small(self):
        """Very small T_ratio still runs (clamped to T>=3)."""
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=400, seed=5, T_ratio=0.001)
        result = algo.run(problem)
        assert result.F.shape[0] > 0

    def test_history_tracks_generations(self):
        problem = ZDT1(n_var=10)
        algo = MOEAD(pop_size=20, max_evals=1000, seed=0)
        result = algo.run(problem)
        assert len(result.history["gen"]) > 0
        assert len(result.history["n_evals"]) > 0
        # n_evals entries should be non-decreasing
        evals = result.history["n_evals"]
        assert all(evals[k] <= evals[k + 1] for k in range(len(evals) - 1))


# ---------------------------------------------------------------------------
# Algorithm registry tests
# ---------------------------------------------------------------------------


class TestAlgorithmRegistry:
    def test_registry_contains_mosade(self):
        assert "MOSADE" in ALGORITHM_REGISTRY

    def test_registry_contains_nsga2(self):
        assert "NSGA2" in ALGORITHM_REGISTRY

    def test_registry_contains_moead(self):
        assert "MOEAD" in ALGORITHM_REGISTRY

    def test_mosade_class_correct(self):
        assert ALGORITHM_REGISTRY["MOSADE"] is MOSADE

    def test_nsga2_class_correct(self):
        assert ALGORITHM_REGISTRY["NSGA2"] is NSGA2

    def test_moead_class_correct(self):
        assert ALGORITHM_REGISTRY["MOEAD"] is MOEAD

    def test_registry_classes_are_callable(self):
        """All registered classes can be instantiated with minimal args."""
        for name, cls in ALGORITHM_REGISTRY.items():
            obj = cls(pop_size=20, max_evals=100, seed=0)
            assert obj is not None, f"{name} instantiation failed"
