from __future__ import annotations

import numpy as np

from mosade.algorithm.mosade import MOSADE, _update_running_extrema
from mosade.algorithm.selection import nondominated_mask
from mosade.algorithm.strategies import polynomial_mutation
from mosade.metrics import hypervolume, igd
from mosade.problems import ZDT3


class TestPolynomialMutationRegression:
    def test_boundary_mutation_stays_in_bounds_and_can_move_inward(self):
        rng = np.random.default_rng(123)
        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])
        x = np.array([0.0, 1.0])

        samples = np.vstack([
            polynomial_mutation(x, lower, upper, rng, pm=1.0, eta_m=20.0)
            for _ in range(256)
        ])

        assert np.all(samples >= lower)
        assert np.all(samples <= upper)
        assert np.any(samples[:, 0] > 0.0)
        assert np.any(samples[:, 1] < 1.0)


class TestRunningExtremaRegression:
    def test_update_running_extrema_never_shrinks_bounds(self):
        z_ideal = np.array([0.2, 0.4])
        z_nadir = np.array([3.5, 4.5])

        restarted_pop = np.array([
            [0.6, 0.9],
            [1.2, 1.8],
            [2.1, 2.4],
        ])

        next_ideal, next_nadir = _update_running_extrema(z_ideal, z_nadir, restarted_pop)

        np.testing.assert_allclose(next_ideal, z_ideal)
        np.testing.assert_allclose(next_nadir, z_nadir)

    def test_update_running_extrema_absorbs_better_ideal_and_worse_nadir(self):
        z_ideal = np.array([0.2, 0.4])
        z_nadir = np.array([3.5, 4.5])

        new_points = np.array([
            [0.1, 1.0],
            [2.0, 4.8],
        ])

        next_ideal, next_nadir = _update_running_extrema(z_ideal, z_nadir, new_points)

        np.testing.assert_allclose(next_ideal, np.array([0.1, 0.4]))
        np.testing.assert_allclose(next_nadir, np.array([3.5, 4.8]))


class TestZDT3SanityRegression:
    def test_mosade_zdt3_smoke_front_is_nondominated_and_reasonable(self):
        problem = ZDT3(n_var=30)
        pf = problem.pareto_front(500)
        ref = pf.max(axis=0) * 1.1 + 1e-6

        result = MOSADE(
            pop_size=40,
            max_evals=5000,
            seed=42,
            stag_ratio=0.05,
            track_interval=200,
        ).run(problem, pf=pf, ref_point=ref)

        assert result.F.shape[1] == 2
        assert np.all(np.isfinite(result.F))
        assert nondominated_mask(result.F).all()
        assert hypervolume(result.F, ref) > 0.4
        assert igd(result.F, pf) < 0.4
