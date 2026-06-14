"""Smoke tests for the pymoo algorithm wrappers.

Tests that NSGA3, SPEA2, SMSEMOA, and MOEAD_DE can run on ZDT1 and return
a valid MOSADEResult with finite objective values.
"""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip("pymoo", reason="pymoo not installed")

from mosade.algorithm.pymoo_wrapper import PymooAlgorithm
from mosade.problems import get_problem


POP_SIZE = 20
MAX_EVALS = 2_000
SEED = 0


@pytest.fixture(scope="module")
def zdt1():
    return get_problem("ZDT1")


def _run(algo_name: str, problem: object) -> object:
    algo = PymooAlgorithm(algo_name, pop_size=POP_SIZE, max_evals=MAX_EVALS, seed=SEED)
    return algo.run(problem)


def _check_result(result, n_obj: int) -> None:
    assert result.F is not None, "result.F is None"
    assert result.X is not None, "result.X is None"
    assert result.F.ndim == 2, f"F.ndim={result.F.ndim}, expected 2"
    assert result.F.shape[1] == n_obj, f"F.shape={result.F.shape}"
    assert result.F.shape[0] > 0, "Empty Pareto approximation set"
    assert np.all(np.isfinite(result.F)), "Non-finite values in F"
    assert np.all(np.isfinite(result.X)), "Non-finite values in X"
    assert result.n_evals > 0, "eval counter is zero"


def test_nsga3_zdt1(zdt1):
    result = _run("NSGA3", zdt1)
    _check_result(result, n_obj=2)


def test_spea2_zdt1(zdt1):
    result = _run("SPEA2", zdt1)
    _check_result(result, n_obj=2)


def test_smsemoa_zdt1(zdt1):
    result = _run("SMSEMOA", zdt1)
    _check_result(result, n_obj=2)


def test_moead_de_zdt1(zdt1):
    result = _run("MOEAD_DE", zdt1)
    _check_result(result, n_obj=2)


def test_unknown_algo_raises(zdt1):
    with pytest.raises(ValueError, match="Unknown pymoo algorithm"):
        _run("NONEXISTENT_ALGO", zdt1)


def test_registry_includes_pymoo_algos():
    """All pymoo algorithms appear in the central registry."""
    from mosade.algorithm import ALGORITHM_REGISTRY

    for name in ("NSGA3", "SPEA2", "SMSEMOA", "MOEAD_DE"):
        assert name in ALGORITHM_REGISTRY, f"{name} missing from ALGORITHM_REGISTRY"
