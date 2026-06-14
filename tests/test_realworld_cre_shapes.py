from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mosade.problems import CRE21, CRE22, CRE23, CRE31, CRE32, get_problem


PROBLEMS = [
    (CRE21, 3, 2, 3),
    (CRE22, 4, 2, 4),
    (CRE23, 4, 2, 4),
    (CRE31, 7, 3, 10),
    (CRE32, 6, 3, 9),
]


def _load_reproblem_module():
    source = Path("external/reproblems/reproblem_python_ver/reproblem.py")
    if not source.exists():
        pytest.skip("external/reproblems is not available for numeric cross-checks")
    spec = importlib.util.spec_from_file_location("reproblem_source", source)
    if spec is None or spec.loader is None:
        pytest.skip("failed to import external reproblem source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("cls", "n_var", "n_obj", "n_constr"), PROBLEMS)
def test_realworld_problem_shapes_and_bounds(cls, n_var, n_obj, n_constr):
    problem = cls()
    assert problem.n_var == n_var
    assert problem.n_obj == n_obj
    assert problem.n_constr == n_constr
    assert problem.lower.shape == (n_var,)
    assert problem.upper.shape == (n_var,)
    assert np.all(problem.lower <= problem.upper)

    midpoint = ((problem.lower + problem.upper) / 2.0).reshape(1, -1)
    lower = problem.lower.reshape(1, -1)
    upper = problem.upper.reshape(1, -1)
    rng = np.random.default_rng(123)
    random_batch = rng.uniform(problem.lower, problem.upper, size=(20, n_var))
    X = np.vstack([lower, upper, midpoint, random_batch])

    F_raw, G_raw = problem._evaluate(X)
    assert F_raw.shape == (X.shape[0], n_obj)
    assert G_raw.shape == (X.shape[0], n_constr)
    assert np.isfinite(F_raw).all()
    assert np.isfinite(G_raw).all()

    F, CV = problem.evaluate(X)
    assert F.shape == (X.shape[0], n_obj)
    assert CV.shape == (X.shape[0],)
    assert np.isfinite(F).all()
    assert np.isfinite(CV).all()


@pytest.mark.parametrize(("cls", "name"), [(row[0], row[0].__name__) for row in PROBLEMS])
def test_realworld_problem_matches_reproblem_source(cls, name):
    reproblems = _load_reproblem_module()
    source_cls = getattr(reproblems, name)
    problem = cls()
    source = source_cls()

    rng = np.random.default_rng(20260423)
    X = rng.uniform(problem.lower, problem.upper, size=(20, problem.n_var))
    F, G = problem._evaluate(X)
    ours_violation = np.maximum(0.0, G)

    src_F = []
    src_violation = []
    for x in X:
        f_row, g_row = source.evaluate(x)
        src_F.append(np.asarray(f_row, dtype=float))
        src_violation.append(np.asarray(g_row, dtype=float))

    src_F_arr = np.vstack(src_F)
    src_violation_arr = np.vstack(src_violation)

    assert np.max(np.abs(F - src_F_arr)) < 1e-8
    assert np.max(np.abs(ours_violation - src_violation_arr)) < 1e-8


@pytest.mark.parametrize("name", ["CRE21", "CRE22", "CRE23", "CRE31", "CRE32"])
def test_get_problem_supports_realworld_names(name):
    problem = get_problem(name)
    assert problem.__class__.__name__ == name
