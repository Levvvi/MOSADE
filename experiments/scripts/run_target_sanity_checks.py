"""Run lightweight MOSADE A1-A7 semantic sanity checks."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mosade.algorithm.registry import ALGORITHM_REGISTRY  # noqa: E402
from mosade.problems.base import Problem  # noqa: E402
from mosade.problems import get_problem  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "audit"


class AlwaysInfeasibleToyProblem(Problem):
    """Small constrained problem with a strictly positive initial epsilon."""

    def __init__(self) -> None:
        super().__init__(
            n_var=2,
            n_obj=2,
            n_constr=1,
            lower=np.zeros(2),
            upper=np.ones(2),
        )

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        f1 = X[:, 0]
        f2 = 1.0 - X[:, 0] + X[:, 1]
        g = 1.0 + 0.25 * X[:, 0]
        return np.column_stack([f1, f2]), g.reshape(-1, 1)


def _digest(obj: object) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _run_variant(name: str, problem_name: str = "DASCMOP7", **problem_kwargs):
    problem = get_problem(problem_name, **problem_kwargs)
    algo = ALGORITHM_REGISTRY[name](pop_size=20, max_evals=80, seed=123, track_interval=0)
    return algo.run(problem)


def _run_eps_variant(name: str):
    problem = AlwaysInfeasibleToyProblem()
    algo = ALGORITHM_REGISTRY[name](pop_size=20, max_evals=80, seed=123, track_interval=0)
    return algo.run(problem)


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"status": "ok", "checks": []}

    variants = ["MOSADE", "MOSADE_fixed_eps_initial", "MOSADE_eps_zero"]
    eps = {}
    for name in variants:
        result = _run_eps_variant(name)
        eps_hist = [float(v) for v in result.history["epsilon"]]
        eps[name] = {
            "eps_mode": result.metadata["eps_mode"],
            "epsilon_initial": result.metadata["epsilon_initial"],
            "epsilon_final": result.metadata["epsilon_final"],
            "epsilon_min": result.metadata["epsilon_min"],
            "epsilon_max": result.metadata["epsilon_max"],
            "epsilon_num_updates": result.metadata["epsilon_num_updates"],
            "epsilon_history_digest": _digest(eps_hist),
            "epsilon_history_head": eps_hist[:5],
        }
    results["epsilon_modes"] = eps
    assert eps["MOSADE"]["eps_mode"] == "adaptive"
    assert eps["MOSADE_fixed_eps_initial"]["eps_mode"] == "fixed_initial"
    assert eps["MOSADE_eps_zero"]["eps_mode"] == "zero"
    assert eps["MOSADE_eps_zero"]["epsilon_max"] == 0.0
    assert eps["MOSADE"]["epsilon_num_updates"] > 0
    assert eps["MOSADE_fixed_eps_initial"]["epsilon_initial"] > 0.0
    assert eps["MOSADE_fixed_eps_initial"]["epsilon_min"] == eps["MOSADE_fixed_eps_initial"]["epsilon_max"]

    memory_default = _run_variant("MOSADE", difficulty=7)
    memory_shared = _run_variant("MOSADE_shared_memory", difficulty=7)
    results["memory"] = {
        "default_scope": memory_default.metadata["memory_scope"],
        "shared_scope": memory_shared.metadata["memory_scope"],
        "default_digest": memory_default.metadata["memory_history_digest"],
        "shared_digest": memory_shared.metadata["memory_history_digest"],
    }
    assert memory_default.metadata["memory_scope"] == "per_strategy"
    assert memory_shared.metadata["memory_scope"] == "shared"
    assert (
        memory_default.metadata["memory_history_digest"]
        != memory_shared.metadata["memory_history_digest"]
    )

    restart_off = _run_variant("MOSADE_no_restart", difficulty=7)
    results["restart"] = {
        "restart_enabled": restart_off.metadata["restart_enabled"],
        "restart_count": restart_off.metadata["restart_count"],
    }
    assert restart_off.metadata["restart_enabled"] is False
    assert restart_off.metadata["restart_count"] == 0

    dom = _run_variant("MOSADE_domselect", difficulty=7)
    results["selection"] = {
        "selection_mode": dom.metadata["selection_mode"],
        "n_fronts_seen": int(np.max(dom.history["n_fronts"])) if dom.history["n_fronts"] else 0,
    }
    assert dom.metadata["selection_mode"] == "dominance"

    (AUDIT / "target_sanity_checks.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (AUDIT / "target_sanity_checks_cn.md").write_text(
        "\n".join(
            [
                "# A1-A7 目标语义 smoke 检查",
                "",
                "- epsilon 三模式均可运行，并写出互斥 `eps_mode`。",
                "- `MOSADE_shared_memory` 使用 shared memory，且 memory digest 与默认不同。",
                "- `MOSADE_no_restart` 写出 `restart_enabled=false` 且 smoke run 中 `restart_count=0`。",
                "- `MOSADE_domselect` 写出 `selection_mode=dominance`。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("target sanity checks passed")


if __name__ == "__main__":
    main()
