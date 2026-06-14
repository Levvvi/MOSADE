from __future__ import annotations

from mosade.algorithm.registry import ALGORITHM_REGISTRY


def test_single_strategy_presets_exist() -> None:
    for name in (
        "MOSADE_single_S1",
        "MOSADE_single_S2",
        "MOSADE_single_S3",
        "MOSADE_single_S4",
        "MOSADE_fixed_eps_initial",
        "MOSADE_eps_zero",
        "MOSADE_fixed_eps_zero",
        "MOSADE_shared_memory",
        "MOSADE_no_restart",
        "MOSADE_domselect",
        "MOSADE_uniform_strategy",
    ):
        assert name in ALGORITHM_REGISTRY


def test_deprecated_fixed_eps_alias_is_explicit_fixed_zero() -> None:
    assert "MOSADE_fixed_eps" in ALGORITHM_REGISTRY
    algo = ALGORITHM_REGISTRY["MOSADE_fixed_eps"](pop_size=20, max_evals=40, seed=1)
    assert algo.eps_mode == "zero"
    assert algo.deprecated_variant_label == "MOSADE_fixed_eps"


def test_new_ablation_presets_have_distinct_runtime_semantics() -> None:
    assert ALGORITHM_REGISTRY["MOSADE_eps_zero"](pop_size=20, max_evals=40).eps_mode == "zero"
    assert (
        ALGORITHM_REGISTRY["MOSADE_shared_memory"](pop_size=20, max_evals=40).memory_scope
        == "shared"
    )
    assert (
        ALGORITHM_REGISTRY["MOSADE_no_restart"](pop_size=20, max_evals=40).restart_enabled
        is False
    )
    assert (
        ALGORITHM_REGISTRY["MOSADE_domselect"](pop_size=20, max_evals=40).selection_mode
        == "dominance"
    )
