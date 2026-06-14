"""Central algorithm registry for the MOSADE experiment runner.

The registry maps string type names (as used in the ``type:`` YAML key) to
callables with the signature::

    callable(seed: int, pop_size: int, max_evals: int, **kwargs) -> algo

Both algorithm classes (whose ``__init__`` accepts these kwargs) and lambda
wrappers around :class:`~mosade.algorithm.pymoo_wrapper.PymooAlgorithm` are
supported.

To add a new algorithm, append an entry here and optionally expose it from
:mod:`mosade.algorithm.__init__`.
"""

from __future__ import annotations

from typing import Any

from mosade.algorithm.mosade import MOSADE
from mosade.algorithm.moead import MOEAD
from mosade.algorithm.nsga2 import NSGA2
from mosade.algorithm.pymoo_wrapper import PymooAlgorithm


def _pymoo(name: str) -> Any:
    """Return a factory lambda for a named pymoo algorithm."""
    return lambda **kw: PymooAlgorithm(name, **kw)


def _mosade_preset(**presets: Any) -> type:
    """Create a MOSADE subclass with preset constructor arguments.

    YAML kwargs override presets, so configs can still fine-tune if needed.
    """

    class _Preset(MOSADE):
        _PRESETS: dict[str, Any] = presets

        def __init__(self, **kwargs: Any) -> None:
            merged = {**self._PRESETS, **kwargs}
            super().__init__(**merged)

    _Preset.__name__ = _Preset.__qualname__ = "MOSADE"
    return _Preset


#: Maps algorithm type names to callables ``(seed, pop_size, max_evals, **kw) -> algo``.
ALGORITHM_REGISTRY: dict[str, Any] = {
    # Built-in MOSADE implementations
    "MOSADE": MOSADE,
    "NSGA2": NSGA2,
    "MOEAD": MOEAD,
    # pymoo wrappers
    "NSGA3": _pymoo("NSGA3"),
    "SPEA2": _pymoo("SPEA2"),
    "SMSEMOA": _pymoo("SMSEMOA"),
    "MOEAD_DE": _pymoo("MOEAD_DE"),
    # Ablation variants (used by configs/ablation.yaml)
    "MOSADE_single_S1": _mosade_preset(fixed_strategy=0),
    "MOSADE_single_S2": _mosade_preset(fixed_strategy=1),
    "MOSADE_single_S3": _mosade_preset(fixed_strategy=2),
    "MOSADE_single_S4": _mosade_preset(fixed_strategy=3),
    "MOSADE_fixed_eps_initial": _mosade_preset(eps_mode="fixed_initial"),
    "MOSADE_eps_zero": _mosade_preset(eps_mode="zero"),
    "MOSADE_fixed_eps_zero": _mosade_preset(eps_mode="zero"),
    "MOSADE_shared_memory": _mosade_preset(memory_scope="shared"),
    "MOSADE_no_restart": _mosade_preset(restart_enabled=False),
    "MOSADE_domselect": _mosade_preset(selection_mode="dominance"),
    # Deprecated compatibility alias.  It is retained so historical configs
    # remain runnable, but table-generation code refuses to export this label.
    "MOSADE_fixed_eps": _mosade_preset(
        eps_mode="zero",
        deprecated_variant_label="MOSADE_fixed_eps",
    ),
    "MOSADE_uniform_strategy": _mosade_preset(disable_credit=True),
}
