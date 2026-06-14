"""MOSADE algorithm components.

``ALGORITHM_REGISTRY`` maps string type names (as used in the ``type:`` YAML
key) to callables so the experiment runner can instantiate algorithms by name
from YAML configs::

    algorithms:
      - name: MOSADE_run
        type: MOSADE      # optional; defaults to the name if omitted
        pop_size: 100
      - name: NSGA3_baseline
        type: NSGA3
        pop_size: 100

See :mod:`mosade.algorithm.registry` for the full registry and instructions on
adding new algorithms.
"""

from mosade.algorithm.mosade import MOSADE, MOSADEResult
from mosade.algorithm.nsga2 import NSGA2
from mosade.algorithm.moead import MOEAD
from mosade.algorithm.pymoo_wrapper import PymooAlgorithm
from mosade.algorithm.registry import ALGORITHM_REGISTRY

__all__ = [
    "MOSADE",
    "MOSADEResult",
    "NSGA2",
    "MOEAD",
    "PymooAlgorithm",
    "ALGORITHM_REGISTRY",
]
