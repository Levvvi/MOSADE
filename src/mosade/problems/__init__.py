"""Benchmark multi-objective optimization problems."""

from __future__ import annotations

from mosade.problems.base import Problem
from mosade.problems.zdt import ZDT1, ZDT2, ZDT3, ZDT4, ZDT6
from mosade.problems.dtlz import DTLZ1, DTLZ2, DTLZ3, DTLZ4
from mosade.problems.wfg import WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9
from mosade.problems.dascmop import (
    DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6,
    DASCMOP7, DASCMOP8, DASCMOP9,
)
from mosade.problems.realworld_cre import CRE21, CRE22, CRE23, CRE31, CRE32

__all__ = [
    "Problem",
    "ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6",
    "DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4",
    "WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9",
    "DASCMOP1", "DASCMOP2", "DASCMOP3", "DASCMOP4", "DASCMOP5", "DASCMOP6",
    "DASCMOP7", "DASCMOP8", "DASCMOP9",
    "CRE21", "CRE22", "CRE23", "CRE31", "CRE32",
]

# Registry: name -> class.  Used by experiment runner to instantiate by config string.
PROBLEM_REGISTRY: dict[str, type[Problem]] = {
    "ZDT1": ZDT1, "ZDT2": ZDT2, "ZDT3": ZDT3, "ZDT4": ZDT4, "ZDT6": ZDT6,
    "DTLZ1": DTLZ1, "DTLZ2": DTLZ2, "DTLZ3": DTLZ3, "DTLZ4": DTLZ4,
    "WFG1": WFG1, "WFG2": WFG2, "WFG3": WFG3, "WFG4": WFG4, "WFG5": WFG5,
    "WFG6": WFG6, "WFG7": WFG7, "WFG8": WFG8, "WFG9": WFG9,
    "DASCMOP1": DASCMOP1, "DASCMOP2": DASCMOP2, "DASCMOP3": DASCMOP3,
    "DASCMOP4": DASCMOP4, "DASCMOP5": DASCMOP5, "DASCMOP6": DASCMOP6,
    "DASCMOP7": DASCMOP7, "DASCMOP8": DASCMOP8, "DASCMOP9": DASCMOP9,
    "CRE21": CRE21, "CRE22": CRE22, "CRE23": CRE23, "CRE31": CRE31, "CRE32": CRE32,
}


def get_problem(name: str, **kwargs) -> Problem:
    """Instantiate a problem by its registry name."""
    if name not in PROBLEM_REGISTRY:
        raise ValueError(
            f"Unknown problem '{name}'. Available: {sorted(PROBLEM_REGISTRY)}"
        )
    return PROBLEM_REGISTRY[name](**kwargs)
