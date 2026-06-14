"""Thesis-experiment tooling for MOSADE.

This package holds analysis, plotting, statistics, and result-aggregation
utilities used to produce the thesis experiments. It depends on the shippable
``mosade`` library but is deliberately *not* part of the distributed wheel --
``[tool.setuptools.packages.find]`` only discovers ``mosade`` under ``src/``.
"""
