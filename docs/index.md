\# MOSADE



\[!\[CI](https://github.com/Levvvi/MOSADE/actions/workflows/ci.yml/badge.svg)](https://github.com/Levvvi/MOSADE/actions/workflows/ci.yml)



\*\*Multi-Objective Self-Adaptive Differential Evolution\*\* — a Python library for

multi-objective optimisation over real-valued decision variables, including

constrained problems.



MOSADE combines several differential-evolution mutation strategies under an

online, self-adaptive credit-assignment scheme, decomposition-based

environmental selection, per-strategy parameter memories, and ε-constraint

handling. It interoperates with \[pymoo](https://pymoo.org): the comparison

baselines are pymoo's own implementations, run on MOSADE's problems and scored

by the same indicators, so head-to-head results are fair by construction.



\## Installation



```bash

pip install mosade

```



For the comparison baselines (pymoo) and the analysis tooling:



```bash

pip install "mosade\[baselines,analysis]"

```



See \[Usage](usage.md) to get started, or the \[API reference](api.md).

