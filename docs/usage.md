# Usage



A minimal run on a benchmark problem, scored against its analytical Pareto front:



```python

import numpy as np

from mosade.problems import ZDT1

from mosade.algorithm import MOSADE

from mosade.metrics import hypervolume, igd



problem = ZDT1(n_var=30)

result = MOSADE(pop_size=100, max_evals=25_000, seed=0).run(problem)



print("non-dominated solutions:", result.F.shape)

print("function evaluations:   ", result.n_evals)



pf = problem.pareto_front(200)

print("hypervolume:", round(hypervolume(result.F, ref=np.array([1.1, 1.1])), 4))

print("IGD:        ", round(igd(result.F, pf), 4))

```



A fixed `seed` makes the run deterministic within the pinned, tested

environment. `result.F` and `result.X` are

the objective and decision vectors of the final non-dominated set; `result.history`

holds per-generation diagnostics.

