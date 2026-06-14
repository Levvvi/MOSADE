"""Real-world constrained engineering design problems.

This module ports the core analytical CRE problems used in the RE benchmark
suite into MOSADE's native :class:`~mosade.problems.base.Problem` interface.

Source provenance:
- ``external/reproblems/reproblem_python_ver/reproblem.py`` (primary analytic
  source and numeric cross-check target)
- ``external/MultiObjectiveProjects/functions/problems_definitions/*.R``
  (secondary provenance cross-check)

Notes
-----
The external reference implementations often compute a positive *feasibility
margin* and then convert negative margins into positive violation magnitudes.
MOSADE uses the convention ``g_j(x) <= 0`` means feasible.  Therefore each
ported constraint here returns the negated reference margin so that
``np.maximum(0, G)`` matches the original violation magnitudes exactly.
"""

from __future__ import annotations

import numpy as np

from mosade.problems.base import Problem


class CRE21(Problem):
    """CRE21 - Two bar truss design.

    Source provenance:
    - ``external/reproblems/reproblem_python_ver/reproblem.py`` (``class CRE21``)
    - ``external/MultiObjectiveProjects/functions/problems_definitions/CRE21.R``
    """

    def __init__(self) -> None:
        lower = np.array([1.0e-5, 1.0e-5, 1.0], dtype=float)
        upper = np.array([100.0, 100.0, 3.0], dtype=float)
        super().__init__(n_var=3, n_obj=2, n_constr=3, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]

        f1 = x1 * np.sqrt(16.0 + x3**2) + x2 * np.sqrt(1.0 + x3**2)
        f2 = (20.0 * np.sqrt(16.0 + x3**2)) / (x1 * x3)

        margin1 = 0.1 - f1
        margin2 = 100000.0 - f2
        margin3 = 100000.0 - ((80.0 * np.sqrt(1.0 + x3**2)) / (x3 * x2))

        F = np.column_stack([f1, f2])
        G = -np.column_stack([margin1, margin2, margin3])
        return F, G


class CRE22(Problem):
    """CRE22 - Welded beam design.

    Source provenance:
    - ``external/reproblems/reproblem_python_ver/reproblem.py`` (``class CRE22``)
    - ``external/MultiObjectiveProjects/functions/problems_definitions/CRE22.R``
    """

    def __init__(self) -> None:
        lower = np.array([0.125, 0.1, 0.1, 0.125], dtype=float)
        upper = np.array([5.0, 10.0, 10.0, 5.0], dtype=float)
        super().__init__(n_var=4, n_obj=2, n_constr=4, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        P = 6000.0
        L = 14.0
        E = 30.0e6
        G_shear = 12.0e6
        tau_max = 13600.0
        sigma_max = 30000.0

        f1 = (1.10471 * x1**2 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        f2 = (4.0 * P * L**3) / (E * x4 * x3**3)

        M = P * (L + (x2 / 2.0))
        R = np.sqrt((x2**2) / 4.0 + ((x1 + x3) / 2.0) ** 2)
        J = 2.0 * np.sqrt(2.0) * x1 * x2 * (((x2**2) / 12.0) + ((x1 + x3) / 2.0) ** 2)
        tau_dd = (M * R) / J
        tau_d = P / (np.sqrt(2.0) * x1 * x2)
        tau = np.sqrt(
            tau_d**2
            + ((2.0 * tau_d * tau_dd * x2) / (2.0 * R))
            + tau_dd**2
        )
        sigma = (6.0 * P * L) / (x4 * x3**2)
        pc = (
            4.013
            * E
            * np.sqrt((x3**2 * x4**6) / 36.0)
            / (L**2)
            * (1.0 - (x3 / (2.0 * L)) * np.sqrt(E / (4.0 * G_shear)))
        )

        margin1 = tau_max - tau
        margin2 = sigma_max - sigma
        margin3 = x4 - x1
        margin4 = pc - P

        F = np.column_stack([f1, f2])
        G = -np.column_stack([margin1, margin2, margin3, margin4])
        return F, G


class CRE23(Problem):
    """CRE23 - Disc brake design.

    Source provenance:
    - ``external/reproblems/reproblem_python_ver/reproblem.py`` (``class CRE23``)
    - ``external/MultiObjectiveProjects/functions/problems_definitions/CRE23.R``
    """

    def __init__(self) -> None:
        lower = np.array([55.0, 75.0, 1000.0, 11.0], dtype=float)
        upper = np.array([80.0, 110.0, 3000.0, 20.0], dtype=float)
        super().__init__(n_var=4, n_obj=2, n_constr=4, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        f1 = 4.9e-5 * (x2**2 - x1**2) * (x4 - 1.0)
        f2 = (9.82e6 * (x2**2 - x1**2)) / (x3 * x4 * (x2**3 - x1**3))

        term2 = x3 / (3.14 * (x2**2 - x1**2))
        term3 = (2.22e-3 * x3 * (x2**3 - x1**3)) / ((x2**2 - x1**2) ** 2)
        term4 = (2.66e-2 * x3 * x4 * (x2**3 - x1**3)) / (x2**2 - x1**2)

        margin1 = (x2 - x1) - 20.0
        margin2 = 0.4 - term2
        margin3 = 1.0 - term3
        margin4 = term4 - 900.0

        F = np.column_stack([f1, f2])
        G = -np.column_stack([margin1, margin2, margin3, margin4])
        return F, G


class CRE31(Problem):
    """CRE31 - Car side impact design.

    Source provenance:
    - ``external/reproblems/reproblem_python_ver/reproblem.py`` (``class CRE31``)
    - ``external/MultiObjectiveProjects/functions/problems_definitions/CRE31.R``
    """

    def __init__(self) -> None:
        lower = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=float)
        upper = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=float)
        super().__init__(n_var=7, n_obj=3, n_constr=10, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x1, x2, x3, x4, x5, x6, x7 = (X[:, i] for i in range(7))

        f1 = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 1.0e-5 * x6 + 2.73 * x7
        f2 = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f3 = 0.5 * (vmbp + vfd)

        margin1 = 1.0 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        margin2 = 0.32 - (
            0.261
            - 0.0159 * x1 * x2
            - 0.06486 * x1
            - 0.019 * x2 * x7
            + 0.0144 * x3 * x5
            + 0.0154464 * x6
        )
        margin3 = 0.32 - (
            0.214
            + 0.00817 * x5
            - 0.045195 * x1
            - 0.0135168 * x1
            + 0.03099 * x2 * x6
            - 0.018 * x2 * x7
            + 0.007176 * x3
            + 0.023232 * x3
            - 0.00364 * x5 * x6
            - 0.018 * x2**2
        )
        margin4 = 0.32 - (
            0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2**2
        )
        margin5 = 32.0 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        margin6 = 32.0 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        margin7 = 32.0 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        margin8 = 4.0 - f2
        margin9 = 9.9 - vmbp
        margin10 = 15.7 - vfd

        F = np.column_stack([f1, f2, f3])
        G = -np.column_stack([
            margin1,
            margin2,
            margin3,
            margin4,
            margin5,
            margin6,
            margin7,
            margin8,
            margin9,
            margin10,
        ])
        return F, G


class CRE32(Problem):
    """CRE32 - Conceptual marine design.

    Source provenance:
    - ``external/reproblems/reproblem_python_ver/reproblem.py`` (``class CRE32``)
    - ``external/MultiObjectiveProjects/functions/problems_definitions/CRE32.R``
    """

    def __init__(self) -> None:
        lower = np.array([150.0, 20.0, 13.0, 10.0, 14.0, 0.63], dtype=float)
        upper = np.array([274.32, 32.31, 25.0, 11.71, 18.0, 0.75], dtype=float)
        super().__init__(n_var=6, n_obj=3, n_constr=9, lower=lower, upper=upper)

    def _evaluate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        x_l, x_b, x_d, x_t, x_vk, x_cb = (X[:, i] for i in range(6))

        displacement = 1.025 * x_l * x_b * x_t * x_cb
        velocity = 0.5144 * x_vk
        gravity = 9.8065
        fn = velocity / np.sqrt(gravity * x_l)
        a = 4977.06 * x_cb**2 - 8105.61 * x_cb + 4456.51
        b = -10847.2 * x_cb**2 + 12817.0 * x_cb - 6960.32

        power = (displacement ** (2.0 / 3.0) * x_vk**3) / (a + b * fn)
        outfit_weight = x_l**0.8 * x_b**0.6 * x_d**0.3 * x_cb**0.1
        steel_weight = 0.034 * x_l**1.7 * x_b**0.7 * x_d**0.4 * x_cb**0.5
        machinery_weight = 0.17 * power**0.9
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * (
            2000.0 * steel_weight**0.85 + 3500.0 * outfit_weight + 2400.0 * power**0.8
        )
        capital_costs = 0.2 * ship_cost

        dwt = displacement - light_ship_weight
        running_costs = 40000.0 * dwt**0.3

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_vk
        handling_rate = 8000.0

        daily_consumption = (0.19 * power * 24.0) / 1000.0 + 0.2
        fuel_cost = 1.05 * daily_consumption * sea_days * 100.0
        port_cost = 6.3 * dwt**0.8

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_dwt = 2.0 * dwt**0.5
        cargo_dwt = dwt - fuel_carried - miscellaneous_dwt
        port_days = 2.0 * ((cargo_dwt / handling_rate) + 0.5)
        rtpa = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * rtpa
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_dwt * rtpa

        f1 = annual_costs / annual_cargo
        f2 = light_ship_weight
        f3 = -annual_cargo

        kb = 0.53 * x_t
        bmt = ((0.085 * x_cb - 0.002) * x_b**2) / (x_t * x_cb)
        kg = 1.0 + 0.52 * x_d

        margin1 = (x_l / x_b) - 6.0
        margin2 = -(x_l / x_d) + 15.0
        margin3 = -(x_l / x_t) + 19.0
        margin4 = 0.45 * dwt**0.31 - x_t
        margin5 = 0.7 * x_d + 0.7 - x_t
        margin6 = 500000.0 - dwt
        margin7 = dwt - 3000.0
        margin8 = 0.32 - fn
        margin9 = (kb + bmt - kg) - (0.07 * x_b)

        F = np.column_stack([f1, f2, f3])
        G = -np.column_stack([
            margin1,
            margin2,
            margin3,
            margin4,
            margin5,
            margin6,
            margin7,
            margin8,
            margin9,
        ])
        return F, G
