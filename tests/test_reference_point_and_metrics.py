from __future__ import annotations

import numpy as np

from mosade.runner.experiment import _dominating_reference_point, _fe_to_threshold


def test_dominating_reference_point_handles_negative_maxima() -> None:
    maxima = np.array([-5.0, -0.2, 0.0, 3.0])
    ref = _dominating_reference_point(maxima)
    assert np.all(ref > maxima)
    assert np.isfinite(ref).all()


def test_fe_threshold_returns_none_when_final_hv_non_positive() -> None:
    hist = [{"n_evals": 100, "hv": 0.0}, {"n_evals": 200, "hv": 0.0}]
    assert _fe_to_threshold(hist, 0.0, 0.80) is None
    assert _fe_to_threshold(hist, float("nan"), 0.80) is None
