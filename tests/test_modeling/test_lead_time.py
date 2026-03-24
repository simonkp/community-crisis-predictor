import pandas as pd

from src.modeling.evaluate import _compute_detection_lead_time


def test_lead_time_uses_crisis_min_for_binary():
    predictions = pd.Series([0, 1, 1, 0, 1], dtype=float)
    actuals = pd.Series([0, 1, 1, 0, 1], dtype=float)
    out = _compute_detection_lead_time(predictions, actuals, crisis_min=1)
    assert out["mean"] >= 0
    assert isinstance(out["distribution"], list)


def test_lead_time_uses_crisis_min_for_multiclass():
    predictions = pd.Series([1, 2, 2, 1, 3], dtype=float)
    actuals = pd.Series([1, 1, 2, 1, 3], dtype=float)
    out = _compute_detection_lead_time(predictions, actuals, crisis_min=2)
    # Crisis starts are only where actual >= 2 (indexes 2 and 4)
    assert len(out["distribution"]) == 2
