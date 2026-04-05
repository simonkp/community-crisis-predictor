"""Tests for shared dashboard view helpers."""

from src.dashboard.view_helpers import available_models_for_sub, resolve_model_results


def test_available_models_for_sub_order():
    sub = {"lstm": {"f1": 0.1}, "xgb": {"f1": 0.2}}
    assert available_models_for_sub(sub) == ["LSTM", "XGBoost", "Ensemble"]


def test_resolve_model_results_prefers_choice():
    sub = {"lstm": {"per_week": {"predictions": [0.0]}}, "xgb": {"per_week": {"predictions": [1.0]}}}
    out, choice = resolve_model_results(sub, "XGBoost")
    assert choice == "XGBoost"
    assert out["per_week"]["predictions"] == [1.0]
