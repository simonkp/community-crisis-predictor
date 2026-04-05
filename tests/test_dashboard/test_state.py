"""Tests for dashboard state helpers."""

import pytest

from src.dashboard.state import merge_ensemble_results, pick_model_results


def test_merge_ensemble_averages_predictions_and_probs():
    sub = {
        "lstm": {
            "recall": 0.4,
            "precision": 0.3,
            "f1": 0.35,
            "pr_auc": 0.5,
            "per_week": {
                "predictions": [0.0, 2.0],
                "probabilities": [0.2, 0.8],
                "actuals": [0.0, 2.0],
            },
        },
        "xgb": {
            "recall": 0.6,
            "precision": 0.5,
            "f1": 0.55,
            "pr_auc": 0.6,
            "per_week": {
                "predictions": [0.0, 2.0],
                "probabilities": [0.4, 0.6],
                "actuals": [0.0, 2.0],
            },
        },
    }
    merged = merge_ensemble_results(sub)
    assert merged["recall"] == pytest.approx(0.5)
    pw = merged["per_week"]
    assert pw["predictions"] == [0.0, 2.0]
    assert pw["probabilities"][0] == pytest.approx(0.3)
    assert pw["probabilities"][1] == pytest.approx(0.7)


def test_pick_model_results_ensemble():
    sub = {
        "lstm": {"per_week": {"predictions": [1.0]}},
        "xgb": {"per_week": {"predictions": [3.0]}},
    }
    out = pick_model_results(sub, "Ensemble")
    assert "per_week" in out
    assert out["per_week"]["predictions"][0] == 3.0  # ensemble uses max (safety-first), not mean


def test_pick_model_results_fallback():
    sub = {"xgb": {"recall": 0.1, "per_week": {}}}
    assert pick_model_results(sub, "LSTM") == {}
    assert pick_model_results(sub, "XGBoost")["recall"] == 0.1
