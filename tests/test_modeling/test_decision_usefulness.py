import numpy as np
import pytest

from src.modeling.evaluate import (
    compute_decision_usefulness,
    top_k_alert_recall,
)


def test_top_k_perfect_ranking():
    y_true = np.array([1, 0, 1, 0, 1])
    y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    out = top_k_alert_recall(y_true, y_prob, k_values=[1, 2, 3])
    assert out["1"]["captured"] == 1
    assert out["1"]["total_positives"] == 3
    assert out["1"]["recall"] == pytest.approx(1 / 3)
    assert out["3"]["captured"] == 3
    assert out["3"]["recall"] == 1.0


def test_top_k_empty_positives():
    y_true = np.zeros(5, dtype=int)
    y_prob = np.linspace(0.1, 0.5, 5)
    out = top_k_alert_recall(y_true, y_prob, k_values=[2])
    assert out["2"]["total_positives"] == 0
    assert out["2"]["recall"] == 0.0


def test_compute_decision_usefulness_random_matches_formula():
    y_true = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    y_prob = np.linspace(0.1, 1.0, 10)
    du = compute_decision_usefulness(y_true, y_prob, k_values=[2])
    n = len(y_true)
    assert du["n_weeks"] == n
    assert du["n_elevated_distress_weeks"] == int(y_true.sum())
    assert du["random_expected_recall"]["2"] == pytest.approx(2 / n)


def test_compute_decision_usefulness_persistence():
    # Week t alerts if prev elevated; force high persistence capture
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_prob = np.ones(6) * 0.5
    du = compute_decision_usefulness(y_true, y_prob, k_values=[2])
    pers = du["persistence"]["2"]
    assert pers["total_positives"] == 2
    assert pers["captured"] >= 1
