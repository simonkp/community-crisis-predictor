import numpy as np

from src.modeling.calibration import apply_binary_calibrator, fit_binary_calibrator


def test_identity_calibrator_when_too_few_samples():
    probs = np.array([0.1, 0.2, 0.3])
    labels = np.array([0, 1, 0])
    cal = fit_binary_calibrator(probs, labels, min_samples=20)
    assert cal["type"] == "identity"
    out = apply_binary_calibrator(probs, cal)
    np.testing.assert_allclose(out, probs)


def test_platt_calibrator_outputs_probabilities():
    rng = np.random.RandomState(42)
    probs = rng.uniform(0.01, 0.99, size=80)
    labels = (probs > 0.5).astype(int)
    cal = fit_binary_calibrator(probs, labels, method="platt", min_samples=20, min_class_count=5)
    assert cal["type"] in {"platt", "identity"}
    out = apply_binary_calibrator(probs, cal)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_isotonic_calibrator_outputs_probabilities():
    rng = np.random.RandomState(0)
    probs = rng.uniform(0.01, 0.99, size=120)
    labels = (probs > 0.6).astype(int)
    cal = fit_binary_calibrator(probs, labels, method="isotonic", min_samples=20, min_class_count=5)
    assert cal["type"] in {"isotonic", "identity"}
    out = apply_binary_calibrator(probs, cal)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
