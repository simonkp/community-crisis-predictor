from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def fit_binary_calibrator(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    *,
    method: str = "platt",
    min_samples: int = 20,
    min_class_count: int = 3,
) -> dict:
    """
    Fit a lightweight probability calibrator on a held-out calibration window.

    Returns a serializable dict:
      - {"type": "identity", "reason": "..."}
      - {"type": "platt", "coef": ..., "intercept": ...}
      - {"type": "isotonic", "x": [...], "y": [...]}
    """
    probs = np.asarray(y_prob, dtype=float).reshape(-1)
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    valid = ~(np.isnan(probs) | np.isnan(labels))
    probs = probs[valid]
    labels = labels[valid]

    if len(probs) < min_samples:
        return {"type": "identity", "reason": "insufficient_samples"}
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos < min_class_count or n_neg < min_class_count:
        return {"type": "identity", "reason": "insufficient_class_support"}

    mode = str(method).strip().lower()
    mode = mode if mode in {"platt", "isotonic"} else "platt"

    if mode == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs, labels)
        return {
            "type": "isotonic",
            "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_],
        }

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(probs.reshape(-1, 1), labels)
    return {
        "type": "platt",
        "coef": float(lr.coef_[0, 0]),
        "intercept": float(lr.intercept_[0]),
    }


def apply_binary_calibrator(y_prob: np.ndarray, calibrator: dict | None) -> np.ndarray:
    """Apply a calibrator dict returned by fit_binary_calibrator."""
    probs = np.asarray(y_prob, dtype=float).reshape(-1)
    if not calibrator or calibrator.get("type") in {None, "identity"}:
        return np.clip(probs, 0.0, 1.0)

    ctype = calibrator.get("type")
    if ctype == "platt":
        coef = float(calibrator.get("coef", 1.0))
        intercept = float(calibrator.get("intercept", 0.0))
        logits = coef * probs + intercept
        return 1.0 / (1.0 + np.exp(-logits))

    if ctype == "isotonic":
        xs = np.asarray(calibrator.get("x", []), dtype=float)
        ys = np.asarray(calibrator.get("y", []), dtype=float)
        if len(xs) < 2 or len(xs) != len(ys):
            return np.clip(probs, 0.0, 1.0)
        return np.interp(np.clip(probs, xs.min(), xs.max()), xs, ys)

    return np.clip(probs, 0.0, 1.0)
