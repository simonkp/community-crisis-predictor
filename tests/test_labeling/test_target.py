import numpy as np
import pandas as pd
from src.labeling.target import CrisisLabeler


def test_crisis_labeler_fit():
    scores = pd.Series([0.1, 0.2, 0.3, 0.5, 0.8, 0.1, 0.2])
    labeler = CrisisLabeler(threshold_std=1.5)
    labeler.fit(scores)

    assert labeler.threshold is not None
    assert labeler.mean is not None
    assert labeler.std is not None
    assert labeler.threshold > labeler.mean


def test_crisis_labeler_forward_shift():
    scores = pd.Series([0.0, 0.0, 0.0, 0.0, 5.0])
    labeler = CrisisLabeler(threshold_std=1.0)
    labeler.fit(scores)

    labels = labeler.label(scores)
    # Week before the spike (index 3) should be labeled >= 1 (some distress state)
    # With 4-class labels [0.5, 1.0, 2.0] sigma thresholds, the spike (5.0) falls
    # in class 2 (Elevated Distress) for typical mean/std of this series.
    assert labels.iloc[3] >= 1
    # Last week should be NaN (no future data)
    assert np.isnan(labels.iloc[-1])


def test_crisis_labeler_no_fit_raises():
    labeler = CrisisLabeler()
    import pytest
    with pytest.raises(ValueError):
        labeler.label(pd.Series([1, 2, 3]))
