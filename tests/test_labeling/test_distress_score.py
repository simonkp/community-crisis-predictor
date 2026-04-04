import numpy as np
import pandas as pd
from src.labeling.distress_score import compute_distress_score


def test_compute_distress_score(sample_feature_matrix):
    scores = compute_distress_score(sample_feature_matrix)
    assert len(scores) == len(sample_feature_matrix)
    assert scores.name == "distress_score"
    # Z-scored, so mean should be near 0
    assert abs(scores.mean()) < 0.5


def test_distress_score_higher_for_negative():
    df = pd.DataFrame({
        "avg_negative": [0.1, 0.1, 0.5, 0.5],
        "hopelessness_density": [0.01, 0.01, 0.1, 0.1],
        "help_seeking_density": [0.01, 0.01, 0.08, 0.08],
    })
    scores = compute_distress_score(df)
    # Later rows (more distressed) should have higher scores
    assert scores.iloc[2] > scores.iloc[0]


def test_compute_distress_score_without_normalization():
    df = pd.DataFrame(
        {
            "avg_negative": [0.1, 0.5],
            "hopelessness_density": [0.01, 0.09],
            "help_seeking_density": [0.02, 0.07],
        }
    )
    scores = compute_distress_score(df, normalize=False)
    assert scores.iloc[1] > scores.iloc[0]
