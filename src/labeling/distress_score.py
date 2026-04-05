import numpy as np
import pandas as pd


def compute_distress_score(
    feature_df: pd.DataFrame,
    weights: dict[str, float] = None,
    *,
    normalize: bool = True,
) -> pd.Series:
    if weights is None:
        weights = {
            "neg_sentiment": 0.25,
            "hopelessness": 0.20,
            "help_seeking": 0.15,
            "suicidality": 0.20,
            "isolation": 0.10,
            "economic_stress": 0.05,
            "domestic_stress": 0.05,
        }

    # Map weight keys to feature columns
    col_map = {
        "neg_sentiment": "avg_negative",
        "hopelessness": "hopelessness_density",
        "help_seeking": "help_seeking_density",
        "suicidality": "suicidality_density",
        "isolation": "isolation_density",
        "economic_stress": "economic_stress_density",
        "domestic_stress": "domestic_stress_density",
    }

    components = {}
    for key, col in col_map.items():
        if col in feature_df.columns:
            values = feature_df[col].values.astype(float)
            if normalize:
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    components[key] = (values - mean) / std
                else:
                    components[key] = np.zeros_like(values)
            else:
                components[key] = values

    score = np.zeros(len(feature_df))
    for key, weight in weights.items():
        if key in components:
            score += weight * components[key]

    return pd.Series(score, index=feature_df.index, name="distress_score")
