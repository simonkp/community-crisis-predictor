import numpy as np
import pandas as pd


class CrisisLabeler:
    def __init__(self, threshold_std: float = 1.5):
        self.threshold_std = threshold_std
        self.threshold: float | None = None
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, distress_scores: pd.Series) -> "CrisisLabeler":
        self.mean = float(distress_scores.mean())
        self.std = float(distress_scores.std())
        self.threshold = self.mean + self.threshold_std * self.std
        return self

    def label(self, distress_scores: pd.Series) -> pd.Series:
        if self.threshold is None:
            raise ValueError("Must call fit() before label()")

        # Shift by -1: label at time t = 1 if distress at t+1 exceeds threshold
        next_week_scores = distress_scores.shift(-1)
        labels = (next_week_scores > self.threshold).astype(int)
        # Last week has no future data — mark as NaN
        labels.iloc[-1] = np.nan
        return labels.rename("crisis_label")

    def get_crisis_weeks(self, distress_scores: pd.Series) -> pd.Series:
        if self.threshold is None:
            raise ValueError("Must call fit() before get_crisis_weeks()")
        return (distress_scores > self.threshold).astype(int).rename("is_crisis_week")
