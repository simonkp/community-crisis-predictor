import numpy as np
import pandas as pd

STATE_NAMES = {
    0: "Stable",
    1: "Emerging Distress",
    2: "Acute Risk",
    3: "Critical Escalation",
}


class CrisisLabeler:
    def __init__(
        self,
        threshold_std: float = 1.5,
        thresholds_std: list[float] | None = None,
    ):
        # thresholds_std takes priority; threshold_std kept for backward compat
        self.thresholds_std = thresholds_std if thresholds_std is not None else [0.5, 1.0, 2.0]
        self.threshold_std = threshold_std
        self.thresholds: list[float] | None = None
        self.threshold: float | None = None  # backward compat = thresholds[2] (crisis-level)
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, distress_scores: pd.Series) -> "CrisisLabeler":
        self.mean = float(distress_scores.mean())
        self.std = float(distress_scores.std())
        self.thresholds = [self.mean + t * self.std for t in self.thresholds_std]
        self.threshold = self.thresholds[2]  # backward compat
        return self

    def label(self, distress_scores: pd.Series) -> pd.Series:
        if self.thresholds is None:
            raise ValueError("Must call fit() before label()")

        next_week_scores = distress_scores.shift(-1)

        def _classify(score):
            if np.isnan(score):
                return np.nan
            if score >= self.thresholds[2]:
                return 3
            elif score >= self.thresholds[1]:
                return 2
            elif score >= self.thresholds[0]:
                return 1
            return 0

        labels = next_week_scores.apply(_classify)
        labels.iloc[-1] = np.nan
        return labels.rename("crisis_label")

    def get_crisis_weeks(self, distress_scores: pd.Series) -> pd.Series:
        if self.thresholds is None:
            raise ValueError("Must call fit() before get_crisis_weeks()")
        return (distress_scores > self.threshold).astype(int).rename("is_crisis_week")
