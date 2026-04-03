import numpy as np
import pandas as pd

DRIFT_SIGNALS = [
    "avg_negative",
    "hopelessness_density",
    "topic_shift_jsd",
    "topic_shift_jsd_4w",
    "suicidality_total",
    "isolation_total",
    "post_volume",
]

ALERT_LEVELS = {0: "normal", 1: "warning", 2: "alert", 3: "critical"}

THRESHOLDS = [1.0, 2.0, 3.0]  # sigma cutoffs for warning / alert / critical


class DriftDetector:
    def __init__(self, baseline_weeks: int = 12):
        self.baseline_weeks = baseline_weeks

    def detect(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        n = len(feature_df)
        available_signals = [s for s in DRIFT_SIGNALS if s in feature_df.columns]

        results = []
        for t in range(n):
            start = max(0, t - self.baseline_weeks)
            baseline = feature_df.iloc[start:t]

            z_scores: dict[str, float] = {}
            for sig in available_signals:
                if len(baseline) < 2:
                    z_scores[sig] = 0.0
                else:
                    bm = float(baseline[sig].mean())
                    bs = float(baseline[sig].std())
                    if bs > 0:
                        z_scores[sig] = float(
                            (float(feature_df.iloc[t][sig]) - bm) / bs
                        )
                    else:
                        z_scores[sig] = 0.0

            max_z = max(z_scores.values()) if z_scores else 0.0
            if max_z >= THRESHOLDS[2]:
                level = 3
            elif max_z >= THRESHOLDS[1]:
                level = 2
            elif max_z >= THRESHOLDS[0]:
                level = 1
            else:
                level = 0

            dominant = max(z_scores, key=z_scores.get) if z_scores else "none"

            row: dict = {f"z_{sig}": z_scores.get(sig, 0.0) for sig in available_signals}
            week_val = feature_df.iloc[t].get("week_start", str(t))
            row["week_start"] = week_val
            row["aggregate_level"] = level
            row["alert_level_name"] = ALERT_LEVELS[level]
            row["dominant_signal"] = dominant
            results.append(row)

        return pd.DataFrame(results)
