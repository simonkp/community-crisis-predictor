import numpy as np
import pandas as pd
from scipy.stats import entropy

from src.features.progress_util import iter_weeks


def _posting_time_entropy(hours: list[int]) -> float:
    if not hours:
        return 0.0
    bins = np.zeros(24)
    for h in hours:
        bins[h % 24] += 1
    total = bins.sum()
    if total == 0:
        return 0.0
    probs = bins / total
    probs = probs[probs > 0]
    return float(entropy(probs))


def extract_behavioral_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in iter_weeks(weekly_df, desc="  Behavioral"):
        post_count = row.get("post_count", 0)
        hours = row.get("post_hours", [])

        rows.append({
            "post_volume": post_count,
            "avg_comments": row.get("total_comments", 0) / max(post_count, 1),
            "unique_posters": row.get("unique_authors", 0),
            "new_poster_ratio": row.get("new_author_ratio", 0.0),
            "posting_time_entropy": _posting_time_entropy(hours),
        })

    return pd.DataFrame(rows, index=weekly_df.index)
