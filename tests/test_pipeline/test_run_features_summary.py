import pandas as pd

from src.pipeline.run_features import _counts_by_subreddit


def test_counts_by_subreddit_normalizes_case():
    df = pd.DataFrame(
        {
            "subreddit": ["SuicideWatch", "suicidewatch", " suicidewatch ", "Anxiety"],
        }
    )

    counts = _counts_by_subreddit(df)

    assert int(counts["suicidewatch"]) == 3
    assert int(counts["anxiety"]) == 1
