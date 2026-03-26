import pandas as pd
from src.processing.weekly_aggregator import WeeklyAggregator


def test_aggregate_groups_by_week(sample_raw_posts):
    # Add clean_text and author_hash columns
    sample_raw_posts["clean_text"] = sample_raw_posts["title"] + ". " + sample_raw_posts["selftext"]
    sample_raw_posts["author_hash"] = sample_raw_posts["author"].apply(lambda x: x[:8])

    aggregator = WeeklyAggregator()
    result = aggregator.aggregate(sample_raw_posts)

    assert len(result) > 0
    assert "texts" in result.columns
    assert "post_count" in result.columns
    assert "unique_authors" in result.columns
    assert "new_author_ratio" in result.columns
    assert all(result["post_count"] > 0)


def test_aggregate_tracks_new_authors():
    df = pd.DataFrame({
        "subreddit": ["test"] * 6,
        "created_utc": [1704067200, 1704067201, 1704067202,  # week 1
                        1704672000, 1704672001, 1704672002],  # week 2
        "created_utc_dt": pd.to_datetime([
            1704067200, 1704067201, 1704067202,
            1704672000, 1704672001, 1704672002
        ], unit="s"),
        "author_hash": ["a", "b", "c", "a", "b", "d"],
        "clean_text": ["text"] * 6,
        "score": [1] * 6,
        "num_comments": [1] * 6,
    })

    aggregator = WeeklyAggregator()
    result = aggregator.aggregate(df)

    # Week 2 should have d as new (a, b are returning)
    if len(result) > 1:
        assert result.iloc[1]["new_author_ratio"] < 1.0


def test_aggregate_fills_missing_created_utc_dt_from_created_utc():
    df = pd.DataFrame(
        {
            "subreddit": ["mix", "mix"],
            "created_utc": [1704067200, 1704672000],  # both valid unix seconds
            "created_utc_dt": [pd.Timestamp("2024-01-01T00:00:00Z"), pd.NaT],  # second row missing dt
            "author_hash": ["a", "b"],
            "clean_text": ["x", "y"],
            "score": [1, 1],
            "num_comments": [0, 0],
        }
    )
    out = WeeklyAggregator().aggregate(df)
    assert len(out) == 2
    assert int(out["post_count"].sum()) == 2
