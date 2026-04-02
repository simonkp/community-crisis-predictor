import tempfile
from pathlib import Path

import pandas as pd
import pytest
from src.collector.storage import save_raw, load_raw, save_processed, load_processed


def test_save_and_load_raw():
    df = pd.DataFrame({
        "post_id": ["a", "b"],
        "created_utc": [1704067200, 1704067201],
        "selftext": ["hello", "world"],
        "subreddit": ["depression", "depression"],
        "author_hash": ["u1", "u2"],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        save_raw(df, tmpdir, "depression")
        loaded = load_raw(tmpdir, "depression")
        assert len(loaded) == 2
        assert list(loaded.columns) == list(df.columns)


def test_save_and_load_processed():
    df = pd.DataFrame(
        {
            "subreddit": ["depression", "depression"],
            "iso_year": [2024, 2024],
            "iso_week": [1, 2],
            "week_start": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "texts": [["a"], ["b"]],
            "post_count": [1, 1],
            "unique_authors": [1, 1],
            "new_author_ratio": [1.0, 0.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        save_processed(df, tmpdir, "weekly")
        loaded = load_processed(tmpdir, "weekly")
        assert len(loaded) == 2


def test_save_raw_raises_on_missing_required_columns():
    df = pd.DataFrame({"post_id": ["a"]})
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            save_raw(df, tmpdir, "depression")


def test_save_processed_features_raises_without_feature_columns():
    df = pd.DataFrame(
        {
            "subreddit": ["depression"],
            "iso_year": [2024],
            "iso_week": [1],
            "week_start": pd.to_datetime(["2024-01-01"]),
        }
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError):
            save_processed(df, tmpdir, "features")
