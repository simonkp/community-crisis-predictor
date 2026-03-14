import tempfile
from pathlib import Path

import pandas as pd
from src.collector.storage import save_raw, load_raw, save_processed, load_processed


def test_save_and_load_raw():
    df = pd.DataFrame({
        "post_id": ["a", "b"],
        "title": ["hello", "world"],
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        save_raw(df, tmpdir, "depression")
        loaded = load_raw(tmpdir, "depression")
        assert len(loaded) == 2
        assert list(loaded.columns) == list(df.columns)


def test_save_and_load_processed():
    df = pd.DataFrame({"feature1": [1.0, 2.0], "feature2": [3.0, 4.0]})
    with tempfile.TemporaryDirectory() as tmpdir:
        save_processed(df, tmpdir, "weekly")
        loaded = load_processed(tmpdir, "weekly")
        assert len(loaded) == 2
