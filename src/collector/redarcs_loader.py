"""
Load a pre-downloaded pushshift / RedArcs CSV export.

Usage:
    from src.collector.redarcs_loader import RedArcsLoader
    df = RedArcsLoader().load("path/to/depression.csv", subreddit="depression")
"""

import pandas as pd


class RedArcsLoader:
    """Load pre-downloaded subreddit CSV and normalise to the project schema."""

    # Common column name aliases used by different dump sources
    _ALIASES = {
        "body": "selftext",
        "content": "selftext",
        "text": "selftext",
        "created": "created_utc",
        "posted_at": "created_utc",
        "author_name": "author",
        "submission_id": "post_id",
        "id": "post_id",
        "num_comments": "num_comments",
        "ups": "score",
    }

    def load(self, csv_path: str, subreddit: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)

        # Rename aliases to canonical column names
        rename_map = {
            alias: target
            for alias, target in self._ALIASES.items()
            if alias in df.columns and target not in df.columns
        }
        if rename_map:
            df = df.rename(columns=rename_map)

        # Validate required columns
        for col in ("created_utc", "selftext"):
            if col not in df.columns:
                raise ValueError(
                    f"CSV is missing required column '{col}'. "
                    f"Available columns: {list(df.columns)}"
                )

        if subreddit and "subreddit" not in df.columns:
            df["subreddit"] = subreddit

        # Parse timestamps
        df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
        df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")

        # Drop rows with unparseable timestamps
        df = df.dropna(subset=["created_utc"])

        return df.reset_index(drop=True)
