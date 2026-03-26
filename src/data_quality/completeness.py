import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def check_weekly_completeness(
    df: pd.DataFrame,
    subreddit: str,
    rolling_window: int = 4,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "subreddit",
                "week_start",
                "post_count",
                "rolling_avg_post_count",
                "completeness_score",
                "is_gap",
            ]
        )

    work = df.copy()
    if "week_start" not in work.columns:
        if "created_utc_dt" not in work.columns and "created_utc" not in work.columns:
            raise ValueError("Data must include week_start or created_utc_dt for completeness checks.")

        if "created_utc_dt" in work.columns:
            work["created_utc_dt"] = pd.to_datetime(work["created_utc_dt"], errors="coerce", utc=True)
        else:
            work["created_utc_dt"] = pd.NaT
        if "created_utc" in work.columns:
            fallback_dt = pd.to_datetime(work["created_utc"], unit="s", errors="coerce", utc=True)
            work["created_utc_dt"] = work["created_utc_dt"].fillna(fallback_dt)

        # Convert to timezone-naive before Period conversion to avoid pandas timezone warning.
        work["created_utc_dt"] = work["created_utc_dt"].dt.tz_localize(None)
        # Use Monday-start weeks to align with expected_freq default W-MON.
        work = work[work["created_utc_dt"].notna()].copy()
        work["week_start"] = work["created_utc_dt"].dt.to_period("W-SUN").dt.start_time
    else:
        work["week_start"] = pd.to_datetime(work["week_start"])

    weekly_counts = (
        work.groupby("week_start", as_index=False)
        .size()
        .rename(columns={"size": "post_count"})
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    rolling_avg = (
        weekly_counts["post_count"]
        .rolling(window=rolling_window, min_periods=1)
        .mean()
        .replace(0, np.nan)
    )
    weekly_counts["rolling_avg_post_count"] = rolling_avg
    weekly_counts["completeness_score"] = (
        weekly_counts["post_count"] / weekly_counts["rolling_avg_post_count"]
    ).fillna(0.0)
    weekly_counts["is_gap"] = weekly_counts["completeness_score"] < 0.5
    weekly_counts["subreddit"] = subreddit
    return weekly_counts


def flag_missing_weeks(
    weekly_df: pd.DataFrame,
    subreddit: str,
    start_date: str,
    end_date: str,
    expected_freq: str = "W-MON",
) -> list[str]:
    if weekly_df is None or weekly_df.empty:
        present_weeks = set()
    else:
        work = weekly_df.copy()
        if "subreddit" in work.columns:
            work = work[work["subreddit"] == subreddit]
        present_weeks = set(pd.to_datetime(work["week_start"]).dt.strftime("%Y-%m-%d"))

    expected = pd.date_range(start=start_date, end=end_date, freq=expected_freq).strftime("%Y-%m-%d")
    expected_weeks = list(expected)
    return [w for w in expected_weeks if w not in present_weeks]


def log_source_provenance(
    subreddit: str,
    week: str,
    source: str,
    db_path: str = "data/quality.db",
) -> None:
    db = Path(db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                subreddit TEXT NOT NULL,
                week TEXT NOT NULL,
                source TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO data_provenance (timestamp, subreddit, week, source)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                subreddit,
                week,
                source,
            ),
        )
        conn.commit()
