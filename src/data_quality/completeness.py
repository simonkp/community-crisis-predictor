import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _utc_dt_from_created_utc(series: pd.Series) -> pd.Series:
    """Build UTC datetimes from integer unix timestamps (seconds, ms, or ns)."""
    num = pd.to_numeric(series, errors="coerce")
    if not num.notna().any():
        return pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    mx = float(num.abs().max())
    if mx > 1e15:
        num = num / 1.0e9
    elif mx > 1e12:
        num = num / 1.0e3
    return pd.to_datetime(num, unit="s", errors="coerce", utc=True)


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
            fallback_dt = _utc_dt_from_created_utc(work["created_utc"])
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


def cross_source_validate(
    df: pd.DataFrame,
    subreddit: str,
) -> dict:
    """
    Check for cross-source discrepancies for a given subreddit.

    Identifies weeks where data from multiple sources overlap and flags
    weeks where the post-count ratio between sources exceeds 2× (indicating
    a likely double-count or gap-fill alignment problem).

    Returns a structured report dict with:
      - sources_present: list of source names seen
      - total_weeks: number of distinct weeks in the data
      - weeks_with_multiple_sources: weeks covered by >1 source
      - discrepancies: list of weeks where per-source counts diverge >2×
      - n_discrepancies: count of flagged weeks
    """
    if "data_source" not in df.columns or "created_utc" not in df.columns:
        return {"status": "skipped", "reason": "missing data_source or created_utc", "subreddit": subreddit}

    work = df[df["subreddit"] == subreddit].copy() if "subreddit" in df.columns else df.copy()
    if work.empty:
        return {"status": "empty", "subreddit": subreddit, "n_discrepancies": 0}

    work["_dt"] = _utc_dt_from_created_utc(work["created_utc"])
    work = work[work["_dt"].notna()].copy()
    work["_week"] = work["_dt"].dt.tz_localize(None).dt.to_period("W-SUN").dt.start_time

    week_source = (
        work.groupby(["_week", "data_source"])
        .size()
        .reset_index(name="post_count")
    )

    multi = week_source.groupby("_week").filter(lambda g: len(g) > 1)

    discrepancies: list[dict] = []
    for week, group in multi.groupby("_week"):
        counts = group.set_index("data_source")["post_count"]
        max_c, min_c = int(counts.max()), int(counts.min())
        ratio = max_c / min_c if min_c > 0 else float("inf")
        if ratio > 2.0:
            discrepancies.append({
                "week": str(week.date()),
                "sources": {k: int(v) for k, v in counts.items()},
                "ratio": round(float(ratio), 2),
            })

    return {
        "status": "ok",
        "subreddit": subreddit,
        "sources_present": sorted(work["data_source"].unique().tolist()),
        "total_weeks": int(week_source["_week"].nunique()),
        "weeks_with_multiple_sources": int(multi["_week"].nunique()),
        "discrepancies": discrepancies,
        "n_discrepancies": len(discrepancies),
    }


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
                source TEXT NOT NULL,
                UNIQUE(subreddit, week, source)
            )
            """
        )
        # Migration-safe dedupe + unique index for idempotent upserts.
        conn.execute(
            """
            DELETE FROM data_provenance
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM data_provenance
                GROUP BY subreddit, week, source
            )
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_data_provenance_sub_week_source
            ON data_provenance (subreddit, week, source)
            """
        )
        conn.execute(
            """
            INSERT INTO data_provenance (timestamp, subreddit, week, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(subreddit, week, source)
            DO UPDATE SET timestamp = excluded.timestamp
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                subreddit,
                week,
                source,
            ),
        )
        conn.commit()
