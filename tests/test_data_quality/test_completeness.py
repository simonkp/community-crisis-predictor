from pathlib import Path

import pandas as pd

from src.data_quality.completeness import (
    check_weekly_completeness,
    flag_missing_weeks,
    log_source_provenance,
)


def test_check_weekly_completeness_flags_low_weeks():
    rows = []
    for week_start, count in [("2024-01-01", 100), ("2024-01-08", 110), ("2024-01-15", 20)]:
        for i in range(count):
            rows.append(
                {
                    "subreddit": "depression",
                    "week_start": week_start,
                    "post_id": f"{week_start}-{i}",
                }
            )
    df = pd.DataFrame(rows)
    out = check_weekly_completeness(df, "depression")
    assert len(out) == 3
    assert bool(out.iloc[-1]["is_gap"]) is True


def test_flag_missing_weeks_detects_gap():
    weekly_df = pd.DataFrame(
        {
            "subreddit": ["depression", "depression"],
            "week_start": ["2024-01-01", "2024-01-15"],
        }
    )
    missing = flag_missing_weeks(
        weekly_df,
        subreddit="depression",
        start_date="2024-01-01",
        end_date="2024-01-22",
    )
    assert len(missing) >= 1


def test_log_source_provenance_writes_row(tmp_path):
    db_path = tmp_path / "quality.db"
    log_source_provenance("depression", "2024-01-01/2024-01-07", "pushshift", str(db_path))
    assert Path(db_path).exists()


def test_check_weekly_completeness_falls_back_to_created_utc():
    dt = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-08"], utc=True)
    df = pd.DataFrame(
        {
            "subreddit": ["anxiety", "anxiety", "anxiety"],
            "created_utc": (dt.astype("int64") // 10**9).astype(int),
            # Simulate mixed-source: missing created_utc_dt for rows that should still count.
            "created_utc_dt": [pd.NaT, pd.NaT, pd.NaT],
        }
    )
    out = check_weekly_completeness(df, "anxiety")
    assert len(out) == 2
