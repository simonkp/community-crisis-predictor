import pandas as pd


class WeeklyAggregator:
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "subreddit" in df.columns:
            df["subreddit"] = df["subreddit"].astype(str).str.strip().str.lower()

        # Normalize datetime.
        # If mixed sources produced partial `created_utc_dt` (e.g., Zenodo has it, Arctic Shift doesn't),
        # fill missing values from `created_utc` so rows from all sources survive aggregation.
        if "created_utc_dt" in df.columns:
            df["created_utc_dt"] = pd.to_datetime(df["created_utc_dt"], errors="coerce", utc=True)
            fallback_dt = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
            df["created_utc_dt"] = df["created_utc_dt"].fillna(fallback_dt)
        else:
            df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce", utc=True)
        df = df[df["created_utc_dt"].notna()].copy()
        if df.empty:
            return pd.DataFrame()

        rows = []
        # Convert to timezone-naive before Period conversion to avoid pandas warning.
        df["created_utc_dt"] = df["created_utc_dt"].dt.tz_localize(None)
        df["week_start"] = df["created_utc_dt"].dt.to_period("W-SUN").dt.start_time
        for subreddit, sub_df in df.groupby("subreddit", sort=True):
            sub_df = sub_df.sort_values("created_utc_dt").reset_index(drop=True)
            weekly_map: dict[pd.Timestamp, pd.DataFrame] = {
                wk: g.reset_index(drop=True) for wk, g in sub_df.groupby("week_start", sort=True)
            }
            if not weekly_map:
                continue
            full_weeks = pd.date_range(
                start=min(weekly_map.keys()),
                end=max(weekly_map.keys()),
                freq="W-MON",
            )
            seen_authors: set[str] = set()
            for week_start in full_weeks:
                group = weekly_map.get(week_start)
                iso = week_start.isocalendar()
                if group is None or group.empty:
                    rows.append(
                        {
                            "subreddit": subreddit,
                            "iso_year": int(iso.year),
                            "iso_week": int(iso.week),
                            "week_start": week_start,
                            "texts": [],
                            "post_count": 0,
                            "avg_score": 0.0,
                            "total_comments": 0.0,
                            "unique_authors": 0,
                            "new_author_ratio": 0.0,
                            "post_hours": [],
                            "is_missing_week": True,
                        }
                    )
                    continue

                authors = set(group["author_hash"].unique()) if "author_hash" in group.columns else set()
                new_authors = authors - seen_authors
                seen_authors.update(authors)
                unique_count = len(authors)
                new_ratio = len(new_authors) / unique_count if unique_count > 0 else 0.0
                hours = group["created_utc_dt"].dt.hour.tolist()

                rows.append(
                    {
                        "subreddit": subreddit,
                        "iso_year": int(iso.year),
                        "iso_week": int(iso.week),
                        "week_start": week_start,
                        "texts": group["clean_text"].tolist() if "clean_text" in group.columns else [],
                        "post_count": int(len(group)),
                        "avg_score": float(group["score"].mean()) if "score" in group.columns else 0.0,
                        "total_comments": float(group["num_comments"].sum()) if "num_comments" in group.columns else 0.0,
                        "unique_authors": unique_count,
                        "new_author_ratio": new_ratio,
                        "post_hours": hours,
                        "is_missing_week": False,
                    }
                )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["subreddit", "iso_year", "iso_week"]).reset_index(drop=True)
        return result
