import pandas as pd


class WeeklyAggregator:
    def __init__(self):
        self._seen_authors: dict[str, set] = {}

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

        df["iso_year"] = df["created_utc_dt"].dt.isocalendar().year.astype(int)
        df["iso_week"] = df["created_utc_dt"].dt.isocalendar().week.astype(int)

        rows = []
        for (subreddit, year, week), group in df.groupby(
            ["subreddit", "iso_year", "iso_week"]
        ):
            sub_key = str(subreddit)
            if sub_key not in self._seen_authors:
                self._seen_authors[sub_key] = set()

            authors = set(group["author_hash"].unique()) if "author_hash" in group.columns else set()
            new_authors = authors - self._seen_authors[sub_key]
            self._seen_authors[sub_key].update(authors)

            unique_count = len(authors)
            new_ratio = len(new_authors) / unique_count if unique_count > 0 else 0.0

            week_start = group["created_utc_dt"].min()

            # Collect hour-of-day for each post
            hours = group["created_utc_dt"].dt.hour.tolist()

            rows.append({
                "subreddit": subreddit,
                "iso_year": year,
                "iso_week": week,
                "week_start": week_start,
                "texts": group["clean_text"].tolist() if "clean_text" in group.columns else [],
                "post_count": len(group),
                "avg_score": group["score"].mean() if "score" in group.columns else 0,
                "total_comments": group["num_comments"].sum() if "num_comments" in group.columns else 0,
                "unique_authors": unique_count,
                "new_author_ratio": new_ratio,
                "post_hours": hours,
            })

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values(["subreddit", "iso_year", "iso_week"]).reset_index(drop=True)
        return result
