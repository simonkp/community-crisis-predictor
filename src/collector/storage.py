from pathlib import Path

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: set[str], artifact_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{artifact_name} is missing required columns: {missing}")


def _validate_raw_schema(df: pd.DataFrame) -> None:
    required = {"post_id", "created_utc", "selftext", "subreddit"}
    _ensure_columns(df, required, "raw posts artifact")
    if df["post_id"].isna().any() or (df["post_id"].astype(str).str.strip() == "").any():
        raise ValueError("raw posts artifact contains null/empty `post_id` values")
    created = pd.to_numeric(df["created_utc"], errors="coerce")
    if created.isna().any():
        raise ValueError("raw posts artifact contains non-numeric `created_utc` values")
    if df["subreddit"].isna().any() or (df["subreddit"].astype(str).str.strip() == "").any():
        raise ValueError("raw posts artifact contains null/empty `subreddit` values")


def _validate_weekly_schema(df: pd.DataFrame) -> None:
    required = {
        "subreddit",
        "iso_year",
        "iso_week",
        "week_start",
        "texts",
        "post_count",
        "unique_authors",
        "new_author_ratio",
    }
    _ensure_columns(df, required, "weekly artifact")


def _validate_features_schema(df: pd.DataFrame) -> None:
    meta = {"subreddit", "iso_year", "iso_week", "week_start"}
    _ensure_columns(df, meta, "feature artifact")
    feature_cols = [c for c in df.columns if c not in meta]
    if not feature_cols:
        raise ValueError("feature artifact must contain at least one feature column")


def validate_source_compatibility(dfs: dict[str, pd.DataFrame]) -> None:
    """
    Fail-fast check before merging DataFrames from multiple collectors.

    Raises ValueError if any source is missing required columns or has an
    incompatible `created_utc` dtype, preventing silent corruption when a
    new source adds unexpected fields or represents timestamps differently.
    """
    required = {"post_id", "created_utc", "selftext", "subreddit"}
    for source_name, df in dfs.items():
        if df.empty:
            continue
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"Source '{source_name}' is missing columns required for merge: {missing}. "
                "Aborting to prevent silent data corruption."
            )
        numeric = pd.to_numeric(df["created_utc"], errors="coerce")
        bad_frac = float(numeric.isna().mean())
        if bad_frac > 0.1:
            raise ValueError(
                f"Source '{source_name}' has {bad_frac:.1%} non-numeric created_utc values "
                "(expected <10%). Aborting merge."
            )


def save_raw(df: pd.DataFrame, base_path: str, subreddit: str) -> Path:
    _validate_raw_schema(df)
    path = Path(base_path) / subreddit
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / "posts.parquet"
    df.to_parquet(file_path, index=False, engine="pyarrow")
    return file_path


def load_raw(base_path: str, subreddit: str) -> pd.DataFrame:
    file_path = Path(base_path) / subreddit / "posts.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"No data found at {file_path}")
    return pd.read_parquet(file_path, engine="pyarrow")


def load_all_raw(base_path: str, subreddits: list[str]) -> pd.DataFrame:
    dfs = []
    for sub in subreddits:
        try:
            df = load_raw(base_path, sub)
            dfs.append(df)
        except FileNotFoundError:
            continue
    if not dfs:
        raise FileNotFoundError(f"No data found in {base_path}")
    return pd.concat(dfs, ignore_index=True)


def save_processed(df: pd.DataFrame, base_path: str, name: str) -> Path:
    if name == "weekly":
        _validate_weekly_schema(df)
    elif name == "features":
        _validate_features_schema(df)
    path = Path(base_path)
    path.mkdir(parents=True, exist_ok=True)
    file_path = path / f"{name}.parquet"
    df.to_parquet(file_path, index=False, engine="pyarrow")
    return file_path


def load_processed(base_path: str, name: str) -> pd.DataFrame:
    file_path = Path(base_path) / f"{name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"No processed data at {file_path}")
    return pd.read_parquet(file_path, engine="pyarrow")
