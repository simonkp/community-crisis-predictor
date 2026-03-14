from pathlib import Path

import pandas as pd


def save_raw(df: pd.DataFrame, base_path: str, subreddit: str) -> Path:
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
