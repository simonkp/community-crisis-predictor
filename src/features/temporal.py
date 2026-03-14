import numpy as np
import pandas as pd


def add_temporal_features(
    feature_df: pd.DataFrame,
    rolling_windows: list[int] = None,
) -> pd.DataFrame:
    if rolling_windows is None:
        rolling_windows = [2, 4]

    df = feature_df.copy()

    # Identify numeric columns to compute deltas and rolling averages
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude columns that are identifiers or already temporal
    exclude = {"iso_year", "iso_week", "dominant_topic"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    # Week-over-week deltas
    for col in numeric_cols:
        df[f"{col}_delta"] = df[col].diff()

    # Rolling averages
    for window in rolling_windows:
        for col in numeric_cols:
            df[f"{col}_roll{window}w"] = df[col].rolling(window=window, min_periods=1).mean()

    # Cyclical week-of-year encoding
    if "iso_week" in df.columns:
        df["week_sin"] = np.sin(2 * np.pi * df["iso_week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["iso_week"] / 52)

    return df
