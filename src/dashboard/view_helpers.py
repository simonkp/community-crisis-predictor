"""Pure helpers shared by analyst (app.py) and end-user dashboard pages."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.dashboard.state import pick_model_results


def format_week_label(value) -> str:
    try:
        dt = pd.to_datetime(value)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(value)[:10]


def to_naive_ts(value):
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert(None)


def resolve_week_index_for_sub(sub_df: pd.DataFrame, replay_week_ts) -> int:
    """Map a global replay calendar week to this subreddit's nearest available week."""
    if sub_df.empty:
        return 0
    n = len(sub_df)
    if {"iso_year", "iso_week"}.issubset(sub_df.columns) and pd.notna(replay_week_ts):
        replay_iso = pd.Timestamp(replay_week_ts).isocalendar()
        sub_iso_year = pd.to_numeric(sub_df["iso_year"], errors="coerce").fillna(-1).astype(int).to_numpy()
        sub_iso_week = pd.to_numeric(sub_df["iso_week"], errors="coerce").fillna(-1).astype(int).to_numpy()
        exact = np.where((sub_iso_year == int(replay_iso.year)) & (sub_iso_week == int(replay_iso.week)))[0]
        if len(exact):
            return int(exact[0])
        sub_keys = sub_iso_year * 100 + sub_iso_week
        replay_key = int(replay_iso.year) * 100 + int(replay_iso.week)
        not_after = np.where(sub_keys <= replay_key)[0]
        if len(not_after):
            return int(not_after[-1])
        return 0
    if "week_start" not in sub_df.columns or pd.isna(replay_week_ts):
        return n - 1
    sub_weeks = pd.to_datetime(sub_df["week_start"], errors="coerce", utc=True).dt.tz_convert(None)
    exact = np.where(sub_weeks == replay_week_ts)[0]
    if len(exact):
        return int(exact[0])
    not_after = np.where(sub_weeks <= replay_week_ts)[0]
    if len(not_after):
        return int(not_after[-1])
    return 0


def available_models_for_sub(sub_results: dict) -> list[str]:
    out: list[str] = []
    if sub_results.get("lstm"):
        out.append("LSTM")
    if sub_results.get("xgb"):
        out.append("XGBoost")
    if sub_results.get("lstm") and sub_results.get("xgb"):
        out.append("Ensemble")
    return out or ["LSTM"]


def resolve_model_results(sub_results: dict, preferred: str):
    avail = available_models_for_sub(sub_results)
    choice = preferred if preferred in avail else avail[0]
    return pick_model_results(sub_results, choice), choice


def build_global_replay_weeks(feature_df: pd.DataFrame) -> np.ndarray:
    """ISO-calendar or week_start unique timeline aligned with analyst dashboard."""
    if {"iso_year", "iso_week"}.issubset(feature_df.columns):
        _global_iso = (
            feature_df[["iso_year", "iso_week"]]
            .dropna()
            .astype({"iso_year": int, "iso_week": int})
            .drop_duplicates()
            .sort_values(["iso_year", "iso_week"])
        )
        return np.array(
            [
                pd.Timestamp.fromisocalendar(int(r.iso_year), int(r.iso_week), 1)
                for r in _global_iso.itertuples(index=False)
            ]
        )
    _global_weeks_series = pd.to_datetime(
        feature_df["week_start"],
        errors="coerce",
        utc=True,
    ).dropna().dt.tz_convert(None)
    return np.array(sorted(_global_weeks_series.unique()))
