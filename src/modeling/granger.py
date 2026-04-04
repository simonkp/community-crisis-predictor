"""
Granger causality analysis between subreddit distress signals.

Tests whether the distress time-series of subreddit A provides statistically
significant predictive information about subreddit B beyond B's own history,
using the standard F-test formulation (via OLS regression of lagged values).

Requires: statsmodels (already in the dependency tree via scikit-learn / scipy).
Falls back gracefully if statsmodels is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ols_granger_test(x: np.ndarray, y: np.ndarray, max_lag: int) -> list[dict]:
    """
    OLS-based Granger causality test: does x Granger-cause y?

    For each lag k in 1..max_lag, fits:
      restricted:   y_t = c + sum_i a_i * y_{t-i}  (i=1..k)
      unrestricted: y_t = c + sum_i a_i * y_{t-i} + sum_j b_j * x_{t-j}  (j=1..k)
    and reports the F-statistic and p-value.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return [{"lag": k, "f_stat": None, "p_value": None, "error": "statsmodels_not_installed"} for k in range(1, max_lag + 1)]

    data = np.column_stack([y, x])  # statsmodels convention: [effect, cause]
    valid = ~(np.isnan(data).any(axis=1))
    data = data[valid]
    if len(data) < 2 * max_lag + 5:
        return [{"lag": k, "f_stat": None, "p_value": None, "error": "too_few_observations"} for k in range(1, max_lag + 1)]

    results = []
    try:
        gc_out = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        for k in range(1, max_lag + 1):
            test = gc_out[k][0].get("ssr_ftest", (None, None, None, None))
            results.append({
                "lag": k,
                "f_stat": round(float(test[0]), 4) if test[0] is not None else None,
                "p_value": round(float(test[1]), 4) if test[1] is not None else None,
            })
    except Exception as e:
        results = [{"lag": k, "f_stat": None, "p_value": None, "error": str(e)} for k in range(1, max_lag + 1)]
    return results


def compute_granger_causality(
    feature_df: pd.DataFrame,
    subreddits: list[str],
    distress_col: str = "distress_score",
    max_lag: int = 4,
) -> dict:
    """
    For each ordered pair (A, B) of subreddits, test whether A's weekly
    distress signal Granger-causes B's weekly distress signal.

    Parameters
    ----------
    feature_df : DataFrame with a `subreddit` column and the distress signal column.
    subreddits  : List of subreddit names to test.
    distress_col: Column name of the scalar distress signal (added by caller).
    max_lag     : Maximum number of weekly lags to test.

    Returns
    -------
    Dict of {source_sub: {target_sub: [{lag, f_stat, p_value}, ...]}}.
    """
    # Build per-subreddit distress time-series aligned to ISO week index
    series: dict[str, np.ndarray] = {}
    for sub in subreddits:
        sub_df = feature_df[feature_df["subreddit"] == sub].copy()
        if sub_df.empty or distress_col not in sub_df.columns:
            continue
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        series[sub] = sub_df[distress_col].to_numpy(dtype=float)

    results: dict = {}
    for src in series:
        results[src] = {}
        for tgt in series:
            if tgt == src:
                continue
            x = series[src]
            y = series[tgt]
            # Align to the shorter series (conservative)
            n = min(len(x), len(y))
            if n < 2 * max_lag + 5:
                results[src][tgt] = [{"lag": k, "skipped": True, "reason": "series_too_short"} for k in range(1, max_lag + 1)]
                continue
            results[src][tgt] = _ols_granger_test(x[-n:], y[-n:], max_lag)

    return results


def save_granger_report(results: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
