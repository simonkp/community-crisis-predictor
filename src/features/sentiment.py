import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from tqdm import tqdm

from src.features.progress_util import iter_weeks

_DEFAULT_BINS = {"very_negative": -0.5, "negative": -0.05, "positive": 0.05}

_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def _resolve_parallel_workers(requested: int) -> int:
    """0 = auto (cap 8), 1 = serial, >1 = cap to requested."""
    if requested < 0:
        return 1
    if requested == 0:
        return max(1, min(8, os.cpu_count() or 4))
    return max(1, requested)


def _empty_sentiment_row() -> dict:
    return {
        "avg_compound": 0.0,
        "avg_positive": 0.0,
        "avg_negative": 0.0,
        "avg_neutral": 0.0,
        "pct_very_negative": 0.0,
        "pct_negative": 0.0,
        "pct_neutral": 0.0,
        "pct_positive": 0.0,
    }


def _sentiment_for_week(texts: list[str], bins: dict, analyzer: SentimentIntensityAnalyzer) -> dict:
    if not texts:
        return _empty_sentiment_row()

    compounds = []
    positives = []
    negatives = []
    neutrals = []

    for text in texts:
        scores = analyzer.polarity_scores(text)
        compounds.append(scores["compound"])
        positives.append(scores["pos"])
        negatives.append(scores["neg"])
        neutrals.append(scores["neu"])

    n = len(compounds)
    compounds_arr = np.array(compounds, dtype=float)

    pct_very_neg = np.sum(compounds_arr < bins["very_negative"]) / n
    pct_neg = np.sum(
        (compounds_arr >= bins["very_negative"]) & (compounds_arr < bins["negative"])
    ) / n
    pct_neutral = np.sum(
        (compounds_arr >= bins["negative"]) & (compounds_arr <= bins["positive"])
    ) / n
    pct_pos = np.sum(compounds_arr > bins["positive"]) / n

    return {
        "avg_compound": float(np.mean(compounds)),
        "avg_positive": float(np.mean(positives)),
        "avg_negative": float(np.mean(negatives)),
        "avg_neutral": float(np.mean(neutrals)),
        "pct_very_negative": float(pct_very_neg),
        "pct_negative": float(pct_neg),
        "pct_neutral": float(pct_neutral),
        "pct_positive": float(pct_pos),
    }


# Worker-process local analyzer (one per fork/spawn child, lazy init)
_proc_analyzer: SentimentIntensityAnalyzer | None = None


def _get_worker_analyzer() -> SentimentIntensityAnalyzer:
    global _proc_analyzer
    if _proc_analyzer is None:
        _proc_analyzer = SentimentIntensityAnalyzer()
    return _proc_analyzer


def _sentiment_worker_payload(payload: tuple[list[str], dict]) -> dict:
    """Top-level for pickling; runs in worker process."""
    texts, bins = payload
    return _sentiment_for_week(texts, bins, _get_worker_analyzer())


def extract_sentiment_features(
    weekly_df: pd.DataFrame,
    bins: dict | None = None,
    *,
    parallel_workers: int = 1,
) -> pd.DataFrame:
    if bins is None:
        bins = dict(_DEFAULT_BINS)

    n_workers = _resolve_parallel_workers(parallel_workers)
    n_weeks = len(weekly_df)
    mode = "parallel" if n_workers > 1 and n_weeks > 1 else "serial"
    print(f"  Sentiment mode: {mode} (workers={min(n_workers, n_weeks) if n_weeks else n_workers})")

    if n_workers <= 1 or n_weeks <= 1:
        analyzer = _get_analyzer()
        rows = []
        for _, row in iter_weeks(weekly_df, desc="  Sentiment"):
            texts = row["texts"]
            if not isinstance(texts, list):
                texts = []
            rows.append(_sentiment_for_week(texts, bins, analyzer))
        return pd.DataFrame(rows, index=weekly_df.index)

    tasks: list[tuple[list[str], dict]] = []
    for _, row in weekly_df.iterrows():
        texts = row["texts"]
        if not isinstance(texts, list):
            texts = []
        tasks.append((texts, bins))

    max_proc = min(n_workers, n_weeks)
    chunksize = max(1, n_weeks // (max_proc * 4))
    with ProcessPoolExecutor(max_workers=max_proc) as ex:
        mapped = ex.map(_sentiment_worker_payload, tasks, chunksize=chunksize)
        rows = list(
            tqdm(
                mapped,
                total=n_weeks,
                desc="  Sentiment",
                unit="wk",
                leave=True,
                disable=not sys.stdout.isatty(),
                file=sys.stdout,
                mininterval=0.5,
            )
        )

    return pd.DataFrame(rows, index=weekly_df.index)
