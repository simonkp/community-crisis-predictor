import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.features.progress_util import iter_weeks


_analyzer = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def extract_sentiment_features(
    weekly_df: pd.DataFrame,
    bins: dict = None,
) -> pd.DataFrame:
    if bins is None:
        bins = {"very_negative": -0.5, "negative": -0.05, "positive": 0.05}

    analyzer = _get_analyzer()
    rows = []

    for idx, row in iter_weeks(weekly_df, desc="  Sentiment"):
        texts = row["texts"]
        if not texts:
            rows.append({
                "avg_compound": 0, "avg_positive": 0,
                "avg_negative": 0, "avg_neutral": 0,
                "pct_very_negative": 0, "pct_negative": 0,
                "pct_neutral": 0, "pct_positive": 0,
            })
            continue

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
        compounds_arr = np.array(compounds)

        pct_very_neg = np.sum(compounds_arr < bins["very_negative"]) / n
        pct_neg = np.sum(
            (compounds_arr >= bins["very_negative"]) & (compounds_arr < bins["negative"])
        ) / n
        pct_neutral = np.sum(
            (compounds_arr >= bins["negative"]) & (compounds_arr <= bins["positive"])
        ) / n
        pct_pos = np.sum(compounds_arr > bins["positive"]) / n

        rows.append({
            "avg_compound": np.mean(compounds),
            "avg_positive": np.mean(positives),
            "avg_negative": np.mean(negatives),
            "avg_neutral": np.mean(neutrals),
            "pct_very_negative": pct_very_neg,
            "pct_negative": pct_neg,
            "pct_neutral": pct_neutral,
            "pct_positive": pct_pos,
        })

    return pd.DataFrame(rows, index=weekly_df.index)
