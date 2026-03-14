import pandas as pd
from src.features.sentiment import extract_sentiment_features


def test_extract_sentiment_features(sample_weekly_df):
    result = extract_sentiment_features(sample_weekly_df)
    assert len(result) == len(sample_weekly_df)
    assert "avg_compound" in result.columns
    assert "pct_very_negative" in result.columns


def test_known_sentiment():
    df = pd.DataFrame({
        "texts": [["Everything is terrible and hopeless and awful"]],
    })
    result = extract_sentiment_features(df)
    assert result["avg_compound"].iloc[0] < 0


def test_positive_sentiment():
    df = pd.DataFrame({
        "texts": [["I am so happy and grateful and wonderful today"]],
    })
    result = extract_sentiment_features(df)
    assert result["avg_compound"].iloc[0] > 0
