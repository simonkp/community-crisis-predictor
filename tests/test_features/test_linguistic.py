import pandas as pd
from src.features.linguistic import extract_linguistic_features


def test_extract_linguistic_features(sample_weekly_df):
    result = extract_linguistic_features(sample_weekly_df)
    assert len(result) == len(sample_weekly_df)
    assert "avg_word_count" in result.columns
    assert "first_person_singular_ratio" in result.columns
    assert "avg_flesch_kincaid" in result.columns
    assert all(result["avg_word_count"] >= 0)


def test_pronoun_detection():
    df = pd.DataFrame({
        "texts": [["I feel bad. My life is hard. I need help me."]],
    })
    result = extract_linguistic_features(df)
    assert result["first_person_singular_ratio"].iloc[0] > 0
