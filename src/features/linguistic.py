import numpy as np
import pandas as pd
import textstat

from src.features.progress_util import iter_weeks

_FIRST_PERSON_SINGULAR = {"i", "me", "my", "myself", "mine"}
_FIRST_PERSON_PLURAL = {"we", "us", "our", "ourselves", "ours"}


def _compute_post_metrics(text: str) -> dict:
    words = text.split()
    n_words = len(words)
    if n_words == 0:
        return {
            "word_count": 0,
            "char_count": 0,
            "type_token_ratio": 0.0,
            "flesch_kincaid": 0.0,
            "first_person_singular_ratio": 0.0,
            "first_person_plural_ratio": 0.0,
        }

    unique_words = set(words)
    lower_words = [w.lower() for w in words]

    singular_count = sum(1 for w in lower_words if w in _FIRST_PERSON_SINGULAR)
    plural_count = sum(1 for w in lower_words if w in _FIRST_PERSON_PLURAL)

    try:
        fk = textstat.flesch_kincaid_grade(text)
    except Exception:
        fk = 0.0

    return {
        "word_count": n_words,
        "char_count": len(text),
        "type_token_ratio": len(unique_words) / n_words,
        "flesch_kincaid": fk,
        "first_person_singular_ratio": singular_count / n_words,
        "first_person_plural_ratio": plural_count / n_words,
    }


def extract_linguistic_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in iter_weeks(weekly_df, desc="  Linguistic"):
        texts = row["texts"]
        if not texts:
            rows.append({
                "avg_word_count": 0, "std_word_count": 0,
                "avg_char_count": 0, "avg_type_token_ratio": 0,
                "avg_flesch_kincaid": 0,
                "first_person_singular_ratio": 0,
                "first_person_plural_ratio": 0,
            })
            continue

        metrics = [_compute_post_metrics(t) for t in texts]
        word_counts = [m["word_count"] for m in metrics]

        rows.append({
            "avg_word_count": np.mean(word_counts),
            "std_word_count": np.std(word_counts),
            "avg_char_count": np.mean([m["char_count"] for m in metrics]),
            "avg_type_token_ratio": np.mean([m["type_token_ratio"] for m in metrics]),
            "avg_flesch_kincaid": np.mean([m["flesch_kincaid"] for m in metrics]),
            "first_person_singular_ratio": np.mean([m["first_person_singular_ratio"] for m in metrics]),
            "first_person_plural_ratio": np.mean([m["first_person_plural_ratio"] for m in metrics]),
        })

    return pd.DataFrame(rows, index=weekly_df.index)
