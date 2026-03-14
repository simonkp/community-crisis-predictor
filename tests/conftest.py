import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_raw_posts():
    rng = np.random.RandomState(42)
    posts = []
    start = datetime(2024, 1, 1)
    authors = [f"user_{i}" for i in range(20)]

    for i in range(100):
        post_time = start + timedelta(days=rng.randint(0, 90), hours=rng.randint(0, 24))
        posts.append({
            "post_id": str(uuid.uuid4())[:12],
            "created_utc": int(post_time.timestamp()),
            "title": f"Test post title {i}",
            "selftext": f"This is post body number {i}. I feel really anxious and hopeless today. "
                        f"Does anyone else struggle with this? https://example.com/link",
            "score": rng.randint(1, 50),
            "num_comments": rng.randint(0, 20),
            "subreddit": "depression" if i % 2 == 0 else "anxiety",
            "author": authors[rng.randint(0, len(authors))],
            "is_self": True,
        })

    df = pd.DataFrame(posts)
    df["created_utc_dt"] = pd.to_datetime(df["created_utc"], unit="s")
    return df


@pytest.fixture
def sample_weekly_df():
    weeks = []
    start = datetime(2024, 1, 1)

    for i in range(20):
        week_start = start + timedelta(weeks=i)
        texts = [
            f"I feel {'hopeless and anxious' if i % 4 == 0 else 'okay today'}. "
            f"Does anyone else feel this way? I need help."
            for _ in range(10)
        ]

        weeks.append({
            "subreddit": "depression",
            "iso_year": 2024,
            "iso_week": i + 1,
            "week_start": week_start,
            "texts": texts,
            "post_count": 10,
            "avg_score": 8.5,
            "total_comments": 45,
            "unique_authors": 8,
            "new_author_ratio": 0.3,
            "post_hours": [h % 24 for h in range(10)],
        })

    return pd.DataFrame(weeks)


@pytest.fixture
def sample_feature_matrix():
    rng = np.random.RandomState(42)
    n = 52

    df = pd.DataFrame({
        "subreddit": ["depression"] * n,
        "iso_year": [2024] * n,
        "iso_week": list(range(1, n + 1)),
        "week_start": pd.date_range("2024-01-01", periods=n, freq="W"),
        "avg_word_count": rng.normal(80, 20, n),
        "avg_compound": rng.normal(-0.2, 0.3, n),
        "avg_negative": rng.uniform(0.05, 0.3, n),
        "hopelessness_density": rng.uniform(0.0, 0.05, n),
        "help_seeking_density": rng.uniform(0.0, 0.04, n),
        "distress_density": rng.uniform(0.01, 0.08, n),
        "post_volume": rng.randint(50, 200, n),
        "posting_time_entropy": rng.uniform(2.5, 3.2, n),
        "topic_entropy": rng.uniform(1.0, 3.0, n),
    })

    return df


@pytest.fixture
def config():
    return {
        "reddit": {
            "subreddits": ["depression", "anxiety"],
            "date_range": {"start": "2024-01-01", "end": "2025-01-01"},
        },
        "collection": {"privacy_salt": "test_salt_123"},
        "processing": {"min_post_length_chars": 20, "excluded_selftext": ["[deleted]", "[removed]"]},
        "features": {
            "sentiment": {"bins": {"very_negative": -0.5, "negative": -0.05, "positive": 0.05}},
            "topics": {"model_name": "all-MiniLM-L6-v2", "n_topics": 5, "min_topic_size": 5, "max_posts_per_week": 50},
            "temporal": {"rolling_windows": [2, 4]},
        },
        "labeling": {
            "distress_weights": {"neg_sentiment": 0.4, "hopelessness": 0.35, "help_seeking": 0.25},
            "crisis_threshold_std": 1.5,
        },
        "modeling": {
            "xgboost": {
                "scale_pos_weight": "auto",
                "param_grid": {"max_depth": [3, 5], "learning_rate": [0.1], "n_estimators": [50]},
                "n_search_iter": 5,
            },
            "walk_forward": {"min_train_weeks": 10, "gap_weeks": 1},
        },
        "evaluation": {"primary_metric": "recall", "probability_threshold": 0.5},
        "synthetic": {"n_weeks": 30, "posts_per_week_range": [20, 40], "crisis_frequency": 0.15},
        "random_seed": 42,
        "paths": {
            "raw_data": "/tmp/crisis_test/raw",
            "processed_data": "/tmp/crisis_test/processed",
            "features": "/tmp/crisis_test/features",
            "models": "/tmp/crisis_test/models",
            "reports": "/tmp/crisis_test/reports",
        },
    }
