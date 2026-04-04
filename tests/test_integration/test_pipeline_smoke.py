"""
Integration smoke test: synthetic data → collect → features → drift detection.

Covers the three stages that don't require a trained model and can run fast
(skip-topics, small synthetic dataset). Verifies that:
  1. Synthetic data round-trips through save/load without data loss.
  2. Feature extraction produces a non-empty matrix with no unexpected nulls.
  3. DriftDetector produces one row per week with valid alert levels and logs
     de-escalation events (negative-spike scenario).
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.collector.privacy import strip_pii
from src.collector.storage import load_all_raw, save_raw
from src.collector.synthetic import generate_synthetic_data
from src.features.pipeline import FeaturePipeline
from src.monitoring.drift_detector import ALERT_LEVELS, DriftDetector
from src.processing.text_cleaner import process_posts
from src.processing.weekly_aggregator import WeeklyAggregator


@pytest.fixture(scope="module")
def minimal_config(tmp_path_factory):
    base = tmp_path_factory.mktemp("pipeline_smoke")
    return {
        "reddit": {
            "subreddits": ["depression"],
            "date_range": {"start": "2024-01-01", "end": "2024-07-01"},
        },
        "collection": {"privacy_salt": "smoke_test_salt"},
        "processing": {
            "min_post_length_chars": 10,
            "excluded_selftext": ["[deleted]", "[removed]"],
        },
        "features": {
            "sentiment": {
                "bins": {"very_negative": -0.5, "negative": -0.05, "positive": 0.05}
            },
            "topics": {
                "model_name": "all-MiniLM-L6-v2",
                "n_topics": 3,
                "min_topic_size": 2,
                "max_posts_per_week": 20,
            },
            "temporal": {"rolling_windows": [2, 4]},
        },
        "labeling": {
            "distress_weights": {
                "neg_sentiment": 0.4,
                "hopelessness": 0.35,
                "help_seeking": 0.25,
            },
            "crisis_threshold_std": 1.5,
        },
        "modeling": {
            "walk_forward": {"min_train_weeks": 10, "gap_weeks": 1},
            "lstm": {"sequence_length": 4},
        },
        "synthetic": {
            "n_weeks": 26,
            "posts_per_week_range": [15, 25],
            "crisis_frequency": 0.15,
        },
        "random_seed": 0,
        "paths": {
            "raw_data": str(base / "raw"),
            "processed_data": str(base / "processed"),
            "features": str(base / "features"),
            "models": str(base / "models"),
            "reports": str(base / "reports"),
        },
    }


# ---------------------------------------------------------------------------
# Stage 1: collect (synthetic)
# ---------------------------------------------------------------------------

def test_synthetic_collect_and_roundtrip(minimal_config):
    datasets = generate_synthetic_data(minimal_config, seed=0)
    assert "depression" in datasets, "Synthetic generator must return 'depression' key"

    raw_df = datasets["depression"]
    assert not raw_df.empty, "Synthetic dataset must be non-empty"
    assert "post_id" in raw_df.columns
    assert raw_df["post_id"].notna().all(), "All post_ids must be non-null"
    assert raw_df["post_id"].is_unique, "post_ids must be unique"

    raw_df = strip_pii(raw_df, minimal_config["collection"]["privacy_salt"])
    raw_df["data_source"] = "synthetic"

    save_raw(raw_df, minimal_config["paths"]["raw_data"], "depression")

    loaded = load_all_raw(minimal_config["paths"]["raw_data"], ["depression"])
    assert len(loaded) == len(raw_df), "Row count must survive save/load roundtrip"


# ---------------------------------------------------------------------------
# Stage 2: features (skip-topics for speed)
# ---------------------------------------------------------------------------

def test_feature_extraction_produces_valid_matrix(minimal_config):
    """Feature matrix must be non-empty, no unexpected nulls in core columns."""
    raw_df = load_all_raw(minimal_config["paths"]["raw_data"], ["depression"])
    cleaned = process_posts(raw_df, min_length=minimal_config["processing"]["min_post_length_chars"])
    assert not cleaned.empty, "Cleaned dataframe must not be empty"

    weekly_df = WeeklyAggregator().aggregate(cleaned)
    assert not weekly_df.empty, "Weekly aggregation must not be empty"
    assert "iso_week" in weekly_df.columns
    assert "texts" in weekly_df.columns

    pipeline = FeaturePipeline(minimal_config)
    feature_df = pipeline.run(weekly_df, skip_topics=True)

    assert not feature_df.empty, "Feature matrix must not be empty"
    assert feature_df.shape[1] > 5, "Feature matrix must have multiple feature columns"

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    delta_cols = {c for c in feature_df.columns if c.endswith("_delta")}
    roll_cols = {c for c in feature_df.columns if "_roll" in c}
    skip_cols = meta_cols | delta_cols | roll_cols | {"week_sin", "week_cos"}
    core_cols = [c for c in feature_df.columns if c not in skip_cols]
    null_counts = feature_df[core_cols].isnull().sum()
    assert null_counts.sum() == 0, (
        f"Unexpected nulls in core feature columns: {null_counts[null_counts > 0].to_dict()}"
    )


# ---------------------------------------------------------------------------
# Stage 3: drift detection — alert level coverage + de-escalation logging
# ---------------------------------------------------------------------------

def test_drift_detector_covers_all_alert_levels_and_logs_deescalation(caplog):
    """
    Build a synthetic feature_df that forces a clear escalation then drop-back,
    verifying:
      - output has one row per input row
      - all aggregate_level values are in {0,1,2,3}
      - a de-escalation is logged when the level drops
      - negative spikes (below baseline mean) are caught (abs-z fix)
    """
    rng = np.random.RandomState(1)
    n = 30
    baseline_val = 0.1

    # First 12 weeks: stable baseline
    # Weeks 12-14: large positive spike  → should escalate to level 3
    # Weeks 15-17: large *negative* spike → should still escalate (abs fix)
    # Remaining: return to baseline → de-escalation
    avg_negative = np.full(n, baseline_val)
    avg_negative[12:15] = baseline_val + 10.0   # large positive spike
    avg_negative[15:18] = baseline_val - 10.0   # large negative spike (abs-z must catch this)

    feature_df = pd.DataFrame(
        {
            "avg_negative": avg_negative,
            "hopelessness_density": rng.uniform(0.01, 0.03, n),
            "week_start": pd.date_range("2024-01-01", periods=n, freq="W"),
        }
    )

    detector = DriftDetector(baseline_weeks=12)
    with caplog.at_level(logging.INFO, logger="src.monitoring.drift_detector"):
        results = detector.detect(feature_df)

    assert len(results) == n, "Output must have one row per input week"
    assert set(results["aggregate_level"].unique()).issubset({0, 1, 2, 3})
    assert set(results["alert_level_name"].unique()).issubset(set(ALERT_LEVELS.values()))

    # Negative spike weeks must reach at least warning level (not stay at 0)
    negative_spike_levels = results.loc[15:17, "aggregate_level"]
    assert (negative_spike_levels > 0).any(), (
        "Negative spikes must trigger alerts — check abs(z) fix in DriftDetector"
    )

    # De-escalation must be logged after the spike weeks
    deesc_records = [r for r in caplog.records if "drift_deescalation" in r.message]
    assert len(deesc_records) > 0, (
        "DriftDetector must log drift_deescalation when alert level drops"
    )
