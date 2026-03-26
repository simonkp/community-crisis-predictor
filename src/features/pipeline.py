import pandas as pd

from src.features.progress_util import iter_groupby_subreddit
from src.features.linguistic import extract_linguistic_features
from src.features.sentiment import extract_sentiment_features
from src.features.distress import DistressScorer
from src.features.behavioral import extract_behavioral_features
from src.features.topics import TopicFeatureExtractor
from src.features.temporal import add_temporal_features


class FeaturePipeline:
    def __init__(self, config: dict):
        self.config = config
        feat_cfg = config.get("features", {})
        sent_cfg = feat_cfg.get("sentiment", {})
        self.sentiment_bins = sent_cfg.get("bins")
        self.sentiment_parallel_workers = sent_cfg.get("parallel_workers", 0)
        self.rolling_windows = feat_cfg.get("temporal", {}).get("rolling_windows", [2, 4])

        topic_cfg = feat_cfg.get("topics", {})
        self.topic_extractor = TopicFeatureExtractor(
            model_name=topic_cfg.get("model_name", "all-MiniLM-L6-v2"),
            n_topics=topic_cfg.get("n_topics", 15),
            min_topic_size=topic_cfg.get("min_topic_size", 10),
            max_posts_per_week=topic_cfg.get("max_posts_per_week", 200),
        )
        self.distress_scorer = DistressScorer(
            lexicon_dir=config.get("_lexicon_dir", "config/lexicons")
        )

    def run(self, weekly_df: pd.DataFrame, skip_topics: bool = False) -> pd.DataFrame:
        print("  Extracting linguistic features...")
        linguistic = extract_linguistic_features(weekly_df)

        print("  Extracting sentiment features...")
        sentiment = extract_sentiment_features(
            weekly_df,
            bins=self.sentiment_bins,
            parallel_workers=self.sentiment_parallel_workers,
        )

        print("  Extracting distress features...")
        distress = self.distress_scorer.extract_distress_features(weekly_df)

        print("  Extracting behavioral features...")
        behavioral = extract_behavioral_features(weekly_df)

        if skip_topics:
            print("  Skipping topic features (skip_topics=True)...")
            topic_features = pd.DataFrame({
                "dominant_topic": [0] * len(weekly_df),
                "topic_entropy": [0.0] * len(weekly_df),
                "topic_shift_jsd": [0.0] * len(weekly_df),
            }, index=weekly_df.index)
        else:
            print("  Extracting topic features (this may take a while)...")
            topic_features = self.topic_extractor.fit_and_extract(weekly_df)

        # Join all feature families
        meta_cols = ["subreddit", "iso_year", "iso_week", "week_start"]
        meta = weekly_df[meta_cols].copy()

        feature_df = pd.concat(
            [meta, linguistic, sentiment, distress, behavioral, topic_features],
            axis=1,
        )

        # Add temporal features (deltas, rolling, seasonality)
        print("  Adding temporal features...")

        # Group by subreddit for proper temporal computation
        parts = []
        for sub, group in iter_groupby_subreddit(feature_df, "subreddit", "  Temporal"):
            group = group.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
            group = add_temporal_features(group, self.rolling_windows)
            parts.append(group)

        feature_df = pd.concat(parts, ignore_index=True)

        # Fill NaN from deltas and rolling (first rows)
        feature_df = feature_df.fillna(0.0)

        print(f"  Feature matrix: {feature_df.shape[0]} weeks x {feature_df.shape[1]} columns")
        return feature_df
