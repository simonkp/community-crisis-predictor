import argparse

from src.config import load_config
from src.collector.storage import load_all_raw, save_processed
from src.processing.text_cleaner import process_posts
from src.processing.weekly_aggregator import WeeklyAggregator
from src.features.pipeline import FeaturePipeline


def main():
    parser = argparse.ArgumentParser(description="Extract features from collected data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-topics", action="store_true",
                        help="Skip BERTopic feature extraction (faster)")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading raw data...")
    df = load_all_raw(config["paths"]["raw_data"], config["reddit"]["subreddits"])
    print(f"  {len(df)} total posts loaded")

    print("Cleaning text...")
    min_len = config["processing"].get("min_post_length_chars", 20)
    df = process_posts(df, min_length=min_len)
    print(f"  {len(df)} posts after cleaning")

    print("Aggregating by week...")
    aggregator = WeeklyAggregator()
    weekly_df = aggregator.aggregate(df)
    print(f"  {len(weekly_df)} weeks")

    # Save processed weekly data
    save_processed(weekly_df, config["paths"]["processed_data"], "weekly")

    print("Extracting features...")
    pipeline = FeaturePipeline(config)
    feature_df = pipeline.run(weekly_df, skip_topics=args.skip_topics)

    save_processed(feature_df, config["paths"]["features"], "features")
    print(f"Feature matrix saved: {feature_df.shape}")
    print("Feature extraction complete.")


if __name__ == "__main__":
    main()
