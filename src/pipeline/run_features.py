import argparse
import json
import sys
import time
from pathlib import Path

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
    stage_start = time.perf_counter()

    print("Loading raw data...")
    df = load_all_raw(config["paths"]["raw_data"], config["reddit"]["subreddits"])
    print(f"  {len(df)} total posts loaded")

    print("Cleaning text...")
    min_len = config["processing"].get("min_post_length_chars", 20)
    df = process_posts(df, min_length=min_len)
    print(f"  {len(df)} posts after cleaning")
    min_posts_after_cleaning = config.get("processing", {}).get("min_posts_after_cleaning", 50)
    if len(df) < min_posts_after_cleaning:
        print(
            "ERROR: Too few posts after cleaning "
            f"({len(df)} < required {min_posts_after_cleaning})."
        )
        print("Hint: rerun collection (`python -m src.pipeline.run_collect`) or lower cleaning thresholds.")
        sys.exit(1)

    print("Aggregating by week...")
    aggregator = WeeklyAggregator()
    weekly_df = aggregator.aggregate(df)
    print(f"  {len(weekly_df)} weeks")
    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    min_train_weeks = int(wf_cfg.get("min_train_weeks", 26))
    gap_weeks = int(wf_cfg.get("gap_weeks", 1))
    seq_len = int(config.get("modeling", {}).get("lstm", {}).get("sequence_length", 8))
    min_weeks_required = min_train_weeks + gap_weeks + seq_len
    if len(weekly_df) < min_weeks_required:
        print(
            "ERROR: Weekly history too short for modeling "
            f"({len(weekly_df)} < required {min_weeks_required})."
        )
        print(
            "Hint: extend collection date range or use synthetic mode "
            "(`python -m src.pipeline.run_all --synthetic`)."
        )
        sys.exit(1)

    # Save processed weekly data
    save_processed(weekly_df, config["paths"]["processed_data"], "weekly")

    print("Extracting features...")
    pipeline = FeaturePipeline(config)
    feature_df = pipeline.run(weekly_df, skip_topics=args.skip_topics)

    save_processed(feature_df, config["paths"]["features"], "features")
    print(f"Feature matrix saved: {feature_df.shape}")
    _append_profile(
        config,
        {
            "stage": "features",
            "elapsed_seconds": round(time.perf_counter() - stage_start, 3),
            "rows_processed": int(len(df)),
            "weeks_generated": int(len(weekly_df)),
            "feature_rows": int(feature_df.shape[0]),
            "feature_cols": int(feature_df.shape[1]),
        },
    )
    print("Feature extraction complete.")


def _append_profile(config: dict, entry: dict) -> None:
    reports_root = Path(config["paths"]["reports"])
    reports_root.mkdir(parents=True, exist_ok=True)
    profile_path = reports_root / "pipeline_profile.json"
    payload = []
    if profile_path.exists():
        with open(profile_path, encoding="utf-8") as f:
            payload = json.load(f)
            if not isinstance(payload, list):
                payload = [payload]
    payload.append(entry)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
