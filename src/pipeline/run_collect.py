import argparse
import sys
from datetime import datetime

from src.config import load_config
from src.collector.privacy import strip_pii
from src.collector.storage import save_raw
from src.collector.synthetic import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Collect Reddit data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic data instead of using Reddit API")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.synthetic:
        print("Generating synthetic data...")
        datasets = generate_synthetic_data(config, seed=config.get("random_seed", 42))
        for subreddit, df in datasets.items():
            print(f"  {subreddit}: {len(df)} posts")
            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")
    else:
        # Real collection via PRAW
        try:
            from src.collector.reddit_client import RedditCollector
        except ImportError:
            print("PRAW not installed. Use --synthetic or install praw.", file=sys.stderr)
            sys.exit(1)

        collector = RedditCollector(config)
        date_range = config["reddit"]["date_range"]
        after = datetime.fromisoformat(date_range["start"])
        before = datetime.fromisoformat(date_range["end"])

        for subreddit in config["reddit"]["subreddits"]:
            print(f"Collecting r/{subreddit}...")
            df = collector.collect_subreddit(subreddit, after, before)
            if df.empty:
                print(f"  No posts found for r/{subreddit}")
                continue
            print(f"  {len(df)} posts collected")
            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")

    print("Collection complete.")


if __name__ == "__main__":
    main()
