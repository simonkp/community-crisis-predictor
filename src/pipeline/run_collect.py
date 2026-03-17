import argparse
import sys
from datetime import datetime, timezone

from src.config import load_config
from src.collector.privacy import strip_pii
from src.collector.storage import save_raw
from src.collector.synthetic import generate_synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Collect Reddit data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of using Reddit API",
    )
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
        # Real collection via PushshiftLoader (PullPush.io — free, no auth needed)
        from src.collector.historical_loader import PushshiftLoader

        date_range = config["reddit"]["date_range"]
        after_dt = datetime.fromisoformat(date_range["start"])
        before_dt = datetime.fromisoformat(date_range["end"])
        after_ts = int(after_dt.replace(tzinfo=timezone.utc).timestamp())
        before_ts = int(before_dt.replace(tzinfo=timezone.utc).timestamp())

        pushshift_url = config["collection"].get(
            "pushshift_base_url", "https://api.pullpush.io"
        )
        batch_size = config["collection"].get("batch_size", 500)

        loader = PushshiftLoader(base_url=pushshift_url)

        for subreddit in config["reddit"]["subreddits"]:
            print(f"Collecting r/{subreddit} via PullPush.io ...")
            print(f"  Date range: {date_range['start']} → {date_range['end']}")
            print("  (This may take 20–40 min due to rate limiting — ~1 req/sec)")

            try:
                df = loader.load_range(
                    subreddit=subreddit,
                    after=after_ts,
                    before=before_ts,
                    batch_size=batch_size,
                )
            except Exception as e:
                print(f"  PullPush failed: {e}")
                print("  Falling back to PRAW...")
                df = _collect_via_praw(config, subreddit, after_dt, before_dt)

            if df is None or df.empty:
                print(f"  No posts found for r/{subreddit}")
                continue

            print(f"  {len(df)} posts collected")
            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")

    print("Collection complete.")


def _collect_via_praw(config, subreddit, after, before):
    """PRAW fallback — only fetches recent posts (~1000 limit)."""
    try:
        from src.collector.reddit_client import RedditCollector
    except ImportError:
        print("PRAW not installed. Install praw or use --synthetic.", file=sys.stderr)
        return None

    collector = RedditCollector(config)
    return collector.collect_subreddit(subreddit, after, before)


if __name__ == "__main__":
    main()
