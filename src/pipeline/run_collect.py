import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_config
from src.data_quality.completeness import (
    check_weekly_completeness,
    flag_missing_weeks,
    log_source_provenance,
)
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
    reports_root = Path(config["paths"]["reports"])
    reports_root.mkdir(parents=True, exist_ok=True)
    profile_path = reports_root / "pipeline_profile.json"
    stage_start = time.perf_counter()
    profile_entries: list[dict] = []

    if args.synthetic:
        print("Generating synthetic data...")
        datasets = generate_synthetic_data(config, seed=config.get("random_seed", 42))
        for subreddit, df in datasets.items():
            print(f"  {subreddit}: {len(df)} posts")
            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")
            _run_data_quality_and_log(
                df=df,
                subreddit=subreddit,
                source="synthetic",
                date_range=config["reddit"]["date_range"],
                reports_root=reports_root,
            )
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
        rate_limit_rps = config["collection"].get("rate_limit_rps", 1.0)

        loader = PushshiftLoader(base_url=pushshift_url, rate_limit_rps=rate_limit_rps)

        for subreddit in config["reddit"]["subreddits"]:
            sub_start = time.perf_counter()
            print(f"Collecting r/{subreddit} via PullPush.io ...")
            print(f"  Date range: {date_range['start']} → {date_range['end']}")
            print("  (This may take 20–40 min due to rate limiting — ~1 req/sec)")
            source = "pushshift"

            try:
                df, summary = loader.load_range(
                    subreddit=subreddit,
                    after=after_ts,
                    before=before_ts,
                    batch_size=batch_size,
                )
            except Exception as e:
                print(f"  PullPush failed: {e}")
                print("  Falling back to PRAW...")
                df = _collect_via_praw(config, subreddit, after_dt, before_dt)
                source = "praw"
                summary = None

            if df is None or df.empty:
                print(f"  No posts found for r/{subreddit}")
                elapsed = time.perf_counter() - sub_start
                profile_entries.append(
                    {
                        "stage": "collect",
                        "subreddit": subreddit,
                        "elapsed_seconds": round(elapsed, 3),
                        "rows_processed": 0,
                        "throughput_rows_per_sec": 0.0,
                        "source": source,
                    }
                )
                continue

            print(f"  {len(df)} posts collected")
            if summary is not None:
                print(
                    f"  Requests: {summary.request_count}, retries: {summary.retry_count}, "
                    f"truncated: {summary.truncated}"
                )
            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")
            _run_data_quality_and_log(
                df=df,
                subreddit=subreddit,
                source=source,
                date_range=config["reddit"]["date_range"],
                reports_root=reports_root,
            )
            elapsed = time.perf_counter() - sub_start
            profile_entries.append(
                {
                    "stage": "collect",
                    "subreddit": subreddit,
                    "elapsed_seconds": round(elapsed, 3),
                    "rows_processed": int(len(df)),
                    "throughput_rows_per_sec": round(len(df) / max(elapsed, 1e-9), 3),
                    "source": source,
                    "retry_count": int(summary.retry_count) if summary is not None else 0,
                    "truncated": bool(summary.truncated) if summary is not None else False,
                }
            )

    total_elapsed = time.perf_counter() - stage_start
    _append_profile(
        profile_path,
        {
            "stage": "collect_total",
            "elapsed_seconds": round(total_elapsed, 3),
            "subreddit_runs": profile_entries,
        },
    )
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


def _run_data_quality_and_log(
    df,
    subreddit: str,
    source: str,
    date_range: dict,
    reports_root: Path,
) -> None:
    completeness_df = check_weekly_completeness(df, subreddit)
    missing_weeks = flag_missing_weeks(
        completeness_df,
        subreddit=subreddit,
        start_date=date_range["start"],
        end_date=date_range["end"],
    )
    for wk in completeness_df["week_start"].astype(str).tolist():
        log_source_provenance(subreddit=subreddit, week=wk, source=source)

    sub_report_dir = reports_root / subreddit
    sub_report_dir.mkdir(parents=True, exist_ok=True)
    completeness_csv = sub_report_dir / "weekly_completeness.csv"
    completeness_df.to_csv(completeness_csv, index=False)

    report = {
        "subreddit": subreddit,
        "source": source,
        "total_weeks_observed": int(len(completeness_df)),
        "gap_weeks_below_50pct": int(completeness_df["is_gap"].sum()) if not completeness_df.empty else 0,
        "missing_week_count": len(missing_weeks),
        "missing_weeks": missing_weeks,
        "avg_completeness_score": float(completeness_df["completeness_score"].mean())
        if not completeness_df.empty
        else 0.0,
    }
    with open(sub_report_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"  Data quality: {report['gap_weeks_below_50pct']} gap week(s), "
        f"{report['missing_week_count']} missing week(s)"
    )


def _append_profile(profile_path: Path, entry: dict) -> None:
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
