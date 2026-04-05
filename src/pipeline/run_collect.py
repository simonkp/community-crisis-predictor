import argparse
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.collector.arctic_shift_loader import ArcticShiftLoader, parse_arctic_shift_filename
from src.config import load_config
from src.data_quality.completeness import (
    check_weekly_completeness,
    cross_source_validate,
    flag_missing_weeks,
    log_source_provenance,
)
from src.collector.manifest import (
    is_file_entry_valid,
    load_manifest,
    record_file_entry,
    record_subreddit_ingestion,
    save_manifest,
)
from src.collector.privacy import strip_pii
from src.collector.storage import save_raw, validate_source_compatibility
from src.collector.synthetic import generate_synthetic_data

# Presentation artifact legend:
# - Primary output      -> data/raw/{subreddit}/posts.parquet
# - Quality artifacts   -> data/reports/{subreddit}/weekly_completeness.csv, data_quality_report.json
# - Provenance logging  -> data/quality.db (source-per-week records)
# - Ingestion manifests -> data/staging/zenodo/manifest.json, data/ingestion_manifest.json
# - Stage telemetry     -> data/reports/pipeline_profile.json


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
    source = config.get("collection", {}).get("source", "reddit_api")
    if args.synthetic:
        source = "synthetic"
    reports_root = Path(config["paths"]["reports"])
    reports_root.mkdir(parents=True, exist_ok=True)
    profile_path = reports_root / "pipeline_profile.json"
    stage_start = time.perf_counter()
    profile_entries: list[dict] = []

    # Single entrypoint for all collectors; downstream stages always consume
    # canonical parquet outputs under paths.raw_data regardless of upstream source.
    print(f"Collection source: {source}")
    if source == "synthetic":
        # Presentation checkpoint: synthetic branch
        # Output artifact per subreddit -> data/raw/{subreddit}/posts.parquet
        print("Generating synthetic data...")
        datasets = generate_synthetic_data(config, seed=config.get("random_seed", 42))
        for subreddit, df in datasets.items():
            print(f"  {subreddit}: {len(df)} posts")
            df = strip_pii(df, config["collection"]["privacy_salt"])
            df["data_source"] = "synthetic"
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")
            _run_data_quality_and_log(
                df=df,
                subreddit=subreddit,
                source="synthetic",
                date_range=config["reddit"]["date_range"],
                reports_root=reports_root,
                quality_db_path=config.get("paths", {}).get("quality_db", "data/quality.db"),
            )
    elif source == "zenodo_covid":
        # Presentation checkpoint: Zenodo (+ optional Arctic Shift gap-fill) branch
        # Output artifact per subreddit -> data/raw/{subreddit}/posts.parquet
        from src.collector.zenodo_loader import ZenodoLoader

        zen = config["collection"].get("zenodo", {})
        arctic_cfg = config["collection"].get("arctic_shift", {})
        date_range = zen.get("date_range", config["reddit"]["date_range"])
        loader = ZenodoLoader(
            dataset_url=zen["dataset_url"],
            archive_dir=zen.get(
                "local_archive_dir", config.get("paths", {}).get("external_zenodo", "data/external/zenodo")
            ),
            staging_dir=zen.get(
                "staging_dir", config.get("paths", {}).get("staging_zenodo", "data/staging/zenodo")
            ),
            record_id=int(zen.get("record_id", 3941387)),
            timeframes=zen.get("timeframes", ["2018", "2019", "pre", "post"]),
        )
        manifest_path = zen.get(
            "manifest_path",
            str(
                Path(config.get("paths", {}).get("staging_zenodo", "data/staging/zenodo")) / "manifest.json"
            ),
        )
        # Manifest drives idempotency: if source files + subreddit status are valid,
        # we skip expensive re-download/re-ingestion safely.
        manifest = load_manifest(manifest_path)
        ingestion_manifest_path = config["collection"].get(
            "ingestion_manifest_path", "data/ingestion_manifest.json"
        )
        ingestion_manifest = load_manifest(ingestion_manifest_path)
        _init_ingestion_source_entry(
            ingestion_manifest,
            source_name="zenodo",
            url=zen.get("dataset_url", ""),
        )
        arctic_url = f"https://drive.google.com/uc?id={arctic_cfg.get('gdrive_file_id', '')}"
        _init_ingestion_source_entry(
            ingestion_manifest,
            source_name="arctic_shift",
            url=arctic_url,
        )
        arctic_stage_dir = Path(arctic_cfg.get("staging_dir", "data/staging/arctic_shift"))
        arctic_zip_filename = arctic_cfg.get("zip_filename", "arctic_shift_gap_fill_v1.zip")
        arctic_zip_path = arctic_stage_dir / arctic_zip_filename
        arctic_subreddits = set(arctic_cfg.get("subreddits", zen.get("subreddits", [])))
        arctic_loader = ArcticShiftLoader(min_selftext_chars=10)
        arctic_enabled = bool(arctic_cfg.get("gdrive_file_id"))
        if arctic_enabled:
            _ensure_arctic_shift_ready(
                arctic_cfg=arctic_cfg,
                zip_path=arctic_zip_path,
                stage_dir=arctic_stage_dir,
                ingestion_manifest=ingestion_manifest,
            )
            # Download v2 zip (2021-2024 extension) if configured
            gdrive_v2 = arctic_cfg.get("gdrive_file_id_v2", "").strip()
            if gdrive_v2:
                zip_v2_path = arctic_stage_dir / arctic_cfg.get("zip_filename_v2", "arctic_shift_fill_v2.zip")
                _ensure_arctic_shift_ready(
                    arctic_cfg={"gdrive_file_id": gdrive_v2},
                    zip_path=zip_v2_path,
                    stage_dir=arctic_stage_dir,
                    ingestion_manifest=ingestion_manifest,
                )
            save_manifest(ingestion_manifest_path, ingestion_manifest)

        subreddits = zen.get("subreddits", config["reddit"]["subreddits"])
        arctic_total_rows = 0
        arctic_loaded_any = False
        for subreddit in subreddits:
            sub_start = time.perf_counter()
            raw_path = Path(config["paths"]["raw_data"]) / subreddit / "posts.parquet"
            sub_manifest = manifest.get("subreddits", {}).get(subreddit, {})
            arctic_targeted_sub = bool(arctic_enabled and subreddit in arctic_subreddits)
            arctic_already_loaded_for_sub = bool(sub_manifest.get("arctic_shift_loaded", False))
            can_skip_for_arctic = (not arctic_targeted_sub) or arctic_already_loaded_for_sub
            # Fast-path skip for already ingested subreddit snapshots.
            if raw_path.exists() and sub_manifest.get("status") == "ingested" and can_skip_for_arctic:
                files_ok = True
                for file_path in sub_manifest.get("files", []):
                    if not is_file_entry_valid(manifest, file_path):
                        files_ok = False
                        break
                if files_ok:
                    reason = "already ingested, manifest valid, and Arctic gap-fill already merged"
                    if not arctic_targeted_sub:
                        reason = "already ingested and manifest valid"
                    print(f"Skipping r/{subreddit}: {reason}.")
                    # Keep dashboard data-quality artifacts in sync even when collection is skipped.
                    try:
                        existing_df = pd.read_parquet(raw_path)
                        _run_data_quality_and_log(
                            df=existing_df,
                            subreddit=subreddit,
                            source="zenodo_covid",
                            date_range=date_range,
                            reports_root=reports_root,
                            quality_db_path=config.get("paths", {}).get("quality_db", "data/quality.db"),
                        )
                    except Exception as exc:
                        print(f"  Warning: failed to refresh data-quality artifacts for r/{subreddit}: {exc}")
                    elapsed = time.perf_counter() - sub_start
                    profile_entries.append(
                        {
                            "stage": "collect",
                            "subreddit": subreddit,
                            "elapsed_seconds": round(elapsed, 3),
                            "rows_processed": int(
                                manifest.get("subreddits", {}).get(subreddit, {}).get("rows", 0)
                            ),
                            "throughput_rows_per_sec": 0.0,
                            "source": "zenodo_covid",
                            "status": "skipped_manifest_valid",
                        }
                    )
                    continue

            print(f"Collecting r/{subreddit} via Zenodo dataset ...")
            print(f"  Date range filter: {date_range.get('start')} -> {date_range.get('end')}")
            try:
                downloaded_files = loader.ensure_subreddit_files(subreddit)
            except RuntimeError as exc:
                print(f"\n  [COLLECTION ERROR] r/{subreddit}: {exc}")
                print(f"  Skipping r/{subreddit} — other subreddits will continue.")
                record_subreddit_ingestion(
                    manifest,
                    subreddit=subreddit,
                    rows=0,
                    min_created_utc=None,
                    max_created_utc=None,
                    status="connection_error",
                )
                save_manifest(manifest_path, manifest)
                continue
            print(f"  Files prepared for r/{subreddit}: {len(downloaded_files)}")
            for p in downloaded_files:
                record_file_entry(manifest, p)
            manifest.setdefault("subreddits", {}).setdefault(subreddit, {})["files"] = [
                str(p) for p in downloaded_files
            ]
            save_manifest(manifest_path, manifest)

            # Loader normalizes raw source columns into pipeline's canonical schema.
            df = loader.load_subreddit_posts(
                subreddit=subreddit,
                start_date=date_range.get("start"),
                end_date=date_range.get("end"),
            )
            if df.empty:
                print(f"  No posts found for r/{subreddit} in Zenodo staging files")
                record_subreddit_ingestion(
                    manifest,
                    subreddit=subreddit,
                    rows=0,
                    min_created_utc=None,
                    max_created_utc=None,
                    status="empty",
                )
                save_manifest(manifest_path, manifest)
                continue

            # Optional gap-fill merge for missing windows; dedupe on post_id to avoid
            # double counting when primary and gap-fill sources overlap.
            if arctic_enabled and subreddit in arctic_subreddits:
                arctic_df, file_stats = _load_arctic_shift_for_subreddit(
                    subreddit=subreddit,
                    stage_dir=arctic_stage_dir,
                    loader=arctic_loader,
                )
                for fs in file_stats:
                    print(
                        "  [ArcticShift] "
                        f"{fs['file']} | parsed={fs['parsed']} kept={fs['kept']} inserted={fs['inserted']} "
                        f"skipped={fs['skipped']}"
                    )
                if not arctic_df.empty:
                    try:
                        validate_source_compatibility({"zenodo": df, "arctic_shift": arctic_df})
                    except ValueError as exc:
                        print(f"  [ArcticShift] WARNING: compatibility check failed — skipping merge: {exc}")
                        arctic_df = pd.DataFrame()
                if not arctic_df.empty:
                    before_len = len(df)
                    df = pd.concat([df, arctic_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
                    inserted_rows = len(df) - before_len
                    if inserted_rows > 0:
                        arctic_total_rows += inserted_rows
                        arctic_loaded_any = True

            df = strip_pii(df, config["collection"]["privacy_salt"])
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved {len(df)} posts to {path}")

            min_ts = int(df["created_utc"].min()) if "created_utc" in df.columns else None
            max_ts = int(df["created_utc"].max()) if "created_utc" in df.columns else None
            record_subreddit_ingestion(
                manifest,
                subreddit=subreddit,
                rows=len(df),
                min_created_utc=min_ts,
                max_created_utc=max_ts,
                status="ingested",
            )
            manifest.setdefault("subreddits", {}).setdefault(subreddit, {})["arctic_shift_loaded"] = bool(
                arctic_targeted_sub
            )
            save_manifest(manifest_path, manifest)

            # Side artifacts for reporting/dashboard quality tab.
            _run_data_quality_and_log(
                df=df,
                subreddit=subreddit,
                source="zenodo_covid",
                date_range=date_range,
                reports_root=reports_root,
                quality_db_path=config.get("paths", {}).get("quality_db", "data/quality.db"),
            )
            elapsed = time.perf_counter() - sub_start
            print(f"  r/{subreddit} collection finished in {elapsed:.2f}s")
            profile_entries.append(
                {
                    "stage": "collect",
                    "subreddit": subreddit,
                    "elapsed_seconds": round(elapsed, 3),
                    "rows_processed": int(len(df)),
                    "throughput_rows_per_sec": round(len(df) / max(elapsed, 1e-9), 3),
                    "source": "zenodo_covid",
                }
            )
        if arctic_enabled:
            _update_ingestion_source_entry(
                ingestion_manifest,
                source_name="arctic_shift",
                downloaded=bool(arctic_zip_path.exists()),
                loaded=arctic_loaded_any,
                row_count=arctic_total_rows,
            )
            save_manifest(ingestion_manifest_path, ingestion_manifest)

    else:
        # Real collection via PushshiftLoader (PullPush.io — free, no auth needed)
        # Output artifact per subreddit -> data/raw/{subreddit}/posts.parquet
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
            print(f"  Date range: {date_range['start']} -> {date_range['end']}")
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
            df["data_source"] = source
            path = save_raw(df, config["paths"]["raw_data"], subreddit)
            print(f"  Saved to {path}")
            _run_data_quality_and_log(
                df=df,
                subreddit=subreddit,
                source=source,
                date_range=config["reddit"]["date_range"],
                reports_root=reports_root,
                quality_db_path=config.get("paths", {}).get("quality_db", "data/quality.db"),
            )
            elapsed = time.perf_counter() - sub_start
            print(f"  r/{subreddit} collection finished in {elapsed:.2f}s")
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


def _init_ingestion_source_entry(manifest: dict, source_name: str, url: str) -> None:
    manifest.setdefault(source_name, {})
    section = manifest[source_name]
    section.setdefault("url", url)
    section.setdefault("expected_sha256", "")
    section.setdefault("downloaded", False)
    section.setdefault("loaded", False)
    section.setdefault("row_count", 0)
    section.setdefault("loaded_at", None)


def _update_ingestion_source_entry(
    manifest: dict,
    source_name: str,
    downloaded: bool,
    loaded: bool,
    row_count: int,
) -> None:
    _init_ingestion_source_entry(manifest, source_name=source_name, url=manifest.get(source_name, {}).get("url", ""))
    section = manifest[source_name]
    section["downloaded"] = bool(downloaded)
    section["loaded"] = bool(loaded)
    section["row_count"] = int(row_count)
    section["loaded_at"] = datetime.now(timezone.utc).isoformat()


def _ensure_arctic_shift_ready(
    arctic_cfg: dict,
    zip_path: Path,
    stage_dir: Path,
    ingestion_manifest: dict,
) -> None:
    gdrive_file_id = arctic_cfg.get("gdrive_file_id", "").strip()
    if not gdrive_file_id:
        return
    stage_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        try:
            import gdown  # type: ignore
        except ImportError as exc:
            raise RuntimeError("gdown is required for Arctic Shift download. Install dependencies again.") from exc
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        print(f"  [ArcticShift] Downloading gap-fill zip: {url}")
        gdown.download(url, str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            target = stage_dir / member
            if not target.exists():
                zf.extract(member, path=stage_dir)
    _update_ingestion_source_entry(
        ingestion_manifest,
        source_name="arctic_shift",
        downloaded=True,
        loaded=bool(ingestion_manifest.get("arctic_shift", {}).get("loaded", False)),
        row_count=int(ingestion_manifest.get("arctic_shift", {}).get("row_count", 0)),
    )


def _load_arctic_shift_for_subreddit(
    subreddit: str,
    stage_dir: Path,
    loader: ArcticShiftLoader,
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    stats_out: list[dict] = []
    # Search recursively so newer gap-fill drops extracted into nested folders
    # (e.g. arctic_shift_fill_v2/) are merged without manual file copies.
    for path in sorted(stage_dir.rglob("arctic_shift_*_*.jsonl")):
        sub_from_name = parse_arctic_shift_filename(path)
        if sub_from_name is None:
            continue
        if sub_from_name.lower() != subreddit.lower():
            continue
        df, stats = loader.load_jsonl(path, subreddit=subreddit)
        stats["inserted"] = int(len(df))
        stats["skipped"] = int(
            stats.get("skipped_json", 0)
            + stats.get("skipped_non_self", 0)
            + stats.get("skipped_body", 0)
            + stats.get("skipped_subreddit", 0)
        )
        stats_out.append(stats)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["post_id", "created_utc", "selftext", "subreddit", "author", "data_source"]), stats_out
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["created_utc", "selftext", "post_id"])
    out["created_utc"] = pd.to_numeric(out["created_utc"], errors="coerce")
    out = out.dropna(subset=["created_utc"])
    out["created_utc"] = out["created_utc"].astype(int)
    out = out.drop_duplicates(subset=["post_id"]).reset_index(drop=True)
    return out, stats_out


def _run_data_quality_and_log(
    df,
    subreddit: str,
    source: str,
    date_range: dict,
    reports_root: Path,
    quality_db_path: str,
) -> None:
    # Data-quality artifacts are first-class outputs consumed by dashboard/reporting.
    # We regenerate them during collection so downstream stages stay lightweight.
    completeness_df = check_weekly_completeness(df, subreddit)
    missing_weeks = flag_missing_weeks(
        completeness_df,
        subreddit=subreddit,
        start_date=date_range["start"],
        end_date=date_range["end"],
    )
    for wk in completeness_df["week_start"].astype(str).tolist():
        log_source_provenance(
            subreddit=subreddit,
            week=wk,
            source=source,
            db_path=quality_db_path,
        )

    sub_report_dir = reports_root / subreddit
    sub_report_dir.mkdir(parents=True, exist_ok=True)
    completeness_csv = sub_report_dir / "weekly_completeness.csv"
    completeness_df.to_csv(completeness_csv, index=False)

    cross_src = cross_source_validate(df, subreddit)

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
        "cross_source_validation": cross_src,
    }
    if cross_src.get("n_discrepancies", 0) > 0:
        print(
            f"  Warning: {cross_src['n_discrepancies']} cross-source count discrepancy week(s) "
            f"detected for r/{subreddit} — see data_quality_report.json for details."
        )
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
