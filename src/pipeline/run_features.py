import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.collector.storage import load_all_raw, save_processed
from src.features.progress_util import iter_groupby_subreddit
from src.processing.text_cleaner import process_posts
from src.processing.weekly_aggregator import WeeklyAggregator
from src.features.pipeline import FeaturePipeline
from src.reporting.eda import generate_eda_report

# Presentation artifact legend:
# - Input artifact      -> data/raw/{subreddit}/posts.parquet
# - Intermediate output -> data/processed/weekly.parquet
# - Primary output      -> data/features/features.parquet
# - Cache metadata      -> data/features/feature_build_meta.json
# - Stage telemetry     -> data/reports/pipeline_profile.json


def main():
    parser = argparse.ArgumentParser(description="Extract features from collected data")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-topics", action="store_true",
                        help="Skip BERTopic feature extraction (faster)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full feature extraction even when inputs/config are unchanged",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stage_start = time.perf_counter()
    cache_meta_path = Path(config["paths"]["features"]) / "feature_build_meta.json"
    # Fingerprint guards expensive feature recomputation and keeps rebuilds deterministic.
    current_fingerprint = _compute_feature_fingerprint(config, args.config, args.skip_topics)
    if not args.force and _is_feature_cache_valid(cache_meta_path, current_fingerprint):
        print("Feature extraction skipped: upstream data/config unchanged.")
        print(f"  Cache metadata: {cache_meta_path}")
        print("  Use --force to recompute features.")
        return

    # Presentation checkpoint: input artifact from collection stage.
    print("Loading raw data...")
    df = load_all_raw(config["paths"]["raw_data"], config["reddit"]["subreddits"])
    loaded_counts = _counts_by_subreddit(df)
    print(f"  {len(df)} total posts loaded")

    print("Cleaning text...")
    # Cleaning is done per subreddit to preserve progress visibility and isolate failures.
    min_len = config["processing"].get("min_post_length_chars", 20)
    cleaned_chunks = []
    for _, sub_df in iter_groupby_subreddit(df, "subreddit", desc="Cleaning text"):
        cleaned_chunks.append(process_posts(sub_df, min_length=min_len))
    if cleaned_chunks:
        df = pd.concat(cleaned_chunks, ignore_index=True)
    else:
        df = df.iloc[0:0].copy()
    cleaned_counts = _counts_by_subreddit(df)
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
    # This is the handoff boundary from post-level to week-level modeling granularity.
    aggregator = WeeklyAggregator()
    weekly_df = aggregator.aggregate(df)
    week_counts = _counts_by_subreddit(weekly_df)
    print(f"  {len(weekly_df)} week-rows (one row per subreddit per ISO week with posts)")
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

    # Intermediate artifact used for audit/debug and optional reuse.
    # Path: data/processed/weekly.parquet
    save_processed(weekly_df, config["paths"]["processed_data"], "weekly")

    _print_subreddit_summary_table(
        config["reddit"]["subreddits"],
        loaded_counts,
        cleaned_counts,
        week_counts,
        weekly_df,
        feature_df=None,
        include_feat_rows=False,
        title="Per-subreddit summary (before feature extraction)",
    )

    print("Extracting features...")
    # FeaturePipeline expands weekly aggregates into the model-ready matrix used by train/evaluate.
    pipeline = FeaturePipeline(config)
    feature_df = pipeline.run(weekly_df, skip_topics=args.skip_topics)

    # Canonical model input artifact consumed by run_train and run_evaluate.
    # Path: data/features/features.parquet
    save_processed(feature_df, config["paths"]["features"], "features")
    print(f"Feature matrix saved: {feature_df.shape}")

    # EDA reports: one per subreddit — outlier detection, trend, feature distributions.
    # Outputs: data/reports/{sub}/eda_report.json + eda_summary.html
    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feat_cols = [c for c in feature_df.columns if c not in meta_cols]
    reports_root = Path(config["paths"]["reports"])
    print("Generating EDA reports...")
    for sub, sub_df in feature_df.groupby("subreddit"):
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        try:
            generate_eda_report(
                feature_df=sub_df,
                config=config,
                feature_columns=feat_cols,
                subreddit=str(sub),
                output_dir=reports_root / str(sub),
            )
        except Exception as exc:
            print(f"  Warning: EDA report failed for r/{sub}: {exc}")
    _print_subreddit_summary_table(
        config["reddit"]["subreddits"],
        loaded_counts,
        cleaned_counts,
        week_counts,
        weekly_df,
        feature_df=feature_df,
        include_feat_rows=True,
        title="Per-subreddit summary (after feature extraction)",
    )
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
    _save_feature_cache_meta(cache_meta_path, current_fingerprint)
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


def _counts_by_subreddit(df) -> pd.Series:
    if "subreddit" not in df.columns or df.empty:
        return pd.Series(dtype=int)
    keys = df["subreddit"].astype(str).str.strip().str.lower()
    return keys.groupby(keys).size()


def _print_subreddit_summary_table(
    subreddits: list[str],
    loaded: pd.Series,
    cleaned: pd.Series,
    week_rows: pd.Series,
    weekly_df: pd.DataFrame,
    feature_df: pd.DataFrame | None,
    *,
    include_feat_rows: bool = True,
    title: str = "Per-subreddit summary",
) -> None:
    """ASCII table: loaded, cleaned, dropped, week_rows, avg posts/week, optional feat_rows."""
    feat_counts = (
        _counts_by_subreddit(feature_df)
        if feature_df is not None and not feature_df.empty
        else pd.Series(dtype=int)
    )
    unique_iso_weeks = 0
    if not weekly_df.empty and {"iso_year", "iso_week"}.issubset(weekly_df.columns):
        unique_iso_weeks = int(
            weekly_df[["iso_year", "iso_week"]].drop_duplicates().shape[0]
        )
    sum_week_rows = int(week_rows.sum()) if not week_rows.empty else 0

    print("")
    print(f"  {title}")
    print(
        f"  Week-rows sum across subs = {sum_week_rows} "
        f"(same calendar week counted once per subreddit; not deduplicated across subs)."
    )
    if unique_iso_weeks:
        print(
            f"  Unique ISO weeks (union across all subs): {unique_iso_weeks}"
        )

    rows_out: list[dict] = []
    t_loaded = t_cleaned = t_dropped = t_weeks = t_feat = 0
    for sub in subreddits:
        sub_key = str(sub).strip().lower()
        pl = int(loaded.get(sub_key, 0)) if not loaded.empty else 0
        pc = int(cleaned.get(sub_key, 0)) if not cleaned.empty else 0
        wr = int(week_rows.get(sub_key, 0)) if not week_rows.empty else 0
        fr = int(feat_counts.get(sub_key, 0)) if not feat_counts.empty else 0
        dr = pl - pc
        avg = (pc / wr) if wr else 0.0
        row = {
            "sub": f"r/{sub}",
            "loaded": pl,
            "cleaned": pc,
            "dropped": dr,
            "week_rows": wr,
            "avg": avg,
            "feat": fr,
        }
        rows_out.append(row)
        t_loaded += pl
        t_cleaned += pc
        t_dropped += dr
        t_weeks += wr
        t_feat += fr

    def _w(label: str, vals: list, is_float: bool = False, is_str: bool = False) -> int:
        m = len(label)
        for v in vals:
            if is_str:
                s = str(v)
            elif is_float:
                s = f"{v:.1f}"
            else:
                s = str(int(v))
            m = max(m, len(s))
        return m

    labs_base = ["subreddit", "loaded", "cleaned", "dropped", "week_rows", "avg_posts/wk"]
    if include_feat_rows:
        labs = labs_base + ["feat_rows"]
    else:
        labs = labs_base

    def col_vals(key: str) -> list:
        return [r[key] for r in rows_out]

    w0 = _w(labs[0], col_vals("sub"), is_str=True)
    w1 = _w(labs[1], col_vals("loaded"))
    w2 = _w(labs[2], col_vals("cleaned"))
    w3 = _w(labs[3], col_vals("dropped"))
    w4 = _w(labs[4], col_vals("week_rows"))
    w5 = _w(labs[5], col_vals("avg"), is_float=True)
    widths = [w0, w1, w2, w3, w4, w5]
    if include_feat_rows:
        w6 = _w(labs[6], col_vals("feat"))
        widths.append(w6)

    parts_hdr = [f"{labs[i]:>{widths[i]}}" if i else f"{labs[0]:<{widths[0]}}" for i in range(len(labs))]
    hdr = "  " + "  ".join(parts_hdr)
    sep_parts = ["-" * widths[i] for i in range(len(labs))]
    sep = "  " + "  ".join(sep_parts)
    print(hdr)
    print(sep)
    for r in rows_out:
        line = (
            f"  {r['sub']:<{w0}}  {r['loaded']:>{w1}}  {r['cleaned']:>{w2}}  {r['dropped']:>{w3}}  "
            f"{r['week_rows']:>{w4}}  {r['avg']:>{w5}.1f}"
        )
        if include_feat_rows:
            line += f"  {r['feat']:>{w6}}"
        print(line)
    total_avg = (t_cleaned / t_weeks) if t_weeks else 0.0
    tot = (
        f"  {'TOTAL':<{w0}}  {t_loaded:>{w1}}  {t_cleaned:>{w2}}  {t_dropped:>{w3}}  "
        f"{t_weeks:>{w4}}  {total_avg:>{w5}.1f}"
    )
    if include_feat_rows:
        tot += f"  {t_feat:>{w6}}"
    print(tot)


def _compute_feature_fingerprint(config: dict, config_path: str, skip_topics: bool) -> dict:
    # Cache key includes raw parquet signatures + relevant config slices.
    # If any input changes, digest changes and feature rebuild is triggered.
    raw_root = Path(config["paths"]["raw_data"])
    subreddits = list(config["reddit"]["subreddits"])
    raw_files = []
    for sub in subreddits:
        p = raw_root / sub / "posts.parquet"
        if p.exists():
            st = p.stat()
            raw_files.append(
                {
                    "path": str(p),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
    cfg_relevant = {
        "subreddits": list(config.get("reddit", {}).get("subreddits", [])),
        "processing": config.get("processing", {}),
        "features": config.get("features", {}),
        "modeling_lstm_sequence_length": config.get("modeling", {}).get("lstm", {}).get("sequence_length", 8),
    }
    cfg_sig = hashlib.sha256(
        json.dumps(cfg_relevant, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    base = {
        "subreddits": subreddits,
        "skip_topics": bool(skip_topics),
        "raw_files": raw_files,
        "config_sig": cfg_sig,
    }
    digest = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"version": 1, "fingerprint": digest, "base": base}


def _is_feature_cache_valid(meta_path: Path, current: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return (
        isinstance(saved, dict)
        and saved.get("version") == current.get("version")
        and saved.get("fingerprint") == current.get("fingerprint")
    )


def _save_feature_cache_meta(meta_path: Path, payload: dict) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
