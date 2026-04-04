import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.core.logging_config import configure_logging
from src.core.ui_config import PIPELINE_COPY

# Presentation artifact legend:
# - Stage 1 (collect)  -> data/raw/{subreddit}/posts.parquet
# - Stage 2 (features) -> data/processed/weekly.parquet, data/features/features.parquet
# - Stage 3 (train)    -> data/models/{sub}_xgb.pkl, {sub}_lstm.pt, {sub}_feature_stats.json, eval_results.json
# - Stage 4 (evaluate) -> data/reports/{sub}/ (shap.csv, drift_alerts.json, weekly_briefs.json, timeline.html, etc.)
# - Pipeline telemetry -> data/reports/pipeline_profile.json


def main():
    parser = argparse.ArgumentParser(description=PIPELINE_COPY["run_all_description"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--skip-topics", action="store_true",
                        help="Skip BERTopic (faster)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip hyperparameter search")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full feature extraction even when cache says unchanged",
    )
    args = parser.parse_args()
    configure_logging()

    print("=" * 60)
    print(PIPELINE_COPY["run_all_banner"])
    print("=" * 60)
    stage_profile: list[dict] = []

    # Presentation checkpoint [1/4]
    # Input: source connectors (synthetic / zenodo / reddit_api)
    # Output artifact: data/raw/{subreddit}/posts.parquet
    # We call each stage's CLI main directly and override sys.argv so
    # stage-specific argparse parsing behaves exactly like standalone execution.
    print("\n[1/4] DATA COLLECTION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_collect", "--config", args.config]
    if args.synthetic:
        sys.argv.append("--synthetic")
    from src.pipeline.run_collect import main as collect_main
    collect_main()
    stage_profile.append({"stage": "collect", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Presentation checkpoint [2/4]
    # Input artifact: data/raw/{subreddit}/posts.parquet
    # Output artifacts: data/processed/weekly.parquet + data/features/features.parquet
    # Feature stage may short-circuit via cache fingerprint unless --force is passed.
    print("\n[2/4] FEATURE EXTRACTION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_features", "--config", args.config]
    if args.skip_topics:
        sys.argv.append("--skip-topics")
    if args.force:
        sys.argv.append("--force")
    from src.pipeline.run_features import main as features_main
    features_main()
    stage_profile.append({"stage": "features", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Presentation checkpoint [3/4]
    # Input artifact: data/features/features.parquet
    # Output artifacts: data/models/*.pkl, *.pt, *_feature_stats.json, eval_results.json
    # Produces eval metrics plus serialized model artifacts under paths.models.
    print("\n[3/4] MODEL TRAINING & EVALUATION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_train", "--config", args.config]
    if args.skip_search:
        sys.argv.append("--skip-search")
    from src.pipeline.run_train import main as train_main
    train_main()
    stage_profile.append({"stage": "train", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Presentation checkpoint [4/4]
    # Input artifacts: data/features/features.parquet + data/models/eval_results.json
    # Output artifacts: data/reports/{sub}/ (timeline, SHAP, drift, briefs, dashboard HTML)
    # Generates deploy-facing report artifacts under paths.reports.
    print("\n[4/4] VISUALIZATION & REPORTING")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_evaluate", "--config", args.config]
    from src.pipeline.run_evaluate import main as evaluate_main
    evaluate_main()
    stage_profile.append({"stage": "evaluate", "elapsed_seconds": round(time.perf_counter() - t0, 3)})
    _append_profile(args.config, {"stage": "run_all", "steps": stage_profile})
    _print_stage_table(stage_profile)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    cfg = load_config(args.config)
    print(f"Reports saved to: {cfg['paths']['reports']}")


def _append_profile(config_path: str, entry: dict) -> None:
    # pipeline_profile.json is append-only operational telemetry used by dashboard/reporting.
    cfg = load_config(config_path)
    reports_root = Path(cfg["paths"]["reports"])
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


def _print_stage_table(stage_profile: list[dict]) -> None:
    print("\nPipeline stage timing")
    print("  stage        elapsed_seconds")
    print("  -----------  ---------------")
    total = 0.0
    for row in stage_profile:
        sec = float(row.get("elapsed_seconds", 0.0))
        total += sec
        print(f"  {row.get('stage', ''):<11}  {sec:>15.3f}")
    print(f"  {'total':<11}  {total:>15.3f}")


if __name__ == "__main__":
    main()
