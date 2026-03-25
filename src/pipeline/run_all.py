import argparse
import json
import sys
import time
from pathlib import Path

from src.config import load_config
from src.core.ui_config import PIPELINE_COPY


def main():
    parser = argparse.ArgumentParser(description=PIPELINE_COPY["run_all_description"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data")
    parser.add_argument("--skip-topics", action="store_true",
                        help="Skip BERTopic (faster)")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip hyperparameter search")
    args = parser.parse_args()

    print("=" * 60)
    print(PIPELINE_COPY["run_all_banner"])
    print("=" * 60)
    stage_profile: list[dict] = []

    # Step 1: Collect
    print("\n[1/4] DATA COLLECTION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_collect", "--config", args.config]
    if args.synthetic:
        sys.argv.append("--synthetic")
    from src.pipeline.run_collect import main as collect_main
    collect_main()
    stage_profile.append({"stage": "collect", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Step 2: Features
    print("\n[2/4] FEATURE EXTRACTION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_features", "--config", args.config]
    if args.skip_topics:
        sys.argv.append("--skip-topics")
    from src.pipeline.run_features import main as features_main
    features_main()
    stage_profile.append({"stage": "features", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Step 3: Train
    print("\n[3/4] MODEL TRAINING & EVALUATION")
    print("-" * 40)
    t0 = time.perf_counter()
    sys.argv = ["run_train", "--config", args.config]
    if args.skip_search:
        sys.argv.append("--skip-search")
    from src.pipeline.run_train import main as train_main
    train_main()
    stage_profile.append({"stage": "train", "elapsed_seconds": round(time.perf_counter() - t0, 3)})

    # Step 4: Visualize
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
