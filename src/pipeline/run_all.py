import argparse
import sys

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

    # Step 1: Collect
    print("\n[1/4] DATA COLLECTION")
    print("-" * 40)
    sys.argv = ["run_collect", "--config", args.config]
    if args.synthetic:
        sys.argv.append("--synthetic")
    from src.pipeline.run_collect import main as collect_main
    collect_main()

    # Step 2: Features
    print("\n[2/4] FEATURE EXTRACTION")
    print("-" * 40)
    sys.argv = ["run_features", "--config", args.config]
    if args.skip_topics:
        sys.argv.append("--skip-topics")
    from src.pipeline.run_features import main as features_main
    features_main()

    # Step 3: Train
    print("\n[3/4] MODEL TRAINING & EVALUATION")
    print("-" * 40)
    sys.argv = ["run_train", "--config", args.config]
    if args.skip_search:
        sys.argv.append("--skip-search")
    from src.pipeline.run_train import main as train_main
    train_main()

    # Step 4: Visualize
    print("\n[4/4] VISUALIZATION & REPORTING")
    print("-" * 40)
    sys.argv = ["run_evaluate", "--config", args.config]
    from src.pipeline.run_evaluate import main as evaluate_main
    evaluate_main()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Reports saved to: {args.config.replace('config/default.yaml', 'data/reports/')}")


if __name__ == "__main__":
    main()
