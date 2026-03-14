import argparse
import json
from pathlib import Path

from src.config import load_config
from src.collector.storage import load_processed
from src.modeling.evaluate import evaluate_walk_forward


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate crisis prediction model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip hyperparameter search (use defaults)")
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading feature matrix...")
    feature_df = load_processed(config["paths"]["features"], "features")
    print(f"  {feature_df.shape[0]} weeks x {feature_df.shape[1]} features")

    # Identify feature columns (exclude meta columns)
    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in feature_df.columns if c not in meta_cols]

    # Run evaluation per subreddit
    all_results = {}
    for sub, sub_df in feature_df.groupby("subreddit"):
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        print(f"\nEvaluating r/{sub} ({len(sub_df)} weeks)...")

        results = evaluate_walk_forward(
            sub_df, config, feature_columns, skip_search=args.skip_search
        )

        if "error" in results:
            print(f"  Error: {results['error']}")
            continue

        print(f"  Recall:    {results['recall']:.3f}")
        print(f"  Precision: {results['precision']:.3f}")
        print(f"  F1:        {results['f1']:.3f}")
        print(f"  PR-AUC:    {results['pr_auc']:.3f}")
        print(f"  Avg lead time: {results['avg_detection_lead_time_weeks']:.1f} weeks")

        all_results[str(sub)] = results

    # Save results
    models_path = Path(config["paths"]["models"])
    models_path.mkdir(parents=True, exist_ok=True)
    results_path = models_path / "eval_results.json"

    # Convert numpy types for JSON serialization
    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
