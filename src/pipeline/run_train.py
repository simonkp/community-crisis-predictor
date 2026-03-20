import argparse
import json
from pathlib import Path

from src.config import load_config
from src.collector.storage import load_processed
from src.core.ui_config import PIPELINE_COPY
from src.modeling.evaluate import evaluate_walk_forward, evaluate_walk_forward_lstm
from src.labeling.target import STATE_NAMES


def main():
    parser = argparse.ArgumentParser(description=PIPELINE_COPY["run_train_description"])
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--skip-search", action="store_true", help="Skip XGBoost hyperparameter search"
    )
    parser.add_argument(
        "--skip-lstm", action="store_true", help="Skip LSTM training (faster)"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Loading feature matrix...")
    feature_df = load_processed(config["paths"]["features"], "features")
    print(f"  {feature_df.shape[0]} weeks x {feature_df.shape[1]} features")

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in feature_df.columns if c not in meta_cols]

    all_results: dict = {}

    for sub, sub_df in feature_df.groupby("subreddit"):
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        print(f"\n{'='*50}")
        print(f"Evaluating r/{sub} ({len(sub_df)} weeks)...")
        print("=" * 50)

        # --- XGBoost baseline ---
        print(f"\n{PIPELINE_COPY['xgb_section_title']}")
        xgb_results = evaluate_walk_forward(
            sub_df, config, feature_columns, skip_search=args.skip_search
        )
        if "error" in xgb_results:
            print(f"  XGBoost error: {xgb_results['error']}")
            xgb_results = {}

        # --- LSTM primary ---
        lstm_results: dict = {}
        if not args.skip_lstm:
            print("\n[LSTM — 4-class primary model]")
            lstm_results = evaluate_walk_forward_lstm(sub_df, config, feature_columns)
            if "error" in lstm_results:
                print(f"  LSTM error: {lstm_results['error']}")
                lstm_results = {}

        # --- Comparison table ---
        _print_comparison(sub, xgb_results, lstm_results)

        all_results[str(sub)] = {"xgb": xgb_results, "lstm": lstm_results}

    # Save results
    models_path = Path(config["paths"]["models"])
    models_path.mkdir(parents=True, exist_ok=True)
    results_path = models_path / "eval_results.json"

    def convert(obj):
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)

    print(f"\nResults saved to {results_path}")


def _print_comparison(sub: str, xgb: dict, lstm: dict) -> None:
    print(f"\n{'-'*52}")
    print(f"  Model comparison - r/{sub}")
    print(f"  {'Metric':<28} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'-'*48}")

    metrics = ["recall", "precision", "f1", "pr_auc", "avg_detection_lead_time_weeks"]
    labels = [PIPELINE_COPY["recall_metric_label"], "Precision", "F1", "PR-AUC", "Avg lead time (weeks)"]
    for key, label in zip(metrics, labels):
        xv = xgb.get(key, float("nan"))
        lv = lstm.get(key, float("nan"))
        xv_str = f"{xv:.3f}" if isinstance(xv, float) and not (xv != xv) else "—"
        lv_str = f"{lv:.3f}" if isinstance(lv, float) and not (lv != lv) else "—"
        print(f"  {label:<28} {xv_str:>10} {lv_str:>10}")

    if lstm:
        print(f"\n  LSTM per-class recall:")
        for cls in range(4):
            val = lstm.get(f"recall_class_{cls}", float("nan"))
            val_str = f"{val:.3f}" if isinstance(val, float) and not (val != val) else "—"
            print(f"    Class {cls} ({STATE_NAMES[cls]:<22}): {val_str}")

    print(f"{'-'*52}")


if __name__ == "__main__":
    main()
