import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config
from src.collector.storage import load_processed
from src.labeling.distress_score import compute_distress_score
from src.labeling.target import CrisisLabeler
from src.modeling.train_xgb import XGBCrisisModel
from src.modeling.explain import compute_shap_importance
from src.visualization.timeline import plot_backtest_timeline
from src.visualization.feature_importance import plot_feature_importance
from src.visualization.case_study import CaseStudyGenerator
from src.visualization.dashboard import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    reports_path = Path(config["paths"]["reports"])
    reports_path.mkdir(parents=True, exist_ok=True)

    # Load data
    feature_df = load_processed(config["paths"]["features"], "features")
    results_path = Path(config["paths"]["models"]) / "eval_results.json"

    if not results_path.exists():
        print("No evaluation results found. Run `make train` first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in feature_df.columns if c not in meta_cols]

    labeling_cfg = config.get("labeling", {})
    weights = labeling_cfg.get("distress_weights")
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)

    for sub, results in all_results.items():
        if "error" in results:
            continue

        print(f"\nGenerating visualizations for r/{sub}...")
        sub_df = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

        # Compute distress scores
        distress_scores = compute_distress_score(sub_df, weights)
        labeler = CrisisLabeler(threshold_std)
        labeler.fit(distress_scores)

        # Train a final model for SHAP on the full training data
        valid_labels = labeler.label(distress_scores)
        valid_mask = ~valid_labels.isna()
        X_full = sub_df.loc[valid_mask, feature_columns]
        y_full = valid_labels[valid_mask].astype(int)

        if len(y_full) < 10 or y_full.sum() < 2:
            print(f"  Insufficient data for r/{sub}")
            continue

        model = XGBCrisisModel(config)
        model.train(X_full, y_full, do_search=False)

        # SHAP importance
        shap_df = compute_shap_importance(model, X_full, feature_columns)

        # Timeline
        timeline_path = reports_path / f"{sub}_timeline.html"
        plot_backtest_timeline(
            sub_df, distress_scores, results, labeler.threshold, timeline_path
        )
        print(f"  Timeline: {timeline_path}")

        # Feature importance
        importance_path = reports_path / f"{sub}_feature_importance.html"
        plot_feature_importance(shap_df, top_n=20, output_path=importance_path)
        print(f"  Feature importance: {importance_path}")

        # Case studies — find correctly predicted crisis weeks
        predictions = np.array(results["per_week"]["predictions"])
        actuals = np.array(results["per_week"]["actuals"])
        correct_crisis = np.where(
            (predictions == 1) & (actuals == 1)
        )[0]

        case_study_paths = []
        for i, crisis_idx in enumerate(correct_crisis[:3]):
            cs_path = reports_path / f"{sub}_case_study_{i+1}.md"
            generator = CaseStudyGenerator(sub_df, distress_scores, results, shap_df)
            generator.generate(crisis_idx, cs_path)
            case_study_paths.append(cs_path)
            print(f"  Case study {i+1}: {cs_path}")

        # Dashboard
        dashboard_path = reports_path / f"{sub}_dashboard.html"
        generate_html_report(
            timeline_path, importance_path, case_study_paths, results, dashboard_path
        )
        print(f"  Dashboard: {dashboard_path}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
