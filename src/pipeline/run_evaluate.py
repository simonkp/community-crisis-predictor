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
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.alert_engine import AlertEngine
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
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])

    alert_engine = AlertEngine(db_path="data/alerts.db")
    drift_detector = DriftDetector(baseline_weeks=12)

    for sub, sub_results in all_results.items():
        # Handle new {"xgb": ..., "lstm": ...} format — prefer LSTM for visualization
        if "lstm" in sub_results or "xgb" in sub_results:
            results = sub_results.get("lstm") or sub_results.get("xgb", {})
        else:
            results = sub_results  # backward compat: flat dict

        if not results or "error" in results:
            print(f"  Skipping r/{sub}: no valid results")
            continue

        print(f"\nGenerating visualizations for r/{sub}...")
        sub_df = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

        # Distress scores + labeler
        distress_scores = compute_distress_score(sub_df, weights)
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        labeler.fit(distress_scores)

        # --- Drift detection ---
        drift_df = drift_detector.detect(sub_df)
        drift_path = reports_path / f"{sub}_drift_alerts.json"
        drift_df.to_json(drift_path, orient="records", indent=2)
        print(f"  Drift alerts: {drift_path}")

        # --- Alert engine: log actual state escalations (ground truth) ---
        per_week = results.get("per_week", {})
        # Use actual 4-class labels so the DB reflects real community transitions
        actual_states_raw = per_week.get("actuals", [])
        weekly_scores_raw = per_week.get("probabilities", [])
        if actual_states_raw:
            alert_engine.process_week_sequence(
                subreddit=sub,
                weekly_states=[
                    int(s) if not (isinstance(s, float) and np.isnan(s)) else None
                    for s in actual_states_raw
                ],
                weekly_scores=[
                    float(p) if not (isinstance(p, float) and np.isnan(p)) else 0.0
                    for p in weekly_scores_raw
                ],
                feature_df=sub_df,
            )

        # --- Train XGB on full data for SHAP (binary labels) ---
        valid_labels = labeler.label(distress_scores)
        valid_mask = ~valid_labels.isna()
        X_full = sub_df.loc[valid_mask, feature_columns]
        y_full_4class = valid_labels[valid_mask].astype(int)
        y_full = (y_full_4class >= 2).astype(int)  # binary for SHAP via XGB

        if len(y_full) < 10 or y_full.sum() < 2:
            print(f"  Insufficient data for r/{sub}")
            continue

        xgb_model = XGBCrisisModel(config)
        xgb_model.train(X_full, y_full, do_search=False)

        shap_df = compute_shap_importance(xgb_model, X_full, feature_columns)

        # Save SHAP for dashboard
        shap_csv = reports_path / f"{sub}_shap.csv"
        shap_df.to_csv(shap_csv, index=False)
        print(f"  SHAP: {shap_csv}")

        # --- Timeline ---
        timeline_path = reports_path / f"{sub}_timeline.html"
        plot_backtest_timeline(
            sub_df,
            distress_scores,
            results,
            labeler.threshold,
            timeline_path,
            thresholds=labeler.thresholds,
        )
        print(f"  Timeline: {timeline_path}")

        # --- Feature importance ---
        importance_path = reports_path / f"{sub}_feature_importance.html"
        plot_feature_importance(shap_df, top_n=20, output_path=importance_path)
        print(f"  Feature importance: {importance_path}")

        # --- Case studies ---
        predictions = np.array(per_week.get("predictions", []))
        actuals = np.array(per_week.get("actuals", []))

        # Crisis = state >= 2 (for LSTM) or == 1 (for XGB binary)
        if predictions.max() > 1:  # LSTM multiclass
            pred_crisis = predictions >= 2
            actual_crisis = actuals >= 2
        else:
            pred_crisis = predictions == 1
            actual_crisis = actuals == 1

        correct_crisis = np.where(pred_crisis & actual_crisis)[0]

        case_study_paths = []
        for i, crisis_idx in enumerate(correct_crisis[:3]):
            cs_path = reports_path / f"{sub}_case_study_{i + 1}.md"
            generator = CaseStudyGenerator(sub_df, distress_scores, results, shap_df)
            generator.generate(crisis_idx, cs_path)
            case_study_paths.append(cs_path)
            print(f"  Case study {i + 1}: {cs_path}")

        # --- Dashboard HTML ---
        dashboard_path = reports_path / f"{sub}_dashboard.html"
        generate_html_report(
            timeline_path, importance_path, case_study_paths, results, dashboard_path
        )
        print(f"  Dashboard: {dashboard_path}")

    print("\nVisualization complete.")
    print(f"Alert transitions logged to: data/alerts.db")


if __name__ == "__main__":
    main()
