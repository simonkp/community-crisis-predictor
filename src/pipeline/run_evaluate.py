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
from src.narration.narrative_generator import generate_weekly_briefs_for_subreddit
from src.modeling.eda import (
    generate_pre_training_eda,
    generate_fold_diagnostics_eda,
    generate_post_training_eda,
    write_modelling_eda_html,
)
from src.modeling.evaluate import (
    evaluate_cross_subreddit_generalization,
    evaluate_feature_family_ablation,
)
from src.modeling.granger import compute_granger_causality, save_granger_report
from src.modeling.label_audit import audit_labels_with_llm, save_audit_report
from src.prescriptive.lp_allocator import run_allocation, format_allocation_text, save_allocation_report

# Presentation artifact legend:
# - Input artifacts     -> data/features/features.parquet, data/models/eval_results.json
# - Per-subreddit outputs under data/reports/{sub}/:
#   - shap.csv, drift_alerts.json, weekly_briefs.json
#   - timeline.html, feature_importance.html, dashboard.html
#   - case_studies/case_study_*.md
# - Cross-subreddit output -> data/alerts.db (transition log)


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

    alerts_db_path = config.get("paths", {}).get("alerts_db", "data/alerts.db")
    alert_engine = AlertEngine(db_path=alerts_db_path)
    drift_detector = DriftDetector(baseline_weeks=12)

    # Evaluate/report generation is artifact-oriented:
    # each subreddit gets its own report folder under paths.reports/{sub}/.
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
        sub_reports_path = reports_path / sub
        sub_reports_path.mkdir(parents=True, exist_ok=True)
        sub_df = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

        # Distress scores + labeler
        distress_scores = compute_distress_score(sub_df, weights)
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        labeler.fit(distress_scores)

        # --- Drift detection ---
        # Output artifact: data/reports/{sub}/drift_alerts.json
        drift_df = drift_detector.detect(sub_df)
        drift_path = sub_reports_path / "drift_alerts.json"
        drift_df.to_json(drift_path, orient="records", indent=2)
        print(f"  Drift alerts: {drift_path}")

        # --- Alert engine: log actual state escalations (ground truth) ---
        # Output artifact: data/alerts.db (state transition timeline)
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
        # SHAP is recomputed here for explanation artifacts, independent of walk-forward folds.
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

        # Save SHAP for dashboard and serving model-info endpoint.
        # Output artifact: data/reports/{sub}/shap.csv
        shap_csv = sub_reports_path / "shap.csv"
        shap_df.to_csv(shap_csv, index=False)
        print(f"  SHAP: {shap_csv}")

        # --- Weekly narratives (structured context + optional LLM) ---
        # Stored as one JSON file per subreddit: weekly_briefs.json (week-keyed entries).
        preds_for_brief = np.array(per_week.get("predictions", []))
        n_brief, _ = generate_weekly_briefs_for_subreddit(
            sub,
            sub_df,
            distress_scores,
            preds_for_brief,
            shap_df,
            reports_path,
        )
        print(f"  Weekly briefs: {n_brief} week entries in {sub_reports_path / 'weekly_briefs.json'}")

        # --- Timeline ---
        # Output artifact: data/reports/{sub}/timeline.html
        timeline_path = sub_reports_path / "timeline.html"
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
        # Output artifact: data/reports/{sub}/feature_importance.html
        importance_path = sub_reports_path / "feature_importance.html"
        plot_feature_importance(shap_df, top_n=20, output_path=importance_path)
        print(f"  Feature importance: {importance_path}")

        # --- Case studies ---
        # Output artifact: data/reports/{sub}/case_studies/case_study_*.md
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
        case_studies_dir = sub_reports_path / "case_studies"
        case_studies_dir.mkdir(parents=True, exist_ok=True)
        for i, crisis_idx in enumerate(correct_crisis[:3]):
            cs_path = case_studies_dir / f"case_study_{i + 1}.md"
            generator = CaseStudyGenerator(sub_df, distress_scores, results, shap_df)
            generator.generate(crisis_idx, cs_path)
            case_study_paths.append(cs_path)
            print(f"  Case study {i + 1}: {cs_path}")

        # --- Dashboard HTML ---
        # Output artifact: data/reports/{sub}/dashboard.html
        dashboard_path = sub_reports_path / "dashboard.html"
        generate_html_report(
            timeline_path, importance_path, case_study_paths, results, dashboard_path
        )
        print(f"  Dashboard: {dashboard_path}")

        # ── Modelling EDA (three-stage) ─────────────────────────────────────
        modelling_eda_dir = sub_reports_path / "modelling_eda"
        modelling_eda_dir.mkdir(parents=True, exist_ok=True)

        # Stage 1: Pre-training
        try:
            pre_eda = generate_pre_training_eda(
                feature_df=sub_df,
                labels=labeler.label(distress_scores),
                feature_columns=feature_columns,
                subreddit=sub,
                output_dir=modelling_eda_dir,
            )
        except Exception as e:
            print(f"  Warning: pre-training EDA failed for {sub}: {e}")
            pre_eda = {"subreddit": sub}

        # Stage 2: Fold diagnostics (from both models; prefer LSTM if available)
        xgb_results = sub_results.get("xgb", {})
        lstm_results = sub_results.get("lstm", {})
        fold_records_combined = (
            lstm_results.get("fold_records") or xgb_results.get("fold_records") or []
        )
        try:
            fold_eda = generate_fold_diagnostics_eda(
                fold_records=fold_records_combined,
                subreddit=sub,
                output_dir=modelling_eda_dir,
            )
        except Exception as e:
            print(f"  Warning: fold diagnostics EDA failed for {sub}: {e}")
            fold_eda = {"subreddit": sub}

        # Stage 3: Post-training diagnostics for XGBoost
        post_eda_xgb = None
        xgb_pw = xgb_results.get("per_week", {})
        if xgb_pw.get("actuals") and xgb_pw.get("probabilities"):
            try:
                import numpy as _np
                post_eda_xgb = generate_post_training_eda(
                    y_true=_np.array(xgb_pw["actuals"]),
                    y_prob=_np.array(xgb_pw["probabilities"]),
                    y_pred=_np.array(xgb_pw["predictions"]),
                    subreddit=sub,
                    output_dir=modelling_eda_dir,
                    model_name="xgb",
                )
            except Exception as e:
                print(f"  Warning: post-training XGBoost EDA failed for {sub}: {e}")

        # Stage 3: Post-training diagnostics for LSTM
        post_eda_lstm = None
        lstm_pw = lstm_results.get("per_week", {})
        if lstm_pw.get("actuals") and lstm_pw.get("probabilities"):
            try:
                import numpy as _np
                # LSTM actuals are 4-class; binarise for post-training EDA
                lstm_actuals_bin = (_np.array(lstm_pw["actuals"]) >= 2).astype(float)
                lstm_preds_bin = (_np.array(lstm_pw["predictions"]) >= 2).astype(float)
                post_eda_lstm = generate_post_training_eda(
                    y_true=lstm_actuals_bin,
                    y_prob=_np.array(lstm_pw["probabilities"]),
                    y_pred=lstm_preds_bin,
                    subreddit=sub,
                    output_dir=modelling_eda_dir,
                    model_name="lstm",
                )
            except Exception as e:
                print(f"  Warning: post-training LSTM EDA failed for {sub}: {e}")

        # Write combined HTML
        try:
            eda_html_path = modelling_eda_dir / "modelling_eda_summary.html"
            write_modelling_eda_html(
                pre_eda=pre_eda,
                fold_eda=fold_eda,
                post_eda_xgb=post_eda_xgb,
                post_eda_lstm=post_eda_lstm,
                output_path=eda_html_path,
            )
        except Exception as e:
            print(f"  Warning: modelling EDA HTML failed for {sub}: {e}")

    # ── Cross-subreddit analyses (require ≥2 subreddits) ──────────────────────
    subreddits_with_data = list(all_results.keys())
    all_sub_dfs: dict[str, pd.DataFrame] = {}
    weekly_dfs_by_sub: dict[str, pd.DataFrame] = {}
    for sub in subreddits_with_data:
        sub_df = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        if not sub_df.empty:
            all_sub_dfs[sub] = sub_df

    # --- #10: Cross-subreddit generalization ---
    # Output: data/reports/cross_subreddit_generalization.json
    if len(all_sub_dfs) >= 2:
        print("\nRunning cross-subreddit generalization evaluation...")
        try:
            gen_results = evaluate_cross_subreddit_generalization(
                all_sub_dfs, config, feature_columns
            )
            gen_path = reports_path / "cross_subreddit_generalization.json"
            with open(gen_path, "w", encoding="utf-8") as f:
                json.dump(gen_results, f, indent=2)
            print(f"  Cross-subreddit generalization: {gen_path}")
        except Exception as e:
            print(f"  Warning: cross-subreddit generalization failed: {e}")

    # --- #8: Feature family ablation (per subreddit) ---
    # Output: data/reports/{sub}/ablation_study.json
    print("\nRunning feature family ablation studies...")
    for sub, sub_df in all_sub_dfs.items():
        try:
            ablation = evaluate_feature_family_ablation(sub_df, config, feature_columns)
            ablation_path = reports_path / sub / "ablation_study.json"
            with open(ablation_path, "w", encoding="utf-8") as f:
                json.dump(ablation, f, indent=2)
            print(f"  Ablation study for r/{sub}: {ablation_path}")
        except Exception as e:
            print(f"  Warning: ablation study failed for r/{sub}: {e}")

    # --- #9: Granger causality (requires distress_score column on feature_df) ---
    # Output: data/reports/granger_causality.json
    if len(all_sub_dfs) >= 2:
        print("\nRunning Granger causality analysis...")
        try:
            distress_col = "distress_density"  # proxy; present in all feature matrices
            if distress_col in feature_df.columns:
                granger_results = compute_granger_causality(
                    feature_df, subreddits_with_data, distress_col=distress_col, max_lag=4
                )
                granger_path = reports_path / "granger_causality.json"
                save_granger_report(granger_results, granger_path)
                print(f"  Granger causality: {granger_path}")
            else:
                print(f"  Skipping Granger causality: column '{distress_col}' not in feature matrix")
        except Exception as e:
            print(f"  Warning: Granger causality analysis failed: {e}")

    # --- #14: LLM-assisted label auditing (per subreddit, best-effort) ---
    # Output: data/reports/{sub}/label_audit.json
    # Only runs if an API key is available; skips silently otherwise.
    print("\nRunning LLM label auditing (best-effort)...")
    llm_provider = config.get("evaluation", {}).get("label_audit_provider", "anthropic")
    for sub, sub_df in all_sub_dfs.items():
        try:
            sub_scores = compute_distress_score(sub_df, weights)
            sub_labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
            sub_labeler.fit(sub_scores)
            sub_labels = sub_labeler.label(sub_scores)

            # Load weekly_df for this subreddit (needs `texts` column)
            processed_path = Path(config["paths"]["processed_data"]) / "weekly.parquet"
            if processed_path.exists():
                weekly_full = pd.read_parquet(processed_path, engine="pyarrow")
                weekly_sub = weekly_full[weekly_full["subreddit"] == sub].copy()
                weekly_sub = weekly_sub.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
                audit = audit_labels_with_llm(
                    weekly_sub, sub_labels, subreddit=sub,
                    sample_size=10, provider=llm_provider,
                )
                audit_path = reports_path / sub / "label_audit.json"
                save_audit_report(audit, audit_path)
                rate = audit.get("agreement_rate")
                if rate is not None:
                    print(f"  Label audit for r/{sub}: agreement={rate:.0%} ({audit_path})")
                else:
                    print(f"  Label audit for r/{sub}: {audit.get('status', 'unknown')} ({audit_path})")
        except Exception as e:
            print(f"  Warning: label audit skipped for r/{sub}: {e}")

    # --- #7: Prescriptive LP — moderator resource allocation ---
    # Output: data/reports/allocation.json
    print("\nRunning prescriptive LP allocation...")
    try:
        allocation_report = run_allocation(all_results, config)
        allocation_path = reports_path / "allocation.json"
        save_allocation_report(allocation_report, allocation_path)
        print(format_allocation_text(allocation_report))
        print(f"  Allocation report: {allocation_path}")
    except Exception as e:
        print(f"  Warning: LP allocation failed: {e}")

    print("\nVisualization complete.")
    print(f"Alert transitions logged to: {alerts_db_path}")


if __name__ == "__main__":
    main()
