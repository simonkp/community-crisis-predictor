import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import load_config
from src.collector.storage import load_processed
from src.core.ui_config import PIPELINE_COPY
from src.modeling.evaluate import compute_decision_usefulness, evaluate_walk_forward, evaluate_walk_forward_lstm
from src.labeling.target import STATE_NAMES

# Presentation artifact legend:
# - Input artifact      -> data/features/features.parquet
# - Model artifacts     -> data/models/{sub}_xgb.pkl, {sub}_lstm.pt
# - Drift stats         -> data/models/{sub}_feature_stats.json
# - Metrics artifact    -> data/models/eval_results.json


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
    try:
        feature_df = load_processed(config["paths"]["features"], "features")
    except FileNotFoundError:
        print(
            "\nERROR: Feature matrix not found. Run the features stage first:\n"
            "  python -m src.pipeline.run_features --config config/default.yaml\n"
            "Or run the full pipeline:\n"
            "  python -m src.pipeline.run_all --config config/default.yaml --synthetic"
        )
        raise SystemExit(1)
    if feature_df.empty:
        print("\nERROR: Feature matrix is empty — no weeks to train on.")
        raise SystemExit(1)
    print(f"  {feature_df.shape[0]} weeks x {feature_df.shape[1]} features")

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in feature_df.columns if c not in meta_cols]

    # Ensure models directory exists before the loop so save_dir is valid
    models_path = Path(config["paths"]["models"])
    models_path.mkdir(parents=True, exist_ok=True)

    all_results: dict = {}

    # Train/evaluate independently per subreddit so artifacts and metrics remain
    # community-specific and easy to explain in demos.
    for sub, sub_df in feature_df.groupby("subreddit"):
        sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        print(f"\n{'='*50}")
        print(f"Evaluating r/{sub} ({len(sub_df)} weeks)...")
        print("=" * 50)
        selected_features = _select_features_for_subreddit(config, str(sub), feature_columns)
        print(f"  Feature selection: {len(selected_features)}/{len(feature_columns)} features")

        # --- XGBoost baseline ---
        # Produces binary crisis metrics and persists {sub}_xgb.pkl + feature stats.
        print(f"\n{PIPELINE_COPY['xgb_section_title']}")
        xgb_results = evaluate_walk_forward(
            sub_df,
            config,
            selected_features,
            skip_search=args.skip_search,
            save_dir=models_path,
            sub=str(sub),
        )
        if "error" in xgb_results:
            print(f"  XGBoost error: {xgb_results['error']}")
            xgb_results = {}

        # --- LSTM primary ---
        # Produces 4-class sequential metrics and persists {sub}_lstm.pt.
        lstm_results: dict = {}
        if not args.skip_lstm:
            print("\n[LSTM — 4-class primary model]")
            lstm_results = evaluate_walk_forward_lstm(
                sub_df,
                config,
                selected_features,
                save_dir=models_path,
                sub=str(sub),
            )
            if "error" in lstm_results:
                print(f"  LSTM error: {lstm_results['error']}")
                lstm_results = {}
        ensemble_results = _build_ensemble_results(xgb_results, lstm_results, config)

        # --- Comparison table ---
        _print_comparison(sub, xgb_results, lstm_results)
        if ensemble_results:
            print(f"  Ensemble: PR-AUC={ensemble_results.get('pr_auc', float('nan')):.3f} "
                  f"Recall={ensemble_results.get('recall', float('nan')):.3f}")

        # Persist both model families in one payload; evaluate stage can prefer LSTM
        # while keeping XGB as an explicit baseline for comparison.
        all_results[str(sub)] = {
            "xgb": xgb_results,
            "lstm": lstm_results,
            "ensemble": ensemble_results,
            "selected_features": selected_features,
        }

    _print_section3_summary(all_results)

    # Presentation checkpoint: consolidated metrics artifact
    # Path: data/models/eval_results.json (consumed by dashboard + serving API).
    # eval_results.json is the canonical metrics artifact consumed by dashboard/API.
    # Keep serialization robust for numpy scalar/array types.
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


def _select_features_for_subreddit(config: dict, subreddit: str, feature_columns: list[str]) -> list[str]:
    fs_cfg = config.get("modeling", {}).get("feature_selection", {})
    if not bool(fs_cfg.get("enabled", True)):
        return list(feature_columns)
    ratio = float(fs_cfg.get("shap_min_ratio_to_top", 0.01))
    min_features = int(fs_cfg.get("min_features", 20))
    max_features = int(fs_cfg.get("max_features", len(feature_columns)))
    reports_root = Path(config.get("paths", {}).get("reports", "data/reports"))
    shap_path = reports_root / subreddit / "shap.csv"
    if not shap_path.exists():
        return list(feature_columns)
    try:
        import pandas as pd

        shap_df = pd.read_csv(shap_path)
    except Exception:
        return list(feature_columns)
    if shap_df.empty or "feature" not in shap_df.columns or "mean_abs_shap" not in shap_df.columns:
        return list(feature_columns)
    shap_df = shap_df[shap_df["feature"].isin(feature_columns)].copy()
    if shap_df.empty:
        return list(feature_columns)
    top_shap = float(shap_df["mean_abs_shap"].max())
    threshold = top_shap * ratio
    keep_df = shap_df[shap_df["mean_abs_shap"] >= threshold].copy()
    if len(keep_df) < min_features:
        keep_df = shap_df.nlargest(min_features, "mean_abs_shap")
    keep_df = keep_df.nlargest(max_features, "mean_abs_shap")
    keep = keep_df["feature"].tolist()
    return keep if keep else list(feature_columns)


def _build_ensemble_results(xgb: dict, lstm: dict, config: dict) -> dict:
    if not xgb or not lstm:
        return {}
    xgb_pw = xgb.get("per_week", {})
    lstm_pw = lstm.get("per_week", {})
    if not xgb_pw or not lstm_pw:
        return {}
    xgb_probs = np.asarray(xgb_pw.get("probabilities", []), dtype=float)
    lstm_probs = np.asarray(lstm_pw.get("probabilities", []), dtype=float)
    xgb_actual = np.asarray(xgb_pw.get("actuals", []), dtype=float)
    lstm_actual = np.asarray(lstm_pw.get("actuals", []), dtype=float)
    n = max(len(xgb_probs), len(lstm_probs))
    if n == 0:
        return {}

    def _pad(arr: np.ndarray, n_out: int) -> np.ndarray:
        if len(arr) >= n_out:
            return arr[:n_out]
        return np.concatenate([arr, np.full(n_out - len(arr), np.nan)])

    xgb_probs = _pad(xgb_probs, n)
    lstm_probs = _pad(lstm_probs, n)
    xgb_actual = _pad(xgb_actual, n)
    lstm_actual = _pad(lstm_actual, n)
    ens_prob = np.nanmean(np.vstack([xgb_probs, lstm_probs]), axis=0)
    actual_4 = np.where(np.isfinite(lstm_actual), lstm_actual, np.where(np.isfinite(xgb_actual), xgb_actual * 2, np.nan))
    valid = ~(np.isnan(ens_prob) | np.isnan(actual_4))
    if valid.sum() < 5:
        return {}
    prob_threshold = float(config.get("evaluation", {}).get("probability_threshold", 0.5))
    y_true_4 = actual_4[valid].astype(int)
    y_true_bin = (y_true_4 >= 2).astype(int)
    y_prob = ens_prob[valid]
    y_pred_bin = (y_prob >= prob_threshold).astype(int)
    y_pred_4 = np.where(y_pred_bin == 1, 2, 0)
    result = {
        "n_valid_predictions": int(valid.sum()),
        "n_crisis_actual": int(y_true_bin.sum()),
        "n_crisis_predicted": int(y_pred_bin.sum()),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_bin, y_prob)) if y_true_bin.sum() > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true_bin, y_prob))
        if (y_true_bin.sum() > 0 and (y_true_bin == 0).sum() > 0)
        else None,
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist(),
        "confusion_matrix_4class": confusion_matrix(y_true_4, y_pred_4, labels=[0, 1, 2, 3]).tolist(),
        "decision_usefulness": compute_decision_usefulness(y_true_bin, y_prob),
        "per_week": {
            "predictions": np.where(np.isnan(ens_prob), np.nan, np.where(ens_prob >= prob_threshold, 2, 0)).tolist(),
            "probabilities": ens_prob.tolist(),
            "actuals": actual_4.tolist(),
        },
    }
    return result


def _print_comparison(sub: str, xgb: dict, lstm: dict) -> None:
    print(f"\n{'-'*52}")
    print(f"  Model comparison - r/{sub}")
    print(f"  {'Metric':<28} {'XGBoost':>10} {'LSTM':>10}")
    print(f"  {'-'*48}")

    metrics = ["recall", "precision", "f1", "pr_auc", "roc_auc", "avg_detection_lead_time_weeks"]
    labels = [PIPELINE_COPY["recall_metric_label"], "Precision", "F1", "PR-AUC", "ROC-AUC", "Avg lead time (weeks)"]
    for key, label in zip(metrics, labels):
        xv = xgb.get(key, float("nan"))
        lv = lstm.get(key, float("nan"))
        xv_str = f"{xv:.3f}" if isinstance(xv, float) and xv == xv else "—"
        lv_str = f"{lv:.3f}" if isinstance(lv, float) and lv == lv else "—"
        print(f"  {label:<28} {xv_str:>10} {lv_str:>10}")

    if lstm:
        print(f"\n  LSTM per-class recall:")
        for cls in range(4):
            val = lstm.get(f"recall_class_{cls}", float("nan"))
            val_str = f"{val:.3f}" if isinstance(val, float) and not (val != val) else "—"
            print(f"    Class {cls} ({STATE_NAMES[cls]:<22}): {val_str}")
    _print_anomalies(xgb, lstm)

    print(f"{'-'*52}")


def _print_anomalies(xgb: dict, lstm: dict) -> None:
    anomalies: list[str] = []
    for model_name, metrics in (("XGBoost", xgb), ("LSTM", lstm)):
        if not metrics:
            anomalies.append(f"{model_name}: missing metrics payload")
            continue
        n_crisis = metrics.get("n_crisis_actual")
        if isinstance(n_crisis, (int, float)) and n_crisis < 10:
            anomalies.append(f"{model_name}: low crisis support ({int(n_crisis)} weeks)")
        recall = metrics.get("recall")
        if isinstance(recall, (int, float)) and recall <= 0.0:
            anomalies.append(f"{model_name}: recall is zero")
        pr_auc = metrics.get("pr_auc")
        if isinstance(pr_auc, (int, float)) and pr_auc < 0.12:
            anomalies.append(f"{model_name}: PR-AUC very low ({pr_auc:.3f})")
        roc_auc = metrics.get("roc_auc")
        if isinstance(roc_auc, float) and roc_auc < 0.5:
            anomalies.append(
                f"{model_name}: ROC-AUC below random ({roc_auc:.3f}) — "
                "model may need more training data or real temporal patterns"
            )
    if anomalies:
        print("\n  Anomaly flags:")
        for a in anomalies:
            print(f"    - {a}")


def _print_section3_summary(all_results: dict) -> None:
    print("\n" + "=" * 72)
    print("[3/4] Section Summary Table")
    print("=" * 72)
    hdr = (
        f"  {'subreddit':<15} {'xgb_pr_auc':>10} {'xgb_recall':>10} "
        f"{'lstm_pr_auc':>11} {'lstm_recall':>12} {'alerts':>8}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for sub, payload in sorted(all_results.items(), key=lambda kv: kv[0]):
        xgb = payload.get("xgb", {}) if isinstance(payload, dict) else {}
        lstm = payload.get("lstm", {}) if isinstance(payload, dict) else {}
        xa = xgb.get("pr_auc", float("nan"))
        xr = xgb.get("recall", float("nan"))
        la = lstm.get("pr_auc", float("nan"))
        lr = lstm.get("recall", float("nan"))
        alerts = 0
        if isinstance(xgb.get("n_crisis_actual"), (int, float)) and xgb.get("n_crisis_actual", 0) < 10:
            alerts += 1
        if isinstance(lstm.get("n_crisis_actual"), (int, float)) and lstm.get("n_crisis_actual", 0) < 10:
            alerts += 1
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) and v == v else "—"
        print(
            f"  {sub:<15} {fmt(xa):>10} {fmt(xr):>10} "
            f"{fmt(la):>11} {fmt(lr):>12} {alerts:>8}"
        )

    _print_performance_bands(all_results)


def _print_performance_bands(all_results: dict) -> None:
    """
    Classify communities into High / Medium / Low performer bands.
    Mirrors the segment-level bias analysis from L3.2 / L4.0:
    - Which subreddits the model detects crises most reliably?
    - Which are hardest to predict (low PR-AUC) and need more attention?
    - Which model family (XGB vs LSTM) wins per community?
    """
    HIGH_THRESHOLD = 0.45   # PR-AUC >= 0.45  -> High performance
    LOW_THRESHOLD = 0.20    # PR-AUC <  0.20  -> Low performance

    rows = []
    for sub, payload in all_results.items():
        xgb = payload.get("xgb", {}) if isinstance(payload, dict) else {}
        lstm = payload.get("lstm", {}) if isinstance(payload, dict) else {}
        xa = xgb.get("pr_auc") if isinstance(xgb.get("pr_auc"), float) else float("nan")
        la = lstm.get("pr_auc") if isinstance(lstm.get("pr_auc"), float) else float("nan")
        # Best PR-AUC across model families
        best = max((v for v in [xa, la] if v == v), default=float("nan"))
        winner = "xgb" if (xa == xa and la != la) or (xa == xa and la == la and xa >= la) else "lstm"
        if la == la and xa == xa:
            winner = "lstm" if la > xa else "xgb"
        elif la == la:
            winner = "lstm"
        elif xa == xa:
            winner = "xgb"
        else:
            winner = "—"
        rows.append({"sub": sub, "best_pr_auc": best, "xgb_pr_auc": xa, "lstm_pr_auc": la, "winner": winner})

    if not rows:
        return

    rows.sort(key=lambda r: r["best_pr_auc"] if r["best_pr_auc"] == r["best_pr_auc"] else -1, reverse=True)

    high = [r for r in rows if r["best_pr_auc"] == r["best_pr_auc"] and r["best_pr_auc"] >= HIGH_THRESHOLD]
    low  = [r for r in rows if r["best_pr_auc"] == r["best_pr_auc"] and r["best_pr_auc"] < LOW_THRESHOLD]
    mid  = [r for r in rows if r not in high and r not in low]

    print("\n" + "=" * 72)
    print("[4/4] High vs Low Performance Identification")
    print("      (by best PR-AUC across XGB + LSTM)")
    print("=" * 72)

    def _band_block(label: str, band_rows: list[dict]) -> None:
        if not band_rows:
            return
        print(f"\n  [{label}]")
        for r in band_rows:
            fmt = lambda v: f"{v:.3f}" if v == v else "—"
            note = f"best model: {r['winner']}"
            print(f"    r/{r['sub']:<15}  PR-AUC={fmt(r['best_pr_auc'])}  ({note})")

    _band_block("HIGH performers  - model reliably detects crises", high)
    _band_block("MEDIUM performers - moderate signal quality",        mid)
    _band_block("LOW performers   - crisis signal hard to detect",    low)

    # Cross-learning recommendations grounded in the data
    print("\n  Cross-learning recommendations:")
    if high and low:
        high_subs = ", ".join(f"r/{r['sub']}" for r in high)
        low_subs  = ", ".join(f"r/{r['sub']}" for r in low)
        print(f"    - Features driving high performance in {high_subs} should be")
        print(f"      reviewed for applicability in {low_subs}.")
        print(f"      Check SHAP reports (data/reports/{{sub}}/shap.csv) to identify")
        print(f"      which features contribute most to correct crisis predictions.")
    lstm_wins = [r for r in rows if r["winner"] == "lstm"]
    xgb_wins  = [r for r in rows if r["winner"] == "xgb"]
    if lstm_wins and xgb_wins:
        lstm_subs = ", ".join(f"r/{r['sub']}" for r in lstm_wins)
        xgb_subs  = ", ".join(f"r/{r['sub']}" for r in xgb_wins)
        print(f"    - LSTM outperforms XGB in {lstm_subs}: sequential week-over-week")
        print(f"      context captures escalation patterns better for these communities.")
        print(f"    - XGB outperforms LSTM in {xgb_subs}: tabular weekly features are")
        print(f"      sufficient; long-term memory not needed (or limited training data).")
    if low:
        low_subs = ", ".join(f"r/{r['sub']}" for r in low)
        print(f"    - Priority for data quality review: {low_subs}.")
        print(f"      Consider expanding feature set or collecting more crisis weeks.")
    print("  " + "-" * 68)


if __name__ == "__main__":
    main()
