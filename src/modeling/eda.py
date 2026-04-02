"""
Modelling EDA — three-stage exploratory analysis tied directly to the training pipeline.

Stage 1 — Pre-training  : feature-label correlations, class imbalance, multicollinearity
Stage 2 — Fold-level    : walk-forward fold health (class balance, size, skip rate)
Stage 3 — Post-training : calibration, confusion matrix, error timeline, threshold sweep

Each stage writes a JSON facts file and contributes to one combined HTML report at:
  data/reports/{sub}/modelling_eda/modelling_eda_summary.html
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Pre-training EDA
# ─────────────────────────────────────────────────────────────────────────────

def generate_pre_training_eda(
    feature_df: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    subreddit: str,
    output_dir: Path,
) -> dict:
    """
    Run EDA on the feature matrix before any model is trained.

    Returns a dict with: correlations, class_balance, multicollinear_pairs,
    distribution_by_state, missing_pct.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_cols = [c for c in feature_columns if c in feature_df.columns]
    X = feature_df[valid_cols].copy()

    # Binary label (crisis = state 2+3) aligned to feature_df index
    y_bin = (labels >= 2).astype(float)
    y_4 = labels.copy()

    report: dict = {"subreddit": subreddit, "n_weeks": len(X)}

    # ------------------------------------------------------------------
    # 1a. Feature-label Spearman correlations
    # ------------------------------------------------------------------
    corr_rows = []
    for col in valid_cols:
        col_vals = X[col].fillna(X[col].median())
        valid = ~(col_vals.isna() | y_bin.isna())
        if valid.sum() < 8:
            continue
        rho, pval = scipy_stats.spearmanr(col_vals[valid], y_bin[valid])
        corr_rows.append(
            {
                "feature": col,
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 6),
                "significant": bool(pval < 0.05),
                "abs_rho": round(abs(float(rho)), 4),
            }
        )
    corr_rows.sort(key=lambda r: -r["abs_rho"])
    report["feature_label_correlations"] = corr_rows
    report["top_10_features"] = [r["feature"] for r in corr_rows[:10]]

    # ------------------------------------------------------------------
    # 1b. Class imbalance profile (binary + 4-class)
    # ------------------------------------------------------------------
    valid_labels = y_4.dropna()
    n_total = len(valid_labels)
    class_counts = {str(int(c)): int((valid_labels == c).sum()) for c in range(4)}
    crisis_count = int((valid_labels >= 2).sum())
    report["class_balance"] = {
        "n_total_labeled": n_total,
        "counts_4class": class_counts,
        "crisis_weeks": crisis_count,
        "stable_weeks": int((valid_labels < 2).sum()),
        "crisis_rate": round(crisis_count / n_total, 4) if n_total > 0 else 0.0,
        "imbalance_ratio": round((n_total - crisis_count) / max(crisis_count, 1), 2),
    }

    # ------------------------------------------------------------------
    # 1c. Multicollinear feature pairs (|ρ| > 0.9)
    # ------------------------------------------------------------------
    X_filled = X.fillna(X.median())
    if len(X_filled) >= 8 and len(valid_cols) >= 2:
        corr_matrix = X_filled.corr(method="spearman")
        multi_pairs = []
        cols = list(corr_matrix.columns)
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                val = corr_matrix.loc[c1, c2]
                if abs(val) >= 0.9:
                    multi_pairs.append(
                        {"feature_a": c1, "feature_b": c2, "spearman_rho": round(float(val), 4)}
                    )
        multi_pairs.sort(key=lambda p: -abs(p["spearman_rho"]))
        report["multicollinear_pairs"] = multi_pairs[:20]
    else:
        report["multicollinear_pairs"] = []

    # ------------------------------------------------------------------
    # 1d. Distribution by crisis state (top-5 correlated features)
    # ------------------------------------------------------------------
    top5 = [r["feature"] for r in corr_rows[:5]]
    dist_by_state: dict[str, dict] = {}
    for col in top5:
        by_state: dict[str, dict] = {}
        for state in range(4):
            mask = y_4 == state
            vals = X.loc[mask, col].dropna()
            if len(vals) == 0:
                continue
            by_state[str(state)] = {
                "n": int(len(vals)),
                "mean": round(float(vals.mean()), 4),
                "std": round(float(vals.std()), 4),
                "median": round(float(vals.median()), 4),
            }
        dist_by_state[col] = by_state
    report["distribution_by_state"] = dist_by_state

    # ------------------------------------------------------------------
    # 1e. Missing value matrix
    # ------------------------------------------------------------------
    missing_pct = {col: round(float(X[col].isna().mean() * 100), 2) for col in valid_cols}
    report["missing_pct"] = missing_pct
    report["n_high_missing_features"] = int(sum(1 for v in missing_pct.values() if v > 20))

    # Save JSON
    json_path = output_dir / "pre_training_eda.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Pre-training EDA -> {json_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Fold-level EDA
# ─────────────────────────────────────────────────────────────────────────────

def generate_fold_diagnostics_eda(
    fold_records: list[dict],
    subreddit: str,
    output_dir: Path,
) -> dict:
    """
    Analyse the walk-forward fold health from accumulated fold records.

    Each fold_record should have keys:
      fold_i, n_train, n_crisis_train, crisis_rate_train, skipped, skip_reason
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ran = [r for r in fold_records if not r.get("skipped", False)]
    skipped = [r for r in fold_records if r.get("skipped", False)]

    skip_reasons: dict[str, int] = {}
    for r in skipped:
        reason = r.get("skip_reason", "unknown")
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    crisis_rates = [r["crisis_rate_train"] for r in ran if "crisis_rate_train" in r]
    train_sizes = [r["n_train"] for r in ran if "n_train" in r]

    low_balance_folds = [
        {"fold": r["fold_i"], "crisis_rate": r["crisis_rate_train"]}
        for r in ran
        if r.get("crisis_rate_train", 1.0) <= 0.05
    ]

    report: dict = {
        "subreddit": subreddit,
        "n_total_folds": len(fold_records),
        "n_ran_folds": len(ran),
        "n_skipped_folds": len(skipped),
        "skip_rate_pct": round(100 * len(skipped) / max(len(fold_records), 1), 1),
        "skip_reasons": skip_reasons,
        "crisis_rate_per_fold": [round(r, 4) for r in crisis_rates],
        "train_size_per_fold": train_sizes,
        "mean_fold_crisis_rate": round(float(np.mean(crisis_rates)), 4) if crisis_rates else 0.0,
        "min_fold_crisis_rate": round(float(np.min(crisis_rates)), 4) if crisis_rates else 0.0,
        "max_fold_crisis_rate": round(float(np.max(crisis_rates)), 4) if crisis_rates else 0.0,
        "low_balance_folds": low_balance_folds,
        "n_low_balance_folds": len(low_balance_folds),
        "mean_train_size": round(float(np.mean(train_sizes)), 1) if train_sizes else 0.0,
    }

    json_path = output_dir / "fold_diagnostics_eda.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Fold diagnostics EDA -> {json_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Post-training EDA
# ─────────────────────────────────────────────────────────────────────────────

def generate_post_training_eda(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    subreddit: str,
    output_dir: Path,
    model_name: str = "model",
) -> dict:
    """
    Post-training analysis: calibration, confusion breakdown,
    error timeline, and threshold sensitivity sweep.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(y_true) | np.isnan(y_prob) | np.isnan(y_pred))
    yt = y_true[valid].astype(int)
    yp = y_prob[valid]
    yd = y_pred[valid].astype(int)

    report: dict = {"subreddit": subreddit, "model": model_name, "n_valid": int(valid.sum())}

    # ------------------------------------------------------------------
    # 3a. Calibration curve (10 probability bins)
    # ------------------------------------------------------------------
    if len(yp) >= 20:
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        calibration_bins = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (yp >= lo) & (yp < hi) if i < n_bins - 1 else (yp >= lo) & (yp <= hi)
            if mask.sum() == 0:
                continue
            calibration_bins.append(
                {
                    "bin_low": round(float(lo), 2),
                    "bin_high": round(float(hi), 2),
                    "n": int(mask.sum()),
                    "mean_predicted": round(float(yp[mask].mean()), 4),
                    "mean_actual": round(float(yt[mask].mean()), 4),
                    "calibration_gap": round(float(yp[mask].mean() - yt[mask].mean()), 4),
                }
            )
        # Overall calibration: mean predicted vs mean actual
        overall_gap = float(yp.mean()) - float(yt.mean())
        calibration_verdict = (
            "over-confident" if overall_gap > 0.05
            else ("under-confident" if overall_gap < -0.05 else "well-calibrated")
        )
        report["calibration"] = {
            "bins": calibration_bins,
            "overall_mean_predicted": round(float(yp.mean()), 4),
            "overall_mean_actual": round(float(yt.mean()), 4),
            "overall_calibration_gap": round(overall_gap, 4),
            "verdict": calibration_verdict,
        }
    else:
        report["calibration"] = {"verdict": "insufficient_data"}

    # ------------------------------------------------------------------
    # 3b. Confusion matrix — FP / FN breakdown
    # ------------------------------------------------------------------
    tp = int(((yd == 1) & (yt == 1)).sum())
    fp = int(((yd == 1) & (yt == 0)).sum())
    fn = int(((yd == 0) & (yt == 1)).sum())
    tn = int(((yd == 0) & (yt == 0)).sum())
    total = tp + fp + fn + tn
    report["confusion_breakdown"] = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "false_alarm_rate": round(fp / max(fp + tn, 1), 4),
        "miss_rate": round(fn / max(fn + tp, 1), 4),
        "precision": round(tp / max(tp + fp, 1), 4),
        "recall": round(tp / max(tp + fn, 1), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "summary": (
            f"{tp} true alerts, {fp} false alarms, {fn} missed crises, {tn} correct non-alerts "
            f"out of {total} predicted weeks."
        ),
    }

    # ------------------------------------------------------------------
    # 3c. Error timeline — FP / FN positions by week index
    # ------------------------------------------------------------------
    week_indices = np.where(valid)[0]
    error_timeline = []
    for idx_pos, global_idx in enumerate(week_indices):
        actual, pred_label = int(yt[idx_pos]), int(yd[idx_pos])
        error_type = None
        if pred_label == 1 and actual == 0:
            error_type = "false_positive"
        elif pred_label == 0 and actual == 1:
            error_type = "false_negative"
        if error_type:
            error_timeline.append({"week_index": int(global_idx), "error_type": error_type})
    report["error_timeline"] = error_timeline
    report["n_false_positives"] = fp
    report["n_false_negatives"] = fn

    # ------------------------------------------------------------------
    # 3d. Threshold sensitivity sweep
    # ------------------------------------------------------------------
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    threshold_sweep = []
    for thresh in thresholds:
        yd_t = (yp >= thresh).astype(int)
        tp_t = int(((yd_t == 1) & (yt == 1)).sum())
        fp_t = int(((yd_t == 1) & (yt == 0)).sum())
        fn_t = int(((yd_t == 0) & (yt == 1)).sum())
        prec = tp_t / max(tp_t + fp_t, 1)
        rec = tp_t / max(tp_t + fn_t, 1)
        f1 = 2 * tp_t / max(2 * tp_t + fp_t + fn_t, 1)
        threshold_sweep.append(
            {
                "threshold": thresh,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "n_alerts": int(yd_t.sum()),
            }
        )
    report["threshold_sweep"] = threshold_sweep

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    fname = f"post_training_eda_{model_name}.json"
    json_path = output_dir / fname
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Post-training EDA ({model_name}) -> {json_path}")
    return report


# ─────────────────────────────────────────────────────────────────────────────
# HTML Report Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_modelling_eda_html(
    pre_eda: dict,
    fold_eda: dict,
    post_eda_xgb: dict | None,
    post_eda_lstm: dict | None,
    output_path: Path,
) -> None:
    """Write a self-contained HTML summary combining all three EDA stages."""

    sub = pre_eda.get("subreddit", "unknown")

    # ── Pre-training: top-10 correlation table ──────────────────────────────
    corr_rows_html = ""
    for r in pre_eda.get("feature_label_correlations", [])[:15]:
        bg = "#d9534f" if not r["significant"] else ("#5cb85c" if r["abs_rho"] >= 0.3 else "#f4f4f4")
        sig_marker = "✓" if r["significant"] else "–"
        corr_rows_html += (
            f"<tr><td>{r['feature']}</td>"
            f"<td style='text-align:center'>{r['spearman_rho']:+.4f}</td>"
            f"<td style='text-align:center;background:{bg}'>{sig_marker}</td></tr>\n"
        )

    # ── Class balance cards ─────────────────────────────────────────────────
    cb = pre_eda.get("class_balance", {})
    state_names = ["Stable", "Early Signal", "Elevated", "Severe"]
    state_colors = ["#5cb85c", "#f0ad4e", "#e67e22", "#d9534f"]
    class_bar_html = ""
    n_total = cb.get("n_total_labeled", 1)
    counts = cb.get("counts_4class", {})
    for cls in range(4):
        n = counts.get(str(cls), 0)
        pct = 100 * n / max(n_total, 1)
        class_bar_html += (
            f"<div style='margin-bottom:6px'><span style='display:inline-block;width:130px;font-size:0.85em'>"
            f"State {cls} — {state_names[cls]}</span>"
            f"<span style='display:inline-block;width:{max(pct * 2.5, 2):.0f}px;height:14px;"
            f"background:{state_colors[cls]};vertical-align:middle;border-radius:3px'></span>"
            f" <small>{n} ({pct:.1f}%)</small></div>\n"
        )

    # ── Multicollinear pairs ─────────────────────────────────────────────────
    multi_pairs = pre_eda.get("multicollinear_pairs", [])
    multi_html = ""
    if multi_pairs:
        for p in multi_pairs[:10]:
            multi_html += (
                f"<tr><td>{p['feature_a']}</td><td>{p['feature_b']}</td>"
                f"<td style='text-align:center'>{p['spearman_rho']:+.4f}</td></tr>\n"
            )
    else:
        multi_html = "<tr><td colspan='3' style='text-align:center;color:#888'>No highly correlated pairs found</td></tr>"

    # ── Fold diagnostics ─────────────────────────────────────────────────────
    fold_crisis_rates = fold_eda.get("crisis_rate_per_fold", [])
    fold_bar_html = ""
    for i, rate in enumerate(fold_crisis_rates):
        bar_color = "#d9534f" if rate <= 0.05 else "#5bc0de"
        width = max(rate * 300, 2)
        fold_bar_html += (
            f"<div style='margin-bottom:3px'><small style='display:inline-block;width:60px'>Fold {i + 1}</small>"
            f"<span style='display:inline-block;width:{width:.0f}px;height:12px;background:{bar_color};"
            f"border-radius:2px;vertical-align:middle'></span>"
            f" <small>{rate:.1%}</small></div>\n"
        )
    if not fold_bar_html:
        fold_bar_html = "<p style='color:#888'>No fold data recorded.</p>"

    skip_reasons = fold_eda.get("skip_reasons", {})
    skip_html = ""
    if skip_reasons:
        for reason, count in skip_reasons.items():
            skip_html += f"<li><code>{reason}</code>: {count} fold(s)</li>"
    else:
        skip_html = "<li>No skipped folds.</li>"

    # ── Post-training calibration ─────────────────────────────────────────
    def _calibration_table(post_eda: dict | None, label: str) -> str:
        if not post_eda or "calibration" not in post_eda:
            return f"<p>No post-training EDA for {label}.</p>"
        cal = post_eda["calibration"]
        if cal.get("verdict") == "insufficient_data":
            return f"<p>Insufficient data for calibration curve ({label}).</p>"
        verdict_color = {
            "well-calibrated": "#5cb85c",
            "over-confident": "#d9534f",
            "under-confident": "#f0ad4e",
        }.get(cal.get("verdict", ""), "#888")
        rows = ""
        for b in cal.get("bins", []):
            gap = b["calibration_gap"]
            gap_color = "#d9534f" if abs(gap) > 0.1 else "#333"
            rows += (
                f"<tr><td>[{b['bin_low']:.1f}–{b['bin_high']:.1f}]</td>"
                f"<td>{b['n']}</td>"
                f"<td>{b['mean_predicted']:.3f}</td>"
                f"<td>{b['mean_actual']:.3f}</td>"
                f"<td style='color:{gap_color}'>{gap:+.3f}</td></tr>\n"
            )
        return (
            f"<p><strong>Verdict:</strong> <span style='color:{verdict_color};font-weight:bold'>"
            f"{cal['verdict']}</span> | Mean predicted: {cal['overall_mean_predicted']:.3f} "
            f"vs actual: {cal['overall_mean_actual']:.3f} "
            f"(gap: {cal['overall_calibration_gap']:+.3f})</p>"
            "<table><tr><th>Prob Bin</th><th>N</th><th>Mean Pred</th><th>Mean Actual</th><th>Gap</th></tr>"
            f"{rows}</table>"
        )

    def _threshold_table(post_eda: dict | None, label: str) -> str:
        if not post_eda or "threshold_sweep" not in post_eda:
            return f"<p>No threshold data for {label}.</p>"
        rows = ""
        best_f1 = max((r["f1"] for r in post_eda["threshold_sweep"]), default=0)
        for r in post_eda["threshold_sweep"]:
            bold = "font-weight:bold;background:#eaf7ea" if r["f1"] == best_f1 else ""
            rows += (
                f"<tr style='{bold}'><td>{r['threshold']}</td>"
                f"<td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td>"
                f"<td>{r['f1']:.3f}</td><td>{r['n_alerts']}</td></tr>\n"
            )
        return (
            "<table><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>F1</th><th>#Alerts</th></tr>"
            f"{rows}</table>"
        )

    def _confusion_summary(post_eda: dict | None, label: str) -> str:
        if not post_eda or "confusion_breakdown" not in post_eda:
            return f"<p>No confusion data for {label}.</p>"
        cd = post_eda["confusion_breakdown"]
        return (
            f"<p>{cd['summary']}</p>"
            f"<table><tr><th></th><th>Predicted Crisis</th><th>Predicted Stable</th></tr>"
            f"<tr><td><strong>Actual Crisis</strong></td>"
            f"<td style='background:#d4edda'>{cd['tp']} TP</td>"
            f"<td style='background:#f8d7da'>{cd['fn']} FN</td></tr>"
            f"<tr><td><strong>Actual Stable</strong></td>"
            f"<td style='background:#f8d7da'>{cd['fp']} FP</td>"
            f"<td style='background:#d4edda'>{cd['tn']} TN</td></tr></table>"
            f"<p>Precision: <strong>{cd['precision']:.3f}</strong> | "
            f"Recall: <strong>{cd['recall']:.3f}</strong> | "
            f"F1: <strong>{cd['f1']:.3f}</strong> | "
            f"Miss rate: <strong>{cd['miss_rate']:.1%}</strong> | "
            f"False alarm rate: <strong>{cd['false_alarm_rate']:.1%}</strong></p>"
        )

    cal_xgb_html = _calibration_table(post_eda_xgb, "XGBoost")
    cal_lstm_html = _calibration_table(post_eda_lstm, "LSTM")
    thresh_xgb_html = _threshold_table(post_eda_xgb, "XGBoost")
    thresh_lstm_html = _threshold_table(post_eda_lstm, "LSTM")
    conf_xgb_html = _confusion_summary(post_eda_xgb, "XGBoost")
    conf_lstm_html = _confusion_summary(post_eda_lstm, "LSTM")

    # ── Assemble HTML ──────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Modelling EDA — r/{sub}</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f8f9fa; color: #333; }}
  .container {{ max-width: 1100px; margin: 0 auto; padding: 2em; }}
  h1 {{ color: #1a252f; border-bottom: 3px solid #2c3e50; padding-bottom: 0.4em; }}
  h2 {{ color: #2c3e50; margin-top: 1.8em; border-left: 4px solid #3498db; padding-left: 0.6em; }}
  h3 {{ color: #34495e; margin-top: 1.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.88em; }}
  th {{ background: #2c3e50; color: white; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .section {{ background: white; border-radius: 8px; padding: 1.5em; margin-bottom: 1.5em;
              box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5em; }}
  .stat {{ display: inline-block; background: #eaf0fb; border-radius: 6px; padding: 0.5em 1em;
           margin: 0.3em; font-size: 0.9em; }}
  .stat strong {{ color: #2c3e50; }}
  .badge {{ display: inline-block; padding: 2px 10px; border-radius: 10px; color: white;
            font-size: 0.82em; font-weight: bold; }}
  .green {{ background: #5cb85c; }} .red {{ background: #d9534f; }}
  .orange {{ background: #f0ad4e; color:#333; }} .blue {{ background: #5bc0de; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; }}
  footer {{ color: #aaa; font-size: 0.78em; margin-top: 2em; border-top: 1px solid #ddd; padding-top: 0.8em; }}
</style>
</head>
<body>
<div class="container">
<h1>🔬 Modelling EDA Report — r/{sub}</h1>
<p>Three-stage exploratory analysis: <strong>pre-training</strong> feature profiling →
<strong>fold-level</strong> walk-forward health → <strong>post-training</strong> model diagnostics.</p>

<!-- ══════════════ STAGE 1: PRE-TRAINING ══════════════ -->
<h2>Stage 1 — Pre-training Feature Analysis</h2>

<div class="section">
<h3>Class Balance</h3>
<div class="stat">Total labeled weeks: <strong>{cb.get('n_total_labeled', 0)}</strong></div>
<div class="stat">Crisis weeks (State 2+3): <strong>{cb.get('crisis_weeks', 0)}</strong></div>
<div class="stat">Crisis rate: <strong>{cb.get('crisis_rate', 0):.1%}</strong></div>
<div class="stat">Imbalance ratio: <strong>{cb.get('imbalance_ratio', 0):.1f}:1</strong></div>
<br><br>
{class_bar_html}
</div>

<div class="section">
<h3>Feature–Label Correlations (Spearman ρ vs binary crisis label)</h3>
<p>Top 15 features ranked by |ρ|. Green = significant (p&lt;0.05) and |ρ|≥0.3, grey = significant but weak, red = not significant.</p>
<table>
  <tr><th>Feature</th><th>ρ with crisis label</th><th>Significant?</th></tr>
  {corr_rows_html}
</table>
</div>

<div class="section">
<h3>Highly Collinear Feature Pairs (|ρ| ≥ 0.90)</h3>
<p>Redundant features may inflate SHAP uncertainty. Consider removing one from each pair.</p>
<table>
  <tr><th>Feature A</th><th>Feature B</th><th>Spearman ρ</th></tr>
  {multi_html}
</table>
</div>

<!-- ══════════════ STAGE 2: FOLD DIAGNOSTICS ══════════════ -->
<h2>Stage 2 — Walk-forward Fold Health</h2>

<div class="section">
<div class="stat">Total folds: <strong>{fold_eda.get('n_total_folds', 0)}</strong></div>
<div class="stat">Ran: <strong>{fold_eda.get('n_ran_folds', 0)}</strong></div>
<div class="stat">Skipped: <strong>{fold_eda.get('n_skipped_folds', 0)}</strong>
  ({fold_eda.get('skip_rate_pct', 0):.1f}%)</div>
<div class="stat">Mean crisis rate across folds: <strong>{fold_eda.get('mean_fold_crisis_rate', 0):.1%}</strong></div>
<div class="stat">Low-balance folds (≤5%): <strong>{fold_eda.get('n_low_balance_folds', 0)}</strong></div>
<br><br>
<h3>Crisis Rate per Fold (red = low-balance ≤5%)</h3>
{fold_bar_html}
<h3>Skip Reasons</h3>
<ul>{skip_html}</ul>
</div>

<!-- ══════════════ STAGE 3: POST-TRAINING ══════════════ -->
<h2>Stage 3 — Post-training Model Diagnostics</h2>

<div class="two-col">
  <div>
    <h3>XGBoost — Confusion Matrix</h3>
    <div class="section">{conf_xgb_html}</div>
  </div>
  <div>
    <h3>LSTM — Confusion Matrix</h3>
    <div class="section">{conf_lstm_html}</div>
  </div>
</div>

<div class="two-col">
  <div>
    <h3>XGBoost — Probability Calibration</h3>
    <div class="section">{cal_xgb_html}</div>
  </div>
  <div>
    <h3>LSTM — Probability Calibration</h3>
    <div class="section">{cal_lstm_html}</div>
  </div>
</div>

<div class="two-col">
  <div>
    <h3>XGBoost — Threshold Sensitivity</h3>
    <div class="section">{thresh_xgb_html}</div>
  </div>
  <div>
    <h3>LSTM — Threshold Sensitivity</h3>
    <div class="section">{thresh_lstm_html}</div>
  </div>
</div>

<footer>
Generated by community-crisis-predictor modelling EDA module.
Methodology: Spearman correlations, IQR collinearity detection, walk-forward fold tracking,
probability calibration curves, threshold sensitivity sweep.
</footer>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Modelling EDA HTML -> {output_path}")
