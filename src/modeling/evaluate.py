import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)

from src.labeling.distress_score import compute_distress_score
from src.labeling.target import CrisisLabeler
from src.modeling.splits import WalkForwardSplitter
from src.modeling.train_xgb import XGBCrisisModel


def evaluate_walk_forward(
    feature_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
    skip_search: bool = False,
) -> dict:
    labeling_cfg = config.get("labeling", {})
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    weights = labeling_cfg.get("distress_weights")

    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    splitter = WalkForwardSplitter(
        min_train_weeks=wf_cfg.get("min_train_weeks", 26),
        gap_weeks=wf_cfg.get("gap_weeks", 1),
    )

    prob_threshold = config.get("evaluation", {}).get("probability_threshold", 0.5)

    # Compute distress scores on full data (z-scores refit per fold below)
    distress_scores = compute_distress_score(feature_df, weights)

    X = feature_df[feature_columns].values
    n_samples = len(X)

    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full(n_samples, np.nan)
    all_actuals = np.full(n_samples, np.nan)

    n_folds = splitter.n_splits(n_samples)
    print(f"  Running {n_folds} walk-forward folds...")

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(n_samples)):
        # Refit crisis labeler on training data only
        labeler = CrisisLabeler(threshold_std=threshold_std)
        train_scores = distress_scores.iloc[train_idx]
        labeler.fit(train_scores)

        # Label the full range needed (train + test)
        all_idx = np.concatenate([train_idx, test_idx])
        labels = labeler.label(distress_scores.iloc[all_idx])

        # Map back to positions
        train_labels = labels.iloc[:len(train_idx)]
        test_labels = labels.iloc[len(train_idx):]

        # Drop NaN labels
        valid_train = ~train_labels.isna()
        X_train = feature_df.iloc[train_idx][feature_columns].values[valid_train]
        y_train = train_labels[valid_train].astype(int)

        if len(y_train) < 10 or y_train.sum() < 2:
            continue

        # Train model
        model = XGBCrisisModel(config)
        model.train(
            pd.DataFrame(X_train, columns=feature_columns),
            y_train,
            do_search=not skip_search,
        )

        # Predict
        for ti in test_idx:
            X_test = feature_df.iloc[[ti]][feature_columns]
            prob = model.predict_proba(X_test)[0]
            all_probs[ti] = prob
            all_preds[ti] = int(prob >= prob_threshold)

            # Get actual label
            if ti < n_samples - 1:
                actual_label = labeler.label(distress_scores.iloc[train_idx[0]:ti + 2])
                if not actual_label.isna().iloc[-2]:
                    all_actuals[ti] = actual_label.iloc[-2]

    # Compute metrics on valid predictions
    valid = ~(np.isnan(all_preds) | np.isnan(all_actuals))
    if valid.sum() < 5:
        return {"error": "Too few valid predictions", "n_valid": int(valid.sum())}

    y_true = all_actuals[valid].astype(int)
    y_pred = all_preds[valid].astype(int)
    y_prob = all_probs[valid]

    metrics = {
        "n_folds": n_folds,
        "n_valid_predictions": int(valid.sum()),
        "n_crisis_actual": int(y_true.sum()),
        "n_crisis_predicted": int(y_pred.sum()),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    # Detection lead time
    lead_time = _compute_detection_lead_time(
        pd.Series(all_preds), pd.Series(all_actuals)
    )
    metrics["avg_detection_lead_time_weeks"] = lead_time

    # Store per-week results
    metrics["per_week"] = {
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "actuals": all_actuals.tolist(),
    }

    return metrics


def _compute_detection_lead_time(predictions: pd.Series, actuals: pd.Series) -> float:
    crisis_starts = []
    in_crisis = False

    for i in range(len(actuals)):
        if actuals.iloc[i] == 1 and not in_crisis:
            crisis_starts.append(i)
            in_crisis = True
        elif actuals.iloc[i] != 1:
            in_crisis = False

    if not crisis_starts:
        return 0.0

    lead_times = []
    for start in crisis_starts:
        lead = 0
        for j in range(start - 1, -1, -1):
            if predictions.iloc[j] == 1:
                lead += 1
            else:
                break
        lead_times.append(lead)

    return float(np.mean(lead_times)) if lead_times else 0.0
