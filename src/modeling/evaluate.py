import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    weights = labeling_cfg.get("distress_weights")

    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    splitter = WalkForwardSplitter(
        min_train_weeks=wf_cfg.get("min_train_weeks", 26),
        gap_weeks=wf_cfg.get("gap_weeks", 1),
    )
    prob_threshold = config.get("evaluation", {}).get("probability_threshold", 0.5)

    distress_scores = compute_distress_score(feature_df, weights)
    n_samples = len(feature_df)

    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full(n_samples, np.nan)
    all_actuals = np.full(n_samples, np.nan)

    n_folds = splitter.n_splits(n_samples)
    print(f"  Running {n_folds} walk-forward folds (XGBoost)...")

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(n_samples)):
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        train_scores = distress_scores.iloc[train_idx]
        labeler.fit(train_scores)

        all_idx = np.concatenate([train_idx, test_idx])
        labels = labeler.label(distress_scores.iloc[all_idx])

        train_labels = labels.iloc[: len(train_idx)]
        valid_train = ~train_labels.isna()
        X_train = feature_df.iloc[train_idx][feature_columns].values[valid_train]
        y_train_4class = train_labels[valid_train].astype(int)
        # XGB is binary baseline: crisis = states 2+3
        y_train = (y_train_4class >= 2).astype(int)

        if len(y_train) < 10 or y_train.sum() < 2:
            continue

        model = XGBCrisisModel(config)
        model.train(
            pd.DataFrame(X_train, columns=feature_columns),
            y_train,
            do_search=not skip_search,
        )

        for ti in test_idx:
            X_test = feature_df.iloc[[ti]][feature_columns]
            prob = model.predict_proba(X_test)[0]
            all_probs[ti] = prob
            all_preds[ti] = int(prob >= prob_threshold)

            if ti < n_samples - 1:
                actual_label = labeler.label(distress_scores.iloc[train_idx[0] : ti + 2])
                if not actual_label.isna().iloc[-2]:
                    all_actuals[ti] = int(actual_label.iloc[-2] >= 2)

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
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if y_true.sum() > 0
        else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }

    metrics["avg_detection_lead_time_weeks"] = _compute_detection_lead_time(
        pd.Series(all_preds), pd.Series(all_actuals), crisis_min=1
    )
    metrics["per_week"] = {
        "predictions": all_preds.tolist(),
        "probabilities": all_probs.tolist(),
        "actuals": all_actuals.tolist(),
    }
    return metrics


def evaluate_walk_forward_lstm(
    feature_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
) -> dict:
    from src.modeling.train_rnn import LSTMCrisisModel

    labeling_cfg = config.get("labeling", {})
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    weights = labeling_cfg.get("distress_weights")

    lstm_cfg = config.get("modeling", {}).get("lstm", {})
    sequence_length = lstm_cfg.get("sequence_length", 8)

    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    min_train = wf_cfg.get("min_train_weeks", 26) + sequence_length
    splitter = WalkForwardSplitter(min_train_weeks=min_train, gap_weeks=1)

    distress_scores = compute_distress_score(feature_df, weights)
    n_samples = len(feature_df)

    all_states = np.full(n_samples, np.nan)
    all_probs = np.full(n_samples, np.nan)
    all_actuals_4class = np.full(n_samples, np.nan)

    n_folds = splitter.n_splits(n_samples)
    print(f"  Running {n_folds} walk-forward folds (LSTM)...")

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(n_samples)):
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        train_scores = distress_scores.iloc[train_idx]
        labeler.fit(train_scores)

        all_idx = np.concatenate([train_idx, test_idx])
        labels = labeler.label(distress_scores.iloc[all_idx])

        train_labels = labels.iloc[: len(train_idx)]
        valid_train = ~train_labels.isna()
        X_train = feature_df.iloc[train_idx][feature_columns][valid_train]
        y_train = train_labels[valid_train].astype(int)  # 0/1/2/3

        if len(y_train) < sequence_length + 5 or y_train.nunique() < 2:
            continue

        model = LSTMCrisisModel(config)
        try:
            model.train(X_train, y_train, walk_forward=True)
        except Exception as e:
            print(f"  LSTM fold {fold_i} failed: {e}")
            continue

        for ti in test_idx:
            seq_start = max(0, ti - sequence_length + 1)
            X_seq = feature_df.iloc[seq_start : ti + 1][feature_columns]
            if len(X_seq) < sequence_length:
                continue
            try:
                all_probs[ti] = model.predict_proba(X_seq)[-1]
                all_states[ti] = model.predict_state(X_seq)[-1]
            except Exception:
                continue

            if ti < n_samples - 1:
                actual_label = labeler.label(distress_scores.iloc[train_idx[0] : ti + 2])
                if not actual_label.isna().iloc[-2]:
                    all_actuals_4class[ti] = actual_label.iloc[-2]

    valid = ~(np.isnan(all_states) | np.isnan(all_actuals_4class))
    if valid.sum() < 5:
        return {"error": "Too few valid predictions", "n_valid": int(valid.sum())}

    y_true_4 = all_actuals_4class[valid].astype(int)
    y_pred_4 = all_states[valid].astype(int)
    y_prob = all_probs[valid]
    y_true_bin = (y_true_4 >= 2).astype(int)
    y_pred_bin = (y_pred_4 >= 2).astype(int)

    metrics: dict = {
        "n_folds": n_folds,
        "n_valid_predictions": int(valid.sum()),
        "n_crisis_actual": int(y_true_bin.sum()),
        "n_crisis_predicted": int(y_pred_bin.sum()),
        "recall": float(recall_score(y_true_bin, y_pred_bin, zero_division=0)),
        "precision": float(precision_score(y_true_bin, y_pred_bin, zero_division=0)),
        "f1": float(f1_score(y_true_bin, y_pred_bin, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_bin, y_prob))
        if y_true_bin.sum() > 0
        else 0.0,
        "confusion_matrix": confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist(),
        "confusion_matrix_4class": confusion_matrix(
            y_true_4, y_pred_4, labels=[0, 1, 2, 3]
        ).tolist(),
    }

    for cls in range(4):
        y_cls_t = (y_true_4 == cls).astype(int)
        y_cls_p = (y_pred_4 == cls).astype(int)
        metrics[f"recall_class_{cls}"] = float(recall_score(y_cls_t, y_cls_p, zero_division=0))
        metrics[f"precision_class_{cls}"] = float(
            precision_score(y_cls_t, y_cls_p, zero_division=0)
        )

    metrics["avg_detection_lead_time_weeks"] = _compute_detection_lead_time(
        pd.Series(all_states), pd.Series((all_actuals_4class >= 2).astype(float)), crisis_min=2
    )
    metrics["per_week"] = {
        "predictions": all_states.tolist(),
        "probabilities": all_probs.tolist(),
        "actuals": all_actuals_4class.tolist(),
    }
    return metrics


def _compute_detection_lead_time(
    predictions: pd.Series,
    actuals: pd.Series,
    crisis_min: int = 1,
) -> float:
    crisis_starts = []
    in_crisis = False

    for i in range(len(actuals)):
        val = actuals.iloc[i]
        if not np.isnan(val) and val >= 1 and not in_crisis:
            crisis_starts.append(i)
            in_crisis = True
        elif np.isnan(val) or val < 1:
            in_crisis = False

    if not crisis_starts:
        return 0.0

    lead_times = []
    for start in crisis_starts:
        lead = 0
        for j in range(start - 1, -1, -1):
            pred = predictions.iloc[j]
            if not np.isnan(pred) and pred >= crisis_min:
                lead += 1
            else:
                break
        lead_times.append(lead)

    return float(np.mean(lead_times)) if lead_times else 0.0
