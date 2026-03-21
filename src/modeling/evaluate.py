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


def top_k_alert_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: tuple[int, ...] | list[int] = (1, 2, 3, 5),
) -> dict[str, dict[str, float | int]]:
    """
    Rank all evaluation weeks by predicted probability (descending), take the top-K
    as \"alerts\", and measure what fraction of actual elevated-distress weeks are covered.

    This answers: if an ops team can only investigate K weeks per history, how much of
    the true elevated-distress load do we capture?

    Parameters
    ----------
    y_true : binary array (1 = elevated-distress week to catch)
    y_prob : same length, higher = model prioritizes that week for alerting

    Returns
    -------
    dict mapping K -> {captured, total_positives, recall}
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return {str(int(k)): {"captured": 0, "total_positives": 0, "recall": 0.0} for k in k_values}

    total_positives = int(y_true.sum())
    if total_positives == 0:
        return {str(int(k)): {"captured": 0, "total_positives": 0, "recall": 0.0} for k in k_values}

    order = np.argsort(-y_prob, kind="stable")
    out: dict[str, dict[str, float | int]] = {}
    for k in k_values:
        kk = min(int(k), n)
        top_idx = order[:kk]
        captured = int(y_true[top_idx].sum())
        # String keys so JSON round-trip via eval_results.json stays consistent
        out[str(int(k))] = {
            "captured": captured,
            "total_positives": total_positives,
            "recall": float(captured / total_positives),
        }
    return out


def _random_baseline_top_k_recall(
    n_weeks: int,
    total_positives: int,
    k_values: tuple[int, ...] | list[int],
) -> dict[str, float]:
    """
    Expected recall if K distinct weeks were chosen uniformly at random (without replacement)
    from the evaluation period: E[captured] = K * P / n, so E[recall] = (K * P / n) / P = K/n
    (for P > 0), capped when K >= n at 1.0 for the capture count expectation / P.
    """
    if n_weeks <= 0 or total_positives <= 0:
        return {str(int(k)): 0.0 for k in k_values}
    out = {}
    for k in k_values:
        kk = min(int(k), n_weeks)
        # Hypergeometric mean captures: kk * P / n
        expected_captured = kk * total_positives / n_weeks
        out[str(int(k))] = float(expected_captured / total_positives)
    return out


def _persistence_baseline_top_k_recall(
    y_true: np.ndarray,
    k_values: tuple[int, ...] | list[int],
) -> dict[str, dict[str, float | int]]:
    """
    Baseline ranking: score week t by whether week t-1 was elevated distress (actual).
    Higher score = alert here; ties keep chronological order (stable sort).
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    total_positives = int(y_true.sum())
    if n == 0 or total_positives == 0:
        return {str(int(k)): {"captured": 0, "total_positives": 0, "recall": 0.0} for k in k_values}

    scores = np.zeros(n, dtype=float)
    for i in range(1, n):
        scores[i] = float(y_true[i - 1])
    order = np.argsort(-scores, kind="stable")
    out: dict[str, dict[str, float | int]] = {}
    for k in k_values:
        kk = min(int(k), n)
        top_idx = order[:kk]
        captured = int(y_true[top_idx].sum())
        out[str(int(k))] = {
            "captured": captured,
            "total_positives": total_positives,
            "recall": float(captured / total_positives),
        }
    return out


def compute_decision_usefulness(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k_values: tuple[int, ...] | list[int] = (1, 2, 3, 5),
) -> dict:
    """Bundle model top-K recall with random and persistence baselines (same K, same labels)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    p = int(y_true.sum())
    model = top_k_alert_recall(y_true, y_prob, k_values)
    random_exp = _random_baseline_top_k_recall(n, p, k_values)
    persistence = _persistence_baseline_top_k_recall(y_true, k_values)
    return {
        "k_values": [int(k) for k in k_values],
        "n_weeks": n,
        "n_elevated_distress_weeks": p,
        "model": model,
        "random_expected_recall": random_exp,
        "persistence": persistence,
    }


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
    metrics["decision_usefulness"] = compute_decision_usefulness(y_true, y_prob)
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
    metrics["decision_usefulness"] = compute_decision_usefulness(y_true_bin, y_prob)
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
