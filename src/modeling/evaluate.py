import datetime
import json
import itertools
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.labeling.distress_score import compute_distress_score
from src.labeling.target import CrisisLabeler
from src.modeling.calibration import apply_binary_calibrator, fit_binary_calibrator
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


def _save_feature_stats(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    sub: str,
    save_dir: Path,
) -> None:
    """Save per-feature training distribution stats for serving-layer drift detection."""
    stats: dict = {
        "subreddit": sub,
        "features": {},
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "n_training_weeks": len(X_train),
    }
    for col in feature_columns:
        if col in X_train.columns:
            vals = X_train[col].dropna()
            stats["features"][col] = {
                "mean": float(vals.mean()) if len(vals) else 0.0,
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "min": float(vals.min()) if len(vals) else 0.0,
                "max": float(vals.max()) if len(vals) else 0.0,
            }
    with open(save_dir / f"{sub}_feature_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _compute_dev_samples(n_samples: int, holdout_weeks: int, min_train_required: int) -> int:
    if n_samples <= min_train_required + 1:
        return n_samples
    hold = max(0, int(holdout_weeks))
    if hold == 0:
        return n_samples
    max_hold = max(0, n_samples - (min_train_required + 1))
    hold = min(hold, max_hold)
    return n_samples - hold


def _split_fit_calibration(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    calibration_frac: float = 0.2,
    min_calibration_samples: int = 12,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    n = len(X)
    n_cal = max(min_calibration_samples, int(round(n * calibration_frac)))
    n_cal = min(max(0, n_cal), max(0, n - 1))
    if n_cal == 0:
        return X, y, X.iloc[0:0], y.iloc[0:0]
    split_at = n - n_cal
    return (
        X.iloc[:split_at].copy(),
        y.iloc[:split_at].copy(),
        X.iloc[split_at:].copy(),
        y.iloc[split_at:].copy(),
    )


def _pack_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "n_valid_predictions": int(len(y_true)),
        "n_crisis_actual": int(y_true.sum()),
        "n_crisis_predicted": int(y_pred.sum()),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if y_true.sum() > 0
        else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_prob))
        if (y_true.sum() > 0 and (y_true == 0).sum() > 0)
        else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "decision_usefulness": compute_decision_usefulness(y_true, y_prob),
    }
    return out


def _with_lstm_overrides(config: dict, overrides: dict | None) -> dict:
    cfg = deepcopy(config)
    if not overrides:
        return cfg
    lstm_cfg = cfg.setdefault("modeling", {}).setdefault("lstm", {})
    for key, val in overrides.items():
        lstm_cfg[key] = val
    return cfg


def _tune_lstm_hyperparams(
    feature_df: pd.DataFrame,
    distress_scores: pd.Series,
    config: dict,
    feature_columns: list[str],
    *,
    dev_n: int,
    threshold_std: float,
    thresholds_std: list[float],
    prob_threshold: float,
) -> tuple[dict, dict]:
    from src.modeling.train_rnn import LSTMCrisisModel

    lstm_cfg = config.get("modeling", {}).get("lstm", {})
    search_cfg = lstm_cfg.get("search", {})
    if not bool(search_cfg.get("enabled", False)):
        return {}, {}
    grid = search_cfg.get("grid", {})
    hidden_sizes = grid.get("hidden_size", [int(lstm_cfg.get("hidden_size", 32))])
    num_layers_list = grid.get("num_layers", [int(lstm_cfg.get("num_layers", 2))])
    learning_rates = grid.get("learning_rate", [float(lstm_cfg.get("learning_rate", 0.001))])
    dropouts = grid.get("dropout", [float(lstm_cfg.get("dropout", 0.2))])
    batch_sizes = grid.get("batch_size", [int(lstm_cfg.get("batch_size", 16))])
    max_trials = int(search_cfg.get("max_trials", 8))
    seq_len = int(lstm_cfg.get("sequence_length", 8))

    val_start = int(max(seq_len + 8, round(dev_n * 0.8)))
    if val_start >= dev_n - 2:
        return {}, {"reason": "insufficient_dev_window"}

    labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
    labeler.fit(distress_scores.iloc[:val_start])
    labels_dev = labeler.label(distress_scores.iloc[:dev_n])

    train_labels = labels_dev.iloc[:val_start]
    train_mask = ~train_labels.isna()
    X_train = feature_df.iloc[:val_start][feature_columns].iloc[train_mask.values].reset_index(drop=True)
    y_train = train_labels[train_mask].astype(int).reset_index(drop=True)
    if len(y_train) < seq_len + 8 or y_train.nunique() < 2:
        return {}, {"reason": "insufficient_train_window"}

    X_dev = feature_df.iloc[:dev_n][feature_columns].reset_index(drop=True)
    val_labels = labels_dev.iloc[val_start:dev_n].reset_index(drop=True)
    candidates = list(itertools.product(hidden_sizes, num_layers_list, learning_rates, dropouts, batch_sizes))[:max_trials]
    if not candidates:
        return {}, {"reason": "empty_search_space"}

    best: dict | None = None
    for hid, n_layers, lr, drp, bsz in candidates:
        params = {
            "hidden_size": int(hid),
            "num_layers": int(n_layers),
            "learning_rate": float(lr),
            "dropout": float(drp),
            "batch_size": int(bsz),
        }
        trial_cfg = _with_lstm_overrides(config, params)
        model = LSTMCrisisModel(trial_cfg)
        try:
            model.train(X_train, y_train, walk_forward=False)
            probs = model.predict_proba(X_dev)[val_start:dev_n]
        except Exception:
            continue
        valid = (~val_labels.isna().values) & np.isfinite(probs)
        if valid.sum() < 5:
            continue
        y_true_4 = val_labels.values[valid].astype(int)
        y_true_bin = (y_true_4 >= 2).astype(int)
        y_prob = probs[valid]
        y_pred = (y_prob >= prob_threshold).astype(int)
        pr_auc = float(average_precision_score(y_true_bin, y_prob)) if y_true_bin.sum() > 0 else 0.0
        recall = float(recall_score(y_true_bin, y_pred, zero_division=0))
        score = pr_auc + (0.05 * recall)
        row = {**params, "pr_auc": pr_auc, "recall": recall, "score": score}
        if best is None or row["score"] > best["score"]:
            best = row

    if best is None:
        return {}, {"reason": "all_trials_failed"}
    selected = {
        "hidden_size": int(best["hidden_size"]),
        "num_layers": int(best["num_layers"]),
        "learning_rate": float(best["learning_rate"]),
        "dropout": float(best["dropout"]),
        "batch_size": int(best["batch_size"]),
    }
    return selected, best


def evaluate_walk_forward(
    feature_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
    skip_search: bool = False,
    save_dir: Path | None = None,
    sub: str = "",
) -> dict:
    labeling_cfg = config.get("labeling", {})
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    weights = labeling_cfg.get("distress_weights")

    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    min_train_weeks = int(wf_cfg.get("min_train_weeks", 26))
    gap_weeks = int(wf_cfg.get("gap_weeks", 1))
    eval_cfg = config.get("evaluation", {})
    prob_threshold = eval_cfg.get("probability_threshold", 0.5)
    holdout_weeks = int(eval_cfg.get("holdout_weeks", 12))
    cal_cfg = eval_cfg.get("calibration", {})
    calibration_frac = float(cal_cfg.get("calibration_frac", 0.2))
    calibration_method = str(cal_cfg.get("method", "platt"))
    calibration_min_samples = int(cal_cfg.get("min_samples", 20))
    calibration_min_class = int(cal_cfg.get("min_class_count", 3))

    # Avoid temporal leakage: do not globally z-score with future weeks included.
    distress_scores = compute_distress_score(feature_df, weights, normalize=False)
    n_samples = len(feature_df)
    dev_n = _compute_dev_samples(
        n_samples=n_samples,
        holdout_weeks=holdout_weeks,
        min_train_required=min_train_weeks + gap_weeks,
    )

    all_preds = np.full(n_samples, np.nan)
    all_probs = np.full(n_samples, np.nan)
    all_actuals = np.full(n_samples, np.nan)

    splitter = WalkForwardSplitter(min_train_weeks=min_train_weeks, gap_weeks=gap_weeks)
    n_folds = splitter.n_splits(dev_n)
    print(f"  Running {n_folds} walk-forward folds (XGBoost) on dev window...")
    fold_diagnostics: list[dict[str, str | int]] = []
    fold_records: list[dict] = []

    final_labeler: CrisisLabeler | None = None
    final_model: XGBCrisisModel | None = None
    final_calibrator: dict | None = None
    final_X_train_df: pd.DataFrame | None = None

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(dev_n)):
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        train_scores = distress_scores.iloc[train_idx]
        labeler.fit(train_scores)

        all_idx = np.concatenate([train_idx, test_idx])
        labels = labeler.label(distress_scores.iloc[all_idx])

        train_labels = labels.iloc[: len(train_idx)]
        valid_train = ~train_labels.isna()
        X_train_df = feature_df.iloc[train_idx][feature_columns].iloc[valid_train.values].reset_index(drop=True)
        y_train_4class = train_labels[valid_train].astype(int).reset_index(drop=True)
        y_train = (y_train_4class >= 2).astype(int)

        if len(y_train) < 10 or y_train.sum() < 2:
            fold_diagnostics.append({"fold": int(fold_i), "reason": "insufficient_training_examples_or_positives"})
            fold_records.append(
                {
                    "fold_i": int(fold_i),
                    "n_train": int(len(y_train)),
                    "n_crisis_train": int(y_train.sum()),
                    "crisis_rate_train": float(y_train.mean()) if len(y_train) > 0 else 0.0,
                    "skipped": True,
                    "skip_reason": "insufficient_training_examples_or_positives",
                }
            )
            continue

        X_fit, y_fit, X_cal, y_cal = _split_fit_calibration(
            X_train_df,
            y_train,
            calibration_frac=calibration_frac,
            min_calibration_samples=calibration_min_samples,
        )
        model = XGBCrisisModel(config)
        try:
            model.train(X_fit, y_fit, do_search=not skip_search)
        except Exception as e:
            fold_diagnostics.append({"fold": int(fold_i), "reason": f"xgb_train_failed: {type(e).__name__}"})
            continue

        calibrator = {"type": "identity", "reason": "not_fitted"}
        if len(X_cal) > 0:
            cal_probs = model.predict_proba(X_cal)
            calibrator = fit_binary_calibrator(
                cal_probs,
                y_cal.values,
                method=calibration_method,
                min_samples=calibration_min_samples,
                min_class_count=calibration_min_class,
            )

        fold_records.append(
            {
                "fold_i": int(fold_i),
                "n_train": int(len(y_train)),
                "n_crisis_train": int(y_train.sum()),
                "crisis_rate_train": round(float(y_train.mean()), 4),
                "calibrator": calibrator.get("type", "identity"),
                "skipped": False,
            }
        )

        for ti in test_idx:
            X_test = feature_df.iloc[[ti]][feature_columns]
            raw_prob = model.predict_proba(X_test)[0]
            prob = float(apply_binary_calibrator(np.array([raw_prob]), calibrator)[0])
            all_probs[ti] = prob
            all_preds[ti] = int(prob >= prob_threshold)
            if ti < n_samples - 1:
                actual_label = labeler.label(distress_scores.iloc[train_idx[0] : ti + 2])
                if not actual_label.isna().iloc[-2]:
                    all_actuals[ti] = int(actual_label.iloc[-2] >= 2)

    # Final model for holdout and serving artifacts.
    if dev_n > (min_train_weeks + gap_weeks):
        final_labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        final_labeler.fit(distress_scores.iloc[:dev_n])
        all_labels_final = final_labeler.label(distress_scores)
        train_labels = all_labels_final.iloc[:dev_n]
        train_mask = ~train_labels.isna()
        final_X_train_df = feature_df.iloc[:dev_n][feature_columns].iloc[train_mask.values].reset_index(drop=True)
        final_y_train = (train_labels[train_mask].astype(int) >= 2).astype(int).reset_index(drop=True)
        X_fit, y_fit, X_cal, y_cal = _split_fit_calibration(
            final_X_train_df,
            final_y_train,
            calibration_frac=calibration_frac,
            min_calibration_samples=calibration_min_samples,
        )
        if len(y_fit) > 0 and int(y_fit.sum()) > 0:
            final_model = XGBCrisisModel(config)
            final_model.train(X_fit, y_fit, do_search=not skip_search)
            final_calibrator = {"type": "identity", "reason": "not_fitted"}
            if len(X_cal) > 0:
                final_calibrator = fit_binary_calibrator(
                    final_model.predict_proba(X_cal),
                    y_cal.values,
                    method=calibration_method,
                    min_samples=calibration_min_samples,
                    min_class_count=calibration_min_class,
                )

            holdout_start = dev_n
            for ti in range(holdout_start, n_samples):
                if ti >= n_samples - 1:
                    continue
                raw = final_model.predict_proba(feature_df.iloc[[ti]][feature_columns])[0]
                prob = float(apply_binary_calibrator(np.array([raw]), final_calibrator)[0])
                all_probs[ti] = prob
                all_preds[ti] = int(prob >= prob_threshold)
                label_val = all_labels_final.iloc[ti]
                if not np.isnan(label_val):
                    all_actuals[ti] = int(int(label_val) >= 2)

    if save_dir is not None and sub and final_model is not None:
        try:
            import joblib

            save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(final_model.model, save_dir / f"{sub}_xgb.pkl")
            with open(save_dir / f"{sub}_xgb_calibrator.json", "w", encoding="utf-8") as f:
                json.dump(final_calibrator or {"type": "identity"}, f, indent=2)
            print(f"  XGB model saved -> {save_dir / f'{sub}_xgb.pkl'}")
        except Exception as e:
            print(f"  Warning: could not save XGB model for {sub}: {e}")
        if final_X_train_df is not None:
            try:
                _save_feature_stats(final_X_train_df, feature_columns, sub, save_dir)
                print(f"  Feature stats saved -> {save_dir / f'{sub}_feature_stats.json'}")
            except Exception as e:
                print(f"  Warning: could not save feature stats for {sub}: {e}")

    valid = ~(np.isnan(all_preds) | np.isnan(all_actuals))
    dev_mask = valid.copy()
    dev_mask[dev_n:] = False
    hold_mask = valid.copy()
    hold_mask[:dev_n] = False
    if dev_mask.sum() < 5:
        return {
            "error": "Too few valid predictions",
            "n_valid": int(dev_mask.sum()),
            "fold_diagnostics": fold_diagnostics,
        }

    y_true = all_actuals[dev_mask].astype(int)
    y_pred = all_preds[dev_mask].astype(int)
    y_prob = all_probs[dev_mask]
    metrics = {"n_folds": n_folds, **_pack_binary_metrics(y_true, y_pred, y_prob)}
    metrics["n_valid_predictions"] = int(dev_mask.sum())
    lead_time = _compute_detection_lead_time(pd.Series(all_preds[:dev_n]), pd.Series(all_actuals[:dev_n]), crisis_min=1)
    metrics["avg_detection_lead_time_weeks"] = float(lead_time["mean"])
    metrics["detection_lead_time_distribution"] = lead_time
    metrics["fold_diagnostics"] = fold_diagnostics
    metrics["fold_records"] = fold_records
    metrics["calibration_method"] = calibration_method
    metrics["holdout_weeks"] = int(n_samples - dev_n)

    if hold_mask.sum() > 0:
        h_true = all_actuals[hold_mask].astype(int)
        h_pred = all_preds[hold_mask].astype(int)
        h_prob = all_probs[hold_mask]
        metrics["holdout"] = _pack_binary_metrics(h_true, h_pred, h_prob)
    else:
        metrics["holdout"] = {"n_valid_predictions": 0}

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
    save_dir: Path | None = None,
    sub: str = "",
) -> dict:
    from src.modeling.train_rnn import LSTMCrisisModel

    labeling_cfg = config.get("labeling", {})
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    weights = labeling_cfg.get("distress_weights")

    lstm_cfg = config.get("modeling", {}).get("lstm", {})
    sequence_length = int(lstm_cfg.get("sequence_length", 8))
    eval_cfg = config.get("evaluation", {})
    holdout_weeks = int(eval_cfg.get("holdout_weeks", 12))
    prob_threshold = float(eval_cfg.get("probability_threshold", 0.5))
    cal_cfg = eval_cfg.get("calibration", {})
    calibration_frac = float(cal_cfg.get("calibration_frac", 0.2))
    calibration_method = str(cal_cfg.get("method", "platt"))
    calibration_min_samples = int(cal_cfg.get("min_samples", 20))
    calibration_min_class = int(cal_cfg.get("min_class_count", 3))

    wf_cfg = config.get("modeling", {}).get("walk_forward", {})
    min_train = int(wf_cfg.get("min_train_weeks", 26)) + sequence_length
    splitter = WalkForwardSplitter(min_train_weeks=min_train, gap_weeks=1)

    # Avoid temporal leakage: do not globally z-score with future weeks included.
    distress_scores = compute_distress_score(feature_df, weights, normalize=False)
    n_samples = len(feature_df)
    dev_n = _compute_dev_samples(n_samples=n_samples, holdout_weeks=holdout_weeks, min_train_required=min_train + 1)
    tuned_params, tuning_summary = _tune_lstm_hyperparams(
        feature_df=feature_df,
        distress_scores=distress_scores,
        config=config,
        feature_columns=feature_columns,
        dev_n=dev_n,
        threshold_std=threshold_std,
        thresholds_std=thresholds_std,
        prob_threshold=prob_threshold,
    )
    lstm_runtime_config = _with_lstm_overrides(config, tuned_params)
    if tuned_params:
        print(f"  LSTM search selected: {tuned_params}")

    all_states = np.full(n_samples, np.nan)
    all_probs = np.full(n_samples, np.nan)
    all_actuals_4class = np.full(n_samples, np.nan)

    n_folds = splitter.n_splits(dev_n)
    print(f"  Running {n_folds} walk-forward folds (LSTM) on dev window...")
    fold_diagnostics: list[dict[str, str | int]] = []
    fold_records_lstm: list[dict] = []
    final_model: "LSTMCrisisModel | None" = None
    final_calibrator: dict | None = None

    for fold_i, (train_idx, test_idx) in enumerate(splitter.split(dev_n)):
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        train_scores = distress_scores.iloc[train_idx]
        labeler.fit(train_scores)

        all_idx = np.concatenate([train_idx, test_idx])
        labels = labeler.label(distress_scores.iloc[all_idx])

        train_labels = labels.iloc[: len(train_idx)]
        valid_train = ~train_labels.isna()
        X_train = feature_df.iloc[train_idx][feature_columns].iloc[valid_train.values].reset_index(drop=True)
        y_train = train_labels[valid_train].astype(int).reset_index(drop=True)

        if len(y_train) < sequence_length + 5 or y_train.nunique() < 2:
            fold_diagnostics.append({"fold": int(fold_i), "reason": "insufficient_sequence_or_class_variety"})
            fold_records_lstm.append(
                {
                    "fold_i": int(fold_i),
                    "n_train": int(len(y_train)),
                    "n_crisis_train": int((y_train >= 2).sum()),
                    "crisis_rate_train": float((y_train >= 2).mean()) if len(y_train) > 0 else 0.0,
                    "skipped": True,
                    "skip_reason": "insufficient_sequence_or_class_variety",
                }
            )
            continue

        X_fit, y_fit, X_cal, y_cal = _split_fit_calibration(
            X_train,
            y_train,
            calibration_frac=calibration_frac,
            min_calibration_samples=calibration_min_samples,
        )
        if len(y_fit) < sequence_length + 2:
            fold_diagnostics.append({"fold": int(fold_i), "reason": "insufficient_fit_window_after_cal_split"})
            continue

        model = LSTMCrisisModel(lstm_runtime_config)
        try:
            model.train(X_fit, y_fit, walk_forward=True)
        except Exception as e:
            print(f"  LSTM fold {fold_i} failed: {e}")
            fold_diagnostics.append({"fold": int(fold_i), "reason": f"lstm_train_failed: {type(e).__name__}"})
            continue

        calibrator = {"type": "identity", "reason": "not_fitted"}
        if len(X_cal) > 0:
            probs_full = model.predict_proba(X_train)
            cal_start = len(X_fit)
            cal_probs = probs_full[cal_start:]
            y_cal_bin = (y_cal.values >= 2).astype(int)
            valid_cal = np.isfinite(cal_probs)
            if valid_cal.any():
                calibrator = fit_binary_calibrator(
                    cal_probs[valid_cal],
                    y_cal_bin[valid_cal],
                    method=calibration_method,
                    min_samples=calibration_min_samples,
                    min_class_count=calibration_min_class,
                )

        fold_records_lstm.append(
            {
                "fold_i": int(fold_i),
                "n_train": int(len(y_train)),
                "n_crisis_train": int((y_train >= 2).sum()),
                "crisis_rate_train": round(float((y_train >= 2).mean()), 4),
                "calibrator": calibrator.get("type", "identity"),
                "skipped": False,
            }
        )

        for ti in test_idx:
            seq_start = max(0, ti - sequence_length + 1)
            X_seq = feature_df.iloc[seq_start : ti + 1][feature_columns]
            if len(X_seq) < sequence_length:
                fold_diagnostics.append({"fold": int(fold_i), "reason": "sequence_too_short_for_prediction"})
                continue
            try:
                raw_prob = float(model.predict_proba(X_seq)[-1])
                all_probs[ti] = float(apply_binary_calibrator(np.array([raw_prob]), calibrator)[0])
                all_states[ti] = model.predict_state(X_seq)[-1]
            except Exception:
                fold_diagnostics.append({"fold": int(fold_i), "reason": "lstm_predict_failed"})
                continue

            if ti < n_samples - 1:
                actual_label = labeler.label(distress_scores.iloc[train_idx[0] : ti + 2])
                if not actual_label.isna().iloc[-2]:
                    all_actuals_4class[ti] = actual_label.iloc[-2]

    # Final model and holdout evaluation.
    if dev_n > (min_train + 1):
        labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        labeler.fit(distress_scores.iloc[:dev_n])
        labels_all = labeler.label(distress_scores)
        train_labels = labels_all.iloc[:dev_n]
        train_mask = ~train_labels.isna()
        X_train = feature_df.iloc[:dev_n][feature_columns].iloc[train_mask.values].reset_index(drop=True)
        y_train = train_labels[train_mask].astype(int).reset_index(drop=True)
        X_fit, y_fit, X_cal, y_cal = _split_fit_calibration(
            X_train,
            y_train,
            calibration_frac=calibration_frac,
            min_calibration_samples=calibration_min_samples,
        )
        if len(y_fit) >= sequence_length + 2:
            final_model = LSTMCrisisModel(lstm_runtime_config)
            final_model.train(X_fit, y_fit, walk_forward=False)
            final_calibrator = {"type": "identity", "reason": "not_fitted"}
            if len(X_cal) > 0:
                probs_full = final_model.predict_proba(X_train)
                cal_probs = probs_full[len(X_fit) :]
                y_cal_bin = (y_cal.values >= 2).astype(int)
                valid_cal = np.isfinite(cal_probs)
                if valid_cal.any():
                    final_calibrator = fit_binary_calibrator(
                        cal_probs[valid_cal],
                        y_cal_bin[valid_cal],
                        method=calibration_method,
                        min_samples=calibration_min_samples,
                        min_class_count=calibration_min_class,
                    )
            for ti in range(dev_n, n_samples):
                if ti >= n_samples - 1:
                    continue
                seq_start = max(0, ti - sequence_length + 1)
                X_seq = feature_df.iloc[seq_start : ti + 1][feature_columns]
                if len(X_seq) < sequence_length:
                    continue
                raw_prob = float(final_model.predict_proba(X_seq)[-1])
                all_probs[ti] = float(apply_binary_calibrator(np.array([raw_prob]), final_calibrator)[0])
                all_states[ti] = final_model.predict_state(X_seq)[-1]
                label_val = labels_all.iloc[ti]
                if not np.isnan(label_val):
                    all_actuals_4class[ti] = label_val

    if save_dir is not None and sub and final_model is not None:
        try:
            import torch

            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": final_model.model.state_dict(),
                    "feature_size": final_model._feature_size,
                    "sequence_length": final_model.sequence_length,
                    "hidden_size": final_model.hidden_size,
                    "num_layers": final_model.num_layers,
                    "num_classes": final_model.num_classes,
                    "dropout": final_model.dropout,
                },
                save_dir / f"{sub}_lstm.pt",
            )
            with open(save_dir / f"{sub}_lstm_calibrator.json", "w", encoding="utf-8") as f:
                json.dump(final_calibrator or {"type": "identity"}, f, indent=2)
            print(f"  LSTM model saved -> {save_dir / f'{sub}_lstm.pt'}")
        except Exception as e:
            print(f"  Warning: could not save LSTM model for {sub}: {e}")

    valid = ~(np.isnan(all_states) | np.isnan(all_actuals_4class))
    dev_mask = valid.copy()
    dev_mask[dev_n:] = False
    hold_mask = valid.copy()
    hold_mask[:dev_n] = False
    if dev_mask.sum() < 5:
        return {"error": "Too few valid predictions", "n_valid": int(dev_mask.sum()), "fold_diagnostics": fold_diagnostics}

    y_true_4 = all_actuals_4class[dev_mask].astype(int)
    y_pred_4 = all_states[dev_mask].astype(int)
    y_prob = all_probs[dev_mask]
    y_true_bin = (y_true_4 >= 2).astype(int)
    y_pred_bin = (y_pred_4 >= 2).astype(int)

    metrics: dict = {"n_folds": n_folds, **_pack_binary_metrics(y_true_bin, y_pred_bin, y_prob)}
    metrics["n_valid_predictions"] = int(dev_mask.sum())
    metrics["confusion_matrix_4class"] = confusion_matrix(y_true_4, y_pred_4, labels=[0, 1, 2, 3]).tolist()

    for cls in range(4):
        y_cls_t = (y_true_4 == cls).astype(int)
        y_cls_p = (y_pred_4 == cls).astype(int)
        metrics[f"recall_class_{cls}"] = float(recall_score(y_cls_t, y_cls_p, zero_division=0))
        metrics[f"precision_class_{cls}"] = float(precision_score(y_cls_t, y_cls_p, zero_division=0))

    lead_time = _compute_detection_lead_time(
        pd.Series(all_states[:dev_n]),
        pd.Series((all_actuals_4class[:dev_n] >= 2).astype(float)),
        crisis_min=2,
    )
    metrics["avg_detection_lead_time_weeks"] = float(lead_time["mean"])
    metrics["detection_lead_time_distribution"] = lead_time
    metrics["fold_diagnostics"] = fold_diagnostics
    metrics["fold_records"] = fold_records_lstm
    metrics["calibration_method"] = calibration_method
    metrics["holdout_weeks"] = int(n_samples - dev_n)
    metrics["lstm_hyperparams"] = {
        "sequence_length": sequence_length,
        "hidden_size": int(lstm_runtime_config.get("modeling", {}).get("lstm", {}).get("hidden_size", 16)),
        "learning_rate": float(lstm_runtime_config.get("modeling", {}).get("lstm", {}).get("learning_rate", 0.001)),
        "dropout": float(lstm_runtime_config.get("modeling", {}).get("lstm", {}).get("dropout", 0.2)),
        "batch_size": int(lstm_runtime_config.get("modeling", {}).get("lstm", {}).get("batch_size", 16)),
    }
    if tuning_summary:
        metrics["lstm_search_summary"] = tuning_summary

    if hold_mask.sum() > 0:
        h_true_4 = all_actuals_4class[hold_mask].astype(int)
        h_pred_4 = all_states[hold_mask].astype(int)
        h_prob = all_probs[hold_mask]
        h_true_bin = (h_true_4 >= 2).astype(int)
        h_pred_bin = (h_pred_4 >= 2).astype(int)
        holdout_metrics = _pack_binary_metrics(h_true_bin, h_pred_bin, h_prob)
        holdout_metrics["confusion_matrix_4class"] = confusion_matrix(h_true_4, h_pred_4, labels=[0, 1, 2, 3]).tolist()
        metrics["holdout"] = holdout_metrics
    else:
        metrics["holdout"] = {"n_valid_predictions": 0}

    metrics["per_week"] = {
        "predictions": all_states.tolist(),
        "probabilities": all_probs.tolist(),
        "actuals": all_actuals_4class.tolist(),
    }
    return metrics


def evaluate_cross_subreddit_generalization(
    all_sub_dfs: dict[str, pd.DataFrame],
    config: dict,
    feature_columns: list[str],
) -> dict:
    """
    For each (source, target) subreddit pair, train an XGBoost model on the
    source subreddit and evaluate it on the target subreddit (zero-shot transfer).

    Returns a dict of {source_sub: {target_sub: {pr_auc, recall, f1, n_test, n_positive}}}.
    Useful for diagnosing whether signal learned in one community generalises across
    communities — a prerequisite for a shared/pooled model.
    """
    labeling_cfg = config.get("labeling", {})
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    weights = labeling_cfg.get("distress_weights")
    prob_threshold = float(config.get("evaluation", {}).get("probability_threshold", 0.5))

    results: dict = {}
    subreddits = list(all_sub_dfs.keys())

    for src in subreddits:
        src_df = all_sub_dfs[src]
        if len(src_df) < 20:
            continue
        src_scores = compute_distress_score(src_df, weights, normalize=False)
        labeler_src = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
        labeler_src.fit(src_scores)
        labels_src = labeler_src.label(src_scores)
        valid_src = ~labels_src.isna()
        X_src = src_df.loc[valid_src, feature_columns]
        y_src = (labels_src[valid_src].astype(int) >= 2).astype(int)
        if len(y_src) < 10 or int(y_src.sum()) < 2:
            continue
        model = XGBCrisisModel(config)
        try:
            model.train(X_src, y_src, do_search=False)
        except Exception as e:
            results.setdefault(src, {})["_train_error"] = str(e)
            continue

        results[src] = {}
        for tgt in subreddits:
            if tgt == src:
                continue
            tgt_df = all_sub_dfs[tgt]
            if len(tgt_df) < 10:
                continue
            tgt_scores = compute_distress_score(tgt_df, weights, normalize=False)
            labeler_tgt = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
            labeler_tgt.fit(tgt_scores)
            labels_tgt = labeler_tgt.label(tgt_scores)
            valid_tgt = ~labels_tgt.isna()
            X_tgt = tgt_df.loc[valid_tgt, feature_columns]
            y_tgt = (labels_tgt[valid_tgt].astype(int) >= 2).astype(int)
            if len(y_tgt) < 5:
                results[src][tgt] = {"skipped": True, "reason": "too_few_target_samples"}
                continue
            try:
                probs = model.predict_proba(X_tgt)
                preds = (probs >= prob_threshold).astype(int)
                pr_auc = float(average_precision_score(y_tgt, probs)) if int(y_tgt.sum()) > 0 else 0.0
                results[src][tgt] = {
                    "pr_auc": round(pr_auc, 4),
                    "recall": round(float(recall_score(y_tgt, preds, zero_division=0)), 4),
                    "f1": round(float(f1_score(y_tgt, preds, zero_division=0)), 4),
                    "n_test": int(len(y_tgt)),
                    "n_positive": int(y_tgt.sum()),
                }
            except Exception as e:
                results[src][tgt] = {"skipped": True, "reason": str(e)}

    return results


def evaluate_feature_family_ablation(
    feature_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
) -> dict:
    """
    Train an XGBoost model on the full feature set, then re-train with each
    feature family held out (one at a time), recording the drop in PR-AUC.

    Feature families are identified by column-name conventions:
      - linguistic:  word_count, char_count, type_token_ratio, flesch_kincaid,
                     first_person_*
      - sentiment:   avg_compound, avg_positive, avg_negative, avg_neutral,
                     pct_*, sentiment_std
      - distress:    *_density, *_total
      - behavioral:  post_volume, avg_comments, unique_posters, new_poster_ratio,
                     posting_time_entropy
      - topic:       dominant_topic, topic_entropy, topic_shift_jsd
      - temporal:    *_delta, *_roll*w, week_sin, week_cos

    Returns {family: {pr_auc, pr_auc_drop, recall, n_features_removed}}.
    """
    labeling_cfg = config.get("labeling", {})
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    weights = labeling_cfg.get("distress_weights")
    prob_threshold = float(config.get("evaluation", {}).get("probability_threshold", 0.5))

    distress_scores = compute_distress_score(feature_df, weights, normalize=False)
    labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
    labeler.fit(distress_scores)
    labels = labeler.label(distress_scores)
    valid = ~labels.isna()
    X_all = feature_df.loc[valid, feature_columns]
    y_all = (labels[valid].astype(int) >= 2).astype(int)

    if len(y_all) < 15 or int(y_all.sum()) < 2:
        return {"error": "insufficient_data_for_ablation"}

    def _pr_auc(X: pd.DataFrame, y: pd.Series) -> float:
        if int(y.sum()) < 2:
            return 0.0
        m = XGBCrisisModel(config)
        try:
            m.train(X, y, do_search=False)
            probs = m.predict_proba(X)
            return float(average_precision_score(y, probs))
        except Exception:
            return 0.0

    full_pr_auc = _pr_auc(X_all, y_all)

    _linguistic = {c for c in feature_columns if c in {
        "word_count", "char_count", "type_token_ratio", "flesch_kincaid",
        "first_person_singular_ratio", "first_person_plural_ratio",
        "avg_word_count", "avg_char_count",
    }}
    _sentiment = {c for c in feature_columns if c.startswith(("avg_compound", "avg_positive",
        "avg_negative", "avg_neutral", "pct_", "sentiment_std", "pct_high"))}
    _distress = {c for c in feature_columns if c.endswith(("_density", "_total"))}
    _behavioral = {c for c in feature_columns if c in {
        "post_volume", "avg_comments", "unique_posters", "new_poster_ratio", "posting_time_entropy",
    }}
    _topic = {c for c in feature_columns if c in {
        "dominant_topic", "topic_entropy", "topic_shift_jsd",
    }}
    _temporal = {c for c in feature_columns if (
        c.endswith("_delta") or "_roll" in c or c in {"week_sin", "week_cos"}
    )}

    families = {
        "linguistic": _linguistic,
        "sentiment": _sentiment,
        "distress": _distress,
        "behavioral": _behavioral,
        "topic": _topic,
        "temporal": _temporal,
    }

    ablation_results: dict = {"full_model_pr_auc": round(full_pr_auc, 4)}
    for family, drop_cols in families.items():
        remaining = [c for c in feature_columns if c not in drop_cols]
        if not remaining:
            ablation_results[family] = {"skipped": True, "reason": "no_remaining_features"}
            continue
        ablated_pr_auc = _pr_auc(X_all[remaining], y_all)
        ablation_results[family] = {
            "pr_auc": round(ablated_pr_auc, 4),
            "pr_auc_drop": round(full_pr_auc - ablated_pr_auc, 4),
            "n_features_removed": len(drop_cols & set(feature_columns)),
        }

    return ablation_results


def _compute_detection_lead_time(
    predictions: pd.Series,
    actuals: pd.Series,
    crisis_min: int = 1,
) -> dict:
    crisis_starts = []
    in_crisis = False

    for i in range(len(actuals)):
        val = actuals.iloc[i]
        if not np.isnan(val) and val >= crisis_min and not in_crisis:
            crisis_starts.append(i)
            in_crisis = True
        elif np.isnan(val) or val < crisis_min:
            in_crisis = False

    if not crisis_starts:
        return {
            "mean": 0.0,
            "distribution": [],
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }

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

    if not lead_times:
        return {
            "mean": 0.0,
            "distribution": [],
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    arr = np.asarray(lead_times, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "distribution": [int(v) for v in lead_times],
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }
