import numpy as np


def trim_to_length(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def clamp_week_idx(current: int, n_weeks: int) -> int:
    if n_weeks <= 0:
        return 0
    return max(0, min(int(current), n_weeks - 1))


def merge_ensemble_per_week(lstm_pw: dict, xgb_pw: dict) -> dict:
    """Average walk-forward predictions and probabilities where both models have values."""
    pred_l = np.array(lstm_pw.get("predictions", []), dtype=float)
    pred_x = np.array(xgb_pw.get("predictions", []), dtype=float)
    prob_l = np.array(lstm_pw.get("probabilities", []), dtype=float)
    prob_x = np.array(xgb_pw.get("probabilities", []), dtype=float)
    act_l = np.array(lstm_pw.get("actuals", []), dtype=float)
    act_x = np.array(xgb_pw.get("actuals", []), dtype=float)
    n = max(len(pred_l), len(pred_x), len(prob_l), len(prob_x))
    if n == 0:
        return {}
    pred_l = trim_to_length(pred_l, n)
    pred_x = trim_to_length(pred_x, n)
    prob_l = trim_to_length(prob_l, n)
    prob_x = trim_to_length(prob_x, n)
    act_l = trim_to_length(act_l, n)
    act_x = trim_to_length(act_x, n)
    merged_p = np.full(n, np.nan)
    merged_pr = np.full(n, np.nan)
    merged_a = np.full(n, np.nan)
    for i in range(n):
        pl, px = pred_l[i], pred_x[i]
        if np.isfinite(pl) and np.isfinite(px):
            merged_p[i] = int(np.clip(np.round((pl + px) / 2.0), 0, 3))
        elif np.isfinite(pl):
            merged_p[i] = pl
        elif np.isfinite(px):
            merged_p[i] = px
        al, ax = prob_l[i], prob_x[i]
        if np.isfinite(al) and np.isfinite(ax):
            merged_pr[i] = (al + ax) / 2.0
        elif np.isfinite(al):
            merged_pr[i] = al
        elif np.isfinite(ax):
            merged_pr[i] = ax
        ac_l, ac_x = act_l[i], act_x[i]
        if np.isfinite(ac_l) and np.isfinite(ac_x):
            merged_a[i] = ac_l if ac_l == ac_x else ac_l
        elif np.isfinite(ac_l):
            merged_a[i] = ac_l
        elif np.isfinite(ac_x):
            merged_a[i] = ac_x
    out = {**lstm_pw}
    out["predictions"] = merged_p.tolist()
    out["probabilities"] = merged_pr.tolist()
    if np.isfinite(merged_a).any():
        out["actuals"] = merged_a.tolist()
    return out


def merge_ensemble_results(sub_results: dict) -> dict:
    lstm = sub_results.get("lstm") or {}
    xgb = sub_results.get("xgb") or {}
    if not lstm and not xgb:
        return {}
    if not lstm:
        return dict(xgb)
    if not xgb:
        return dict(lstm)
    lstm_pw = lstm.get("per_week") or {}
    xgb_pw = xgb.get("per_week") or {}
    merged_pw = merge_ensemble_per_week(lstm_pw, xgb_pw)
    merged = dict(lstm)
    merged["per_week"] = merged_pw
    for k in ("recall", "precision", "f1", "pr_auc"):
        a, b = lstm.get(k), xgb.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            merged[k] = (float(a) + float(b)) / 2.0
        elif isinstance(a, (int, float)):
            merged[k] = float(a)
        elif isinstance(b, (int, float)):
            merged[k] = float(b)
    return merged


def pick_model_results(sub_results: dict, model_choice: str) -> dict:
    if model_choice == "LSTM":
        return sub_results.get("lstm") or {}
    if model_choice == "XGBoost":
        return sub_results.get("xgb") or {}
    if model_choice == "Ensemble":
        return merge_ensemble_results(sub_results)
    return sub_results.get("xgb") or sub_results.get("lstm") or {}


def monitoring_mode(results: dict, min_crisis_weeks: int) -> tuple[bool, int | None]:
    n = results.get("n_crisis_actual", None) if isinstance(results, dict) else None
    is_monitor = isinstance(n, (int, float)) and n < min_crisis_weeks
    return is_monitor, int(n) if isinstance(n, (int, float)) else None
