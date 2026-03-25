import numpy as np


def trim_to_length(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def clamp_week_idx(current: int, n_weeks: int) -> int:
    if n_weeks <= 0:
        return 0
    return max(0, min(int(current), n_weeks - 1))


def pick_model_results(sub_results: dict, model_choice: str) -> dict:
    if model_choice == "LSTM":
        return sub_results.get("lstm", sub_results)
    return sub_results.get("xgb", sub_results)


def monitoring_mode(results: dict, min_crisis_weeks: int) -> tuple[bool, int | None]:
    n = results.get("n_crisis_actual", None) if isinstance(results, dict) else None
    is_monitor = isinstance(n, (int, float)) and n < min_crisis_weeks
    return is_monitor, int(n) if isinstance(n, (int, float)) else None
