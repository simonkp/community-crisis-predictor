from typing import Any, TypedDict


class EvalModelPayload(TypedDict, total=False):
    recall: float
    precision: float
    f1: float
    pr_auc: float
    n_crisis_actual: int
    n_crisis_predicted: int
    per_week: dict[str, Any]

