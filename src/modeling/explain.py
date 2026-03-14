from pathlib import Path

import numpy as np
import pandas as pd
import shap


def compute_shap_importance(
    model,
    X: pd.DataFrame,
    feature_names: list[str] = None,
) -> pd.DataFrame:
    if feature_names is None:
        feature_names = list(X.columns)

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X)

    importance = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": importance,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return result


def generate_shap_waterfall(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer(X_row)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model.model)
    return explainer.shap_values(X)
