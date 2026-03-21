import numpy as np
import pandas as pd

from src.narration.narrative_generator import (
    build_llm_context,
    build_shap_top5_for_week,
    template_fallback,
    week_key_from_row,
)


def test_week_key_from_row():
    row = pd.Series({"iso_year": 2025, "iso_week": 3})
    assert week_key_from_row(row) == "2025-W03"


def test_build_shap_top5_for_week_orders_by_global_shap():
    sub_df = pd.DataFrame(
        {
            "iso_year": [2025, 2025, 2025],
            "iso_week": [1, 2, 3],
            "a": [10.0, 10.0, 30.0],
            "b": [5.0, 5.0, 5.0],
        }
    )
    shap_df = pd.DataFrame(
        {"feature": ["a", "b"], "mean_abs_shap": [0.9, 0.1]}
    ).sort_values("mean_abs_shap", ascending=False)
    top = build_shap_top5_for_week(2, shap_df, sub_df, baseline_window=2)
    assert top[0]["feature"] == "a"
    assert top[0]["direction"] in ("up", "down", "flat")


def test_build_llm_context_maps_state_names():
    sub_df = pd.DataFrame(
        {
            "iso_year": [2025, 2025],
            "iso_week": [10, 11],
            "feat": [0.0, 1.0],
        }
    )
    distress = pd.Series([0.1, 0.5])
    preds = np.array([0.0, 2.0])
    shap_df = pd.DataFrame({"feature": ["feat"], "mean_abs_shap": [1.0]})
    ctx = build_llm_context("depression", 1, sub_df, distress, preds, shap_df)
    assert ctx is not None
    assert ctx["subreddit"] == "r/depression"
    assert ctx["predicted_state"] == "Elevated Distress"
    assert ctx["previous_state"] == "Stable"
    assert "shap_top5" in ctx


def test_build_llm_context_skips_nan_prediction():
    sub_df = pd.DataFrame({"iso_year": [2025], "iso_week": [1], "f": [1.0]})
    distress = pd.Series([0.2])
    preds = np.array([np.nan])
    shap_df = pd.DataFrame({"feature": ["f"], "mean_abs_shap": [1.0]})
    assert build_llm_context("x", 0, sub_df, distress, preds, shap_df) is None


def test_template_fallback_pulls_playbook_bullet():
    playbook = "## Elevated Distress\n- Pin community resources for visibility.\n"
    ctx = {
        "subreddit": "r/test",
        "week": "2025-W01",
        "predicted_state": "Elevated Distress",
        "previous_state": "Stable",
        "distress_score_delta": 0.12,
        "shap_top5": [{"feature": "hopelessness_density", "direction": "up", "delta_pct": 10}],
    }
    out = template_fallback(ctx, playbook)
    assert "Elevated Distress" in out
    assert "community resources" in out
