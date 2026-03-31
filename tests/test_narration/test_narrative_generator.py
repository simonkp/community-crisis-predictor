import json

import numpy as np
import pandas as pd
import pytest

from src.narration.narrative_generator import (
    build_llm_context,
    build_shap_top5_for_week,
    load_weekly_briefs_json,
    template_fallback,
    week_key_from_row,
    write_weekly_brief_json,
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


def test_write_weekly_brief_json_creates_file(tmp_path):
    path = write_weekly_brief_json(tmp_path, "depression", "2025-W01", "Some brief.", source="template")
    assert path == tmp_path / "depression" / "weekly_briefs.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert "2025-W01" in data
    assert data["2025-W01"]["text"] == "Some brief."
    assert data["2025-W01"]["source"] == "template"


def test_write_weekly_brief_json_upserts_existing(tmp_path):
    write_weekly_brief_json(tmp_path, "sub", "2025-W01", "First", source="template")
    write_weekly_brief_json(tmp_path, "sub", "2025-W02", "Second", source="anthropic")
    write_weekly_brief_json(tmp_path, "sub", "2025-W01", "Updated", source="openai")
    briefs = load_weekly_briefs_json(tmp_path, "sub")
    assert briefs["2025-W01"]["text"] == "Updated"
    assert briefs["2025-W01"]["source"] == "openai"
    assert briefs["2025-W02"]["text"] == "Second"
    assert len(briefs) == 2


def test_write_weekly_brief_json_sorted_keys(tmp_path):
    for wk in ["2025-W10", "2025-W03", "2025-W07"]:
        write_weekly_brief_json(tmp_path, "anxiety", wk, f"Brief {wk}")
    briefs = load_weekly_briefs_json(tmp_path, "anxiety")
    assert list(briefs.keys()) == ["2025-W03", "2025-W07", "2025-W10"]


def test_load_weekly_briefs_json_missing(tmp_path):
    assert load_weekly_briefs_json(tmp_path, "nonexistent") == {}


def test_load_weekly_briefs_json_malformed(tmp_path):
    sub_dir = tmp_path / "broken"
    sub_dir.mkdir()
    (sub_dir / "weekly_briefs.json").write_text("not json", encoding="utf-8")
    assert load_weekly_briefs_json(tmp_path, "broken") == {}


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
