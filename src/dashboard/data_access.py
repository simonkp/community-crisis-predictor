import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.narration.narrative_generator import load_weekly_briefs_json


@st.cache_data
def load_app_config():
    return load_config("config/default.yaml")


@st.cache_data
def load_feature_df():
    cfg = load_app_config()
    path = Path(cfg["paths"]["features"]) / "features.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_eval_results():
    cfg = load_app_config()
    path = Path(cfg["paths"]["models"]) / "eval_results.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_shap(sub: str):
    cfg = load_app_config()
    reports_root = Path(cfg["paths"]["reports"])
    path = reports_root / sub / "shap.csv"
    if not path.exists():
        path = reports_root / f"{sub}_shap.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_drift(sub: str):
    cfg = load_app_config()
    reports_root = Path(cfg["paths"]["reports"])
    path = reports_root / sub / "drift_alerts.json"
    if not path.exists():
        path = reports_root / f"{sub}_drift_alerts.json"
    if not path.exists():
        return None
    return pd.read_json(path)


@st.cache_data
def load_data_quality_report(sub: str):
    cfg = load_app_config()
    path = Path(cfg["paths"]["reports"]) / sub / "data_quality_report.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_weekly_completeness(sub: str):
    cfg = load_app_config()
    path = Path(cfg["paths"]["reports"]) / sub / "weekly_completeness.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_pipeline_profile():
    cfg = load_app_config()
    path = Path(cfg["paths"]["reports"]) / "pipeline_profile.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else [payload]


@st.cache_data
def load_weekly_briefs(subreddit_short: str) -> dict[str, dict]:
    """Load consolidated weekly_briefs.json for a subreddit.

    Returns dict keyed by week_key, each value has 'text', 'source', 'generated_at'.
    Falls back to scanning legacy .txt files if JSON is missing.
    """
    cfg = load_app_config()
    reports_root = Path(cfg["paths"]["reports"])

    briefs = load_weekly_briefs_json(reports_root, subreddit_short)
    if briefs:
        return briefs

    # Backward-compat: read legacy txt files if JSON hasn't been generated yet
    txt_dir = reports_root / subreddit_short / "weekly_briefs"
    if not txt_dir.exists():
        return {}
    result: dict[str, dict] = {}
    for f in sorted(txt_dir.glob("*.txt")):
        week_key = f.stem
        try:
            text = f.read_text(encoding="utf-8").strip()
            result[week_key] = {"text": text, "source": "legacy_txt", "generated_at": ""}
        except OSError:
            pass
    return result


def get_brief_text(subreddit_short: str, week_key: str) -> str | None:
    """Convenience: return just the brief text for a week, or None if not found."""
    briefs = load_weekly_briefs(subreddit_short)
    entry = briefs.get(week_key.replace("/", "-"))
    if entry:
        return entry.get("text")
    return None


def load_transitions(n: int = 30) -> list[dict]:
    cfg = load_app_config()
    db = Path(cfg["paths"].get("alerts_db", "data/alerts.db"))
    if not db.exists():
        return []
    with sqlite3.connect(db) as conn:
        cursor = conn.execute(
            """
            SELECT timestamp, subreddit, week_start, from_state, to_state,
                   distress_score, dominant_signal
            FROM transitions ORDER BY timestamp DESC LIMIT ?
            """,
            (n,),
        )
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]


