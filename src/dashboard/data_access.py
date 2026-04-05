import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import load_config
from src.narration.narrative_generator import load_weekly_briefs_json

# Cache TTL: 5 minutes — pipeline outputs are refreshed on each run, not real-time
_CACHE_TTL = 300


@st.cache_data(ttl=_CACHE_TTL)
def load_app_config():
    try:
        return load_config("config/default.yaml")
    except Exception as e:
        st.error(f"Failed to load config/default.yaml: {e}")
        return {}


@st.cache_data(ttl=_CACHE_TTL)
def load_feature_df():
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("features", "data/features")) / "features.parquet"
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        st.warning(f"Could not read features.parquet: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_eval_results():
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("models", "data/models")) / "eval_results.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not read eval_results.json: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_shap(sub: str):
    cfg = load_app_config()
    reports_root = Path(cfg.get("paths", {}).get("reports", "data/reports"))
    path = reports_root / sub / "shap.csv"
    if not path.exists():
        path = reports_root / f"{sub}_shap.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read SHAP data for r/{sub}: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_drift(sub: str):
    cfg = load_app_config()
    reports_root = Path(cfg.get("paths", {}).get("reports", "data/reports"))
    path = reports_root / sub / "drift_alerts.json"
    if not path.exists():
        path = reports_root / f"{sub}_drift_alerts.json"
    if not path.exists():
        return None
    try:
        return pd.read_json(path)
    except Exception as e:
        st.warning(f"Could not read drift alerts for r/{sub}: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_data_quality_report(sub: str):
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("reports", "data/reports")) / sub / "data_quality_report.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not read data quality report for r/{sub}: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_weekly_completeness(sub: str):
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("reports", "data/reports")) / sub / "weekly_completeness.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not read weekly completeness for r/{sub}: {e}")
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_pipeline_profile():
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("reports", "data/reports")) / "pipeline_profile.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, list) else [payload]
    except Exception as e:
        st.warning(f"Could not read pipeline profile: {e}")
        return []


@st.cache_data(ttl=_CACHE_TTL)
def load_pipeline_last_run_time() -> str | None:
    """Return mtime of eval_results.json as a human-readable string, or None."""
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("models", "data/models")) / "eval_results.json"
    if not path.exists():
        return None
    try:
        import datetime
        mtime = path.stat().st_mtime
        dt = datetime.datetime.fromtimestamp(mtime)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return None


@st.cache_data(ttl=_CACHE_TTL)
def load_weekly_briefs(subreddit_short: str) -> dict[str, dict]:
    """Load consolidated weekly_briefs.json for a subreddit.

    Returns dict keyed by week_key, each value has 'text', 'source', 'generated_at'.
    Falls back to scanning legacy .txt files if JSON is missing.
    """
    cfg = load_app_config()
    reports_root = Path(cfg.get("paths", {}).get("reports", "data/reports"))

    try:
        briefs = load_weekly_briefs_json(reports_root, subreddit_short)
        if briefs:
            return briefs
    except Exception:
        pass

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
    try:
        briefs = load_weekly_briefs(subreddit_short)
        entry = briefs.get(week_key.replace("/", "-"))
        if entry:
            return entry.get("text")
    except Exception:
        pass
    return None


@st.cache_data(ttl=_CACHE_TTL)
def load_allocation_report() -> dict | None:
    cfg = load_app_config()
    path = Path(cfg.get("paths", {}).get("reports", "data/reports")) / "allocation.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not read allocation report: {e}")
        return None


def load_transitions(n: int = 30) -> list[dict]:
    cfg = load_app_config()
    db = Path(cfg.get("paths", {}).get("alerts_db", "data/alerts.db"))
    json_path = Path("data/alerts.json")

    # Try SQLite first (local runs); fall back to committed JSON (cloud deploy)
    if db.exists():
        try:
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
        except Exception as e:
            st.warning(f"Could not load alert transitions: {e}")

    if json_path.exists():
        try:
            import json as _json
            rows = _json.loads(json_path.read_text())
            return rows[:n]
        except Exception:
            pass

    return []
