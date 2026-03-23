"""
Streamlit live dashboard for Community Mental Health Crisis Predictor.

Run with:
    streamlit run src/dashboard/app.py
"""

import json
import re
import sqlite3
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.config import load_config
from src.core import ui_config as _ui_cfg
from src.core.ui_config import (
    DASHBOARD_COPY,
    DECISION_USEFULNESS_COPY,
    STATE_COLORS,
    STATE_NAMES,
)

# Fallback if an older ui_config is on disk / bytecode cache is stale.
DEMO_MODE_COPY = getattr(
    _ui_cfg,
    "DEMO_MODE_COPY",
    {
        "sidebar_header": "Live Demo Mode",
        "section_label": "**Scenario Preview (STePS)**",
        "section_caption": "Interactive preview mode for stakeholder walkthroughs (not production behavior).",
        "about_expander_label": "What is live demo mode?",
        "about_markdown": "See project README (STePS demo) for full behavior.",
        "toggle_label": "Enable scenario preview (STePS)",
        "toggle_help_short": "Enables sandbox, timeline markers, and comparison.",
        "where_changes_label": "Where you will see changes: sidebar metrics + timeline overlay + comparison cards.",
        "timeline_overlay_toggle_label": "Show scenario overlay on timeline",
        "demo_active_subheader": "Scenario & overlays",
        "scenario_header": "What-if sandbox",
        "scenario_label": "Scenario mode - not a real prediction",
        "scenario_probability_label": "Scenario high-distress probability",
        "scenario_vs_baseline_header": "Scenario vs baseline",
        "events_header": "Context events",
        "comparison_header": "Subreddit live comparison",
    },
)
from src.dashboard.demo_utils import (
    DemoFeatureMap,
    apply_scenario_adjustments,
    event_in_range,
    parse_demo_events,
    resolve_demo_feature_map,
)
from src.labeling.target import CrisisLabeler
from src.modeling.train_xgb import XGBCrisisModel
from src.narration.narrative_generator import week_key_from_row

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_COPY["page_title"],
    page_icon="🧠",
    layout="wide",
)

st.title(DASHBOARD_COPY["title"])
st.caption(DASHBOARD_COPY["caption"])


# ── Data loading (cached) ─────────────────────────────────────────────
@st.cache_data
def load_feature_df():
    path = Path("data/features/features.parquet")
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def load_eval_results():
    path = Path("data/models/eval_results.json")
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_app_config():
    return load_config("config/default.yaml")


@st.cache_data
def load_shap(sub: str):
    path = Path(f"data/reports/{sub}/shap.csv")
    if not path.exists():
        path = Path(f"data/reports/{sub}_shap.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_drift(sub: str):
    path = Path(f"data/reports/{sub}/drift_alerts.json")
    if not path.exists():
        path = Path(f"data/reports/{sub}_drift_alerts.json")
    if not path.exists():
        return None
    return pd.read_json(path)


def load_transitions(n: int = 30) -> list[dict]:
    db = Path("data/alerts.db")
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


@st.cache_resource
def train_demo_xgb(sub: str, feature_df: pd.DataFrame, config: dict):
    sub_df = feature_df[feature_df["subreddit"] == sub].copy()
    sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    if sub_df.empty:
        return None, []

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in sub_df.columns if c not in meta_cols]
    from src.labeling.distress_score import compute_distress_score

    distress_scores = compute_distress_score(sub_df)
    labeling_cfg = config.get("labeling", {})
    labeler = CrisisLabeler(
        threshold_std=labeling_cfg.get("crisis_threshold_std", 1.5),
        thresholds_std=labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0]),
    )
    labeler.fit(distress_scores)
    labels = labeler.label(distress_scores)
    valid = ~labels.isna()
    if valid.sum() < 10:
        return None, feature_columns
    y = (labels[valid].astype(int) >= 2).astype(int)
    if y.sum() < 2:
        return None, feature_columns
    X = sub_df.loc[valid, feature_columns]
    model = XGBCrisisModel(config)
    model.train(X, y, do_search=False)
    return model, feature_columns


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Controls")

app_config = load_app_config()
demo_cfg = app_config.get("demo_mode", {})
prob_threshold = float(app_config.get("evaluation", {}).get("probability_threshold", 0.5))

# Demo toggle: single key `demo_enabled` (default from YAML once)
if "demo_enabled" not in st.session_state:
    st.session_state.demo_enabled = bool(demo_cfg.get("enabled", False))

feature_df = load_feature_df()
eval_results = load_eval_results()

if feature_df is None or eval_results is None:
    st.error(
        "No data found. Run the pipeline first:\n\n"
        "```\npython -m src.pipeline.run_all --config config/default.yaml --synthetic\n```"
    )
    st.stop()

available_subs = sorted(feature_df["subreddit"].unique().tolist())
subreddit = st.sidebar.selectbox("Subreddit", available_subs)

available_models = []
sub_results = eval_results.get(subreddit, {})
if "lstm" in sub_results and sub_results["lstm"]:
    available_models.append("LSTM")
if "xgb" in sub_results and sub_results["xgb"]:
    available_models.append("XGBoost")
if not available_models:
    available_models = ["LSTM"]

model_choice = st.sidebar.selectbox("Model", available_models)

# Get results for chosen model
if model_choice == "LSTM":
    results = sub_results.get("lstm", sub_results)
else:
    results = sub_results.get("xgb", sub_results)

per_week = results.get("per_week", {})
predictions_all = np.array(per_week.get("predictions", []))
probabilities_all = np.array(per_week.get("probabilities", []))
actuals_all = np.array(per_week.get("actuals", []))

# Filter sub data
sub_df = feature_df[feature_df["subreddit"] == subreddit].copy()
sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
n_weeks = len(sub_df)
weeks = sub_df["week_start"].values if "week_start" in sub_df.columns else np.arange(n_weeks)

# Align arrays to sub_df length
def _trim(arr, n):
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])

predictions_all = _trim(predictions_all, n_weeks)
probabilities_all = _trim(probabilities_all, n_weeks)
actuals_all = _trim(actuals_all, n_weeks)

# Week slider
st.sidebar.markdown("---")
st.sidebar.subheader("Live Replay")

if "week_idx" not in st.session_state:
    st.session_state["week_idx"] = min(40, n_weeks - 1)

week_idx = st.sidebar.slider("Current Week", 0, n_weeks - 1, st.session_state["week_idx"])
st.session_state["week_idx"] = week_idx

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("◀ Back"):
    week_idx = max(0, week_idx - 1)
    st.session_state["week_idx"] = week_idx
    st.rerun()
if col_btn2.button("▶ Advance"):
    week_idx = min(n_weeks - 1, week_idx + 1)
    st.session_state["week_idx"] = week_idx
    st.rerun()

# ── Compute distress score ────────────────────────────────────────────
from src.labeling.distress_score import compute_distress_score  # noqa: E402

distress_scores = compute_distress_score(sub_df)

# ── Sidebar: weekly brief (generated by run_evaluate) ───────────────────
_brief_week_key = week_key_from_row(sub_df.iloc[week_idx])
_brief_path = Path(f"data/reports/{subreddit}/weekly_briefs/{_brief_week_key}.txt")
if not _brief_path.exists():
    _brief_path = Path(f"data/reports/{subreddit}_weekly_brief_{_brief_week_key}.txt")
st.sidebar.markdown("---")
st.sidebar.subheader(DASHBOARD_COPY["weekly_brief_header"])


def _split_brief_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _pick_brief_palette(text: str, is_dark_mode: bool) -> dict[str, str]:
    lower = text.lower()
    if "severe" in lower or "critical" in lower:
        return {
            "border": "#f87171",
            "bg": "#451f24" if is_dark_mode else "#ffe7e7",
            "title": "#ffd7db" if is_dark_mode else "#7f1d1d",
            "text": "#ffeef1" if is_dark_mode else "#4a0f11",
            "action_bg": "#5a2229" if is_dark_mode else "#ffd7d5",
        }
    if "elevated" in lower or "vulnerability" in lower or "warning" in lower:
        return {
            "border": "#f59e0b",
            "bg": "#3c2e15" if is_dark_mode else "#fff3dc",
            "title": "#ffecbf" if is_dark_mode else "#7c4a03",
            "text": "#fff5db" if is_dark_mode else "#4a2e00",
            "action_bg": "#4b3617" if is_dark_mode else "#ffe5bc",
        }
    return {
        "border": "#60a5fa",
        "bg": "#1f3558" if is_dark_mode else "#e8f1ff",
        "title": "#d8e9ff" if is_dark_mode else "#1f3f71",
        "text": "#e9f2ff" if is_dark_mode else "#1d3557",
        "action_bg": "#274268" if is_dark_mode else "#d8e8ff",
    }


def _render_weekly_brief(raw_text: str, week_key: str):
    theme_base_local = st.get_option("theme.base") or "light"
    is_dark_mode = theme_base_local == "dark"
    palette = _pick_brief_palette(raw_text, is_dark_mode)

    sentences = _split_brief_sentences(raw_text)
    summary = sentences[0] if sentences else raw_text
    signals = sentences[1] if len(sentences) > 1 else ""
    action_sentence = ""
    for s in sentences:
        if "recommended action:" in s.lower():
            action_sentence = s
            break
    if not action_sentence and len(sentences) > 2:
        action_sentence = sentences[2]

    action_text = action_sentence
    if action_sentence.lower().startswith("recommended action:"):
        action_text = action_sentence.split(":", 1)[1].strip()

    brief_html = f"""
    <div style="
        border-left:5px solid {palette['border']};
        background:linear-gradient(180deg, {palette['bg']} 0%, rgba(0,0,0,0) 240%);
        border-radius:10px;
        padding:12px 12px 10px 12px;
        margin-top:4px;
    ">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
        <div style="font-size:0.80rem;letter-spacing:0.04em;color:{palette['title']};font-weight:700;">WEEKLY SNAPSHOT</div>
        <div style="font-size:0.74rem;opacity:0.92;color:{palette['title']};">{escape(week_key)}</div>
      </div>

      <div style="margin-top:8px;color:{palette['text']};line-height:1.5;font-size:0.98rem;">
        <b>Summary:</b> {escape(summary)}
      </div>

      <div style="margin-top:8px;color:{palette['text']};line-height:1.45;font-size:0.94rem;">
        <b>Key signals:</b> {escape(signals) if signals else 'Signal details unavailable for this week.'}
      </div>

      <div style="margin-top:10px;padding:8px 10px;border-radius:8px;background:{palette['action_bg']};color:{palette['text']};font-size:0.94rem;line-height:1.45;">
        <b>Recommended action</b><br>{escape(action_text) if action_text else 'Review moderation playbook guidance for this signal level.'}
      </div>
    </div>
    """

    st.sidebar.markdown(brief_html, unsafe_allow_html=True)


if _brief_path.exists():
    _brief_text = _brief_path.read_text(encoding="utf-8").strip()
    _render_weekly_brief(_brief_text, _brief_week_key)
else:
    st.sidebar.caption(DASHBOARD_COPY["weekly_brief_missing"])

st.sidebar.markdown("---")
st.sidebar.markdown(DEMO_MODE_COPY["section_label"])
st.sidebar.caption(DEMO_MODE_COPY.get("section_caption", ""))
_about = DEMO_MODE_COPY.get("about_markdown", "").strip()
if _about:
    with st.sidebar.expander(DEMO_MODE_COPY.get("about_expander_label", "About"), expanded=False):
        st.markdown(_about)
demo_enabled = st.sidebar.checkbox(
    DEMO_MODE_COPY.get("toggle_label", "Enable demo tools"),
    key="demo_enabled",
    help=DEMO_MODE_COPY.get("toggle_help_short", ""),
)

# ── Row 1: Current state badge ────────────────────────────────────────
st.markdown(DASHBOARD_COPY["current_state_header"])

current_pred = predictions_all[week_idx]
current_prob = probabilities_all[week_idx]
current_distress = float(distress_scores.iloc[week_idx])
theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base == "dark"

def _format_week_label(value) -> str:
    try:
        dt = pd.to_datetime(value)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(value)[:10]


def _last_predicted_state_for_sub(sub_name: str) -> int:
    sub_eval = eval_results.get(sub_name, {})
    if "lstm" in sub_eval or "xgb" in sub_eval:
        sub_res = sub_eval.get("lstm") or sub_eval.get("xgb", {})
    else:
        sub_res = sub_eval
    sub_pred = np.array((sub_res.get("per_week", {}) or {}).get("predictions", []))
    sub_df_cmp = feature_df[feature_df["subreddit"] == sub_name].copy()
    sub_df_cmp = sub_df_cmp.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    if len(sub_pred) < len(sub_df_cmp):
        sub_pred = np.concatenate([sub_pred, np.full(len(sub_df_cmp) - len(sub_pred), np.nan)])
    return int(sub_pred[-1]) if len(sub_pred) and not np.isnan(sub_pred[-1]) else 0


week_label = _format_week_label(weeks[week_idx]) if week_idx < len(weeks) else "-"

scenario_prob = None
baseline_prob = None
delta_prob = None
scenario_label = None
baseline_label = None
hop_delta = 0.0
vol_delta = 0.0
late_delta = 0.0
show_scenario_overlay = False

# ── Demo mode: what-if sandbox ────────────────────────────────────────
if demo_enabled:
    st.sidebar.markdown("---")
    st.sidebar.subheader(DEMO_MODE_COPY.get("demo_active_subheader", DEMO_MODE_COPY["sidebar_header"]))
    st.sidebar.caption(DEMO_MODE_COPY["scenario_label"])
    st.sidebar.info(DEMO_MODE_COPY.get("where_changes_label", ""))
    st.sidebar.markdown(f"**{DEMO_MODE_COPY['scenario_header']}**")
    what_if_cfg = demo_cfg.get("what_if", {})
    hop_limit = int(what_if_cfg.get("hopelessness_density_pct", 30))
    vol_limit = int(what_if_cfg.get("post_volume_pct", 50))
    late_limit = int(what_if_cfg.get("late_night_ratio_pct", 20))

    hop_delta = st.sidebar.slider("Hopelessness density (%)", -hop_limit, hop_limit, 0, 1)
    vol_delta = st.sidebar.slider("Post volume (%)", -vol_limit, vol_limit, 0, 1)
    late_delta = st.sidebar.slider("Late-night ratio (%)", -late_limit, late_limit, 0, 1)
    show_scenario_overlay = st.sidebar.checkbox(
        DEMO_MODE_COPY.get("timeline_overlay_toggle_label", "Show scenario overlay on timeline"),
        value=True,
    )

    feature_map: DemoFeatureMap = resolve_demo_feature_map(list(sub_df.columns))
    if not any([feature_map.hopelessness_feature, feature_map.post_volume_feature, feature_map.late_night_feature]):
        st.sidebar.warning("Scenario inputs unavailable: required feature columns not found.")
    else:
        demo_model, demo_feature_columns = train_demo_xgb(subreddit, feature_df, app_config)
        if demo_model is None or not demo_feature_columns:
            st.sidebar.warning("Scenario model unavailable for this subreddit (insufficient training rows).")
        else:
            row_features = sub_df.iloc[week_idx][demo_feature_columns]
            scenario_features = apply_scenario_adjustments(
                row_features,
                feature_map,
                hopelessness_pct=float(hop_delta),
                post_volume_pct=float(vol_delta),
                late_night_pct=float(late_delta),
            )
            scenario_df = pd.DataFrame([scenario_features], columns=demo_feature_columns)
            scenario_prob = float(demo_model.predict_proba(scenario_df)[0])
            baseline_prob = float(current_prob) if not np.isnan(current_prob) else float(demo_model.predict_proba(pd.DataFrame([row_features], columns=demo_feature_columns))[0])
            delta_prob = scenario_prob - baseline_prob
            baseline_label = "High-distress alert" if baseline_prob >= prob_threshold else "No high-distress alert"
            scenario_label = "High-distress alert" if scenario_prob >= prob_threshold else "No high-distress alert"

            st.sidebar.markdown(f"**{DEMO_MODE_COPY['scenario_vs_baseline_header']}**")
            st.sidebar.metric("Baseline probability", f"{baseline_prob:.1%}")
            st.sidebar.metric(DEMO_MODE_COPY["scenario_probability_label"], f"{scenario_prob:.1%}", delta=f"{delta_prob:+.1%}")
            st.sidebar.caption(f"Baseline: {baseline_label} -> Scenario: {scenario_label}")

if not np.isnan(current_pred):
    state = int(current_pred)
    state_name = STATE_NAMES.get(state, "Unknown")
    if is_dark:
        dark_badge_style = {
            0: {"bg": "#153528", "border": "#4ade80", "label": "#d9ffe9", "value": "#86efac"},
            1: {"bg": "#3a2f14", "border": "#facc15", "label": "#fff6d5", "value": "#fde68a"},
            2: {"bg": "#4a2a11", "border": "#fb923c", "label": "#ffedd8", "value": "#fdba74"},
            3: {"bg": "#4a1d22", "border": "#f87171", "label": "#ffe3e6", "value": "#fca5a5"},
        }
        badge_style = dark_badge_style.get(
            state, {"bg": "#1f2937", "border": "#9ca3af", "label": "#e5e7eb", "value": "#f3f4f6"}
        )
    else:
        light_badge_style = {
            0: {"bg": "#e6f7ee", "border": "#22c55e", "label": "#14532d", "value": "#15803d"},
            1: {"bg": "#fff8db", "border": "#eab308", "label": "#713f12", "value": "#a16207"},
            2: {"bg": "#ffecd8", "border": "#f97316", "label": "#7c2d12", "value": "#c2410c"},
            3: {"bg": "#ffe5e5", "border": "#ef4444", "label": "#7f1d1d", "value": "#b91c1c"},
        }
        badge_style = light_badge_style.get(
            state, {"bg": "#f3f4f6", "border": "#9ca3af", "label": "#374151", "value": "#111827"}
        )
else:
    state_name = "No prediction yet"
    badge_style = {
        "bg": "#1f2937" if is_dark else "#f3f4f6",
        "border": "#9ca3af",
        "label": "#e5e7eb" if is_dark else "#374151",
        "value": "#f3f4f6" if is_dark else "#111827",
    }

col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"""
    <div style="background:{badge_style['bg']};border-left:6px solid {badge_style['border']};
    padding:12px 16px;border-radius:6px;">
    <b style="font-size:1.1em;color:{badge_style['label']};">{DASHBOARD_COPY["state_badge_label"]}</b><br>
    <span style="font-size:1.6em;color:{badge_style['value']};font-weight:bold">{state_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
col2.metric(DASHBOARD_COPY["probability_metric_label"], f"{current_prob:.1%}" if not np.isnan(current_prob) else "—")
col3.metric("Distress Score", f"{current_distress:.3f}")
col4.metric("Week", week_label)

if demo_enabled and scenario_prob is not None and baseline_prob is not None and delta_prob is not None:
    st.markdown("### Scenario Impact Preview")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Baseline probability", f"{baseline_prob:.1%}")
    p2.metric("Scenario probability", f"{scenario_prob:.1%}", delta=f"{delta_prob:+.1%}")
    p3.metric("Baseline state", baseline_label or "—")
    p4.metric("Scenario state", scenario_label or "—")
    st.caption(
        f"Active scenario input: hopelessness {hop_delta:+.0f}%, "
        f"post volume {vol_delta:+.0f}%, late-night ratio {late_delta:+.0f}%."
    )

st.markdown("---")

# ── Row 2: Timeline ───────────────────────────────────────────────────
st.markdown(DASHBOARD_COPY["timeline_header"])

is_multiclass = bool(np.nanmax(predictions_all) > 1) if (~np.isnan(predictions_all)).any() else False
fig = go.Figure()

# Show history up to current week
x_hist = weeks[: week_idx + 1]
y_hist = distress_scores.values[: week_idx + 1]

fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=y_hist,
        mode="lines",
        name=DASHBOARD_COPY["timeline_distress_label"],
        line=dict(color="steelblue", width=2),
    )
)

# State-colored markers for predictions up to current week
marker_colors_list = []
for i in range(week_idx + 1):
    p = predictions_all[i]
    if not np.isnan(p):
        marker_colors_list.append(STATE_COLORS.get(int(p), "#95a5a6"))
    else:
        marker_colors_list.append("#bdc3c7")

fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=y_hist,
        mode="markers",
        name=DASHBOARD_COPY["timeline_predicted_state_label"],
        marker=dict(color=marker_colors_list, size=8),
        hovertemplate="%{x}<br>Distress: %{y:.3f}<extra></extra>",
    )
)

# Crisis probability on secondary axis
probs_up_to = probabilities_all[: week_idx + 1]
valid_prob = ~np.isnan(probs_up_to)
if valid_prob.any():
    fig.add_trace(
        go.Scatter(
            x=x_hist[valid_prob],
            y=probs_up_to[valid_prob],
            mode="lines",
            name=DASHBOARD_COPY["timeline_probability_label"],
            line=dict(color="purple", width=1, dash="dot"),
            yaxis="y2",
            opacity=0.6,
        )
    )

if demo_enabled and show_scenario_overlay and not sub_df.empty:
    feature_map = resolve_demo_feature_map(list(sub_df.columns))
    demo_model, demo_feature_columns = train_demo_xgb(subreddit, feature_df, app_config)
    if (
        demo_model is not None
        and demo_feature_columns
        and any([feature_map.hopelessness_feature, feature_map.post_volume_feature, feature_map.late_night_feature])
    ):
        scenario_prob_series: list[float] = []
        for idx in range(week_idx + 1):
            base_row = sub_df.iloc[idx][demo_feature_columns]
            adjusted = apply_scenario_adjustments(
                base_row,
                feature_map,
                hopelessness_pct=float(hop_delta),
                post_volume_pct=float(vol_delta),
                late_night_pct=float(late_delta),
            )
            scenario_df = pd.DataFrame([adjusted], columns=demo_feature_columns)
            scenario_prob_series.append(float(demo_model.predict_proba(scenario_df)[0]))
        fig.add_trace(
            go.Scatter(
                x=x_hist,
                y=np.array(scenario_prob_series),
                mode="lines",
                name="Scenario probability",
                line=dict(color="#ef4444", width=2, dash="dash"),
                yaxis="y2",
                opacity=0.9,
            )
        )

if demo_enabled:
    for event_dt, event_label in parse_demo_events(demo_cfg.get("events", [])):
        if not event_in_range(event_dt, x_hist):
            continue
        fig.add_vline(x=event_dt, line_width=1, line_dash="dash", line_color="#6366f1")
        fig.add_trace(
            go.Scatter(
                x=[event_dt],
                y=[float(np.nanmax(y_hist)) if len(y_hist) else 0.0],
                mode="markers",
                marker=dict(size=10, opacity=0.0),
                name=event_label,
                hovertemplate=f"{event_label}<br>%{{x}}<extra></extra>",
                showlegend=False,
            )
        )

fig.update_layout(
    xaxis_title="Week",
    yaxis_title="Distress Score",
    yaxis2=dict(title=DASHBOARD_COPY["timeline_probability_axis_label"], overlaying="y", side="right", range=[0, 1]),
    hovermode="x unified",
    template="plotly_white",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, width="stretch")

# ── Demo mode: side-by-side subreddit comparison ──────────────────────
if demo_enabled:
    st.markdown(DEMO_MODE_COPY["comparison_header"])
    all_subs = sorted(
        feature_df["subreddit"].unique().tolist(),
        key=lambda s: (-_last_predicted_state_for_sub(s), s),
    )
    cols = st.columns(3)
    for idx, sub_name in enumerate(all_subs):
        col = cols[idx % 3]
        sub_eval = eval_results.get(sub_name, {})
        if "lstm" in sub_eval or "xgb" in sub_eval:
            sub_res = sub_eval.get("lstm") or sub_eval.get("xgb", {})
        else:
            sub_res = sub_eval
        sub_pred = np.array((sub_res.get("per_week", {}) or {}).get("predictions", []))
        sub_df_cmp = feature_df[feature_df["subreddit"] == sub_name].copy()
        sub_df_cmp = sub_df_cmp.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        sub_scores = compute_distress_score(sub_df_cmp)
        if len(sub_pred) < len(sub_df_cmp):
            sub_pred = np.concatenate([sub_pred, np.full(len(sub_df_cmp) - len(sub_pred), np.nan)])
        sub_state = int(sub_pred[-1]) if len(sub_pred) and not np.isnan(sub_pred[-1]) else 0
        sub_state_name = STATE_NAMES.get(sub_state, "Unknown")
        badge_color = STATE_COLORS.get(sub_state, "#95a5a6")

        with col:
            st.markdown(
                f"<div style='border-left:5px solid {badge_color};padding:8px 10px;border-radius:8px;background:rgba(148,163,184,0.12)'>"
                f"<b>r/{sub_name}</b><br><span style='color:{badge_color};font-weight:700'>{sub_state_name}</span></div>",
                unsafe_allow_html=True,
            )
            spark = sub_scores.tail(8).reset_index(drop=True)
            spark_fig = go.Figure(
                go.Scatter(
                    x=list(range(len(spark))),
                    y=spark.values,
                    mode="lines",
                    line=dict(color=badge_color, width=2),
                    fill="tozeroy",
                    fillcolor="rgba(59,130,246,0.10)",
                    hovertemplate="t-%{x}: %{y:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
            spark_fig.update_layout(
                margin=dict(l=8, r=8, t=8, b=8),
                height=110,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                template="plotly_white",
            )
            st.plotly_chart(spark_fig, width="stretch")

# ── Row 3: Drift alert table ──────────────────────────────────────────
st.markdown("### Drift Alerts (up to current week)")

drift_df = load_drift(subreddit)
if drift_df is not None and not drift_df.empty:
    drift_up = drift_df.iloc[: week_idx + 1].copy()
    alert_cols = [c for c in drift_df.columns if not c.startswith("z_")]
    display_drift = drift_up[alert_cols].copy()

    theme_base = st.get_option("theme.base") or "light"
    is_dark = theme_base == "dark"

    # Use explicit text and background colors to preserve contrast in both themes.
    if is_dark:
        frame_color = "#273244"
        header_bg = "#121826"
        header_text = "#e6edf7"
        row_styles = {
            0: ("#0b1220", "#dbe7ff"),
            1: ("#33481f", "#f8ffe8"),
            2: ("#5b3713", "#fff7e6"),
            3: ("#5c1f27", "#fff1f3"),
        }
    else:
        frame_color = "#d8deea"
        header_bg = "#f2f5fb"
        header_text = "#1f2937"
        row_styles = {
            0: ("#ffffff", "#111827"),
            1: ("#fff8db", "#3d2f00"),
            2: ("#ffe9cc", "#4a2b00"),
            3: ("#ffd9d6", "#5a1516"),
        }

    def _format_cell(col: str, value):
        if pd.isna(value):
            return "-"
        if col == "week_start":
            try:
                numeric = float(value)
                if numeric > 1e12:
                    return pd.to_datetime(int(numeric), unit="ms").strftime("%Y-%m-%d")
                if numeric > 1e9:
                    return pd.to_datetime(int(numeric), unit="s").strftime("%Y-%m-%d")
            except Exception:
                pass
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    table_rows = []
    for _, row in display_drift.iterrows():
        level = int(row.get("aggregate_level", 0)) if not pd.isna(row.get("aggregate_level", 0)) else 0
        bg, text = row_styles.get(level, row_styles[0])
        row_cells = "".join(
            f"<td style='padding:10px 12px;border-top:1px solid {frame_color};color:{text};'>{escape(_format_cell(col, row[col]))}</td>"
            for col in display_drift.columns
        )
        table_rows.append(f"<tr style='background:{bg};'>{row_cells}</tr>")

    header_cells = "".join(
        f"<th style='text-align:left;padding:10px 12px;background:{header_bg};color:{header_text};border-bottom:1px solid {frame_color};'>{escape(str(col))}</th>"
        for col in display_drift.columns
    )

    drift_table_html = f"""
    <div style='border:1px solid {frame_color}; border-radius:10px; overflow:auto; max-height:240px;'>
      <table style='border-collapse:separate; border-spacing:0; width:100%; font-size:0.95rem;'>
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(drift_table_html, unsafe_allow_html=True)
else:
    st.info("No drift data found. Run `make evaluate` to generate.")

# ── Row 4: Feature importance ─────────────────────────────────────────
st.markdown("### Feature Importance (SHAP — top 15)")

shap_df = load_shap(subreddit)
if shap_df is not None:
    top15 = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)
    fig_shap = go.Figure(
        go.Bar(
            x=top15["mean_abs_shap"],
            y=top15["feature"],
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig_shap.update_layout(
        xaxis_title="Mean |SHAP|",
        yaxis_title="",
        template="plotly_white",
        height=400,
        margin=dict(l=200),
    )
    st.plotly_chart(fig_shap, width="stretch")
else:
    st.info("No SHAP data found. Run `make evaluate` to generate.")

# ── Row 5: Recent state transitions ───────────────────────────────────
st.markdown(DASHBOARD_COPY["recent_transitions_header"])

transitions = load_transitions(n=20)
if transitions:
    rows = []
    for t in transitions:
        from_name = STATE_NAMES.get(t["from_state"], str(t["from_state"]))
        to_name = STATE_NAMES.get(t["to_state"], str(t["to_state"]))
        rows.append(
            {
                "Timestamp": t["timestamp"][:19],
                "Subreddit": t["subreddit"],
                "Week": t["week_start"],
                "Transition": f"{from_name} -> {to_name}",
                "Distress": f"{t['distress_score']:.3f}",
                "Top Signal": t["dominant_signal"],
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch")
else:
    st.info(
        "No escalations logged yet. Run the full pipeline to populate alerts.db."
    )

# ── Metrics panel ─────────────────────────────────────────────────────
with st.expander("Model Metrics", expanded=False):
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Recall", f"{results.get('recall', 0):.3f}")
    col_b.metric("Precision", f"{results.get('precision', 0):.3f}")
    col_c.metric("F1", f"{results.get('f1', 0):.3f}")
    col_d.metric("PR-AUC", f"{results.get('pr_auc', 0):.3f}")

    if "confusion_matrix_4class" in results:
        st.markdown("**4-class confusion matrix**")
        cm = results["confusion_matrix_4class"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {STATE_NAMES[i]}" for i in range(4)],
            columns=[f"Pred {STATE_NAMES[i]}" for i in range(4)],
        )
        st.dataframe(cm_df)

    if "recall_class_0" in results:
        st.markdown("**Per-class recall**")
        for cls in range(4):
            val = results.get(f"recall_class_{cls}", 0)
            st.write(f"- {STATE_NAMES[cls]}: {val:.3f}")

    _du = results.get("decision_usefulness")
    if _du and isinstance(_du, dict):
        st.markdown(DECISION_USEFULNESS_COPY["title"])
        st.markdown(DECISION_USEFULNESS_COPY["intro"])
        kvals = _du.get("k_values") or []
        model = _du.get("model") or {}
        rnd = _du.get("random_expected_recall") or {}
        pers = _du.get("persistence") or {}
        rows_du = []
        for k in kvals:
            ks = str(k)
            mk = model.get(ks) or model.get(k) or {}
            pk = pers.get(ks) or pers.get(k) or {}
            cap = mk.get("captured", 0)
            tot = mk.get("total_positives", _du.get("n_elevated_distress_weeks", 0))
            rec = mk.get("recall", 0.0)
            r_rnd = rnd.get(str(k), rnd.get(k, 0.0))
            p_cap = pk.get("captured", 0)
            p_rec = pk.get("recall", 0.0)
            rows_du.append(
                {
                    "K": k,
                    "Captured (model)": f"{cap}/{tot}",
                    "Recall@K (model)": f"{float(rec):.1%}",
                    "Expected Recall@K (random)": f"{float(r_rnd):.1%}",
                    "Persistence": f"{p_cap}/{tot} ({float(p_rec):.1%})",
                }
            )
        st.caption(
            f"Evaluation weeks n={_du.get('n_weeks', '—')}, "
            f"elevated-distress weeks P={_du.get('n_elevated_distress_weeks', '—')}."
        )
        st.dataframe(pd.DataFrame(rows_du), width="stretch", hide_index=True)
