"""
Streamlit live dashboard for Community Mental Health Crisis Predictor.

Run with:
    streamlit run src/dashboard/app.py
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── State label config ────────────────────────────────────────────────
STATE_NAMES = {0: "Stable", 1: "Emerging Distress", 2: "Acute Risk", 3: "Critical Escalation"}
STATE_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b"}
BADGE_BG = {0: "#d5f5e3", 1: "#fef9e7", 2: "#fdebd0", 3: "#fadbd8"}

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Community Crisis Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("Community Mental Health Crisis Predictor")
st.caption("Live replay dashboard — use the sidebar controls to step through weeks.")


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
def load_shap(sub: str):
    path = Path(f"data/reports/{sub}_shap.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_drift(sub: str):
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


# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Controls")

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

# ── Row 1: Current state badge ────────────────────────────────────────
st.markdown("### Current Risk State")

current_pred = predictions_all[week_idx]
current_prob = probabilities_all[week_idx]
current_distress = float(distress_scores.iloc[week_idx])

if not np.isnan(current_pred):
    state = int(current_pred)
    state_name = STATE_NAMES.get(state, "Unknown")
    badge_color = STATE_COLORS.get(state, "#95a5a6")
    bg_color = BADGE_BG.get(state, "#ecf0f1")
else:
    state_name = "No prediction yet"
    badge_color = "#bdc3c7"
    bg_color = "#f2f3f4"

col1, col2, col3, col4 = st.columns(4)
col1.markdown(
    f"""
    <div style="background:{bg_color};border-left:6px solid {badge_color};
    padding:12px 16px;border-radius:6px;">
    <b style="font-size:1.1em">Risk State</b><br>
    <span style="font-size:1.6em;color:{badge_color};font-weight:bold">{state_name}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
col2.metric("Crisis Probability", f"{current_prob:.1%}" if not np.isnan(current_prob) else "—")
col3.metric("Distress Score", f"{current_distress:.3f}")
col4.metric("Week", str(weeks[week_idx]) if week_idx < len(weeks) else "—")

st.markdown("---")

# ── Row 2: Timeline ───────────────────────────────────────────────────
st.markdown("### Distress Timeline")

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
        name="Distress Score",
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
        name="Predicted State",
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
            name="Crisis Probability",
            line=dict(color="purple", width=1, dash="dot"),
            yaxis="y2",
            opacity=0.6,
        )
    )

fig.update_layout(
    xaxis_title="Week",
    yaxis_title="Distress Score",
    yaxis2=dict(title="Crisis Prob", overlaying="y", side="right", range=[0, 1]),
    hovermode="x unified",
    template="plotly_white",
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Drift alert table ──────────────────────────────────────────
st.markdown("### Drift Alerts (up to current week)")

drift_df = load_drift(subreddit)
if drift_df is not None and not drift_df.empty:
    drift_up = drift_df.iloc[: week_idx + 1].copy()
    alert_cols = [c for c in drift_df.columns if not c.startswith("z_")]
    display_drift = drift_up[alert_cols].copy()
    # Highlight non-normal rows
    def _color_level(row):
        lvl = row.get("aggregate_level", 0)
        colors = {0: "", 1: "background-color: #fef9e7", 2: "background-color: #fdebd0",
                  3: "background-color: #fadbd8"}
        return [colors.get(lvl, "")] * len(row)

    try:
        st.dataframe(
            display_drift.style.apply(_color_level, axis=1),
            use_container_width=True,
            height=200,
        )
    except Exception:
        st.dataframe(display_drift, use_container_width=True, height=200)
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
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("No SHAP data found. Run `make evaluate` to generate.")

# ── Row 5: Recent state transitions ───────────────────────────────────
st.markdown("### Recent State Escalations")

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
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
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
