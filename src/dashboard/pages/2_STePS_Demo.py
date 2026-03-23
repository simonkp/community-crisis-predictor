"""Dedicated STePS demo page for scenario walkthroughs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import load_config
from src.dashboard.demo_utils import apply_scenario_adjustments, resolve_demo_feature_map
from src.labeling.target import CrisisLabeler
from src.modeling.train_xgb import XGBCrisisModel


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


@st.cache_resource
def train_demo_xgb(sub: str, feature_data: pd.DataFrame, config: dict):
    sub_data = feature_data[feature_data["subreddit"] == sub].copy()
    sub_data = sub_data.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    if sub_data.empty:
        return None, []

    meta_cols = {"subreddit", "iso_year", "iso_week", "week_start"}
    feature_columns = [c for c in sub_data.columns if c not in meta_cols]
    from src.labeling.distress_score import compute_distress_score

    distress_scores = compute_distress_score(sub_data)
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
    x = sub_data.loc[valid, feature_columns]

    model = XGBCrisisModel(config)
    model.train(x, y, do_search=False)
    return model, feature_columns


st.set_page_config(page_title="STePS Demo", page_icon="🧪", layout="wide")
st.title("STePS Demo Workspace")
st.caption("Focused scenario preview page. Changes are in-session only and do not modify saved artifacts.")

feature_df = load_feature_df()
eval_results = load_eval_results()
app_config = load_config("config/default.yaml")
prob_threshold = float(app_config.get("evaluation", {}).get("probability_threshold", 0.5))

if feature_df is None or eval_results is None:
    st.error("No data found. Run training/evaluation first.")
    st.stop()

subreddits = sorted(feature_df["subreddit"].unique().tolist())
subreddit = st.selectbox("Subreddit", subreddits)
sub_df = feature_df[feature_df["subreddit"] == subreddit].copy()
sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)

sub_eval = eval_results.get(subreddit, {})
if "lstm" in sub_eval and sub_eval["lstm"]:
    sub_res = sub_eval["lstm"]
elif "xgb" in sub_eval and sub_eval["xgb"]:
    sub_res = sub_eval["xgb"]
else:
    sub_res = sub_eval

per_week = sub_res.get("per_week", {})
probs = np.array(per_week.get("probabilities", []))
if len(probs) < len(sub_df):
    probs = np.concatenate([probs, np.full(len(sub_df) - len(probs), np.nan)])
else:
    probs = probs[: len(sub_df)]

week_idx = st.slider("Current Week", 0, max(len(sub_df) - 1, 0), min(40, max(len(sub_df) - 1, 0)))
selected_week = str(sub_df.iloc[week_idx].get("week_start", week_idx))[:10] if len(sub_df) else "-"

cfg = app_config.get("demo_mode", {}).get("what_if", {})
hop_limit = int(cfg.get("hopelessness_density_pct", 30))
vol_limit = int(cfg.get("post_volume_pct", 50))
late_limit = int(cfg.get("late_night_ratio_pct", 20))

c1, c2, c3 = st.columns(3)
hop_delta = c1.slider("Hopelessness density (%)", -hop_limit, hop_limit, 0, 1)
vol_delta = c2.slider("Post volume (%)", -vol_limit, vol_limit, 0, 1)
late_delta = c3.slider("Late-night ratio (%)", -late_limit, late_limit, 0, 1)

demo_model, demo_feature_columns = train_demo_xgb(subreddit, feature_df, app_config)
feature_map = resolve_demo_feature_map(list(sub_df.columns))

if demo_model is None or not demo_feature_columns:
    st.warning("Demo model unavailable for this subreddit (insufficient rows).")
    st.stop()

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
baseline_prob = float(probs[week_idx]) if not np.isnan(probs[week_idx]) else float(
    demo_model.predict_proba(pd.DataFrame([row_features], columns=demo_feature_columns))[0]
)
delta_prob = scenario_prob - baseline_prob

baseline_state = "High-distress alert" if baseline_prob >= prob_threshold else "No high-distress alert"
scenario_state = "High-distress alert" if scenario_prob >= prob_threshold else "No high-distress alert"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Week", selected_week)
m2.metric("Baseline probability", f"{baseline_prob:.1%}")
m3.metric("Scenario probability", f"{scenario_prob:.1%}", delta=f"{delta_prob:+.1%}")
m4.metric("State change", f"{baseline_state} -> {scenario_state}")

x_hist = sub_df["week_start"].values[: week_idx + 1]
baseline_series = probs[: week_idx + 1]
scenario_series = []
for idx in range(week_idx + 1):
    base_row = sub_df.iloc[idx][demo_feature_columns]
    adj = apply_scenario_adjustments(
        base_row,
        feature_map,
        hopelessness_pct=float(hop_delta),
        post_volume_pct=float(vol_delta),
        late_night_pct=float(late_delta),
    )
    scenario_series.append(float(demo_model.predict_proba(pd.DataFrame([adj], columns=demo_feature_columns))[0]))

fig = go.Figure()
valid_baseline = ~np.isnan(baseline_series)
if valid_baseline.any():
    fig.add_trace(
        go.Scatter(
            x=x_hist[valid_baseline],
            y=baseline_series[valid_baseline],
            mode="lines",
            name="Baseline probability",
            line=dict(color="#7c3aed", width=2),
        )
    )
fig.add_trace(
    go.Scatter(
        x=x_hist,
        y=np.array(scenario_series),
        mode="lines",
        name="Scenario probability",
        line=dict(color="#ef4444", width=2, dash="dash"),
    )
)
fig.update_layout(
    title="Scenario vs baseline probability (up to selected week)",
    xaxis_title="Week",
    yaxis_title="High-distress probability",
    yaxis=dict(range=[0, 1]),
    template="plotly_white",
    hovermode="x unified",
    height=380,
)
st.plotly_chart(fig, width="stretch")

st.info(
    "This page previews hypothetical slider effects only. "
    "Stored models, eval files, and pipeline outputs are not modified."
)
