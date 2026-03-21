"""UI/report colors and user-facing copy."""

from src.core.domain_config import STATE_NAMES

STATE_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b"}
BADGE_BG = {0: "#d5f5e3", 1: "#fef9e7", 2: "#fdebd0", 3: "#fadbd8"}

TIMELINE_STATE_BAND_COLORS = {
    0: "rgba(0,200,0,0.15)",
    1: "rgba(255,200,0,0.20)",
    2: "rgba(255,100,0,0.25)",
    3: "rgba(220,0,0,0.30)",
}
TIMELINE_STATE_MARKER_COLORS = {0: "green", 1: "gold", 2: "orangered", 3: "darkred"}
TIMELINE_THRESHOLD_COLORS = ["gold", "orangered", "darkred"]

DASHBOARD_COPY = {
    "page_title": "Community Crisis Predictor",
    "title": "Community Mental Health Crisis Predictor",
    "caption": "Live replay dashboard — use the sidebar controls to step through weeks.",
    "current_state_header": "### Current Community Signal State",
    "state_badge_label": "Community Signal State",
    "probability_metric_label": "High-Distress Probability",
    "timeline_header": "### Distress Timeline",
    "timeline_distress_label": "Distress Score",
    "timeline_predicted_state_label": "Predicted State",
    "timeline_probability_label": "High-Distress Probability",
    "timeline_probability_axis_label": "High-Distress Probability",
    "recent_transitions_header": "### Recent State Transitions",
    "weekly_brief_header": "Weekly Brief",
    "weekly_brief_missing": "No brief for this week. Run `python -m src.pipeline.run_evaluate --config config/default.yaml` to generate weekly narratives.",
}

TIMELINE_COPY = {
    "distress_series_name": "Community Distress Score",
    "predicted_prefix": "Predicted",
    "predicted_binary": "Predicted High-Distress Signal",
    "actual_binary": "Observed High-Distress Week",
    "threshold_fallback": "Severe Community Distress Threshold",
    "title": "Community Mental Health Distress Early Warning — Backtesting Timeline",
    "xaxis_title": "Week",
    "yaxis_title": "Distress Score (z-scored)",
    "yaxis2_title": "High-Distress Probability",
}

CASE_STUDY_COPY = {
    "title_prefix": "Case Study: High-Distress Signal Week",
    "what_happened_header": "## What Happened",
    "threshold_sentence_suffix": "exceeding the severe community distress threshold.",
    "summary_header": "## Summary",
    "summary_event_noun": "high-distress event",
}

ALERT_ENGINE_COPY = {
    "transition_prefix": "STATE TRANSITION ALERT",
}

PIPELINE_COPY = {
    "run_all_description": "Run full community distress early-warning pipeline",
    "run_all_banner": "COMMUNITY MENTAL HEALTH DISTRESS EARLY-WARNING PIPELINE",
    "run_train_description": "Train and evaluate community distress prediction models",
    "xgb_section_title": "[XGBoost — binary high-distress baseline]",
    "recall_metric_label": "Recall (high-distress)",
}

# Keep exported for convenience where both semantics and UI are needed.
__all__ = [
    "STATE_NAMES",
    "STATE_COLORS",
    "BADGE_BG",
    "TIMELINE_STATE_BAND_COLORS",
    "TIMELINE_STATE_MARKER_COLORS",
    "TIMELINE_THRESHOLD_COLORS",
    "DASHBOARD_COPY",
    "TIMELINE_COPY",
    "CASE_STUDY_COPY",
    "ALERT_ENGINE_COPY",
    "PIPELINE_COPY",
]
