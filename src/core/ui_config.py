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
    "title": "Community mental health early warning",
    "caption": "Walk-forward backtest · r/depression, r/anxiety, r/lonely, r/mentalhealth, r/SuicideWatch · 2018–2024 · Zenodo + Arctic Shift",
    "current_state_header": "### Predicted state (following week)",
    "state_badge_label": "Predicted state (following week)",
    "probability_metric_label": "P(high-distress, following week)",
    "timeline_header": "### Observed distress score + walk-forward predictions",
    "timeline_distress_label": "Composite distress score (z-scored)",
    "timeline_predicted_state_label": "Predicted state (following week)",
    "timeline_probability_label": "P(high-distress, following week)",
    "timeline_probability_axis_label": "P(high-distress, following week)",
    "recent_transitions_header": "### Recent predicted state transitions",
    "weekly_brief_header": "Weekly Brief",
    "weekly_brief_missing": "No brief for this week. Run `python -m src.pipeline.run_evaluate --config config/default.yaml` to generate weekly narratives.",
}

TIMELINE_COPY = {
    "distress_series_name": "Community Distress Score",
    "predicted_prefix": "Predicted",
    "predicted_binary": "Predicted High-Distress Signal",
    "actual_binary": "Observed High-Distress Week",
    "threshold_fallback": "Severe Community Distress Threshold",
    "title": "Composite distress score + walk-forward backtested predictions",
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

END_USER_COPY = {
    "page_title": "Community Copilot",
    "title": "Weekly summary",
    "subtitle": "Plain-language view for stakeholders. Numbers come from your pipeline outputs; this is not medical advice.",
    "hero_week_prefix": "Week of",
    "community_label": "Community",
    "model_note_expander": "Technical note (optional)",
    "model_note_body": (
        "The headline uses the model’s **predicted state for the following week** and **high-distress probability**. "
        "You can change the model on the **app** (analyst) page; this page stays in sync via the same session settings."
    ),
    "bullet_what_we_see": "What we see in the data",
    "bullet_model_flag": "What the forecast suggests (next week)",
    "bullet_confidence": "How strong that signal is",
    "brief_header": "Weekly brief",
    "brief_missing": "No pre-generated brief for this week. Run evaluation to populate weekly briefs, or use the button below.",
    "llm_button": "Generate plain-language explanation (LLM)",
    "llm_help": "Uses only structured metrics and brief excerpt—no raw posts. Requires API keys in environment or Streamlit secrets.",
    "llm_based_on": "Based on (structured facts)",
    "llm_result_header": "Explanation",
    "llm_footer_note": "Generated text may still contain errors; verify against the facts above.",
    "limitations_expander": "Limitations and responsible use",
    "limitations_body": (
        "- **Population-level only** — describes patterns in a subreddit for a week, not any person.\n"
        "- **Weekly lag** — based on aggregated weekly features, not real-time chat.\n"
        "- **Not clinical** — not a diagnosis or treatment recommendation.\n"
        "- **Data and model limits** — trained on historical Reddit-era data; live behavior can differ.\n"
        "- **Moderation support** — intended to prioritize attention, not replace human judgment."
    ),
    "sidebar_hint": (
        "Moderators: open **Community Copilot** in the app navigation for the triage view. "
        "The **app** entry is the full analyst dashboard (charts, tabs, model picker)."
    ),
    "confidence_low": "Lower — treat as background noise unless sustained.",
    "confidence_moderate": "Moderate — worth monitoring alongside other signals.",
    "confidence_elevated": "Elevated — review internal playbooks and recent context.",
    "distress_vs_prior_up": "Language-based distress signal is **higher** than the prior week.",
    "distress_vs_prior_down": "Language-based distress signal is **lower** than the prior week.",
    "distress_vs_prior_flat": "Language-based distress signal is **similar** to the prior week.",
    "nav_caption": "Week and community (synced with Home)",
}

DECISION_USEFULNESS_COPY = {
    "title": "### Decision usefulness (top-K alert recall)",
    "intro": (
        "Assume the team can only **review K weeks** in the evaluation history. "
        "Weeks are ranked by the model’s **predicted high-distress probability** (highest first); "
        "the top K are treated as alerts. **Recall@K** is the fraction of **true elevated-distress weeks** "
        "(same binary target as PR-AUC: actual state ≥ 2) that fall inside those K alerts. "
        "**Random** = expected recall if K weeks were chosen uniformly at random. "
        "**Persistence** = rank weeks by whether the **previous** week was elevated-distress (simple baseline)."
    ),
    "table_header": "Metric comparison (same K)",
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
    "END_USER_COPY",
    "TIMELINE_COPY",
    "CASE_STUDY_COPY",
    "ALERT_ENGINE_COPY",
    "PIPELINE_COPY",
    "DECISION_USEFULNESS_COPY",
]
