"""Weekly narrative generation (structured context + optional LLM)."""

from src.narration.narrative_generator import (
    build_llm_context,
    build_shap_top5_for_week,
    generate_narrative_from_context,
    generate_weekly_briefs_for_subreddit,
    load_playbook,
    template_fallback,
    week_key_from_row,
    write_weekly_brief,
)

__all__ = [
    "build_llm_context",
    "build_shap_top5_for_week",
    "generate_narrative_from_context",
    "generate_weekly_briefs_for_subreddit",
    "load_playbook",
    "template_fallback",
    "week_key_from_row",
    "write_weekly_brief",
]
