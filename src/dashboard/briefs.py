import re
from html import escape

import streamlit as st


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

