"""
Structured weekly narrative: deterministic context + optional LLM (Claude / GPT-4o) + template fallback.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.domain_config import STATE_NAMES

DEFAULT_PLAYBOOK_PATH = Path("config/intervention_playbook.md")
NARRATIVE_SENTENCES = 3
FLAT_PCT_EPS = 0.5  # treat |delta_pct| below this as "flat"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o"


def week_key_from_row(row: pd.Series) -> str:
    """ISO week label consistent with case_study.py."""
    return f"{int(row['iso_year'])}-W{int(row['iso_week']):02d}"


def load_playbook(path: Path | str | None = None) -> str:
    p = Path(path) if path else DEFAULT_PLAYBOOK_PATH
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def build_shap_top5_for_week(
    week_i: int,
    shap_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    baseline_window: int = 4,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Use global SHAP ranking (mean_abs_shap); derive per-week direction vs rolling baseline.
    Baseline: mean of prior `baseline_window` rows; week 0 uses column mean (full series).
    """
    top = shap_df.head(top_n)
    out: list[dict[str, Any]] = []
    for _, r in top.iterrows():
        feat = str(r["feature"])
        if feat not in sub_df.columns:
            continue
        val = float(sub_df.iloc[week_i][feat])
        if week_i > 0:
            lo = max(0, week_i - baseline_window)
            baseline = float(sub_df[feat].iloc[lo:week_i].mean())
        else:
            baseline = float(sub_df[feat].mean())

        if baseline != 0 and not np.isnan(baseline):
            delta_pct = round((val - baseline) / abs(baseline) * 100.0, 1)
        else:
            delta_pct = 0.0

        if abs(delta_pct) < FLAT_PCT_EPS:
            direction = "flat"
        elif val > baseline:
            direction = "up"
        else:
            direction = "down"

        out.append(
            {
                "feature": feat,
                "direction": direction,
                "delta_pct": int(delta_pct) if float(delta_pct).is_integer() else delta_pct,
            }
        )
    return out


def build_llm_context(
    subreddit_short: str,
    week_i: int,
    sub_df: pd.DataFrame,
    distress_scores: pd.Series,
    predictions: np.ndarray,
    shap_df: pd.DataFrame,
    baseline_window: int = 4,
) -> dict[str, Any] | None:
    """Structured JSON-serializable context for the LLM (model outputs only)."""
    pred = predictions[week_i]
    if not np.isfinite(pred):
        return None

    state_id = int(pred)
    row = sub_df.iloc[week_i]
    prev = predictions[week_i - 1] if week_i > 0 else np.nan
    if np.isfinite(prev):
        previous_state = STATE_NAMES.get(int(prev), str(int(prev)))
    else:
        previous_state = "n/a"

    if week_i > 0:
        d_delta = float(distress_scores.iloc[week_i] - distress_scores.iloc[week_i - 1])
    else:
        d_delta = 0.0

    shap_top5 = build_shap_top5_for_week(week_i, shap_df, sub_df, baseline_window=baseline_window)

    return {
        "subreddit": f"r/{subreddit_short}",
        "week": week_key_from_row(row),
        "predicted_state": STATE_NAMES.get(state_id, str(state_id)),
        "previous_state": previous_state,
        "distress_score_delta": round(d_delta, 4),
        "shap_top5": shap_top5,
    }


def _build_user_prompt(context: dict[str, Any], playbook: str) -> str:
    ctx_json = json.dumps(context, indent=2)
    return f"""Model output (JSON only — this is the sole source of facts):
{ctx_json}

--- Retrieved moderation playbook (use for suggested actions only; do not invent new clinical guidance) ---
{playbook}

Write exactly {NARRATIVE_SENTENCES} sentences:
1) Summarize the community-level signal for the subreddit and week using the JSON only.
2) Mention the top contributing features (from shap_top5) in plain language; reference distress_score_delta only as given.
3) End with one concrete moderation-oriented action phrase beginning with "Recommended action:" chosen ONLY from the playbook text above (paraphrase allowed). Do not add medical or clinical recommendations.

Rules:
- Only use the data provided in the JSON. Do not invent risk scores, diagnoses, or statistics.
- Do not add clinical recommendations or advice for individuals.
- Population-level community signal only; measured language.
"""


SYSTEM_PROMPT = (
    "You summarize structured outputs from a community early-warning dashboard. "
    "Follow the user instructions exactly. "
    "Do not provide clinical recommendations or individual-level advice."
)


def _playbook_action_for_state(playbook: str, state: str) -> str:
    """First bullet under the ## heading that contains the state name."""
    lines = playbook.splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            in_section = state in stripped
            continue
        if in_section:
            if stripped.startswith("## "):
                break
            if stripped.startswith("- ") and len(stripped) > 2:
                return stripped[2:].strip()
    return "Review the moderation playbook for this signal level."


def template_fallback(context: dict[str, Any], playbook: str) -> str:
    """Deterministic brief when API keys are missing or LLM fails."""
    sub = context["subreddit"]
    state = context["predicted_state"]
    week = context["week"]
    delta = context["distress_score_delta"]
    top5 = context.get("shap_top5") or []
    names = ", ".join(t["feature"] for t in top5[:3]) if top5 else "listed features"

    action = _playbook_action_for_state(playbook, state)

    s1 = (
        f"{sub} ({week}) is labeled {state} this week based on model outputs "
        f"(aggregate community-level indicator, not individual assessment)."
    )
    s2 = (
        f"Distress score change vs the prior week is {delta:+.4f}; "
        f"globally important features include {names} (SHAP-ranked; per-week direction in JSON)."
    )
    s3 = f"Recommended action: {action}"
    return f"{s1} {s2} {s3}"


def _call_anthropic(user_prompt: str) -> tuple[str | None, str | None]:
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed"
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None, "ANTHROPIC_API_KEY missing"
    try:
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        block = msg.content[0]
        if block.type == "text":
            return block.text.strip(), None
        return None, "anthropic response had no text block"
    except Exception as e:
        return None, f"anthropic error: {e.__class__.__name__}: {e}"


def _call_openai(user_prompt: str) -> tuple[str | None, str | None]:
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed"
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None, "OPENAI_API_KEY missing"
    try:
        client = OpenAI(api_key=key)
        comp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
        )
        choice = comp.choices[0]
        if choice.message and choice.message.content:
            return choice.message.content.strip(), None
        return None, "openai response had empty content"
    except Exception as e:
        return None, f"openai error: {e.__class__.__name__}: {e}"


def _generate_narrative_with_meta(
    context: dict[str, Any],
    playbook: str,
) -> tuple[str, str, str]:
    """Return narrative text, source, and note for observability."""
    user_prompt = _build_user_prompt(context, playbook)

    text, anth_note = _call_anthropic(user_prompt)
    if text:
        return _normalize_sentences(text), "anthropic", "ok"

    text, openai_note = _call_openai(user_prompt)
    if text:
        if anth_note:
            return _normalize_sentences(text), "openai", f"anthropic_unavailable: {anth_note}"
        return _normalize_sentences(text), "openai", "ok"

    notes = []
    if anth_note:
        notes.append(anth_note)
    if openai_note:
        notes.append(openai_note)
    note = " | ".join(notes) if notes else "both providers unavailable"
    return template_fallback(context, playbook), "template", note


def _append_weekly_brief_log(
    reports_dir: Path | str,
    subreddit_short: str,
    week_key: str,
    source: str,
    note: str,
) -> Path:
    """Append one JSON record per generated brief for audit/debug."""
    reports_dir = Path(reports_dir)
    sub_dir = reports_dir / subreddit_short
    log_dir = sub_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "weekly_brief_calls.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "subreddit": subreddit_short,
        "week": week_key,
        "source": source,
        "note": note,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return log_path


def generate_narrative_from_context(
    context: dict[str, Any],
    playbook: str,
) -> str:
    narrative, _, _ = _generate_narrative_with_meta(context, playbook)
    return narrative


def _normalize_sentences(text: str) -> str:
    """Trim to NARRATIVE_SENTENCES if the model returns extra."""
    text = text.strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p for p in parts if p]
    if len(parts) <= NARRATIVE_SENTENCES:
        return " ".join(parts)
    return " ".join(parts[:NARRATIVE_SENTENCES])


def write_weekly_brief(
    reports_dir: Path | str,
    subreddit_short: str,
    week_key: str,
    narrative: str,
) -> Path:
    reports_dir = Path(reports_dir)
    brief_dir = reports_dir / subreddit_short / "weekly_briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    safe_week = week_key.replace("/", "-")
    path = brief_dir / f"{safe_week}.txt"
    path.write_text(narrative.strip() + "\n", encoding="utf-8")
    return path


def _trim_predictions(predictions: np.ndarray, n: int) -> np.ndarray:
    arr = np.asarray(predictions, dtype=float).reshape(-1)
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.full(n - len(arr), np.nan)])


def generate_weekly_briefs_for_subreddit(
    subreddit_short: str,
    sub_df: pd.DataFrame,
    distress_scores: pd.Series,
    predictions: np.ndarray,
    shap_df: pd.DataFrame,
    reports_path: Path | str,
    playbook_path: Path | str | None = None,
    baseline_window: int = 4,
) -> tuple[int, list[Path]]:
    """
    Write one brief file per week with a finite prediction.
    If env WEEKLY_NARRATIVE_MAX_WEEKS is set, only the last N such weeks are generated (cost guard).
    """
    reports_path = Path(reports_path)
    playbook = load_playbook(playbook_path)

    n = len(sub_df)
    preds = _trim_predictions(predictions, n)

    indices = [i for i in range(n) if np.isfinite(preds[i])]
    max_weeks_env = os.environ.get("WEEKLY_NARRATIVE_MAX_WEEKS", "").strip()
    if max_weeks_env:
        try:
            mw = int(max_weeks_env)
            if mw > 0:
                indices = indices[-mw:]
        except ValueError:
            pass

    written: list[Path] = []
    log_path: Path | None = None
    for i in indices:
        ctx = build_llm_context(
            subreddit_short,
            i,
            sub_df,
            distress_scores,
            preds,
            shap_df,
            baseline_window=baseline_window,
        )
        if ctx is None:
            continue
        narrative, source, note = _generate_narrative_with_meta(ctx, playbook)
        wk = ctx["week"]
        path = write_weekly_brief(reports_path, subreddit_short, wk, narrative)
        log_path = _append_weekly_brief_log(
            reports_path,
            subreddit_short,
            wk,
            source,
            note,
        )
        written.append(path)

    if log_path is not None:
        print(f"  Weekly brief call log: {log_path}")

    return len(written), written
