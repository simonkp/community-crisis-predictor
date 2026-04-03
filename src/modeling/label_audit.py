"""
LLM-assisted label auditing for crisis labels.

Samples a set of weeks from the labeled dataset, constructs a structured
prompt containing post excerpts and the model-assigned crisis label, and
asks an LLM to independently rate the community distress level.  The
agreement rate and disagreement cases are saved as an audit report.

Supports Anthropic (claude-*) and OpenAI (gpt-*) backends; falls back to
a stub report if neither API key is configured.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd


_SYSTEM_PROMPT = """You are a mental health researcher auditing community distress labels.
You will be shown a sample of Reddit posts from a single week in a mental health subreddit,
along with a model-assigned distress label (0=Stable, 1=Early Vulnerability, 2=Elevated Distress, 3=Severe).
Assess the posts and reply with a JSON object: {"label": <0-3>, "confidence": <0.0-1.0>, "reason": "<one sentence>"}.
Base your label only on the post content, not on the provided model label."""

_MAX_CHARS_PER_POST = 300
_MAX_POSTS_PER_WEEK = 5


def _build_prompt(posts: list[str], model_label: int, week: str, subreddit: str) -> str:
    excerpts = "\n\n".join(
        f"Post {i + 1}: {p[:_MAX_CHARS_PER_POST]}" for i, p in enumerate(posts[:_MAX_POSTS_PER_WEEK])
    )
    label_names = {0: "Stable", 1: "Early Vulnerability", 2: "Elevated Distress", 3: "Severe"}
    return (
        f"Subreddit: r/{subreddit}\n"
        f"Week: {week}\n"
        f"Model-assigned label: {model_label} ({label_names.get(model_label, 'Unknown')})\n\n"
        f"Post excerpts:\n{excerpts}\n\n"
        "Reply with JSON only."
    )


def _call_anthropic(prompt: str) -> dict | None:
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        return json.loads(text)
    except Exception:
        return None


def _call_openai(prompt: str) -> dict | None:
    try:
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=256,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception:
        return None


def _call_llm(prompt: str, provider: str) -> dict | None:
    if provider == "anthropic":
        return _call_anthropic(prompt)
    if provider == "openai":
        return _call_openai(prompt)
    return None


def audit_labels_with_llm(
    weekly_df: pd.DataFrame,
    labels: pd.Series,
    subreddit: str,
    sample_size: int = 10,
    provider: str = "anthropic",
    seed: int = 42,
) -> dict:
    """
    Sample `sample_size` labeled weeks, query an LLM for its own label,
    and return an audit report comparing model labels to LLM labels.

    Parameters
    ----------
    weekly_df   : Weekly-aggregate DataFrame with a `texts` column.
    labels      : Series of 4-class crisis labels (0-3), aligned with weekly_df.
    subreddit   : Name of the subreddit (for prompting context).
    sample_size : Number of weeks to audit.
    provider    : "anthropic" or "openai".
    seed        : Random seed for reproducible sampling.

    Returns
    -------
    dict with: agreement_rate, n_audited, n_agreement, disagreements (list),
               audit_entries (list of {week, model_label, llm_label, llm_confidence}).
    """
    valid_mask = ~labels.isna()
    valid_idx = np.where(valid_mask)[0]

    if len(valid_idx) == 0:
        return {"status": "no_labeled_weeks", "subreddit": subreddit}

    rng = random.Random(seed)
    sample_idx = rng.sample(list(valid_idx), min(sample_size, len(valid_idx)))

    audit_entries: list[dict] = []
    disagreements: list[dict] = []
    n_agreement = 0

    for idx in sample_idx:
        row = weekly_df.iloc[idx]
        texts = row.get("texts", []) or []
        model_label = int(labels.iloc[idx])
        week = str(row.get("week_start", idx))

        prompt = _build_prompt(texts, model_label, week, subreddit)
        llm_response = _call_llm(prompt, provider)

        entry: dict = {
            "week": week,
            "model_label": model_label,
            "llm_label": None,
            "llm_confidence": None,
            "llm_reason": None,
            "agreed": None,
        }

        if llm_response and "label" in llm_response:
            llm_label = int(llm_response.get("label", -1))
            agreed = (llm_label == model_label)
            entry.update({
                "llm_label": llm_label,
                "llm_confidence": float(llm_response.get("confidence", 0.0)),
                "llm_reason": str(llm_response.get("reason", "")),
                "agreed": agreed,
            })
            if agreed:
                n_agreement += 1
            else:
                disagreements.append(entry)
        else:
            entry["error"] = "llm_call_failed_or_unparseable"

        audit_entries.append(entry)

    n_audited = len(audit_entries)
    agreement_rate = n_agreement / n_audited if n_audited > 0 else 0.0

    return {
        "status": "ok",
        "subreddit": subreddit,
        "provider": provider,
        "n_audited": n_audited,
        "n_agreement": n_agreement,
        "agreement_rate": round(agreement_rate, 4),
        "disagreements": disagreements,
        "audit_entries": audit_entries,
    }


def save_audit_report(report: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
