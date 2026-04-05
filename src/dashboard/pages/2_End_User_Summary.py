"""Community Copilot — moderator triage board with side-by-side detail panel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# ── bootstrap ─────────────────────────────────────────────────────────────
_bs_path = Path(__file__).resolve().parent.parent / "bootstrap.py"
_spec = importlib.util.spec_from_file_location("_ccp_dashboard_bootstrap", _bs_path)
_bmod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bmod)
_ROOT = _bmod.ensure_repo_root_on_path(__file__)
_cfg = _bmod.cfg_value_from_secrets_or_env

from src.core.ui_config import STATE_NAMES
from src.dashboard.data_access import (
    get_brief_text,
    load_app_config,
    load_eval_results,
    load_feature_df,
    load_shap,
)
from src.dashboard.state import clamp_week_idx, monitoring_mode, trim_to_length
from src.dashboard.view_helpers import (
    available_models_for_sub,
    build_global_replay_weeks,
    format_week_label,
    resolve_model_results,
    resolve_week_index_for_sub,
    to_naive_ts,
)
from src.labeling.distress_score import compute_distress_score
from src.narration.narrative_generator import week_key_from_row

st.set_page_config(
    page_title="Community Copilot",
    page_icon="🤝",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hero card */
.hero-card{
    background:linear-gradient(135deg,#1e3a5f 0%,#2d6a9f 100%);
    border-radius:12px;padding:22px 26px;color:#fff;margin-bottom:.9rem;
}
.hero-card h3{margin:0 0 6px;font-size:1.45rem;font-weight:700;}
.hero-card .bc{opacity:.65;font-size:.8rem;margin-bottom:12px;}
.metric-row{display:flex;gap:12px;margin-top:14px;flex-wrap:wrap;}
.metric-box{
    background:rgba(255,255,255,.14);border-radius:8px;
    padding:10px 16px;min-width:110px;text-align:center;
}
.metric-box .val{font-size:1.3rem;font-weight:700;}
.metric-box .lbl{font-size:.7rem;opacity:.75;margin-top:2px;line-height:1.3;}

/* Signal pills */
.signal-pill{
    display:inline-block;border-radius:20px;padding:3px 12px;
    font-weight:700;font-size:.8rem;letter-spacing:.03em;
}
.pill-0{background:#27ae60;color:#fff;}
.pill-1{background:#f39c12;color:#fff;}
.pill-2{background:#e67e22;color:#fff;}
.pill-3{background:#c0392b;color:#fff;}
.pill-mm{background:#7f8c8d;color:#fff;}

/* Red section divider */
.eu-red-rule{
    border:none;border-top:2px solid #c0392b;margin:8px 0 18px;
    opacity:.9;
}

/* Copilot output — border-left only, no fill, dark-mode safe */
.copilot-out{
    border-left:4px solid #2d6a9f;
    padding:14px 18px;border-radius:0 8px 8px 0;line-height:1.8;
    margin-bottom:4px;
}
.copilot-meta{font-size:.76rem;opacity:.55;margin-top:4px;}

/* Section intro */
.sec-intro{font-size:.85rem;opacity:.65;margin:-4px 0 12px;}
</style>
""", unsafe_allow_html=True)

# ── data load ──────────────────────────────────────────────────────────────
app_config = load_app_config()
feature_df = load_feature_df()
eval_results = load_eval_results()

if feature_df is None or eval_results is None:
    st.error("Pipeline outputs not found. Run the full pipeline first (see Home page).")
    st.stop()

_DASH_SUBORDER = ["mentalhealth", "anxiety", "lonely", "depression", "suicidewatch"]
available_subs = sorted(feature_df["subreddit"].unique().tolist())
visible_subs = [s for s in _DASH_SUBORDER if s in available_subs] or available_subs[:5]

global_replay_weeks = build_global_replay_weeks(feature_df)
if len(global_replay_weeks) == 0:
    st.error("No valid weeks in feature data.")
    st.stop()
n_weeks_max = len(global_replay_weeks)
monitoring_min = int(app_config.get("evaluation", {}).get("monitoring_min_crisis_weeks", 10))

_API_MODE = _cfg("API_MODE", "false").lower() == "true"
_API_URL = _cfg("API_URL", "http://localhost:8000").rstrip("/")

# ── session state ──────────────────────────────────────────────────────────
if "selected_sub" not in st.session_state:
    st.session_state.selected_sub = "depression" if "depression" in visible_subs else visible_subs[0]
if st.session_state.selected_sub not in visible_subs:
    st.session_state.selected_sub = visible_subs[0]

if "current_week" not in st.session_state:
    st.session_state.current_week = min(40, n_weeks_max - 1)
st.session_state.current_week = clamp_week_idx(int(st.session_state.current_week), n_weeks_max)

def _best_model(sub: str) -> str:
    avail = available_models_for_sub(eval_results.get(sub, {}))
    for pref in ("Ensemble", "LSTM", "XGB"):
        if pref in avail:
            return pref
    return avail[0] if avail else "LSTM"

# ── sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤝 Community Copilot")
    st.divider()
    st.markdown("**Select a week to review**",
                help="Browse historical weeks to see how community signals evolved over time.")
    nav_l, nav_r, slider_col = st.columns([0.11, 0.11, 0.78])
    with nav_l:
        if st.button("◀", key="eu_wk_back", use_container_width=True):
            st.session_state.current_week = max(0, int(st.session_state.current_week) - 1)
    with nav_r:
        if st.button("▶", key="eu_wk_fwd", use_container_width=True):
            st.session_state.current_week = min(n_weeks_max - 1, int(st.session_state.current_week) + 1)
    with slider_col:
        st.slider("Week", 0, n_weeks_max - 1, key="current_week", label_visibility="collapsed")

    week_idx = int(st.session_state.current_week)
    _ts = pd.to_datetime(global_replay_weeks[week_idx])
    _lbl = format_week_label(_ts)
    _iso = _ts.isocalendar()
    st.caption(f"Viewing **{_lbl}** (ISO {_iso.year}-W{int(_iso.week):02d})")
    st.divider()
    st.caption(
        "In the left **app navigation**, the page named **app** is the full analyst dashboard "
        "(all charts, tabs, and the model dropdown). Use the link below to return there."
    )
    st.page_link("app.py", label="← Analyst dashboard (app)", icon="📊")

# ── resolve week ───────────────────────────────────────────────────────────
week_idx = int(st.session_state.current_week)
replay_week_ts = to_naive_ts(global_replay_weeks[week_idx])
week_display = format_week_label(pd.to_datetime(global_replay_weeks[week_idx]))

_PILL_CLS = {0: "pill-0", 1: "pill-1", 2: "pill-2", 3: "pill-3"}
_SEVERITY  = {3: 0, 2: 1, 1: 2, 0: 3, -1: 4}
_SIG_EMOJI = {0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", -1: "⚫"}

# ── precompute per-sub snapshot ────────────────────────────────────────────
def _sub_snapshot(sub: str) -> dict:
    sr = eval_results.get(sub, {})
    model = _best_model(sub)
    res, _ = resolve_model_results(sr, model)
    if not isinstance(res, dict) or not res:
        return dict(sub=sub, state=-1, state_name="No data", pill="pill-mm",
                    p_hi=float("nan"), d_delta=0.0,
                    trend_icon="➡️", trend_str="Stable", is_mm=False)
    is_mm, _ = monitoring_mode(res, monitoring_min)
    pw = res.get("per_week", {}) or {}
    sdf = feature_df[feature_df["subreddit"] == sub].copy()
    sdf = (
        sdf.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        if {"iso_year", "iso_week"}.issubset(sdf.columns)
        else sdf.sort_values("week_start").reset_index(drop=True)
    )
    n = len(sdf)
    preds = trim_to_length(np.array(pw.get("predictions", [])), n)
    probs  = trim_to_length(np.array(pw.get("probabilities", [])), n)
    ds = compute_distress_score(sdf)
    wi = resolve_week_index_for_sub(sdf, replay_week_ts)
    pred_i = preds[wi] if len(preds) > wi and np.isfinite(preds[wi]) else np.nan
    state = int(pred_i) if np.isfinite(pred_i) else -1
    sname = STATE_NAMES.get(state, "Trend Monitoring" if is_mm else "—")
    p_hi = float(probs[wi]) if len(probs) > wi and np.isfinite(probs[wi]) else float("nan")
    d_now = float(ds.iloc[wi]) if n else 0.0
    d_prev = float(ds.iloc[wi - 1]) if wi > 0 else d_now
    d_delta = d_now - d_prev
    trend_icon = "📈" if d_delta > 0.05 else ("📉" if d_delta < -0.05 else "➡️")
    trend_str  = "Rising" if d_delta > 0.05 else ("Easing" if d_delta < -0.05 else "Stable")
    pill = "pill-mm" if is_mm else _PILL_CLS.get(state, "pill-1")
    return dict(sub=sub, state=state, state_name=sname, pill=pill,
                p_hi=p_hi, d_delta=d_delta, trend_icon=trend_icon,
                trend_str=trend_str, is_mm=is_mm)

snapshots = [_sub_snapshot(s) for s in visible_subs]
snapshots.sort(key=lambda x: (_SEVERITY.get(x["state"], 4),
                               -(x["p_hi"] if not np.isnan(x["p_hi"]) else 0)))

# ── PAGE HEADER ────────────────────────────────────────────────────────────
st.markdown("## 🤝 Community Copilot")
st.markdown(
    "**Who is this for?** Community managers, moderators, and platform safety staff who need "
    "to monitor mental-health signals across multiple subreddits — without needing a data science background.\n\n"
    "**How to use it:** The left panel shows all monitored communities ranked by forecast severity "
    "for the selected week, so you can spot the most at-risk community at a glance. "
    "Click any community to open its full breakdown on the right: signal strength, trend direction, "
    "the pre-written weekly brief, and — when the live API is connected — an **AI Copilot explanation** "
    "that turns model outputs into plain-language situation summaries and concrete moderation suggestions "
    "grounded in your playbook. Use the week slider in the sidebar to review any past week."
)

# ── Red divider: intro (full width) vs main workspace ──────────────────────
st.markdown('<hr class="eu-red-rule" />', unsafe_allow_html=True)

# ── TWO equal columns: triage list | detail ────────────────────────────────
col_list, col_detail = st.columns(2, gap="medium")

# ── LEFT: aligned community table + Open buttons ───────────────────────────
with col_list:
    st.markdown(f"#### Communities · {week_display}",
                help="Ranked worst-first by the model's forecast for the following week.")
    st.caption("Use **Open** to show a community in the panel on the right.")

    # Column weights: narrow rank & p(hi), room for full signal labels, compact trend + action
    _cg = [0.42, 1.05, 2.35, 0.58, 0.92, 0.55]
    h = st.columns(_cg)
    h[0].markdown("**#**")
    h[1].markdown("**Community**")
    h[2].markdown("**Signal**")
    h[3].markdown("**p(hi)**")
    h[4].markdown("**Trend**")
    h[5].markdown("** **")

    for rank, snap in enumerate(snapshots, 1):
        sub = snap["sub"]
        is_sel = st.session_state.selected_sub == sub
        em = _SIG_EMOJI.get(snap["state"], "⚫")
        p_str = f"{snap['p_hi'] * 100:.0f}%" if not np.isnan(snap["p_hi"]) else "—"
        r = st.columns(_cg)
        with r[0]:
            st.markdown(f"{rank}")
        with r[1]:
            st.markdown(f"**r/{sub}**")
        with r[2]:
            st.markdown(f"{em} {snap['state_name']}")
        with r[3]:
            st.markdown(p_str)
        with r[4]:
            st.markdown(f"{snap['trend_icon']} {snap['trend_str']}")
        with r[5]:
            if st.button(
                "✓" if is_sel else "Open",
                key=f"sel_{sub}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
                help=f"Show r/{sub} in the right panel",
            ):
                st.session_state.selected_sub = sub
                st.rerun()

# ── RIGHT: COMMUNITY DETAIL ─────────────────────────────────────────────────
with col_detail:
    subreddit = st.session_state.selected_sub
    model_choice = _best_model(subreddit)
    snap = next(s for s in snapshots if s["sub"] == subreddit)

    sub_results = eval_results.get(subreddit, {})
    results, _used = resolve_model_results(sub_results, model_choice)
    detail_ok = isinstance(results, dict) and bool(results)
    per_week = {}
    sub_df = pd.DataFrame()
    n_weeks = 0

    if detail_ok:
        per_week = results.get("per_week", {}) or {}
        sub_df = feature_df[feature_df["subreddit"] == subreddit].copy()
        sub_df = (
            sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
            if {"iso_year", "iso_week"}.issubset(sub_df.columns)
            else sub_df.sort_values("week_start").reset_index(drop=True)
        )
        n_weeks = len(sub_df)
        if n_weeks == 0:
            st.error("No rows for this community.")
            detail_ok = False
    else:
        st.warning(f"No evaluation results for r/{subreddit}. Pick another community.")

    if detail_ok:
        predictions_all = trim_to_length(np.array(per_week.get("predictions", [])), n_weeks)
        probabilities_all = trim_to_length(np.array(per_week.get("probabilities", [])), n_weeks)
        distress_scores = compute_distress_score(sub_df)
        is_mm = snap["is_mm"]

        wi = resolve_week_index_for_sub(sub_df, replay_week_ts)
        pred_i = predictions_all[wi] if len(predictions_all) > wi and np.isfinite(predictions_all[wi]) else np.nan
        state_name = snap["state_name"]
        p_hi = snap["p_hi"]
        d_delta = snap["d_delta"]
        trend_icon = snap["trend_icon"]
        trend_str = snap["trend_str"]
        d_now = float(distress_scores.iloc[wi]) if n_weeks else 0.0

        week_key = week_key_from_row(sub_df.iloc[wi])
        row = sub_df.iloc[wi]
        week_label = format_week_label(row["week_start"]) if "week_start" in sub_df.columns else week_key

        prev_pred = (
            predictions_all[wi - 1]
            if wi > 0 and len(predictions_all) > wi - 1 and np.isfinite(predictions_all[wi - 1])
            else np.nan
        )
        prev_state_name = STATE_NAMES.get(int(prev_pred), "—") if np.isfinite(prev_pred) else "n/a"

        p_str = f"{p_hi * 100:.0f}%" if not np.isnan(p_hi) else "n/a"
        conf_str, conf_short = (
            ("High confidence", "High") if p_hi >= 0.6
            else ("Moderate confidence", "Moderate") if p_hi >= 0.4
            else ("Low confidence", "Low")
        )
        delta_word = "rose" if d_delta > 0.05 else ("fell" if d_delta < -0.05 else "held steady")

        st.markdown(
            f"""
            <div class="hero-card">
              <div class="bc">Forecast for week following {week_label}</div>
              <h3>r/{subreddit}</h3>
              <span class="signal-pill {snap['pill']}">{state_name}</span>
              <div class="metric-row">
                <div class="metric-box">
                  <div class="val">{p_str}</div>
                  <div class="lbl">High-distress<br>probability</div>
                </div>
                <div class="metric-box">
                  <div class="val">{trend_icon}</div>
                  <div class="lbl">Distress {trend_str.lower()}<br>vs last week</div>
                </div>
                <div class="metric-box">
                  <div class="val">{conf_short}</div>
                  <div class="lbl">Signal<br>confidence</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_mm:
            st.info(
                "**Trend-monitoring mode** — limited crisis history for this community. "
                "Signals are exploratory; do not escalate based solely on this indicator."
            )

        shap_df = load_shap(subreddit)
        shap_top3: list[str] = []
        if shap_df is not None and not shap_df.empty and "feature" in shap_df.columns:
            shap_top3 = shap_df["feature"].head(3).tolist()

        cache_key = f"{subreddit}|{week_key}|{model_choice}"
        if "eu_copilot_cache" not in st.session_state:
            st.session_state.eu_copilot_cache = {}

        st.markdown(
            "**🤖 AI Copilot**",
            help=(
                "Sends this week's signals to the inference server, which calls an LLM to produce "
                "a plain-language situation summary and moderation suggestions grounded in the playbook. "
                "No API keys are stored in this app — they live only on the server."
            ),
        )

        if not _API_MODE:
            st.caption(
                "AI Copilot is available when the dashboard is connected to the live API. "
                "Set `API_MODE=true` and `API_URL` in your secrets or `.env` file."
            )
        else:
            if st.button("✨ Generate Copilot explanation", key="eu_copilot_btn", type="primary"):
                _payload: dict = {
                    "subreddit": subreddit,
                    "week": week_key,
                    "predicted_state": state_name,
                    "previous_state": prev_state_name,
                }
                if not np.isnan(p_hi):
                    _payload["p_high_distress"] = round(p_hi, 4)
                if not np.isnan(d_now):
                    _payload["distress_score_z"] = round(d_now, 4)
                _payload["distress_score_delta"] = round(d_delta, 4)
                if shap_top3:
                    _payload["shap_top3"] = shap_top3

                with st.spinner("Asking AI Copilot…"):
                    try:
                        resp = requests.post(f"{_API_URL}/brief", json=_payload, timeout=30)
                        resp.raise_for_status()
                        st.session_state.eu_copilot_cache[cache_key] = {
                            "data": resp.json(), "payload": _payload,
                        }
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot reach the API at `{_API_URL}`. Is the server running?")
                    except requests.exceptions.Timeout:
                        st.error("Request timed out (30 s). Try again.")
                    except Exception as exc:
                        st.error(f"API error: {exc}")

            cached = st.session_state.eu_copilot_cache.get(cache_key)
            if cached:
                data = cached["data"]
                _fallback = data.get("fallback", False)
                _src = data.get("source", "unknown")
                _badge = "📝 Template (no LLM key on server)" if _fallback else f"✅ {_src.title()}"
                st.markdown('<div class="copilot-out">', unsafe_allow_html=True)
                st.markdown(data["text"])
                st.markdown("</div>", unsafe_allow_html=True)
                st.caption(
                    f"Source: {_badge} · {data.get('latency_ms', 0):.0f} ms  \n"
                    "AI-generated from model outputs only. Not clinical advice. Apply human judgment."
                )
                with st.expander("📊 Data sent to generate this explanation", expanded=False):
                    st.json(cached["payload"])

        st.divider()

        insight_cards = [
            ("👁️", "What we observed",
             f"Distress signals **{delta_word}** vs the previous week."),
            ("🔮", "Forecast",
             f"**{state_name}** for the week after {week_label}. Last week: {prev_state_name}."),
            ("🎯", "Confidence",
             f"**{conf_str}** — {p_str} probability of elevated distress."),
        ]
        with st.expander("📊 Signal breakdown", expanded=True):
            st.markdown(
                '<p class="sec-intro">Three plain-language cards summarising the observation, '
                "forecast and confidence. Community-level signals only.</p>",
                unsafe_allow_html=True,
            )
            c1, c2, c3 = st.columns(3)
            for col, (icon, title, body) in zip([c1, c2, c3], insight_cards):
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{icon} {title}**")
                        st.markdown(body)

        brief = get_brief_text(subreddit, week_key)
        with st.expander("📋 Weekly Brief", expanded=False):
            st.markdown(
                '<p class="sec-intro">Pre-written narrative produced during the last pipeline '
                "evaluate run. Combines model forecast with top SHAP signals.</p>",
                unsafe_allow_html=True,
            )
            if brief:
                st.info(brief)
            else:
                st.caption("No brief found for this week. Run `make evaluate` to produce briefs.")

        if shap_top3:
            with st.expander("🔬 Top predictive signals (SHAP)", expanded=False):
                st.caption(
                    "Community-level metrics that most influenced the forecast. "
                    "These are aggregate statistics — no individual users tracked."
                )
                for i, feat in enumerate(shap_top3, 1):
                    st.markdown(f"**{i}.** `{feat}`")

# ── Full-width footer (below the two-column workspace) ─────────────────────
st.markdown('<hr class="eu-red-rule" />', unsafe_allow_html=True)
with st.expander("⚠️ Responsible use (all communities)", expanded=False):
    st.markdown("""
- **Aggregate signals only** — no individual users are tracked or identified.
- Signals are probabilistic and can be wrong in both directions.
- Never use this to monitor or take action against specific people.
- Always combine with direct community observation and human judgment.
- For emergencies, follow your platform's safety protocols.
    """)
