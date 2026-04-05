"""
Streamlit live dashboard for Community Mental Health Crisis Predictor.

Run with:
    streamlit run src/dashboard/app.py

Configuration (priority order)
------------------------------
1) Streamlit secrets (`st.secrets`)
2) Environment variables

API_MODE   Set to "true" to show live API connection status in the sidebar.
           Predictions still use local pipeline outputs (eval_results.json),
           but the sidebar shows API health for demonstration.
API_URL    URL of the FastAPI service (e.g. https://your-api.onrender.com).
           Only used when API_MODE=true.
"""

import importlib.util
from pathlib import Path

# Bootstrap repo root before importing other `src.*` packages (works from any cwd).
_bs_path = Path(__file__).resolve().parent / "bootstrap.py"
_spec = importlib.util.spec_from_file_location("_ccp_dashboard_bootstrap", _bs_path)
_bmod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bmod)
_bmod.ensure_repo_root_on_path(__file__)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import requests as _requests

from src.core.ui_config import (
    DASHBOARD_COPY,
    DECISION_USEFULNESS_COPY,
    END_USER_COPY,
    STATE_COLORS,
    STATE_NAMES,
)
from src.dashboard.briefs import _render_weekly_brief
from src.dashboard.charts import build_shap_bar, build_sparkline
from src.dashboard.components import (
    render_drift_table,
    render_model_metrics,
    render_model_metrics_tiles,
)
from src.dashboard.data_access import (
    get_brief_text,
    load_allocation_report,
    load_app_config,
    load_data_quality_report,
    load_drift,
    load_eval_results,
    load_feature_df,
    load_pipeline_last_run_time,
    load_pipeline_profile,
    load_shap,
    load_transitions,
    load_weekly_completeness,
)
from src.dashboard.bootstrap import cfg_value_from_secrets_or_env
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

# ── Dashboard layout constants (issue #32) ─────────────────────────────
# ── API mode configuration ─────────────────────────────────────────────
_API_MODE: bool = cfg_value_from_secrets_or_env("API_MODE", "false").strip().lower() == "true"
_API_URL: str = cfg_value_from_secrets_or_env("API_URL", "").strip().rstrip("/")


@st.cache_data(ttl=30)
def _fetch_api_health(api_url: str) -> dict | None:
    """Fetch /health from the FastAPI service; returns None on failure."""
    try:
        resp = _requests.get(f"{api_url}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _render_api_sidebar() -> None:
    """Add an API connection status indicator to the Streamlit sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Deployment**")
        if _API_MODE and _API_URL:
            health = _fetch_api_health(_API_URL)
            if health:
                st.success(f"API live — {_API_URL}")
                loaded = health.get("models_loaded", [])
                st.caption(f"Models loaded: {', '.join(loaded) if loaded else 'none'}")
                if health.get("mock_mode"):
                    st.warning("API running in mock mode (no real models)")
            else:
                st.warning(f"API unreachable — {_API_URL}")
                st.caption("Dashboard using local pipeline outputs (local mode fallback)")
            if st.button("Refresh API status", key="_api_refresh"):
                _fetch_api_health.clear()
                st.rerun()
        elif _API_MODE:
            st.info("API_MODE=true but API_URL is not set.")
        else:
            st.info("Mode: Local (reading pipeline outputs)")
            st.caption(
                "To connect to the deployed API, set `API_MODE=true` and `API_URL` "
                "in Streamlit Cloud secrets or your environment."
            )


SUBREDDIT_ROLES = {
    "mentalhealth": "General mental health discussion",
    "anxiety": "Anxiety-focused community",
    "lonely": "Loneliness & isolation community",
    "depression": "Depression-focused community",
    "suicidewatch": "Acute crisis support community",
}

SUBREDDIT_ACCENT = {
    # Identity palette only (non-semantic): avoid implying risk level.
    "mentalhealth": "#0EA5E9",
    "anxiety": "#8B5CF6",
    "lonely": "#F59E0B",
    "depression": "#14B8A6",
    "suicidewatch": "#6366F1",
}

DASHBOARD_SUBORDER = ["mentalhealth", "anxiety", "lonely", "depression", "suicidewatch"]

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_COPY["page_title"],
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
/* Card spacing + readability */
div[data-testid="stVerticalBlock"] .community-meta-note {
  font-size: 0.72rem;
  color: #94a3b8;
}
.community-legend {
  font-size: 0.74rem;
  color: #64748b;
  margin-top: -0.15rem;
  margin-bottom: 0.35rem;
}
/* Responsive community cards: allow horizontal scroll on narrow screens */
.community-cards-row {
  display: flex;
  flex-direction: row;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.25rem;
  -webkit-overflow-scrolling: touch;
}
.community-cards-row > div {
  min-width: 160px;
  flex: 1 1 160px;
}
</style>
""",
    unsafe_allow_html=True,
)


def _timeline_annotation_x(weeks_arr: np.ndarray) -> list[tuple[object, str, bool]]:
    """(x, label, is_covid_highlight) for vertical reference lines."""
    out: list[tuple[object, str, bool]] = []
    covid_ts = pd.Timestamp("2020-03-16")
    for w in weeks_arr:
        dt = pd.to_datetime(w)
        if pd.isna(dt):
            continue
        if dt >= covid_ts:
            out.append((dt, "2020-03-16 — COVID-19 lockdowns begin", True))
            break
    for w in weeks_arr:
        dt = pd.to_datetime(w)
        if pd.isna(dt):
            continue
        if dt.year == 2019 and dt.month == 10:
            out.append((dt, "2019-10 — Mental Health Awareness Month", False))
            break
    return out


def _dt_plotly(x) -> object:
    """Single x value for Plotly (avoids add_vline + annotation bugs on datetime axes)."""
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    t = pd.Timestamp(x)
    if pd.isna(t):
        return x
    return t.to_pydatetime()


def _add_timeline_vline(
    fig: go.Figure,
    x,
    *,
    color: str,
    width: int = 1,
    dash: str = "dash",
    annotation: str | None = None,
    y_annotation: float = 1.02,
) -> None:
    """Vertical line via add_shape (add_vline annotations call numpy mean on datetimes and crash)."""
    x0 = _dt_plotly(x)
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x0,
        y0=0,
        y1=1,
        yref="paper",
        xref="x",
        line=dict(color=color, width=width, dash=dash),
    )
    if annotation:
        fig.add_annotation(
            x=x0,
            xref="x",
            y=y_annotation,
            yref="paper",
            text=annotation,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=10),
            xanchor="center",
        )


def _state_badge_html(state_text: str, state_color: str) -> str:
    return (
        "<div style='margin:6px 0 8px 0;'>"
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"font-size:0.78rem;font-weight:700;color:{state_color};"
        f"border:1px solid {state_color};background:{state_color}14;'>"
        f"{state_text}</span></div>"
    )


_render_api_sidebar()

# ── Sidebar: freshness indicator ──────────────────────────────────────
with st.sidebar:
    last_run = load_pipeline_last_run_time()
    if last_run:
        st.caption(f"🕐 Pipeline last run: **{last_run}**")
    else:
        st.caption("🕐 Pipeline: not yet run")
    st.caption("Cache refreshes every 5 min. Use ⋮ → Clear cache to force reload.")
    st.caption(END_USER_COPY["sidebar_hint"])

app_config = load_app_config()
feature_df = load_feature_df()
eval_results = load_eval_results()

# ── Graceful init: checklist error instead of generic stop ────────────
if feature_df is None or eval_results is None:
    st.error("### Pipeline outputs not found")
    st.markdown("The dashboard needs these files to be generated first. Run the missing steps:")
    cfg = app_config or {}
    features_path = Path(cfg.get("paths", {}).get("features", "data/features")) / "features.parquet"
    models_path = Path(cfg.get("paths", {}).get("models", "data/models")) / "eval_results.json"
    st.markdown(
        f"{'✅' if features_path.exists() else '❌'} `{features_path}` — "
        f"{'found' if features_path.exists() else 'missing'}\n\n"
        f"{'✅' if models_path.exists() else '❌'} `{models_path}` — "
        f"{'found' if models_path.exists() else 'missing'}"
    )
    st.markdown("**To generate all outputs, run:**")
    st.code(
        "~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_all "
        "--config config/default.yaml --force",
        language="bash",
    )
    st.markdown("For a quick synthetic test (no real data required):")
    st.code(
        "~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_all "
        "--config config/default.yaml --synthetic --skip-topics --skip-search",
        language="bash",
    )
    st.stop()

available_subs = sorted(feature_df["subreddit"].unique().tolist())
if not available_subs:
    st.error("No subreddits available in feature data. Re-run collection + feature pipeline.")
    st.code("~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_collect --config config/default.yaml")
    st.stop()

visible_subs = [s for s in DASHBOARD_SUBORDER if s in available_subs]
if not visible_subs:
    # Fall back to whatever subreddits are present rather than hard-stopping
    visible_subs = available_subs[:5]
    st.warning(
        f"Expected subreddits not found in features. Showing available: {visible_subs}. "
        "Check `config/default.yaml` → `reddit.subreddits`."
    )


global_replay_weeks = build_global_replay_weeks(feature_df)
if len(global_replay_weeks) == 0:
    st.error("No valid week_start values found in feature data.")
    st.stop()
n_weeks_max = len(global_replay_weeks)
monitoring_min_crisis_weeks = int(app_config.get("evaluation", {}).get("monitoring_min_crisis_weeks", 10))

# ── Session state ─────────────────────────────────────────────────────
if "selected_sub" not in st.session_state:
    st.session_state.selected_sub = "depression" if "depression" in visible_subs else visible_subs[0]
if st.session_state.selected_sub not in visible_subs:
    st.session_state.selected_sub = visible_subs[0]

if "current_week" not in st.session_state:
    st.session_state.current_week = min(40, n_weeks_max - 1)
if "week_idx" in st.session_state:
    st.session_state.current_week = clamp_week_idx(int(st.session_state["week_idx"]), n_weeks_max)
    del st.session_state["week_idx"]

st.session_state.current_week = clamp_week_idx(int(st.session_state.current_week), n_weeks_max)

_ref_sub = st.session_state.selected_sub
if "selected_model" not in st.session_state:
    _ma = available_models_for_sub(eval_results.get(_ref_sub, {}))
    st.session_state.selected_model = "LSTM" if "LSTM" in _ma else _ma[0]

# ── Header: title + week controls ───────────────────────────────────────
hdr_l, hdr_r = st.columns([2, 1])
with hdr_l:
    st.title(DASHBOARD_COPY["title"])
    st.caption(DASHBOARD_COPY["caption"])
with hdr_r:
    st.caption("Select a week to inspect")
    # Buttons must run before the slider: Streamlit forbids mutating a widget key after it is built.
    nav_l, nav_r, slider_col = st.columns([0.11, 0.11, 0.78])
    with nav_l:
        if st.button(
            "◀",
            key="wk_back",
            help="Previous week",
            use_container_width=True,
        ):
            st.session_state.current_week = max(0, int(st.session_state.current_week) - 1)
    with nav_r:
        if st.button(
            "▶",
            key="wk_fwd",
            help="Next week",
            use_container_width=True,
        ):
            st.session_state.current_week = min(n_weeks_max - 1, int(st.session_state.current_week) + 1)
    with slider_col:
        st.slider(
            "Week",
            0,
            n_weeks_max - 1,
            key="current_week",
            label_visibility="collapsed",
        )
    _idx = int(st.session_state.current_week)
    _ts = pd.to_datetime(global_replay_weeks[_idx])
    _lbl = format_week_label(_ts)
    _iso = _ts.isocalendar()
    st.caption(f"Week of {_lbl}  (ISO {_iso.year}-W{int(_iso.week):02d})")

week_idx = int(st.session_state.current_week)
subreddit = st.session_state.selected_sub
replay_week_ts = to_naive_ts(global_replay_weeks[week_idx]) if n_weeks_max else pd.NaT

# Model selector must run before community cards so cards + timeline stay in sync on the same run.
sel_sub_results = eval_results.get(subreddit, {})
models_avail = available_models_for_sub(sel_sub_results)
if st.session_state.selected_model not in models_avail:
    st.session_state.selected_model = models_avail[0]
mod_a, mod_b = st.columns([0.18, 0.45])
with mod_a:
    st.markdown("**Prediction model**")
with mod_b:
    st.selectbox(
        "Model",
        models_avail,
        key="selected_model",
        label_visibility="collapsed",
    )
model_choice = st.session_state.selected_model
st.markdown(
    "<div class='community-legend'>Subreddit color = community identity (not risk level). "
    "State badge = model's predicted state for the <b>following</b> week. "
    "Large tile number = composite distress score (z-scored feature signal, same for every model). "
    "<code>p(high-distress)</code> = model's predicted probability of high-distress next week.</div>",
    unsafe_allow_html=True,
)

# ── Community cards ───────────────────────────────────────────────────
card_cols = st.columns(len(visible_subs))
for i, sub in enumerate(visible_subs):
    with card_cols[i]:
        sub_results = eval_results.get(sub, {})
        results_i, _used_model = resolve_model_results(sub_results, model_choice)
        is_mm, _n_crisis = monitoring_mode(results_i, monitoring_min_crisis_weeks)
        use_trend_pill = is_mm

        sub_df_i = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df_i = sub_df_i.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        n_wi = len(sub_df_i)
        wi_local = resolve_week_index_for_sub(sub_df_i, replay_week_ts)

        distress_i = compute_distress_score(sub_df_i) if n_wi else pd.Series(dtype=float)
        per_w = (results_i or {}).get("per_week", {}) or {}
        preds = np.array(per_w.get("predictions", []))
        probs = np.array(per_w.get("probabilities", []))
        preds = trim_to_length(preds, n_wi)
        probs = trim_to_length(probs, n_wi)

        accent = SUBREDDIT_ACCENT.get(sub, "#64748b")
        role = SUBREDDIT_ROLES.get(sub, "")
        sel = subreddit == sub
        border = f"2.5px solid {accent}" if sel else "1px solid rgba(148,163,184,0.45)"
        bg = f"{accent}12" if sel else "rgba(148,163,184,0.04)"
        shadow = "0 0 0 1px rgba(255,255,255,0.06), 0 8px 24px rgba(2,6,23,0.28)" if sel else "none"

        st.markdown(
            f"<div style='border:{border};box-shadow:{shadow};border-radius:12px;padding:10px;background:{bg};'>"
            f"<div style='display:inline-block;padding:3px 8px;border-radius:999px;"
            f"background:{accent}22;border:1px solid {accent}66;font-weight:700;font-size:0.78rem;'>r/{sub}</div>"
            f"<div style='font-size:0.75rem;color:#64748b;margin-top:4px;margin-bottom:6px'>{role}</div></div>",
            unsafe_allow_html=True,
        )

        if use_trend_pill:
            _card_crisis_n = _n_crisis if _n_crisis is not None else "?"
            status_html = (
                "<div style='min-height:2.6em;line-height:1.35;color:#64748b;font-size:0.79rem'>"
                "<b>Insufficient crisis history</b><br/>"
                f"<span style='opacity:0.9'>Only {_card_crisis_n} crisis weeks in data "
                f"(≥{monitoring_min_crisis_weeks} needed for reliable predictions)</span>"
                "</div>"
            )
        else:
            cp = preds[wi_local] if len(preds) > wi_local and np.isfinite(preds[wi_local]) else np.nan
            state_line = STATE_NAMES.get(int(cp), "—") if not np.isnan(cp) else "—"
            state_color = STATE_COLORS.get(int(cp), "#64748b") if not np.isnan(cp) else "#64748b"
            status_html = _state_badge_html(state_line, state_color)
        st.markdown(status_html, unsafe_allow_html=True)

        _safe_wi = min(wi_local, len(distress_i) - 1) if n_wi else 0
        _raw_d = float(distress_i.iloc[_safe_wi]) if n_wi else float("nan")
        d_score = _raw_d if np.isfinite(_raw_d) else float("nan")
        d_line = f"{d_score:+.3f}" if np.isfinite(d_score) else "—"
        p_hi = float(probs[wi_local]) if len(probs) > wi_local and np.isfinite(probs[wi_local]) else float("nan")
        p_line = f"{(p_hi * 100):.1f}%" if not np.isnan(p_hi) else "—"
        st.markdown(
            f"<div class='community-meta-note' style='margin-bottom:2px'>Distress score (z-scored)</div>"
            f"<div style='font-size:1.35rem;font-weight:600;color:{accent}'>{d_line}</div>"
            f"<div class='community-meta-note'>p(high-distress next wk) {p_line}</div>",
            unsafe_allow_html=True,
        )

        if n_wi:
            start_sp = max(0, wi_local - 11)
            spark = distress_i.iloc[start_sp : wi_local + 1]
            if len(spark) > 0:
                spark_color = "#334155" if sel else "#64748b"
                sp_fig = build_sparkline(spark, spark_color, height=72)
                st.plotly_chart(sp_fig, use_container_width=True, config={"displayModeBar": False})

        help_txt = (
            "Insufficient crisis frequency for reliable prediction (<10 crisis weeks)."
            if use_trend_pill
            else f"Focus dashboard on r/{sub}"
        )
        if st.button(f"Select r/{sub}", key=f"sel_card_{sub}", use_container_width=True, type="primary" if sel else "secondary", help=help_txt):
            st.session_state.selected_sub = sub
            st.rerun()

# ── Selected subreddit: evaluation + dataframe ────────────────────────
sub_results = eval_results.get(subreddit, {})
results, _used = resolve_model_results(sub_results, model_choice)

if not isinstance(results, dict) or not results:
    st.warning(
        f"No evaluation results found for r/{subreddit} ({model_choice}). "
        "Run training/evaluation again to generate per-week predictions."
    )
    st.code(
        "python -m src.pipeline.run_train --config config/default.yaml\n"
        "python -m src.pipeline.run_evaluate --config config/default.yaml"
    )
    st.stop()

per_week = results.get("per_week", {})
predictions_all = np.array(per_week.get("predictions", []))
probabilities_all = np.array(per_week.get("probabilities", []))

if not per_week:
    st.warning(
        f"Evaluation artifact exists but has no per-week outputs for r/{subreddit} ({model_choice}). "
        "Re-run training/evaluation."
    )
    st.code(
        "python -m src.pipeline.run_train --config config/default.yaml\n"
        "python -m src.pipeline.run_evaluate --config config/default.yaml"
    )
    st.stop()

is_monitoring_mode, _actual_crisis_weeks = monitoring_mode(results, monitoring_min_crisis_weeks)

sub_df = feature_df[feature_df["subreddit"] == subreddit].copy()
sub_df = sub_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
n_weeks = len(sub_df)
weeks = sub_df["week_start"].values if "week_start" in sub_df.columns else np.arange(n_weeks)

if n_weeks == 0:
    st.warning(f"No feature rows found for r/{subreddit}. Re-run features/training for this subreddit.")
    st.stop()

week_idx_plot = min(week_idx, n_weeks - 1)
week_idx_plot = resolve_week_index_for_sub(sub_df, replay_week_ts)

raw_pred_len = len(predictions_all)
raw_prob_len = len(probabilities_all)
predictions_all = trim_to_length(predictions_all, n_weeks)
probabilities_all = trim_to_length(probabilities_all, n_weeks)

# Warn on length mismatch (silent wrong-week bug guard)
if raw_pred_len > 0 and raw_pred_len != n_weeks:
    st.caption(
        f"ℹ️ Prediction array length ({raw_pred_len}) ≠ feature rows ({n_weeks}) for "
        f"r/{subreddit} — padded with NaN. Re-run training to realign."
    )

try:
    distress_scores = compute_distress_score(sub_df)
except Exception as e:
    st.warning(f"Could not compute distress score for r/{subreddit}: {e}")
    distress_scores = pd.Series(np.zeros(n_weeks), index=sub_df.index)

theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base == "dark"

week_label = format_week_label(weeks[week_idx_plot]) if week_idx_plot < len(weeks) else "-"

# ── Main row: timeline + brief / metrics ──────────────────────────────
main_l, main_r = st.columns([0.65, 0.35])

with main_l:
    st.markdown("##### Observed distress score + walk-forward predictions")
    accent_sel = SUBREDDIT_ACCENT.get(subreddit, "#378ADD")
    _w_slice = weeks[: week_idx_plot + 1]
    if "week_start" in sub_df.columns:
        x_hist_raw = pd.to_datetime(_w_slice, errors="coerce")
        # Convert to numpy bool array — pandas Series indexing into numpy arrays
        # silently misaligns when the Series carries a non-default index.
        valid_x = ~np.array(pd.isnull(x_hist_raw), dtype=bool)
        _n_nat = int((~valid_x).sum())
        if _n_nat:
            st.caption(f"⚠️ {_n_nat} week_start value(s) could not be parsed and are excluded from the chart.")
        # Only pass valid (non-NaT) entries to Plotly — NaT breaks datetime axes
        x_hist = x_hist_raw[valid_x]
    else:
        x_hist_raw = np.asarray(_w_slice, dtype=float)
        valid_x = np.ones(len(x_hist_raw), dtype=bool)
        x_hist = x_hist_raw
    y_hist_raw = distress_scores.values[: week_idx_plot + 1]
    y_hist = y_hist_raw[valid_x]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_hist,
            y=y_hist,
            mode="lines",
            name=DASHBOARD_COPY["timeline_distress_label"],
            line=dict(color=accent_sel, width=2),
        )
    )

    _preds_slice = predictions_all[: week_idx_plot + 1][valid_x]
    marker_colors_list = []
    for p in _preds_slice:
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
            marker=dict(color=marker_colors_list, size=9),
            hovertemplate="%{x}<br>Distress: %{y:.3f}<extra></extra>",
        )
    )

    probs_up_to = probabilities_all[: week_idx_plot + 1][valid_x]
    valid_prob = ~np.isnan(probs_up_to)
    if valid_prob.any():
        fig.add_trace(
            go.Scatter(
                x=x_hist[valid_prob],
                y=probs_up_to[valid_prob],
                mode="lines",
                name=DASHBOARD_COPY["timeline_probability_label"],
                line=dict(color="#7c3aed", width=1, dash="dot"),
                yaxis="y2",
                opacity=0.65,
            )
        )

    cur_x = weeks[week_idx_plot] if week_idx_plot < len(weeks) else None
    if cur_x is not None:
        _add_timeline_vline(
            fig,
            cur_x,
            color="rgba(100,116,139,0.9)",
            width=1,
            annotation="Current week",
            y_annotation=1.02,
        )

    for i, (ax, lab, is_cov) in enumerate(_timeline_annotation_x(weeks)):
        _add_timeline_vline(
            fig,
            ax,
            color="#dc2626" if is_cov else "rgba(100,116,139,0.55)",
            width=2 if is_cov else 1,
            annotation=lab,
            y_annotation=1.07 + i * 0.06,
        )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Distress score (z-scored)",
        yaxis2=dict(
            title=DASHBOARD_COPY["timeline_probability_axis_label"],
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        hovermode="x unified",
        template="plotly_white" if not is_dark else "plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    try:
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render distress timeline: {e}")

with main_r:
    st.markdown("##### Weekly brief")
    _brief_week_key = week_key_from_row(sub_df.iloc[week_idx_plot])
    _brief_text = get_brief_text(subreddit, _brief_week_key)

    if _brief_text:
        _render_weekly_brief(_brief_text, _brief_week_key, in_sidebar=False)
    else:
        st.caption(DASHBOARD_COPY["weekly_brief_missing"])

    st.markdown("##### Model performance (walk-forward backtest)")
    st.caption("Recall, Precision, F1, PR-AUC from held-out weeks in walk-forward CV (eval_results.json)")
    if is_monitoring_mode:
        _crisis_count_str = str(_actual_crisis_weeks) if _actual_crisis_weeks is not None else "unknown"
        st.info(
            f"Only {_crisis_count_str} crisis weeks found in this subreddit's history "
            f"(need ≥{monitoring_min_crisis_weeks} for reliable walk-forward evaluation). "
            "Predictions are available but performance metrics may not be meaningful at this scale."
        )
    else:
        render_model_metrics_tiles(results)
        _du = results.get("decision_usefulness")
        if _du and isinstance(_du, dict):
            with st.expander("Decision usefulness — top-K alert recall", expanded=False):
                st.markdown(DECISION_USEFULNESS_COPY["intro"])
                kvals = _du.get("k_values") or []
                model_du = _du.get("model") or {}
                rnd_du = _du.get("random_expected_recall") or {}
                pers_du = _du.get("persistence") or {}
                rows_du = []
                for k in kvals:
                    ks = str(k)
                    mk = model_du.get(ks) or model_du.get(k) or {}
                    pk = pers_du.get(ks) or pers_du.get(k) or {}
                    tot = mk.get("total_positives", _du.get("n_elevated_distress_weeks", 0))
                    rows_du.append({
                        "K": k,
                        "Captured (model)": f"{mk.get('captured', 0)}/{tot}",
                        "Recall@K (model)": f"{float(mk.get('recall', 0)):.1%}",
                        "Expected (random)": f"{float(rnd_du.get(str(k), rnd_du.get(k, 0))):.1%}",
                        "Persistence": f"{pk.get('captured', 0)}/{tot} ({float(pk.get('recall', 0)):.1%})",
                    })
                st.caption(
                    f"n={_du.get('n_weeks', '—')} eval weeks, "
                    f"P={_du.get('n_elevated_distress_weeks', '—')} elevated-distress weeks"
                )
                st.dataframe(pd.DataFrame(rows_du), use_container_width=True, hide_index=True)

st.markdown("---")

# ── Bottom tabs ───────────────────────────────────────────────────────
tab_shap, tab_drift, tab_alerts, tab_alloc, tab_dq = st.tabs(["Feature importance", "Drift alerts", "State transitions", "Moderator allocation (LP)", "Data quality"])

with tab_drift:
    st.markdown("##### Drift alerts (up to selected week)")
    st.caption(
        "Each row is one week. The model computes a z-score for each signal relative to the preceding "
        "12-week baseline. **aggregate_level** is the highest severity triggered (0 = normal, 1 = warning, "
        "2 = alert, 3 = critical). **dominant_signal** is the feature with the largest absolute deviation that week."
    )
    drift_df = load_drift(subreddit)
    if drift_df is not None and not drift_df.empty:
        # Clamp to drift_df length to avoid index mismatch with feature_df
        safe_drift_idx = min(week_idx_plot + 1, len(drift_df))
        drift_up = drift_df.iloc[:safe_drift_idx].copy()
        alert_cols = [c for c in drift_df.columns if not c.startswith("z_")]
        display_drift = drift_up[alert_cols].copy()
        try:
            render_drift_table(display_drift)
        except Exception as e:
            st.warning(f"Could not render drift table: {e}")
            st.dataframe(display_drift)
    else:
        st.info(
            "No drift data found. Run the evaluate stage to generate:\n\n"
            "`~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_evaluate "
            "--config config/default.yaml`"
        )

with tab_shap:
    st.markdown("##### Feature importance (SHAP — top 15)")
    st.caption(
        f"Mean absolute SHAP value for r/{subreddit}, averaged across all walk-forward folds. "
        "Longer bar = stronger average influence on the model's high-distress prediction. "
        "Computed by XGBoost; LSTM uses a separate gradient-based attribution."
    )
    shap_df = load_shap(subreddit)
    if shap_df is not None:
        try:
            _shap_sort_col = "mean_abs_shap" if "mean_abs_shap" in shap_df.columns else shap_df.columns[-1]
            top15 = shap_df.head(15).sort_values(_shap_sort_col, ascending=True)
            fig_shap = build_shap_bar(top15)
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render SHAP chart: {e}")
            st.dataframe(shap_df.head(15))
    else:
        st.info(
            "No SHAP data found. Run the evaluate stage to generate:\n\n"
            "`~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_evaluate "
            "--config config/default.yaml`"
        )

with tab_dq:
    st.markdown("##### Data quality")
    st.caption(
        f"Coverage and completeness of the ingested Reddit data for r/{subreddit}. "
        "Gap weeks are weeks where fewer than 50% of the expected posts were recovered — "
        "these can cause distress scores for those weeks to be understated."
    )
    dq_report = load_data_quality_report(subreddit)
    dq_weekly = load_weekly_completeness(subreddit)
    pipeline_profile = load_pipeline_profile()

    if dq_report:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total weeks", dq_report.get("total_weeks_observed", "—"))
        c2.metric("Gap weeks (<50%)", dq_report.get("gap_weeks_below_50pct", 0))
        c3.metric("Avg completeness", f"{dq_report.get('avg_completeness_score', 0.0):.2f}")
        if dq_report.get("missing_weeks"):
            st.caption(f"Missing weeks sample: {', '.join(dq_report['missing_weeks'][:8])}")
    else:
        st.info("No data quality report found. Run `run_collect` first.")

    if dq_weekly is not None and not dq_weekly.empty:
        try:
            fig_c = go.Figure(
                go.Bar(
                    x=dq_weekly["week_start"],
                    y=dq_weekly["completeness_score"],
                    marker_color=[
                        "#e74c3c" if bool(v) else "#2ecc71"
                        for v in (dq_weekly["is_gap"].tolist() if "is_gap" in dq_weekly.columns else [False] * len(dq_weekly))
                    ],
                    name="Completeness score",
                )
            )
            fig_c.update_layout(
                title=f"Weekly data completeness — r/{subreddit} (red bars = gap weeks with <50% expected posts)",
                xaxis_title="",
                yaxis_title="Completeness score",
                template="plotly_white",
                height=280,
            )
            st.plotly_chart(fig_c, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render completeness chart: {e}")

    if pipeline_profile:
        st.markdown("**Pipeline stage timing**")
        flat_rows = []
        for entry in pipeline_profile:
            if "steps" in entry and isinstance(entry["steps"], list):
                for step in entry["steps"]:
                    flat_rows.append(
                        {
                            "stage": step.get("stage", ""),
                            "elapsed_seconds": step.get("elapsed_seconds", 0.0),
                        }
                    )
            elif "subreddit_runs" in entry and isinstance(entry["subreddit_runs"], list):
                for run in entry["subreddit_runs"]:
                    flat_rows.append(
                        {
                            "stage": f"{run.get('stage', 'collect')}:{run.get('subreddit', '')}",
                            "elapsed_seconds": run.get("elapsed_seconds", 0.0),
                            "rows_processed": run.get("rows_processed", 0),
                            "throughput_rows_per_sec": run.get("throughput_rows_per_sec", 0.0),
                        }
                    )
            else:
                flat_rows.append(entry)
        st.dataframe(pd.DataFrame(flat_rows), use_container_width=True, height=220)

    fold_diag = results.get("fold_diagnostics", [])
    if fold_diag:
        st.markdown("**Fold diagnostics**")
        st.dataframe(pd.DataFrame(fold_diag), use_container_width=True, height=180)

    with st.expander("Full model metrics & decision usefulness", expanded=False):
        render_model_metrics(results, STATE_NAMES, DECISION_USEFULNESS_COPY)

    lead = results.get("detection_lead_time_distribution", {})
    lead_dist = lead.get("distribution", []) if isinstance(lead, dict) else []
    if lead_dist:
        try:
            fig_l = go.Figure(go.Histogram(x=lead_dist, nbinsx=10, marker_color="#3498db"))
            fig_l.update_layout(
                title="Detection lead time distribution (weeks)",
                xaxis_title="Lead time (weeks)",
                yaxis_title="Count",
                template="plotly_white",
                height=260,
            )
            st.plotly_chart(fig_l, use_container_width=True)
            st.caption(
                f"p50={lead.get('p50', 0):.2f}, p75={lead.get('p75', 0):.2f}, p90={lead.get('p90', 0):.2f}"
            )
        except Exception as e:
            st.warning(f"Could not render lead time chart: {e}")

with tab_alloc:
    st.markdown("##### Moderator hour allocation — LP optimisation over predicted crisis probabilities")
    alloc = load_allocation_report()

    if alloc is None or "error" in (alloc or {}):
        st.info("No allocation report found. Run `make evaluate` to generate.")
    else:
        cfg_presc = load_app_config().get("prescriptive", {})
        total_h = alloc.get("total_hours", cfg_presc.get("total_moderator_hours", 10))
        objective = alloc.get("objective", 0.0)

        st.caption(
            f"Budget: **{total_h:.0f} hrs/week** — "
            f"LP objective (expected interceptions): **{objective:.4f}**"
        )

        # Allocation table
        sub_data = alloc.get("subreddits", {})
        if sub_data:
            rows = sorted(sub_data.items(), key=lambda kv: kv[1]["hours"], reverse=True)
            table_rows = [
                {
                    "Subreddit": f"r/{s}",
                    "Hours": d["hours"],
                    "State": d["state_label"],
                    "Crisis prob.": f"{d['probability']:.2f}",
                    "Effectiveness": d["effectiveness"],
                }
                for s, d in rows
            ]
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

            # Bar chart of allocation
            fig_a = go.Figure(go.Bar(
                x=[r["Subreddit"] for r in table_rows],
                y=[r["Hours"] for r in table_rows],
                marker_color=[
                    SUBREDDIT_ACCENT.get(s, "#64748b") for s, _ in rows
                ],
                text=[f"{r['Hours']:.1f} h" for r in table_rows],
                textposition="outside",
            ))
            fig_a.update_layout(
                title="Recommended weekly hours per subreddit",
                yaxis_title="Hours allocated",
                template="plotly_white",
                height=300,
                showlegend=False,
            )
            st.plotly_chart(fig_a, use_container_width=True)

        # Sensitivity analysis
        sensitivity = alloc.get("sensitivity", {})
        if sensitivity:
            st.markdown("**Sensitivity: allocation vs. budget (5 → 20 hrs)**")
            try:
                first_val = next(iter(sensitivity.values()), {})
                subs_in_sens = list(first_val.keys()) if isinstance(first_val, dict) else []
                def _budget_key(b):
                    try:
                        return int(b)
                    except (ValueError, TypeError):
                        return 0
                budgets = sorted(sensitivity.keys(), key=_budget_key)
                sens_rows = []
                for b in budgets:
                    row = {"Budget (hrs)": int(b)}
                    bval = sensitivity[b]
                    if isinstance(bval, dict):
                        for s in subs_in_sens:
                            row[f"r/{s}"] = bval.get(s, 0.0)
                    sens_rows.append(row)
                sens_df = pd.DataFrame(sens_rows).set_index("Budget (hrs)")

                fig_s = go.Figure()
                for s in subs_in_sens:
                    col = f"r/{s}"
                    if col in sens_df.columns:
                        fig_s.add_trace(go.Scatter(
                            x=sens_df.index.tolist(),
                            y=sens_df[col].tolist(),
                            name=col,
                            mode="lines+markers",
                            line={"color": SUBREDDIT_ACCENT.get(s, "#64748b")},
                        ))
                fig_s.update_layout(
                    title="Hours allocated per subreddit vs. total budget",
                    xaxis_title="Total budget (hrs/week)",
                    yaxis_title="Hours allocated",
                    template="plotly_white",
                    height=320,
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(fig_s, use_container_width=True)

                with st.expander("Sensitivity data table", expanded=False):
                    st.dataframe(sens_df.reset_index(), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Could not render sensitivity analysis: {e}")

        with st.expander("LP formulation", expanded=False):
            st.markdown("""
**Decision variable:** $x_i$ — hours allocated to subreddit $i$

**Objective (maximise):**
$$\\max \\sum_i p_i \\cdot e_i \\cdot x_i$$
where $p_i$ = latest predicted crisis probability, $e_i$ = intervention effectiveness coefficient.

**Constraints:**
- $\\sum_i x_i \\leq H$ — total budget $H$ hrs/week
- $x_i \\geq m$ — minimum coverage floor $m$ per subreddit
- $x_i \\geq 0$

Solved via `scipy.optimize.linprog` (HiGHS backend). Effectiveness coefficients are configurable per subreddit in `config/default.yaml` under `prescriptive.effectiveness`.
            """)

with tab_alerts:
    st.markdown("##### Predicted state transitions")
    st.caption("Latest 30 transitions from `data/alerts.db` — logged when the model's predicted state for the following week changes. Populated on each pipeline run.")
    try:
        transitions = load_transitions(n=30)
        if transitions:
            trans_df = pd.DataFrame(transitions)
            # Friendly column order
            _col_order = ["timestamp", "subreddit", "week_start", "from_state", "to_state", "distress_score", "dominant_signal"]
            trans_df = trans_df[[c for c in _col_order if c in trans_df.columns]]
            # Map integer state codes → human labels for display
            if "to_state" in trans_df.columns:
                trans_df["to_state"] = trans_df["to_state"].apply(
                    lambda v: STATE_NAMES.get(int(v), str(v)) if pd.notna(v) else v
                )
            if "from_state" in trans_df.columns:
                trans_df["from_state"] = trans_df["from_state"].apply(
                    lambda v: STATE_NAMES.get(int(v), str(v)) if pd.notna(v) else v
                )
            # Colour-code rows by to_state severity using Styler
            _state_bg = {
                STATE_NAMES.get(0, "Stable"): "#f0fdf4",
                STATE_NAMES.get(1, "Early Vulnerability Signal"): "#fefce8",
                STATE_NAMES.get(2, "Elevated Distress"): "#fff7ed",
                STATE_NAMES.get(3, "Severe Community Distress Signal"): "#fff1f2",
            }
            def _style_alert_row(row):
                bg = _state_bg.get(str(row.get("to_state", "")), "")
                return [f"background-color: {bg}" if bg else "" for _ in row]
            try:
                st.dataframe(
                    trans_df.style.apply(_style_alert_row, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception:
                st.dataframe(trans_df, use_container_width=True, hide_index=True)
            # Quick bar: transition counts by subreddit
            if "subreddit" in trans_df.columns and "to_state" in trans_df.columns:
                counts = trans_df.groupby(["subreddit", "to_state"]).size().reset_index(name="count")
                # STATE_NAMES values are the labels now (already mapped above)
                _label_to_code = {v: k for k, v in STATE_NAMES.items()}
                fig_tr = go.Figure()
                for ts in counts["to_state"].unique():
                    sub_cnt = counts[counts["to_state"] == ts]
                    state_code = _label_to_code.get(str(ts), 0)
                    fig_tr.add_trace(go.Bar(
                        x=sub_cnt["subreddit"],
                        y=sub_cnt["count"],
                        name=str(ts),
                        marker_color=STATE_COLORS.get(state_code, "#64748b"),
                    ))
                fig_tr.update_layout(
                    barmode="stack",
                    title="Transition counts by subreddit (last 30)",
                    xaxis_title="Subreddit",
                    yaxis_title="Transitions",
                    template="plotly_white",
                    height=280,
                    legend=dict(orientation="h", y=-0.3),
                )
                st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info(
                "No transitions recorded yet. Run the pipeline end-to-end to populate `data/alerts.db`.\n\n"
                "`~/.pyenv/versions/3.12.11/bin/python -m src.pipeline.run_all --config config/default.yaml`"
            )
    except Exception as e:
        st.warning(f"Could not load alert feed: {e}")
