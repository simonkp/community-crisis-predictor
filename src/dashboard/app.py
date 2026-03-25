"""
Streamlit live dashboard for Community Mental Health Crisis Predictor.

Run with:
    streamlit run src/dashboard/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.ui_config import (
    DASHBOARD_COPY,
    DECISION_USEFULNESS_COPY,
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
    load_app_config,
    load_data_quality_report,
    load_drift,
    load_eval_results,
    load_feature_df,
    load_pipeline_profile,
    load_shap,
    load_weekly_completeness,
)
from src.dashboard.state import clamp_week_idx, monitoring_mode, pick_model_results, trim_to_length
from src.labeling.distress_score import compute_distress_score
from src.narration.narrative_generator import week_key_from_row

# ── Dashboard layout constants (issue #32) ─────────────────────────────
SUBREDDIT_ROLES = {
    "mentalhealth": "General discussion",
    "anxiety": "Early warning signal",
    "lonely": "Isolation indicator",
    "depression": "Core signal",
    "SuicideWatch": "Acute sentinel",
}

SUBREDDIT_ACCENT = {
    "mentalhealth": "#1D9E75",
    "anxiety": "#BA7517",
    "lonely": "#D85A30",
    "depression": "#A32D2D",
    "SuicideWatch": "#534AB7",
}

MONITORING_MODE = {"lonely", "mentalhealth"}

DASHBOARD_SUBORDER = ["mentalhealth", "anxiety", "lonely", "depression", "SuicideWatch"]

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title=DASHBOARD_COPY["page_title"],
    page_icon="🧠",
    layout="wide",
)


def _format_week_label(value) -> str:
    try:
        dt = pd.to_datetime(value)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(value)[:10]


def available_models_for_sub(sub_results: dict) -> list[str]:
    out: list[str] = []
    if sub_results.get("lstm"):
        out.append("LSTM")
    if sub_results.get("xgb"):
        out.append("XGBoost")
    if sub_results.get("lstm") and sub_results.get("xgb"):
        out.append("Ensemble")
    return out or ["LSTM"]


def resolve_model_results(sub_results: dict, preferred: str):
    avail = available_models_for_sub(sub_results)
    choice = preferred if preferred in avail else avail[0]
    return pick_model_results(sub_results, choice), choice


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


app_config = load_app_config()
feature_df = load_feature_df()
eval_results = load_eval_results()

if feature_df is None or eval_results is None:
    st.error(
        "No data found. Run the pipeline first:\n\n"
        "```\npython -m src.pipeline.run_all --config config/default.yaml --synthetic\n```"
    )
    st.stop()

available_subs = sorted(feature_df["subreddit"].unique().tolist())
if not available_subs:
    st.error("No subreddits available in feature data. Run collection + feature pipeline first.")
    st.stop()

visible_subs = [s for s in DASHBOARD_SUBORDER if s in available_subs]
if not visible_subs:
    st.error("None of the expected Zenodo subreddits are present in features. Check config reddit.subreddits.")
    st.stop()


def _n_weeks(sub: str) -> int:
    return int(feature_df[feature_df["subreddit"] == sub].shape[0])


n_weeks_max = max(_n_weeks(s) for s in visible_subs)
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
            st.rerun()
        st.markdown(
            "<div style='text-align:center;font-size:0.72rem;color:#94a3b8;margin-top:-0.35rem'>Previous</div>",
            unsafe_allow_html=True,
        )
    with nav_r:
        if st.button(
            "▶",
            key="wk_fwd",
            help="Next week",
            use_container_width=True,
        ):
            st.session_state.current_week = min(n_weeks_max - 1, int(st.session_state.current_week) + 1)
            st.rerun()
        st.markdown(
            "<div style='text-align:center;font-size:0.72rem;color:#94a3b8;margin-top:-0.35rem'>Next</div>",
            unsafe_allow_html=True,
        )
    with slider_col:
        st.slider(
            "Week",
            0,
            n_weeks_max - 1,
            key="current_week",
            label_visibility="collapsed",
        )
    _idx = int(st.session_state.current_week)
    _ref_df = feature_df[feature_df["subreddit"] == visible_subs[0]].copy()
    _ref_df = _ref_df.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
    _wl = _ref_df["week_start"].values if "week_start" in _ref_df.columns and len(_ref_df) else np.arange(len(_ref_df))
    _wi = min(_idx, len(_wl) - 1) if len(_wl) else 0
    _lbl = _format_week_label(_wl[_wi]) if len(_wl) else "—"
    st.caption(f"Replay week (calendar): {_lbl}")

week_idx = int(st.session_state.current_week)
subreddit = st.session_state.selected_sub

# Model selector must run before community cards so cards + timeline stay in sync on the same run.
sel_sub_results = eval_results.get(subreddit, {})
models_avail = available_models_for_sub(sel_sub_results)
if st.session_state.selected_model not in models_avail:
    st.session_state.selected_model = models_avail[0]
mod_a, mod_b = st.columns([0.12, 0.45])
with mod_a:
    st.markdown("**Model**")
with mod_b:
    st.selectbox(
        "Model",
        models_avail,
        key="selected_model",
        label_visibility="collapsed",
    )
model_choice = st.session_state.selected_model

# ── Community cards ───────────────────────────────────────────────────
card_cols = st.columns(len(visible_subs))
for i, sub in enumerate(visible_subs):
    with card_cols[i]:
        sub_results = eval_results.get(sub, {})
        results_i, _used_model = resolve_model_results(sub_results, model_choice)
        is_mm, _n_crisis = monitoring_mode(results_i, monitoring_min_crisis_weeks)
        use_trend_pill = sub in MONITORING_MODE and is_mm

        sub_df_i = feature_df[feature_df["subreddit"] == sub].copy()
        sub_df_i = sub_df_i.sort_values(["iso_year", "iso_week"]).reset_index(drop=True)
        n_wi = len(sub_df_i)
        wi_local = min(week_idx, n_wi - 1) if n_wi else 0

        distress_i = compute_distress_score(sub_df_i) if n_wi else pd.Series(dtype=float)
        per_w = (results_i or {}).get("per_week", {}) or {}
        preds = np.array(per_w.get("predictions", []))
        probs = np.array(per_w.get("probabilities", []))
        preds = trim_to_length(preds, n_wi)
        probs = trim_to_length(probs, n_wi)

        accent = SUBREDDIT_ACCENT.get(sub, "#64748b")
        role = SUBREDDIT_ROLES.get(sub, "")
        sel = subreddit == sub
        border = f"3px solid {accent}" if sel else "1px solid rgba(148,163,184,0.5)"
        bg = f"{accent}18" if sel else "transparent"

        st.markdown(
            f"<div style='border:{border};border-radius:10px;padding:10px;background:{bg};'>"
            f"<div style='font-weight:600;font-size:0.9rem'>r/{sub}</div>"
            f"<div style='font-size:0.75rem;color:#64748b;margin-bottom:6px'>{role}</div></div>",
            unsafe_allow_html=True,
        )

        if use_trend_pill:
            status_html = (
                "<div style='min-height:2.6em;line-height:1.35;color:#94a3b8;font-size:0.78rem'>"
                "<b>Trend monitoring</b><br/>"
                "<span style='opacity:0.9'>Insufficient crisis weeks for stable eval (&lt;10)</span>"
                "</div>"
            )
        else:
            cp = preds[wi_local] if len(preds) > wi_local and np.isfinite(preds[wi_local]) else np.nan
            state_line = STATE_NAMES.get(int(cp), "—") if not np.isnan(cp) else "—"
            status_html = (
                f"<div style='min-height:2.6em;line-height:1.35;font-size:0.82rem'>{state_line}</div>"
            )
        st.markdown(status_html, unsafe_allow_html=True)

        d_score = float(distress_i.iloc[wi_local]) if n_wi else 0.0
        p_hi = float(probs[wi_local]) if len(probs) > wi_local and np.isfinite(probs[wi_local]) else float("nan")
        p_line = f"{(p_hi * 100):.1f}%" if not np.isnan(p_hi) else "—"
        st.markdown(
            f"<div style='font-size:1.35rem;font-weight:600;color:{accent}'>{d_score:+.3f}</div>"
            f"<div style='font-size:0.75rem;color:#64748b'>p(distress) {p_line}</div>",
            unsafe_allow_html=True,
        )

        if n_wi:
            start_sp = max(0, wi_local - 11)
            spark = distress_i.iloc[start_sp : wi_local + 1]
            if len(spark) > 0:
                sp_fig = build_sparkline(spark, accent, height=72)
                st.plotly_chart(sp_fig, width="stretch", config={"displayModeBar": False})

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

predictions_all = trim_to_length(predictions_all, n_weeks)
probabilities_all = trim_to_length(probabilities_all, n_weeks)

distress_scores = compute_distress_score(sub_df)

theme_base = st.get_option("theme.base") or "light"
is_dark = theme_base == "dark"

week_label = _format_week_label(weeks[week_idx_plot]) if week_idx_plot < len(weeks) else "-"

# ── Main row: timeline + brief / metrics ──────────────────────────────
main_l, main_r = st.columns([0.65, 0.35])

with main_l:
    st.markdown("##### Distress timeline")
    accent_sel = SUBREDDIT_ACCENT.get(subreddit, "#378ADD")
    _w_slice = weeks[: week_idx_plot + 1]
    if "week_start" in sub_df.columns:
        x_hist = pd.to_datetime(_w_slice)
    else:
        x_hist = np.asarray(_w_slice, dtype=float)
    y_hist = distress_scores.values[: week_idx_plot + 1]

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

    marker_colors_list = []
    for i in range(week_idx_plot + 1):
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
            name=DASHBOARD_COPY["timeline_predicted_state_label"],
            marker=dict(color=marker_colors_list, size=9),
            hovertemplate="%{x}<br>Distress: %{y:.3f}<extra></extra>",
        )
    )

    probs_up_to = probabilities_all[: week_idx_plot + 1]
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
        xaxis_title="Week",
        yaxis_title="Distress score",
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
    st.plotly_chart(fig, width="stretch")

with main_r:
    st.markdown("##### Weekly snapshot")
    _brief_week_key = week_key_from_row(sub_df.iloc[week_idx_plot])
    _reports_root = Path(app_config["paths"]["reports"])
    _brief_path = _reports_root / subreddit / "weekly_briefs" / f"{_brief_week_key}.txt"
    if not _brief_path.exists():
        _brief_path = _reports_root / f"{subreddit}_weekly_brief_{_brief_week_key}.txt"

    if _brief_path.exists():
        _brief_text = _brief_path.read_text(encoding="utf-8").strip()
        _render_weekly_brief(_brief_text, _brief_week_key, in_sidebar=False)
    else:
        st.caption(DASHBOARD_COPY["weekly_brief_missing"])

    st.markdown("##### Model performance")
    st.caption("Walk-forward CV (metrics from eval_results.json)")
    if is_monitoring_mode:
        st.info(
            "Prediction disabled — fewer than 10 crisis weeks in dataset.",
        )
    else:
        render_model_metrics_tiles(results)

st.markdown("---")

# ── Bottom tabs ───────────────────────────────────────────────────────
tab_drift, tab_shap, tab_dq = st.tabs(["Drift alerts", "Feature importance", "Data quality"])

with tab_drift:
    st.markdown("##### Drift alerts (up to current week)")
    drift_df = load_drift(subreddit)
    if drift_df is not None and not drift_df.empty:
        drift_up = drift_df.iloc[: week_idx_plot + 1].copy()
        alert_cols = [c for c in drift_df.columns if not c.startswith("z_")]
        display_drift = drift_up[alert_cols].copy()
        render_drift_table(display_drift)
    else:
        st.info("No drift data found. Run `make evaluate` to generate.")

with tab_shap:
    st.markdown("##### Feature importance (SHAP — top 15)")
    shap_df = load_shap(subreddit)
    if shap_df is not None:
        top15 = shap_df.head(15).sort_values("mean_abs_shap", ascending=True)
        fig_shap = build_shap_bar(top15)
        st.plotly_chart(fig_shap, width="stretch")
    else:
        st.info("No SHAP data found. Run `make evaluate` to generate.")

with tab_dq:
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
        fig_c = go.Figure(
            go.Bar(
                x=dq_weekly["week_start"],
                y=dq_weekly["completeness_score"],
                marker_color=[
                    "#e74c3c" if bool(v) else "#2ecc71" for v in dq_weekly["is_gap"].tolist()
                ],
                name="Completeness score",
            )
        )
        fig_c.update_layout(
            title="Weekly completeness score (red = flagged gap)",
            xaxis_title="Week",
            yaxis_title="Completeness score",
            template="plotly_white",
            height=280,
        )
        st.plotly_chart(fig_c, width="stretch")

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
        st.dataframe(pd.DataFrame(flat_rows), width="stretch", height=220)

    fold_diag = results.get("fold_diagnostics", [])
    if fold_diag:
        st.markdown("**Fold diagnostics**")
        st.dataframe(pd.DataFrame(fold_diag), width="stretch", height=180)

    with st.expander("Full model metrics & decision usefulness", expanded=False):
        render_model_metrics(results, STATE_NAMES, DECISION_USEFULNESS_COPY)

    lead = results.get("detection_lead_time_distribution", {})
    lead_dist = lead.get("distribution", []) if isinstance(lead, dict) else []
    if lead_dist:
        fig_l = go.Figure(go.Histogram(x=lead_dist, nbinsx=10, marker_color="#3498db"))
        fig_l.update_layout(
            title="Detection lead time distribution (weeks)",
            xaxis_title="Lead time (weeks)",
            yaxis_title="Count",
            template="plotly_white",
            height=260,
        )
        st.plotly_chart(fig_l, width="stretch")
        st.caption(
            f"p50={lead.get('p50', 0):.2f}, p75={lead.get('p75', 0):.2f}, p90={lead.get('p90', 0):.2f}"
        )
