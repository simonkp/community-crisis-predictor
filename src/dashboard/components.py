from html import escape

import pandas as pd
import streamlit as st


def render_drift_table(display_drift: pd.DataFrame) -> None:
    theme_base = st.get_option("theme.base") or "light"
    is_dark = theme_base == "dark"

    if is_dark:
        frame_color = "#273244"
        header_bg = "#121826"
        header_text = "#e6edf7"
        row_styles = {
            0: ("#0b1220", "#dbe7ff"),
            1: ("#33481f", "#f8ffe8"),
            2: ("#5b3713", "#fff7e6"),
            3: ("#5c1f27", "#fff1f3"),
        }
    else:
        frame_color = "#d8deea"
        header_bg = "#f2f5fb"
        header_text = "#1f2937"
        row_styles = {
            0: ("#ffffff", "#111827"),
            1: ("#fff8db", "#3d2f00"),
            2: ("#ffe9cc", "#4a2b00"),
            3: ("#ffd9d6", "#5a1516"),
        }

    def _format_cell(col: str, value):
        if pd.isna(value):
            return "-"
        if col == "week_start":
            try:
                numeric = float(value)
                if numeric > 1e12:
                    return pd.to_datetime(int(numeric), unit="ms").strftime("%Y-%m-%d")
                if numeric > 1e9:
                    return pd.to_datetime(int(numeric), unit="s").strftime("%Y-%m-%d")
            except Exception:
                pass
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    table_rows = []
    for _, row in display_drift.iterrows():
        level = int(row.get("aggregate_level", 0)) if not pd.isna(row.get("aggregate_level", 0)) else 0
        bg, text = row_styles.get(level, row_styles[0])
        row_cells = "".join(
            f"<td style='padding:10px 12px;border-top:1px solid {frame_color};color:{text};'>{escape(_format_cell(col, row[col]))}</td>"
            for col in display_drift.columns
        )
        table_rows.append(f"<tr style='background:{bg};'>{row_cells}</tr>")

    header_cells = "".join(
        f"<th style='text-align:left;padding:10px 12px;background:{header_bg};color:{header_text};border-bottom:1px solid {frame_color};'>{escape(str(col))}</th>"
        for col in display_drift.columns
    )

    drift_table_html = f"""
    <div style='border:1px solid {frame_color}; border-radius:10px; overflow:auto; max-height:240px;'>
      <table style='border-collapse:separate; border-spacing:0; width:100%; font-size:0.95rem;'>
        <thead>
          <tr>{header_cells}</tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(drift_table_html, unsafe_allow_html=True)


def _fmt_metric(value) -> str:
    """Format a metric value; show '—' when absent or exactly 0 (uncomputed)."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{f:.3f}" if f != 0.0 else "—"


def render_model_metrics_tiles(results: dict) -> None:
    """Compact Recall / Precision / F1 / PR-AUC row for dashboard side column."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", _fmt_metric(results.get("recall")))
    c2.metric("Precision", _fmt_metric(results.get("precision")))
    c3.metric("F1", _fmt_metric(results.get("f1")))
    c4.metric("PR-AUC", _fmt_metric(results.get("pr_auc")))


def render_model_metrics(results: dict, state_names: dict, decision_usefulness_copy: dict) -> None:
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Recall", f"{results.get('recall', 0):.3f}")
    col_b.metric("Precision", f"{results.get('precision', 0):.3f}")
    col_c.metric("F1", f"{results.get('f1', 0):.3f}")
    col_d.metric("PR-AUC", f"{results.get('pr_auc', 0):.3f}")

    if "confusion_matrix_4class" in results:
        st.markdown("**4-class confusion matrix**")
        cm = results["confusion_matrix_4class"]
        cm_df = pd.DataFrame(
            cm,
            index=[f"True {state_names[i]}" for i in range(4)],
            columns=[f"Pred {state_names[i]}" for i in range(4)],
        )
        st.dataframe(cm_df)

    if "recall_class_0" in results:
        st.markdown("**Per-class recall**")
        for cls in range(4):
            val = results.get(f"recall_class_{cls}", 0)
            st.write(f"- {state_names[cls]}: {val:.3f}")

    _du = results.get("decision_usefulness")
    if _du and isinstance(_du, dict):
        st.markdown(decision_usefulness_copy["title"])
        st.markdown(decision_usefulness_copy["intro"])
        kvals = _du.get("k_values") or []
        model = _du.get("model") or {}
        rnd = _du.get("random_expected_recall") or {}
        pers = _du.get("persistence") or {}
        rows_du = []
        for k in kvals:
            ks = str(k)
            mk = model.get(ks) or model.get(k) or {}
            pk = pers.get(ks) or pers.get(k) or {}
            cap = mk.get("captured", 0)
            tot = mk.get("total_positives", _du.get("n_elevated_distress_weeks", 0))
            rec = mk.get("recall", 0.0)
            r_rnd = rnd.get(str(k), rnd.get(k, 0.0))
            p_cap = pk.get("captured", 0)
            p_rec = pk.get("recall", 0.0)
            rows_du.append(
                {
                    "K": k,
                    "Captured (model)": f"{cap}/{tot}",
                    "Recall@K (model)": f"{float(rec):.1%}",
                    "Expected Recall@K (random)": f"{float(r_rnd):.1%}",
                    "Persistence": f"{p_cap}/{tot} ({float(p_rec):.1%})",
                }
            )
        st.caption(
            f"Evaluation weeks n={_du.get('n_weeks', '—')}, "
            f"elevated-distress weeks P={_du.get('n_elevated_distress_weeks', '—')}."
        )
        st.dataframe(pd.DataFrame(rows_du), use_container_width=True, hide_index=True)
