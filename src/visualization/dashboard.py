from pathlib import Path


def _decision_usefulness_table_html(metrics: dict) -> str:
    """Build responsive decision-usefulness table + baselines for static HTML report."""
    du = metrics.get("decision_usefulness")
    if not du or not isinstance(du, dict):
        return "<p><em>Decision usefulness metrics not available. Re-run training to refresh eval_results.json.</em></p>"

    kvals = du.get("k_values") or []
    model = du.get("model") or {}
    rnd = du.get("random_expected_recall") or {}
    pers = du.get("persistence") or {}
    n_w = du.get("n_weeks", 0)
    n_p = du.get("n_elevated_distress_weeks", 0)

    intro = (
        "<p class=\"du-intro\"><strong>What this measures.</strong> "
        "Across all walk-forward weeks with valid labels, we rank weeks by predicted "
        "<em>high-distress probability</em> and treat the top <em>K</em> as the alert budget. "
        "<strong>Recall@K</strong> = fraction of true elevated-distress weeks (actual state ≥ 2) "
        "that appear in those top-K slots. "
        "<strong>Random</strong> = expected recall if <em>K</em> weeks were chosen uniformly at random "
        f"(same pool of <em>n={n_w}</em> weeks, <em>P={n_p}</em> positive weeks). "
        "<strong>Persistence</strong> = rank weeks by whether the previous week was elevated-distress, "
        "then take top-<em>K</em> (tie-aware).</p>"
    )

    rows = []
    for k in kvals:
        ks = str(k)
        mk = model.get(ks) or model.get(k) or {}
        pk = pers.get(ks) or pers.get(k) or {}
        cap = mk.get("captured", 0)
        tot = mk.get("total_positives", n_p)
        rec = mk.get("recall", 0.0)
        r_rnd = rnd.get(str(k), rnd.get(k, 0.0))
        p_cap = pk.get("captured", 0)
        p_rec = pk.get("recall", 0.0)
        rows.append(
            f"<tr>"
            f"<td data-label=\"Alert budget (K)\"><strong>{k}</strong></td>"
            f"<td data-label=\"Elevated-distress captured (model)\">{cap}/{tot}</td>"
            f"<td data-label=\"Recall@K (model)\">{float(rec):.1%}</td>"
            f"<td data-label=\"Expected recall@K (random)\">{float(r_rnd):.1%}</td>"
            f"<td data-label=\"Persistence baseline\">{p_cap}/{tot} ({float(p_rec):.1%})</td>"
            f"</tr>"
        )

    table = (
        "<div class=\"du-table-wrap\" role=\"region\" aria-label=\"Decision usefulness metrics\">"
        "<table class=\"du-table\">"
        "<thead><tr>"
        "<th>Alert budget (K)</th>"
        "<th>Elevated-distress weeks captured (model)</th>"
        "<th>Recall@K (model)</th>"
        "<th>Expected recall@K (random)</th>"
        "<th>Captured / Recall@K (persistence)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table></div>"
    )
    return intro + table


def generate_html_report(
    timeline_path: Path,
    importance_path: Path,
    case_study_paths: list[Path],
    metrics: dict,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read embedded HTML from Plotly charts
    timeline_html = ""
    if timeline_path and Path(timeline_path).exists():
        timeline_html = Path(timeline_path).read_text(encoding="utf-8")

    importance_html = ""
    if importance_path and Path(importance_path).exists():
        importance_html = Path(importance_path).read_text(encoding="utf-8")

    # Read case studies (markdown)
    case_studies_md = []
    for cs_path in (case_study_paths or []):
        if Path(cs_path).exists():
            case_studies_md.append(Path(cs_path).read_text(encoding="utf-8"))

    # Build metrics table
    metrics_rows = ""
    for key in ["recall", "precision", "f1", "pr_auc",
                 "n_crisis_actual", "n_crisis_predicted",
                 "avg_detection_lead_time_weeks"]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.3f}"
        metrics_rows += f"<tr><td>{key}</td><td>{val}</td></tr>\n"

    du_section = _decision_usefulness_table_html(metrics)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Community Crisis Prediction Report</title>
    <style>
        :root {{
            --bg: #fafafa;
            --text: #1f2937;
            --muted: #64748b;
            --card: #ffffff;
            --border: #e2e8f0;
            --th: #f1f5f9;
            --accent: #2563eb;
        }}
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg: #0b1220;
                --text: #e6edf7;
                --muted: #94a3b8;
                --card: #121826;
                --border: #273244;
                --th: #1a2332;
                --accent: #60a5fa;
            }}
        }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; background: var(--bg); color: var(--text); line-height: 1.5; }}
        h1, h2, h3 {{ color: var(--text); }}
        .section {{ background: var(--card); padding: 20px; margin: 20px 0; border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 2px rgba(0,0,0,0.06); }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid var(--border); padding: 10px 12px; text-align: left; }}
        th {{ background: var(--th); color: var(--text); }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
        .metric-card {{ background: var(--th); padding: 16px; border-radius: 8px; text-align: center; border: 1px solid var(--border); }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: var(--accent); }}
        .metric-label {{ font-size: 14px; color: var(--muted); margin-top: 4px; }}
        pre {{ background: var(--th); padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid var(--border); color: var(--text); }}
        iframe {{ border: none; width: 100%; }}
        .du-intro {{ color: var(--muted); margin-bottom: 16px; font-size: 0.95rem; }}
        .du-table-wrap {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
        .du-table {{ min-width: 640px; }}
        @media (max-width: 700px) {{
            .du-table thead {{ display: none; }}
            .du-table tr {{ display: block; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 8px; }}
            .du-table td {{ display: block; text-align: right; padding-left: 50%; position: relative; border: none; border-bottom: 1px solid var(--border); }}
            .du-table td:last-child {{ border-bottom: none; }}
            .du-table td::before {{ content: attr(data-label); position: absolute; left: 12px; text-align: left; font-weight: 600; color: var(--muted); }}
        }}
    </style>
</head>
<body>
    <h1>Community Mental Health Crisis Prediction Report</h1>

    <div class="section">
        <h2>Model Performance</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 0):.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 0):.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1', 0):.3f}</div>
                <div class="metric-label">F1 Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('pr_auc', 0):.3f}</div>
                <div class="metric-label">PR-AUC</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('avg_detection_lead_time_weeks', 0):.1f}</div>
                <div class="metric-label">Avg Lead Time (weeks)</div>
            </div>
        </div>
        <br>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>
    </div>

    <div class="section">
        <h2>Decision usefulness</h2>
        <p>Operational budget framing: recall of true elevated-distress weeks when only <em>K</em> alerts are possible.</p>
        {du_section}
    </div>

    <div class="section">
        <h2>Backtesting Timeline</h2>
        <p>Distress scores over time with predicted and actual crisis markers.</p>
        {'<iframe srcdoc="' + timeline_html.replace('"', '&quot;') + '" height="550"></iframe>' if timeline_html else '<p>Timeline not available</p>'}
    </div>

    <div class="section">
        <h2>Feature Importance</h2>
        {'<iframe srcdoc="' + importance_html.replace('"', '&quot;') + '" height="600"></iframe>' if importance_html else '<p>Importance plot not available</p>'}
    </div>

    <div class="section">
        <h2>Case Studies</h2>
        {''.join(f'<pre>{cs}</pre>' for cs in case_studies_md) if case_studies_md else '<p>No case studies generated</p>'}
    </div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    return output_path
