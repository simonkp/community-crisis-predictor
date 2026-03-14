from pathlib import Path


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
        timeline_html = Path(timeline_path).read_text()

    importance_html = ""
    if importance_path and Path(importance_path).exists():
        importance_html = Path(importance_path).read_text()

    # Read case studies (markdown)
    case_studies_md = []
    for cs_path in (case_study_paths or []):
        if Path(cs_path).exists():
            case_studies_md.append(Path(cs_path).read_text())

    # Build metrics table
    metrics_rows = ""
    for key in ["recall", "precision", "f1", "pr_auc",
                 "n_crisis_actual", "n_crisis_predicted",
                 "avg_detection_lead_time_weeks"]:
        val = metrics.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.3f}"
        metrics_rows += f"<tr><td>{key}</td><td>{val}</td></tr>\n"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Community Crisis Prediction Report</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; margin: 40px; background: #fafafa; }}
        h1, h2, h3 {{ color: #333; }}
        .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f0f0f0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; }}
        .metric-card {{ background: #f8f9fa; padding: 16px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
        .metric-label {{ font-size: 14px; color: #666; margin-top: 4px; }}
        pre {{ background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }}
        iframe {{ border: none; width: 100%; }}
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

    output_path.write_text(html)
    return output_path
