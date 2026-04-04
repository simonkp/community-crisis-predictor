"""
EDA report generator — mirrors the L1.2 data quality + L3.2 EDA pattern from IS5126.

For each subreddit, this generates:
  1. Feature distribution summary (mean, std, IQR, skew)
  2. Outlier weeks flagged via IQR rule (like professor's L1.2 NIH thresholds)
  3. Trend identification: is the community's distress rising or falling?
  4. Crisis rate over time (weekly rate of crisis-state weeks)

Outputs:
  data/reports/{sub}/eda_report.json   — machine-readable summary
  data/reports/{sub}/eda_summary.html  — visual HTML for the project report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.labeling.distress_score import compute_distress_score
from src.labeling.target import CrisisLabeler


def generate_eda_report(
    feature_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
    subreddit: str,
    output_dir: Path,
) -> dict:
    """
    Run exploratory analysis on one subreddit's feature matrix.

    Returns a dict with keys: distributions, outlier_weeks, trend, crisis_rate_over_time.
    Also saves eda_report.json and eda_summary.html to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"subreddit": subreddit, "n_weeks": len(feature_df)}

    # ------------------------------------------------------------------ #
    # 1. Feature distribution summary                                      #
    # ------------------------------------------------------------------ #
    distributions: dict[str, dict] = {}
    for col in feature_columns:
        if col not in feature_df.columns:
            continue
        series = feature_df[col].dropna()
        if len(series) == 0:
            continue
        q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
        distributions[col] = {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "q1": round(q1, 4),
            "median": round(float(series.median()), 4),
            "q3": round(q3, 4),
            "skew": round(float(series.skew()), 4),
            "missing_pct": round(float(feature_df[col].isna().mean() * 100), 2),
            "n_valid": int(len(series)),
        }
    report["distributions"] = distributions

    # ------------------------------------------------------------------ #
    # 2. Outlier detection via IQR rule (L1.2 pattern)                    #
    #    A week is an outlier for feature X if:                           #
    #    value < Q1 - 1.5*IQR  OR  value > Q3 + 1.5*IQR                  #
    # ------------------------------------------------------------------ #
    outlier_weeks: list[dict] = []
    for col in feature_columns:
        if col not in feature_df.columns:
            continue
        series = feature_df[col].dropna()
        if len(series) < 8:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (feature_df[col] < lower) | (feature_df[col] > upper)
        outlier_rows = feature_df[mask].copy()
        for _, row in outlier_rows.iterrows():
            direction = "high" if row[col] > upper else "low"
            outlier_weeks.append(
                {
                    "feature": col,
                    "iso_year": int(row.get("iso_year", 0)),
                    "iso_week": int(row.get("iso_week", 0)),
                    "value": round(float(row[col]), 4),
                    "direction": direction,
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                }
            )
    outlier_weeks.sort(key=lambda x: (x["iso_year"], x["iso_week"]))
    report["outlier_weeks"] = outlier_weeks
    report["n_outlier_weeks"] = len(outlier_weeks)

    # ------------------------------------------------------------------ #
    # 3. Trend identification (L3.2 / L4.0 temporal drift pattern)        #
    #    Fit a linear regression on distress_score vs week index.          #
    #    Report slope, direction, and magnitude.                           #
    # ------------------------------------------------------------------ #
    labeling_cfg = config.get("labeling", {})
    weights = labeling_cfg.get("distress_weights")
    distress_scores = compute_distress_score(feature_df, weights)

    if len(distress_scores) >= 4:
        x = np.arange(len(distress_scores), dtype=float)
        y = distress_scores.values.astype(float)
        valid = ~np.isnan(y)
        if valid.sum() >= 4:
            slope, intercept = np.polyfit(x[valid], y[valid], 1)
            total_change = slope * (valid.sum() - 1)
            # Use std as normalization base — mean can be near-zero making % change meaningless
            std_score = float(np.nanstd(y))
            if std_score > 1e-8:
                # Express change in units of std deviations (sigma) over the period
                sigma_change = total_change / std_score
            else:
                sigma_change = 0.0
            # Direction based on slope magnitude relative to std (>0.1 sigma/period = meaningful)
            if abs(sigma_change) < 0.1:
                direction = "stable"
            elif slope > 0:
                direction = "rising"
            else:
                direction = "declining"
            report["trend"] = {
                "slope_per_week": round(float(slope), 6),
                "total_change_weeks": int(valid.sum()),
                "sigma_change_over_period": round(float(sigma_change), 3),
                "direction": direction,
                "interpretation": (
                    f"r/{subreddit} distress is {direction} "
                    f"({sigma_change:+.2f}\u03c3 over {int(valid.sum())} weeks)"
                ),
            }
        else:
            report["trend"] = {"direction": "insufficient_data"}
    else:
        report["trend"] = {"direction": "insufficient_data"}

    # ------------------------------------------------------------------ #
    # 4. Crisis rate over time (weekly binned)                            #
    #    Shows how the fraction of crisis-state weeks changes year-over-y  #
    # ------------------------------------------------------------------ #
    thresholds_std = labeling_cfg.get("crisis_thresholds_std", [0.5, 1.0, 2.0])
    threshold_std = labeling_cfg.get("crisis_threshold_std", 1.5)
    labeler = CrisisLabeler(threshold_std=threshold_std, thresholds_std=thresholds_std)
    labeler.fit(distress_scores)
    labels = labeler.label(distress_scores)

    crisis_rate_by_year: dict[str, float] = {}
    if "iso_year" in feature_df.columns:
        tmp = feature_df[["iso_year"]].copy()
        tmp["label"] = labels.values
        tmp = tmp.dropna(subset=["label"])
        tmp["is_crisis"] = (tmp["label"] >= 2).astype(int)
        by_year = tmp.groupby("iso_year")["is_crisis"].mean()
        crisis_rate_by_year = {str(int(y)): round(float(r), 4) for y, r in by_year.items()}
    report["crisis_rate_by_year"] = crisis_rate_by_year

    overall_crisis_rate = 0.0
    valid_labels = labels.dropna()
    if len(valid_labels) > 0:
        overall_crisis_rate = float((valid_labels >= 2).mean())
    report["overall_crisis_rate"] = round(overall_crisis_rate, 4)

    # ------------------------------------------------------------------ #
    # 5. Data quality flags (L1.2 NIH threshold pattern)                  #
    # ------------------------------------------------------------------ #
    quality_flags: list[str] = []
    high_missing = [
        col for col, d in distributions.items() if d["missing_pct"] > 20
    ]
    if high_missing:
        quality_flags.append(
            f"High missingness (>20%): {', '.join(high_missing)}"
        )
    high_outlier_features = {}
    for o in outlier_weeks:
        high_outlier_features[o["feature"]] = high_outlier_features.get(o["feature"], 0) + 1
    top_outlier = sorted(high_outlier_features.items(), key=lambda x: -x[1])[:3]
    if top_outlier:
        quality_flags.append(
            "Most outlier-prone features: "
            + ", ".join(f"{f} ({n} weeks)" for f, n in top_outlier)
        )
    if overall_crisis_rate < 0.05:
        quality_flags.append(
            f"Very low crisis rate ({overall_crisis_rate:.1%}) — class imbalance may limit model performance"
        )
    report["quality_flags"] = quality_flags

    # ------------------------------------------------------------------ #
    # Save artifacts                                                       #
    # ------------------------------------------------------------------ #
    json_path = output_dir / "eda_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    html_path = output_dir / "eda_summary.html"
    _write_eda_html(report, html_path)

    print(f"  EDA report -> {json_path}")
    print(f"  EDA HTML   -> {html_path}")
    return report


def _write_eda_html(report: dict, path: Path) -> None:
    """Write a self-contained HTML EDA summary for inclusion in the project report."""
    sub = report["subreddit"]
    n = report["n_weeks"]
    trend = report.get("trend", {})
    overall_rate = report.get("overall_crisis_rate", 0)
    flags = report.get("quality_flags", [])

    dists = report.get("distributions", {})
    dist_rows = ""
    for col, d in sorted(dists.items()):
        missing_color = "#d9534f" if d["missing_pct"] > 20 else ("#f0ad4e" if d["missing_pct"] > 5 else "#5cb85c")
        dist_rows += (
            f"<tr><td>{col}</td><td>{d['mean']:.4f}</td><td>{d['std']:.4f}</td>"
            f"<td>{d['q1']:.4f}</td><td>{d['median']:.4f}</td><td>{d['q3']:.4f}</td>"
            f"<td>{d['skew']:.3f}</td>"
            f"<td style='color:{missing_color}'>{d['missing_pct']:.1f}%</td></tr>\n"
        )

    outliers = report.get("outlier_weeks", [])
    outlier_rows = ""
    for o in outliers[:30]:
        outlier_rows += (
            f"<tr><td>{o['iso_year']}-W{o['iso_week']:02d}</td>"
            f"<td>{o['feature']}</td><td>{o['direction']}</td>"
            f"<td>{o['value']:.4f}</td>"
            f"<td>[{o['lower_bound']:.3f}, {o['upper_bound']:.3f}]</td></tr>\n"
        )

    crisis_by_year = report.get("crisis_rate_by_year", {})
    year_rows = "".join(
        f"<tr><td>{yr}</td><td>{rate:.1%}</td></tr>"
        for yr, rate in sorted(crisis_by_year.items())
    )

    flags_html = "".join(f"<li>{f}</li>" for f in flags) if flags else "<li>None</li>"

    trend_dir = trend.get("direction", "unknown")
    trend_color = {"rising": "#d9534f", "declining": "#5cb85c", "stable": "#5bc0de"}.get(trend_dir, "#777")
    trend_interp = trend.get("interpretation", "N/A")

    # Pre-compute outlier section — nested f-strings with triple quotes are invalid in Python 3.11
    if outliers:
        outlier_section = (
            "<table>\n"
            "  <tr><th>Week</th><th>Feature</th><th>Direction</th><th>Value</th><th>Normal Range</th></tr>\n"
            f"  {outlier_rows}\n"
            "</table>"
        )
    else:
        outlier_section = "<p>No outlier weeks detected.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>EDA Report — r/{sub}</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 2em; color: #333; }}
  h1 {{ color: #2c3e50; }}
  h2 {{ color: #34495e; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5em; font-size: 0.9em; }}
  th {{ background: #2c3e50; color: white; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; color: white;
            font-weight: bold; background: {trend_color}; }}
  .flag-list {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: 0.8em 1em; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1em; margin-bottom: 1.5em; }}
  .stat-card {{ background: #f4f6f8; border-radius: 8px; padding: 1em; text-align: center; }}
  .stat-num {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
  .stat-label {{ color: #777; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Exploratory Data Analysis — r/{sub}</h1>

<div class="summary-grid">
  <div class="stat-card">
    <div class="stat-num">{n}</div>
    <div class="stat-label">Total Weeks Analyzed</div>
  </div>
  <div class="stat-card">
    <div class="stat-num">{overall_rate:.1%}</div>
    <div class="stat-label">Overall Crisis Rate (State 2+3)</div>
  </div>
  <div class="stat-card">
    <div class="stat-num"><span class="badge">{trend_dir}</span></div>
    <div class="stat-label">Distress Trend Direction</div>
  </div>
</div>

<h2>Trend Analysis</h2>
<p><strong>Interpretation:</strong> {trend_interp}</p>
{"<p>Slope per week: " + str(trend.get('slope_per_week','N/A')) + " | Change over period: " + str(trend.get('sigma_change_over_period','N/A')) + "&sigma; (" + str(trend.get('total_change_weeks','?')) + " weeks)</p>" if 'slope_per_week' in trend else ""}

<h2>Crisis Rate by Year</h2>
<table>
  <tr><th>Year</th><th>Crisis Rate (% of weeks in State 2 or 3)</th></tr>
  {year_rows}
</table>

<h2>Data Quality Flags</h2>
<div class="flag-list"><ul>{flags_html}</ul></div>

<h2>Feature Distribution Summary</h2>
<p>IQR-based outlier threshold: values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] are flagged.
Missing % color: <span style="color:#5cb85c">green</span> &lt;5%,
<span style="color:#f0ad4e">amber</span> 5-20%,
<span style="color:#d9534f">red</span> &gt;20%.</p>
<table>
  <tr><th>Feature</th><th>Mean</th><th>Std</th><th>Q1</th><th>Median</th><th>Q3</th><th>Skew</th><th>Missing</th></tr>
  {dist_rows}
</table>

<h2>Outlier Weeks (IQR Rule, top 30 shown)</h2>
{outlier_section}

<p style="color:#aaa; font-size:0.8em; margin-top:2em;">
Generated by community-crisis-predictor EDA module.
Methodology: IQR outlier detection (L1.2), linear trend regression (L3.2), crisis labeling via sigma thresholds.
</p>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
