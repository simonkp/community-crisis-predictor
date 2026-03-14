from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_backtest_timeline(
    feature_df: pd.DataFrame,
    distress_scores: pd.Series,
    eval_results: dict,
    threshold: float,
    output_path: Path,
) -> Path:
    weeks = feature_df["week_start"].values
    predictions = np.array(eval_results["per_week"]["predictions"])
    actuals = np.array(eval_results["per_week"]["actuals"])
    probabilities = np.array(eval_results["per_week"]["probabilities"])

    fig = go.Figure()

    # Distress score line
    fig.add_trace(go.Scatter(
        x=weeks, y=distress_scores.values,
        mode="lines", name="Community Distress Score",
        line=dict(color="steelblue", width=2),
    ))

    # Threshold line
    fig.add_trace(go.Scatter(
        x=[weeks[0], weeks[-1]], y=[threshold, threshold],
        mode="lines", name=f"Crisis Threshold ({threshold:.2f})",
        line=dict(color="orange", width=1, dash="dash"),
    ))

    # Actual crisis weeks
    crisis_mask = actuals == 1
    if crisis_mask.any():
        fig.add_trace(go.Scatter(
            x=weeks[crisis_mask],
            y=distress_scores.values[crisis_mask],
            mode="markers", name="Actual Crisis Week",
            marker=dict(color="red", size=12, symbol="x"),
        ))

    # Predicted crisis weeks
    pred_mask = predictions == 1
    if pred_mask.any():
        fig.add_trace(go.Scatter(
            x=weeks[pred_mask],
            y=distress_scores.values[pred_mask],
            mode="markers", name="Predicted Crisis",
            marker=dict(color="blue", size=10, symbol="triangle-up"),
        ))

    # Probability area
    valid_prob = ~np.isnan(probabilities)
    if valid_prob.any():
        fig.add_trace(go.Scatter(
            x=weeks[valid_prob],
            y=probabilities[valid_prob],
            mode="lines", name="Crisis Probability",
            line=dict(color="purple", width=1),
            yaxis="y2",
            opacity=0.5,
        ))

    fig.update_layout(
        title="Community Mental Health Crisis Prediction — Backtesting Timeline",
        xaxis_title="Week",
        yaxis_title="Distress Score (z-scored)",
        yaxis2=dict(
            title="Crisis Probability",
            overlaying="y", side="right",
            range=[0, 1],
        ),
        hovermode="x unified",
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path
