from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

STATE_COLORS = {
    0: "rgba(0,200,0,0.15)",    # Stable — light green
    1: "rgba(255,200,0,0.20)",  # Emerging Distress — light yellow
    2: "rgba(255,100,0,0.25)",  # Acute Risk — light orange
    3: "rgba(220,0,0,0.30)",    # Critical Escalation — light red
}

STATE_MARKER_COLORS = {
    0: "green",
    1: "gold",
    2: "orangered",
    3: "darkred",
}

STATE_NAMES = {
    0: "Stable",
    1: "Emerging Distress",
    2: "Acute Risk",
    3: "Critical Escalation",
}


def plot_backtest_timeline(
    feature_df: pd.DataFrame,
    distress_scores: pd.Series,
    eval_results: dict,
    threshold: float,
    output_path: Path,
    thresholds: list[float] | None = None,
) -> Path:
    weeks = feature_df["week_start"].values
    per_week = eval_results.get("per_week", {})
    predictions = np.array(per_week.get("predictions", np.full(len(weeks), np.nan)))
    actuals = np.array(per_week.get("actuals", np.full(len(weeks), np.nan)))
    probabilities = np.array(per_week.get("probabilities", np.full(len(weeks), np.nan)))

    is_multiclass = np.nanmax(predictions) > 1 if (~np.isnan(predictions)).any() else False

    fig = go.Figure()

    # Background state bands (multiclass only)
    if is_multiclass:
        for i, w in enumerate(weeks):
            state = predictions[i]
            if np.isnan(state):
                continue
            state_int = int(state)
            color = STATE_COLORS.get(state_int, "rgba(200,200,200,0.1)")
            fig.add_vrect(
                x0=w,
                x1=weeks[min(i + 1, len(weeks) - 1)],
                fillcolor=color,
                line_width=0,
                layer="below",
            )

    # Distress score line
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=distress_scores.values,
            mode="lines",
            name="Community Distress Score",
            line=dict(color="steelblue", width=2),
        )
    )

    # Threshold lines
    if thresholds is not None:
        threshold_labels = ["Emerging (0.5σ)", "Acute (1.0σ)", "Critical (2.0σ)"]
        threshold_colors = ["gold", "orangered", "darkred"]
        for thr, label, color in zip(thresholds, threshold_labels, threshold_colors):
            fig.add_trace(
                go.Scatter(
                    x=[weeks[0], weeks[-1]],
                    y=[thr, thr],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1, dash="dash"),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=[weeks[0], weeks[-1]],
                y=[threshold, threshold],
                mode="lines",
                name=f"Crisis Threshold ({threshold:.2f})",
                line=dict(color="orange", width=1, dash="dash"),
            )
        )

    # Predicted states (colored markers)
    if is_multiclass:
        for state_int, state_name in STATE_NAMES.items():
            if state_int == 0:
                continue  # skip stable — too noisy
            mask = np.array(
                [
                    not np.isnan(predictions[i]) and int(predictions[i]) == state_int
                    for i in range(len(weeks))
                ]
            )
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=weeks[mask],
                        y=distress_scores.values[mask],
                        mode="markers",
                        name=f"Predicted: {state_name}",
                        marker=dict(
                            color=STATE_MARKER_COLORS[state_int],
                            size=10,
                            symbol="triangle-up",
                        ),
                    )
                )
    else:
        pred_mask = predictions == 1
        if pred_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=weeks[pred_mask],
                    y=distress_scores.values[pred_mask],
                    mode="markers",
                    name="Predicted Crisis",
                    marker=dict(color="blue", size=10, symbol="triangle-up"),
                )
            )

    # Actual crisis weeks
    if is_multiclass:
        actual_crisis_mask = np.array(
            [not np.isnan(actuals[i]) and int(actuals[i]) >= 2 for i in range(len(weeks))]
        )
    else:
        actual_crisis_mask = actuals == 1
    if actual_crisis_mask.any():
        fig.add_trace(
            go.Scatter(
                x=weeks[actual_crisis_mask],
                y=distress_scores.values[actual_crisis_mask],
                mode="markers",
                name="Actual Crisis Week",
                marker=dict(color="red", size=12, symbol="x"),
            )
        )

    # Crisis probability (secondary axis)
    valid_prob = ~np.isnan(probabilities)
    if valid_prob.any():
        fig.add_trace(
            go.Scatter(
                x=weeks[valid_prob],
                y=probabilities[valid_prob],
                mode="lines",
                name="Crisis Probability",
                line=dict(color="purple", width=1),
                yaxis="y2",
                opacity=0.5,
            )
        )

    fig.update_layout(
        title="Community Mental Health Crisis Prediction — Backtesting Timeline",
        xaxis_title="Week",
        yaxis_title="Distress Score (z-scored)",
        yaxis2=dict(
            title="Crisis Probability",
            overlaying="y",
            side="right",
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
