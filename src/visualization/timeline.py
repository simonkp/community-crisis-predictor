from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.core.domain_config import STATE_THRESHOLD_LABELS
from src.core.ui_config import (
    STATE_NAMES,
    TIMELINE_COPY,
    TIMELINE_STATE_BAND_COLORS,
    TIMELINE_STATE_MARKER_COLORS,
    TIMELINE_THRESHOLD_COLORS,
)


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
            color = TIMELINE_STATE_BAND_COLORS.get(state_int, "rgba(200,200,200,0.1)")
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
            name=TIMELINE_COPY["distress_series_name"],
            line=dict(color="steelblue", width=2),
        )
    )

    # Threshold lines
    if thresholds is not None:
        for thr, label, color in zip(thresholds, STATE_THRESHOLD_LABELS, TIMELINE_THRESHOLD_COLORS):
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
                name=f"{TIMELINE_COPY['threshold_fallback']} ({threshold:.2f})",
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
                        name=f"{TIMELINE_COPY['predicted_prefix']}: {state_name}",
                        hovertemplate=(
                            "%{x}<br>"
                            "State: " + state_name + "<br>"
                            "Distress score: %{y:.3f}<extra></extra>"
                        ),
                        marker=dict(
                            color=TIMELINE_STATE_MARKER_COLORS[state_int],
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
                    name=TIMELINE_COPY["predicted_binary"],
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
                name=TIMELINE_COPY["actual_binary"],
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
                name=TIMELINE_COPY["yaxis2_title"],
                line=dict(color="purple", width=1),
                yaxis="y2",
                opacity=0.5,
            )
        )

    fig.update_layout(
        title=TIMELINE_COPY["title"],
        xaxis_title=TIMELINE_COPY["xaxis_title"],
        yaxis_title=TIMELINE_COPY["yaxis_title"],
        yaxis2=dict(
            title=TIMELINE_COPY["yaxis2_title"],
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
