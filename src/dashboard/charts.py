import plotly.graph_objects as go


def build_sparkline(spark, line_color: str, *, height: int = 90) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=list(range(len(spark))),
            y=spark.values,
            mode="lines",
            line=dict(color=line_color, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(55,65,81,0.08)",
            hovertemplate="t-%{x}: %{y:.3f}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=4, b=0),
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        template="plotly_white",
    )
    return fig


def build_shap_bar(top15) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=top15["mean_abs_shap"],
            y=top15["feature"],
            orientation="h",
            marker_color="steelblue",
        )
    )
    fig.update_layout(
        xaxis_title="Mean |SHAP|",
        yaxis_title="",
        template="plotly_white",
        height=400,
        margin=dict(l=200),
    )
    return fig
