from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def plot_feature_importance(
    shap_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Path = None,
) -> Path:
    df = shap_df.head(top_n).sort_values("mean_abs_shap", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"].values,
        y=df["feature"].values,
        orientation="h",
        marker_color="steelblue",
    ))

    fig.update_layout(
        title=f"Top {top_n} Features by SHAP Importance",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=max(400, top_n * 25),
        margin=dict(l=200),
    )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

    return output_path
