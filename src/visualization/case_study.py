from pathlib import Path

import numpy as np
import pandas as pd
from src.core.ui_config import CASE_STUDY_COPY


class CaseStudyGenerator:
    def __init__(
        self,
        feature_df: pd.DataFrame,
        distress_scores: pd.Series,
        eval_results: dict,
        shap_df: pd.DataFrame,
    ):
        self.feature_df = feature_df
        self.distress_scores = distress_scores
        self.eval_results = eval_results
        self.shap_df = shap_df

    def generate(self, crisis_week_idx: int, output_path: Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        predictions = np.array(self.eval_results["per_week"]["predictions"])
        probabilities = np.array(self.eval_results["per_week"]["probabilities"])

        week_row = self.feature_df.iloc[crisis_week_idx]
        week_label = f"{int(week_row['iso_year'])}-W{int(week_row['iso_week']):02d}"
        week_start = str(week_row.get("week_start", ""))[:10]

        # Look back 1-3 weeks for warning signals
        lookback_start = max(0, crisis_week_idx - 3)
        lookback_range = range(lookback_start, crisis_week_idx)

        # Top features that contributed
        top_features = self.shap_df.head(10)["feature"].tolist()

        lines = [
            f"# {CASE_STUDY_COPY['title_prefix']} {week_label}",
            f"**Week starting:** {week_start}",
            f"**Distress score:** {self.distress_scores.iloc[crisis_week_idx]:.3f}",
            "",
            CASE_STUDY_COPY["what_happened_header"],
            f"The community distress score spiked to {self.distress_scores.iloc[crisis_week_idx]:.3f}, "
            CASE_STUDY_COPY["threshold_sentence_suffix"],
            "",
            "## Early Warning Signals",
            "",
        ]

        for i in lookback_range:
            w = self.feature_df.iloc[i]
            w_label = f"{int(w['iso_year'])}-W{int(w['iso_week']):02d}"
            prob = probabilities[i] if not np.isnan(probabilities[i]) else 0
            pred = "FLAGGED" if predictions[i] == 1 else "not flagged"

            lines.append(f"### {w_label} ({pred}, probability: {prob:.2f})")

            # Show key feature values for this week
            for feat in top_features[:5]:
                if feat in self.feature_df.columns:
                    val = self.feature_df.iloc[i][feat]
                    baseline = self.feature_df[feat].mean()
                    diff_pct = ((val - baseline) / abs(baseline) * 100) if baseline != 0 else 0
                    direction = "above" if diff_pct > 0 else "below"
                    lines.append(f"- **{feat}**: {val:.4f} ({abs(diff_pct):.1f}% {direction} average)")

            lines.append("")

        lines.extend([
            "## Top Contributing Features (SHAP)",
            "",
            "| Rank | Feature | Importance |",
            "|------|---------|------------|",
        ])
        for rank, (_, row) in enumerate(self.shap_df.head(10).iterrows(), 1):
            lines.append(f"| {rank} | {row['feature']} | {row['mean_abs_shap']:.4f} |")

        lines.extend([
            "",
            CASE_STUDY_COPY["summary_header"],
            "",
            f"The early warning system detected precursor signals "
            f"{crisis_week_idx - lookback_start} weeks before this {CASE_STUDY_COPY['summary_event_noun']}. "
            f"Key indicators included changes in {', '.join(top_features[:3])}.",
        ])

        output_path.write_text("\n".join(lines))
        return output_path
