import tempfile
from pathlib import Path

import pandas as pd
from src.features.distress import DistressScorer


def test_distress_scorer_with_lexicons():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "hopelessness.txt").write_text("hopeless\nno point\n")
        (Path(tmpdir) / "help_seeking.txt").write_text("need advice\nanyone else\n")
        (Path(tmpdir) / "distress.txt").write_text("anxious\ndepressed\n")

        scorer = DistressScorer(tmpdir)

        df = pd.DataFrame({
            "texts": [["I feel hopeless and anxious. Does anyone else feel this way? I need advice."]],
        })

        result = scorer.extract_distress_features(df)
        assert result["hopelessness_density"].iloc[0] > 0
        assert result["help_seeking_density"].iloc[0] > 0
        assert result["distress_density"].iloc[0] > 0


def test_distress_scorer_no_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "hopelessness.txt").write_text("hopeless\n")
        (Path(tmpdir) / "help_seeking.txt").write_text("need advice\n")
        (Path(tmpdir) / "distress.txt").write_text("depressed\n")

        scorer = DistressScorer(tmpdir)

        df = pd.DataFrame({
            "texts": [["The weather is nice today and the sun is shining brightly"]],
        })

        result = scorer.extract_distress_features(df)
        assert result["hopelessness_density"].iloc[0] == 0
