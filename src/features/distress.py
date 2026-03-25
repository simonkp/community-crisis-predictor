from pathlib import Path

import numpy as np
import pandas as pd

from src.features.progress_util import iter_weeks


class DistressScorer:
    def __init__(self, lexicon_dir: str = "config/lexicons"):
        self.lexicon_dir = Path(lexicon_dir)
        self.hopelessness_terms = self._load_lexicon("hopelessness.txt")
        self.help_seeking_terms = self._load_lexicon("help_seeking.txt")
        self.distress_terms = self._load_lexicon("distress.txt")

    def _load_lexicon(self, filename: str) -> list[str]:
        path = self.lexicon_dir / filename
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip().lower() for line in f if line.strip()]

    def _count_matches(self, text: str, terms: list[str]) -> int:
        text_lower = text.lower()
        count = 0
        for term in terms:
            if term in text_lower:
                count += text_lower.count(term)
        return count

    def _density(self, text: str, terms: list[str]) -> float:
        words = text.split()
        if not words:
            return 0.0
        matches = self._count_matches(text, terms)
        return matches / len(words)

    def extract_distress_features(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in iter_weeks(weekly_df, desc="  Distress"):
            texts = row["texts"]
            if not texts:
                rows.append({
                    "hopelessness_density": 0,
                    "help_seeking_density": 0,
                    "distress_density": 0,
                })
                continue

            hope_densities = [self._density(t, self.hopelessness_terms) for t in texts]
            help_densities = [self._density(t, self.help_seeking_terms) for t in texts]
            dist_densities = [self._density(t, self.distress_terms) for t in texts]

            rows.append({
                "hopelessness_density": np.mean(hope_densities),
                "help_seeking_density": np.mean(help_densities),
                "distress_density": np.mean(dist_densities),
            })

        return pd.DataFrame(rows, index=weekly_df.index)
