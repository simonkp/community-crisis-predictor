import re
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
        self.suicidality_terms = self._load_lexicon("suicidality.txt")
        self.isolation_terms = self._load_lexicon("isolation.txt")
        self.economic_stress_terms = self._load_lexicon("economic_stress.txt")
        self.domestic_stress_terms = self._load_lexicon("domestic_stress.txt")
        # Precompile word-boundary patterns to avoid substring false positives
        # (e.g. "panic" matching inside "panicking" or unrelated compound tokens).
        self._hopelessness_re = self._compile_patterns(self.hopelessness_terms)
        self._help_seeking_re = self._compile_patterns(self.help_seeking_terms)
        self._distress_re = self._compile_patterns(self.distress_terms)
        self._suicidality_re = self._compile_patterns(self.suicidality_terms)
        self._isolation_re = self._compile_patterns(self.isolation_terms)
        self._economic_stress_re = self._compile_patterns(self.economic_stress_terms)
        self._domestic_stress_re = self._compile_patterns(self.domestic_stress_terms)

    def _load_lexicon(self, filename: str) -> list[str]:
        path = self.lexicon_dir / filename
        if not path.exists():
            return []
        with open(path) as f:
            return [line.strip().lower() for line in f if line.strip()]

    @staticmethod
    def _compile_patterns(terms: list[str]) -> list[re.Pattern]:
        return [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms]

    def _count_matches(self, text: str, patterns: list[re.Pattern]) -> int:
        text_lower = text.lower()
        return sum(len(p.findall(text_lower)) for p in patterns)

    def _density(self, text: str, patterns: list[re.Pattern]) -> float:
        words = text.split()
        if not words:
            return 0.0
        matches = self._count_matches(text, patterns)
        return matches / len(words)

    def extract_distress_features(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in iter_weeks(weekly_df, desc="  Distress"):
            texts = row["texts"]
            if not texts:
                rows.append({
                    "hopelessness_density": 0.0,
                    "help_seeking_density": 0.0,
                    "distress_density": 0.0,
                    "suicidality_density": 0.0,
                    "isolation_density": 0.0,
                    "economic_stress_density": 0.0,
                    "domestic_stress_density": 0.0,
                })
                continue

            hope_densities = [self._density(t, self._hopelessness_re) for t in texts]
            help_densities = [self._density(t, self._help_seeking_re) for t in texts]
            dist_densities = [self._density(t, self._distress_re) for t in texts]
            suic_totals = [self._count_matches(t, self._suicidality_re) for t in texts]
            isol_totals = [self._count_matches(t, self._isolation_re) for t in texts]
            econ_totals = [self._count_matches(t, self._economic_stress_re) for t in texts]
            dome_totals = [self._count_matches(t, self._domestic_stress_re) for t in texts]

            n_posts = len(texts)
            rows.append({
                "hopelessness_density": np.mean(hope_densities),
                "help_seeking_density": np.mean(help_densities),
                "distress_density": np.mean(dist_densities),
                # Normalise by post count so these are rates (matches/post), not raw counts.
                # Raw sums inflate with volume — a busy week always looks more distressed.
                "suicidality_density": sum(suic_totals) / n_posts,
                "isolation_density": sum(isol_totals) / n_posts,
                "economic_stress_density": sum(econ_totals) / n_posts,
                "domestic_stress_density": sum(dome_totals) / n_posts,
            })

        return pd.DataFrame(rows, index=weekly_df.index)
