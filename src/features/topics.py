import warnings
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from src.features.progress_util import iter_weeks, tqdm_index


class TopicFeatureExtractor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_topics: int = 15,
                 min_topic_size: int = 10, max_posts_per_week: int = 200):
        self.model_name = model_name
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.max_posts_per_week = max_posts_per_week
        self._topic_model = None
        self._n_fitted_topics = None

    def _get_model(self):
        if self._topic_model is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from bertopic import BERTopic
                from sentence_transformers import SentenceTransformer

                embedding_model = SentenceTransformer(self.model_name)
                self._topic_model = BERTopic(
                    embedding_model=embedding_model,
                    nr_topics=self.n_topics,
                    min_topic_size=self.min_topic_size,
                    verbose=False,
                )
        return self._topic_model

    def fit_and_extract(self, weekly_df: pd.DataFrame) -> pd.DataFrame:
        # Collect all texts with week labels
        all_texts = []
        week_labels = []

        for idx, row in iter_weeks(weekly_df, desc="  Topic: collect texts"):
            texts = row.get("texts", [])
            # Subsample if too many posts
            if len(texts) > self.max_posts_per_week:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(texts), self.max_posts_per_week, replace=False)
                texts = [texts[i] for i in indices]
            for t in texts:
                if t and len(t) > 10:
                    all_texts.append(t)
                    week_labels.append(idx)

        if len(all_texts) < self.min_topic_size * 2:
            # Not enough data for topic modeling
            return pd.DataFrame({
                "dominant_topic": [0] * len(weekly_df),
                "topic_entropy": [0.0] * len(weekly_df),
                "topic_shift_jsd": [0.0] * len(weekly_df),
                "topic_shift_jsd_4w": [0.0] * len(weekly_df),
            }, index=weekly_df.index)

        model = self._get_model()
        print("  Topic model: embedding + BERTopic fit_transform (slow; CPU/GPU bound)...")
        topics, probs = model.fit_transform(all_texts)
        self._n_fitted_topics = len(set(topics)) - (1 if -1 in topics else 0)

        # Build per-week topic distributions
        n_topics = max(topics) + 1 if topics else 1
        week_topic_dists = {}

        for topic_id, week_idx in zip(topics, week_labels):
            if week_idx not in week_topic_dists:
                week_topic_dists[week_idx] = np.zeros(max(n_topics, 1))
            if topic_id >= 0:
                week_topic_dists[week_idx][topic_id] += 1

        rows = []
        prev_dist = None
        dist_history: deque = deque(maxlen=4)

        for idx in tqdm_index(
            weekly_df.index,
            total=len(weekly_df),
            desc="  Topic: per-week stats",
        ):
            dist = week_topic_dists.get(idx, np.zeros(max(n_topics, 1)))
            total = dist.sum()

            if total > 0:
                dist_norm = dist / total
                dominant = int(np.argmax(dist_norm))
                ent = float(-np.sum(dist_norm[dist_norm > 0] * np.log2(dist_norm[dist_norm > 0])))
            else:
                dist_norm = dist
                dominant = 0
                ent = 0.0

            if prev_dist is not None and total > 0 and prev_dist.sum() > 0:
                jsd = float(jensenshannon(prev_dist, dist_norm))
                if np.isnan(jsd):
                    jsd = 0.0
            else:
                jsd = 0.0

            if len(dist_history) == 4 and total > 0 and dist_history[0].sum() > 0:
                jsd_4w = float(jensenshannon(dist_history[0], dist_norm))
                if np.isnan(jsd_4w):
                    jsd_4w = 0.0
            else:
                jsd_4w = 0.0

            rows.append({
                "dominant_topic": dominant,
                "topic_entropy": ent,
                "topic_shift_jsd": jsd,
                "topic_shift_jsd_4w": jsd_4w,
            })
            prev_dist = dist_norm if total > 0 else prev_dist
            if total > 0:
                dist_history.append(dist_norm)

        return pd.DataFrame(rows, index=weekly_df.index)
