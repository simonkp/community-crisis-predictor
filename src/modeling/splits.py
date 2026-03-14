from typing import Iterator

import numpy as np


class WalkForwardSplitter:
    def __init__(self, min_train_weeks: int = 26, gap_weeks: int = 1):
        self.min_train_weeks = min_train_weeks
        self.gap_weeks = gap_weeks

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for test_idx in range(self.min_train_weeks + self.gap_weeks, n_samples):
            train_end = test_idx - self.gap_weeks
            train_indices = np.arange(0, train_end)
            test_indices = np.array([test_idx])
            yield train_indices, test_indices

    def n_splits(self, n_samples: int) -> int:
        return max(0, n_samples - self.min_train_weeks - self.gap_weeks)
