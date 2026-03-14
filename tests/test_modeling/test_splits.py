import numpy as np
from src.modeling.splits import WalkForwardSplitter


def test_walk_forward_no_overlap():
    splitter = WalkForwardSplitter(min_train_weeks=5, gap_weeks=1)
    for train_idx, test_idx in splitter.split(20):
        # Train must come before test
        assert train_idx[-1] < test_idx[0]
        # Gap must be respected
        assert test_idx[0] - train_idx[-1] > 1


def test_walk_forward_gap():
    splitter = WalkForwardSplitter(min_train_weeks=5, gap_weeks=2)
    for train_idx, test_idx in splitter.split(20):
        assert test_idx[0] - train_idx[-1] >= 2


def test_walk_forward_min_train():
    splitter = WalkForwardSplitter(min_train_weeks=10, gap_weeks=1)
    for train_idx, test_idx in splitter.split(20):
        assert len(train_idx) >= 10


def test_walk_forward_count():
    splitter = WalkForwardSplitter(min_train_weeks=5, gap_weeks=1)
    n = splitter.n_splits(20)
    splits = list(splitter.split(20))
    assert len(splits) == n
