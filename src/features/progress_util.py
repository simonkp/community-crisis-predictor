"""Optional tqdm progress (disabled for non-TTY / tests)."""

import sys

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # pragma: no cover
    def _tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


def _disable_progress() -> bool:
    return not sys.stdout.isatty()


def iter_weeks(weekly_df, desc: str, unit: str = "wk"):
    """Iterate weekly_df rows with a progress bar when appropriate."""
    n = len(weekly_df)
    yield from _tqdm(
        weekly_df.iterrows(),
        total=n,
        desc=desc,
        unit=unit,
        leave=True,
        disable=_disable_progress(),
        file=sys.stdout,
        mininterval=0.5,
    )


def iter_groupby_subreddit(df, col: str, desc: str):
    """Iterate (name, group) from groupby with a progress bar."""
    groups = list(df.groupby(col, sort=False))
    yield from _tqdm(
        groups,
        total=len(groups),
        desc=desc,
        unit="sub",
        leave=True,
        disable=_disable_progress(),
        file=sys.stdout,
    )


def tqdm_index(index, *, total: int, desc: str, unit: str = "wk"):
    """Progress over a pandas Index or list with known length."""
    return _tqdm(
        index,
        total=total,
        desc=desc,
        unit=unit,
        leave=True,
        disable=_disable_progress(),
        file=sys.stdout,
        mininterval=0.5,
    )
