"""Cross-validation splitters that respect time order and group membership.

Both classes follow the scikit-learn splitter protocol, so they can be passed
as ``cv=`` to ``cross_validate``, ``GridSearchCV`` and friends.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

IndexPair = tuple[NDArray[np.intp], NDArray[np.intp]]


def _n_samples(X: ArrayLike) -> int:
    shape = getattr(X, "shape", None)
    if shape is not None:
        return int(shape[0])
    return len(X)  # type: ignore[arg-type]


class TimeSeriesSplit:
    """Forward-chaining splits of time-ordered rows with a gap before each test block.

    The rows are cut into ``n_splits + 1`` contiguous blocks of near-equal size.
    Split ``k`` tests on block ``k + 1`` and trains on everything before it,
    minus the ``gap`` rows immediately preceding the test block. The gap is what
    keeps a feature built from a trailing window (a 7-day mean, a lag) from
    carrying test-period information into training.

    Parameters
    ----------
    n_splits:
        Number of train/test splits, at least 1.
    gap:
        Rows dropped between the end of training and the start of testing.
    max_train_size:
        If given, training uses only the most recent ``max_train_size`` rows
        (a rolling window) instead of the full history (an expanding window).
    """

    def __init__(self, n_splits: int = 5, *, gap: int = 0, max_train_size: int | None = None):
        if n_splits < 1:
            raise ValueError(f"n_splits must be at least 1, got {n_splits}")
        if gap < 0:
            raise ValueError(f"gap must be non-negative, got {gap}")
        if max_train_size is not None and max_train_size < 1:
            raise ValueError(f"max_train_size must be positive, got {max_train_size}")
        self.n_splits = n_splits
        self.gap = gap
        self.max_train_size = max_train_size

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def split(
        self, X: ArrayLike, y: Any = None, groups: Any = None, *, times: ArrayLike | None = None
    ) -> Iterator[IndexPair]:
        """Yield ``(train_idx, test_idx)`` pairs.

        If ``times`` is given it must be non-decreasing; an unsorted time column
        is the usual way a random split sneaks back in, so it raises rather than
        silently sorting.
        """
        n = _n_samples(X)
        if times is not None:
            t = np.asarray(times)
            if t.shape[0] != n:
                raise ValueError(f"times has {t.shape[0]} rows but X has {n}")
            if n > 1 and np.any(t[1:] < t[:-1]):
                raise ValueError("times must be sorted in non-decreasing order")
        n_blocks = self.n_splits + 1
        if n < n_blocks:
            raise ValueError(f"cannot make {self.n_splits} splits from {n} rows")
        bounds = np.linspace(0, n, n_blocks + 1).astype(np.intp)
        for k in range(1, n_blocks):
            test_start, test_stop = int(bounds[k]), int(bounds[k + 1])
            train_stop = test_start - self.gap
            if train_stop <= 0:
                raise ValueError(
                    f"gap={self.gap} leaves no training rows before test block starting at "
                    f"{test_start}; use fewer splits or a smaller gap"
                )
            train_start = 0
            if self.max_train_size is not None:
                train_start = max(0, train_stop - self.max_train_size)
            yield (
                np.arange(train_start, train_stop, dtype=np.intp),
                np.arange(test_start, test_stop, dtype=np.intp),
            )


class GroupKFold:
    """K-fold over groups: every row of a group lands in the same test fold.

    Groups are assigned to folds greedily, largest first, into whichever fold
    currently has the fewest rows. This is the same balancing rule
    ``sklearn.model_selection.GroupKFold`` uses; the difference is the optional
    ``shuffle``, which randomises the order of equal-sized groups under a seed
    so repeated runs with different seeds give genuinely different folds.

    Parameters
    ----------
    n_splits:
        Number of folds, at least 2 and at most the number of distinct groups.
    shuffle:
        Break ties between equal-sized groups at random.
    random_state:
        Seed for the tie-breaking; ignored unless ``shuffle`` is true.
    """

    def __init__(
        self, n_splits: int = 5, *, shuffle: bool = False, random_state: int | None = None
    ):
        if n_splits < 2:
            raise ValueError(f"n_splits must be at least 2, got {n_splits}")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        return self.n_splits

    def fold_of_row(self, groups: ArrayLike) -> NDArray[np.intp]:
        """Return the test-fold index assigned to each row."""
        g = np.asarray(groups)
        uniq, inverse, counts = np.unique(g, return_inverse=True, return_counts=True)
        if uniq.shape[0] < self.n_splits:
            raise ValueError(
                f"n_splits={self.n_splits} exceeds the number of groups ({uniq.shape[0]})"
            )
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            tiebreak = rng.permutation(uniq.shape[0])
        else:
            tiebreak = np.arange(uniq.shape[0])
        # Sort by size descending, then by tiebreak ascending.
        order = np.lexsort((tiebreak, -counts))
        fold_sizes = np.zeros(self.n_splits, dtype=np.int64)
        group_fold = np.empty(uniq.shape[0], dtype=np.intp)
        for gi in order:
            f = int(np.argmin(fold_sizes))
            group_fold[gi] = f
            fold_sizes[f] += counts[gi]
        return group_fold[inverse.reshape(-1)]

    def split(
        self, X: ArrayLike, y: Any = None, groups: ArrayLike | None = None
    ) -> Iterator[IndexPair]:
        if groups is None:
            raise ValueError("GroupKFold requires the groups argument")
        n = _n_samples(X)
        fold = self.fold_of_row(groups)
        if fold.shape[0] != n:
            raise ValueError(f"groups has {fold.shape[0]} rows but X has {n}")
        idx = np.arange(n, dtype=np.intp)
        for f in range(self.n_splits):
            mask = fold == f
            yield idx[~mask], idx[mask]
