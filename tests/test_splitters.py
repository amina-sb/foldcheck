from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold as SkGroupKFold
from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection import cross_val_score

from foldcheck.splitters import GroupKFold, TimeSeriesSplit


@settings(max_examples=200, deadline=None)
@given(
    n=st.integers(10, 400),
    n_splits=st.integers(1, 8),
    gap=st.integers(0, 5),
    max_train=st.one_of(st.none(), st.integers(1, 50)),
)
def test_time_split_order_gap_and_disjointness(
    n: int, n_splits: int, gap: int, max_train: int | None
) -> None:
    cv = TimeSeriesSplit(n_splits, gap=gap, max_train_size=max_train)
    X = np.zeros((n, 1))
    try:
        splits = list(cv.split(X))
    except ValueError:
        # Only legitimate when the first test block starts within `gap` rows of 0.
        assert n // (n_splits + 1) <= gap + 1
        return
    assert len(splits) == n_splits
    tested = np.concatenate([te for _, te in splits])
    assert np.array_equal(tested, np.arange(tested[0], n))  # contiguous, ends at n
    for tr, te in splits:
        assert tr.size > 0 and te.size > 0
        assert te.min() - tr.max() >= gap + 1
        if max_train is not None:
            assert tr.size <= max_train


def test_time_split_matches_sklearn_test_blocks_without_gap() -> None:
    # sklearn sizes test blocks as n // (n_splits + 1); ours uses linspace, so they
    # agree exactly when n is divisible by n_splits + 1.
    n, k = 120, 5
    ours = list(TimeSeriesSplit(k).split(np.zeros(n)))
    ref = list(SkTimeSeriesSplit(k).split(np.zeros(n)))
    for (a_tr, a_te), (b_tr, b_te) in zip(ours, ref, strict=True):
        assert np.array_equal(a_tr, b_tr)
        assert np.array_equal(a_te, b_te)


def test_unsorted_times_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        list(TimeSeriesSplit(2).split(np.zeros(6), times=[1, 2, 3, 5, 4, 6]))


def test_time_split_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        TimeSeriesSplit(0)
    with pytest.raises(ValueError):
        TimeSeriesSplit(2, gap=-1)
    with pytest.raises(ValueError, match="cannot make"):
        list(TimeSeriesSplit(5).split(np.zeros(3)))


group_layouts = st.lists(st.integers(0, 30), min_size=12, max_size=300)


@settings(max_examples=200, deadline=None)
@given(groups=group_layouts, k=st.integers(2, 6), seed=st.integers(0, 2**16), shuffle=st.booleans())
def test_groups_never_straddle_and_rows_covered_once(
    groups: list[int], k: int, seed: int, shuffle: bool
) -> None:
    g = np.asarray(groups)
    if np.unique(g).size < k:
        return
    cv = GroupKFold(k, shuffle=shuffle, random_state=seed)
    seen = np.zeros(g.size, dtype=int)
    for tr, te in cv.split(np.zeros(g.size), groups=g):
        assert set(g[tr]).isdisjoint(set(g[te]))
        seen[te] += 1
    assert np.all(seen == 1)


@settings(max_examples=100, deadline=None)
@given(groups=group_layouts, k=st.integers(2, 6))
def test_fold_sizes_match_sklearn_balance(groups: list[int], k: int) -> None:
    g = np.asarray(groups)
    if np.unique(g).size < k:
        return
    ours = sorted(te.size for _, te in GroupKFold(k).split(np.zeros(g.size), groups=g))
    ref = sorted(te.size for _, te in SkGroupKFold(k).split(np.zeros(g.size), groups=g))
    # Same greedy rule => same multiset of fold sizes, whatever the tie-breaking.
    assert ours == ref


def test_shuffle_changes_assignment_under_different_seeds() -> None:
    g = np.repeat(np.arange(40), 5)  # all groups equal size: assignment is pure tie-breaking
    a = GroupKFold(4, shuffle=True, random_state=0).fold_of_row(g)
    b = GroupKFold(4, shuffle=True, random_state=1).fold_of_row(g)
    c = GroupKFold(4, shuffle=True, random_state=0).fold_of_row(g)
    assert not np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_group_split_requires_groups_and_enough_of_them() -> None:
    with pytest.raises(ValueError, match="requires"):
        list(GroupKFold(2).split(np.zeros(4)))
    with pytest.raises(ValueError, match="exceeds"):
        list(GroupKFold(3).split(np.zeros(4), groups=[0, 0, 1, 1]))


def test_splitters_work_inside_sklearn_cross_validation() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] > 0).astype(int)
    groups = np.repeat(np.arange(20), 10)
    s1 = cross_val_score(LogisticRegression(), X, y, cv=GroupKFold(5), groups=groups)
    s2 = cross_val_score(LogisticRegression(), X, y, cv=TimeSeriesSplit(4, gap=3))
    assert s1.shape == (5,) and s2.shape == (4,)
    assert s1.min() > 0.8 and s2.min() > 0.8
