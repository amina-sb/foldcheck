from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from foldcheck import GroupKFold, TimeSeriesSplit
from foldcheck.audit import (
    LeakageError,
    audit_cv,
    detect_global_scaling,
    scan_target_leakage,
)


class FrozenScaler(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """A scaler fitted once on all rows whose ``fit`` is a no-op.

    This is what "fit the scaler, then cross-validate the pipeline" amounts to
    when someone protects the prefit scaler from being refit.
    """

    def __init__(self, X_all: NDArray[np.float64] | None = None) -> None:
        self.X_all = X_all

    def fit(self, X: Any, y: Any = None) -> FrozenScaler:
        inner = StandardScaler().fit(self.X_all)
        self.mean_ = inner.mean_
        self.scale_ = inner.scale_
        self.n_samples_seen_ = inner.n_samples_seen_
        return self

    def transform(self, X: Any) -> NDArray[np.float64]:
        out: NDArray[np.float64] = (np.asarray(X) - self.mean_) / self.scale_
        return out


def _data(seed: int, n: int = 300, p: int = 5) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=3.0, scale=2.0, size=(n, p))
    y = (X[:, 0] + rng.normal(size=n) > 3.0).astype(int)
    return X, y


@pytest.mark.parametrize("seed", range(5))
def test_clean_pipeline_has_no_findings(seed: int) -> None:
    X, y = _data(seed)
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression())
    report = audit_cv(pipe, X, y, KFold(5, shuffle=True, random_state=seed))
    assert report.ok, str(report)
    report.raise_if_leaky()


def test_fold_fit_scaler_mean_differs_from_full_fit() -> None:
    X, y = _data(0)
    full_mean = StandardScaler().fit(X).mean_
    train, _ = next(KFold(5, shuffle=True, random_state=0).split(X))
    fold_mean = (
        make_pipeline(StandardScaler(), LogisticRegression()).fit(X[train], y[train])[0].mean_
    )
    assert not np.allclose(fold_mean, full_mean)


def test_frozen_scaler_caught_twice() -> None:
    X, y = _data(1)
    pipe = make_pipeline(FrozenScaler(X), LogisticRegression())
    report = audit_cv(pipe, X, y, KFold(5))
    kinds = sorted({f.kind for f in report.findings})
    assert kinds == ["not_refit", "sample_count"]
    assert all(f.where == "frozenscaler" for f in report.findings)
    with pytest.raises(LeakageError, match="frozenscaler"):
        report.raise_if_leaky()


def test_audit_descends_into_column_transformer() -> None:
    X, y = _data(2)
    ct = ColumnTransformer([("frozen", FrozenScaler(X[:, :2]), [0, 1])], remainder="passthrough")
    report = audit_cv(make_pipeline(ct, LogisticRegression()), X, y, KFold(4))
    assert {f.where for f in report.findings} == {"columntransformer.frozen"}


def test_audit_accepts_group_and_time_splitters() -> None:
    X, y = _data(3)
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    groups = np.repeat(np.arange(30), 10)
    assert audit_cv(pipe, X, y, GroupKFold(5), groups=groups).ok
    assert audit_cv(pipe, X, y, TimeSeriesSplit(4, gap=5)).ok


def test_global_standardisation_and_minmax_detected() -> None:
    X, _ = _data(4)
    assert detect_global_scaling(X).ok
    std = detect_global_scaling(StandardScaler().fit_transform(X))
    assert len(std.findings) == X.shape[1]
    mm = detect_global_scaling(MinMaxScaler().fit_transform(X), column_names=list("abcde"))
    assert [f.where for f in mm.findings] == list("abcde")


def test_scaling_on_training_fold_only_is_not_flagged() -> None:
    X, _ = _data(5)
    train = np.arange(200)
    Xs = StandardScaler().fit(X[train]).transform(X)
    assert detect_global_scaling(Xs).ok


def test_binary_indicators_not_mistaken_for_minmax() -> None:
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(100, 3)).astype(float)
    assert detect_global_scaling(X).ok


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(0, 2**16), scaled=st.booleans())
def test_global_scaling_verdict_invariant_to_row_order(seed: int, scaled: bool) -> None:
    X, _ = _data(seed, n=80, p=3)
    if scaled:
        X = StandardScaler().fit_transform(X)
    perm = np.random.default_rng(seed + 1).permutation(X.shape[0])
    a = detect_global_scaling(X)
    b = detect_global_scaling(X[perm])
    assert [f.where for f in a.findings] == [f.where for f in b.findings]
    assert a.ok is not scaled


def test_target_derived_column_flagged_and_ranked_first() -> None:
    X, y = _data(6)
    rng = np.random.default_rng(6)
    leak = y + rng.normal(scale=0.05, size=y.size)
    Xl = np.column_stack([X, leak])
    report, scores = scan_target_leakage(Xl, y, KFold(5, shuffle=True, random_state=0))
    assert [f.where for f in report.findings] == ["x5"]
    assert scores[5] > 0.99
    # The genuinely predictive feature is useful but not suspicious.
    assert 0.7 < scores[0] < 0.98


@pytest.mark.parametrize("seed", range(5))
def test_no_target_leak_flags_on_noise(seed: int) -> None:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(250, 8))
    y = rng.integers(0, 2, size=250)
    report, scores = scan_target_leakage(X, y, KFold(5, shuffle=True, random_state=seed))
    assert report.ok
    assert np.all(np.abs(scores - 0.5) < 0.15)


def test_target_leak_regression_uses_r2() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(200, 3))
    y = 2.0 * X[:, 0] + rng.normal(size=200)
    Xl = np.column_stack([X, y * 1.0001])
    report, scores = scan_target_leakage(Xl, y, KFold(5, shuffle=True, random_state=0))
    assert [f.where for f in report.findings] == ["x3"]
    assert scores[1] < 0.1


def test_minmax_detected_despite_rounding_in_scaler_output() -> None:
    # MinMaxScaler's max is not exactly 1.0 on a sizeable share of columns; an
    # exact-equality check missed 220 of 1,000 columns across these seeds.
    missed = exact_misses = 0
    for seed in range(200):
        X, _ = _data(seed, n=300, p=5)
        M = MinMaxScaler().fit_transform(X)
        missed += X.shape[1] - len(detect_global_scaling(M).findings)
        exact_misses += int(np.sum((M.min(axis=0) != 0.0) | (M.max(axis=0) != 1.0)))
    assert missed == 0
    assert exact_misses == 220
