from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from foldcheck import GroupKFold
from foldcheck.inference import bootstrap_ci, nested_cv


def _accuracy(a: NDArray[Any], b: NDArray[Any]) -> float:
    return float(np.mean(a == b))


def test_bootstrap_matches_scipy_percentile_interval() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    p = np.where(rng.random(400) < 0.8, y, 1 - y)
    ours = bootstrap_ci(y, p, _accuracy, n_resamples=4000, random_state=1)
    ref = stats.bootstrap(
        (y, p),
        lambda a, b: np.mean(a == b),
        paired=True,
        vectorized=False,
        n_resamples=4000,
        method="percentile",
        random_state=2,
    ).confidence_interval
    # Different RNG streams: endpoints agree to Monte Carlo error, not exactly.
    assert ours.low == pytest.approx(ref.low, abs=0.006)
    assert ours.high == pytest.approx(ref.high, abs=0.006)
    assert ours.estimate == pytest.approx(np.mean(y == p))


def test_bootstrap_interval_reaches_nominal_coverage() -> None:
    # Accuracy of a classifier that is right with probability 0.75, n = 200.
    true_acc, n, reps = 0.75, 200, 300
    rng = np.random.default_rng(123)
    hits = 0
    for r in range(reps):
        y = np.zeros(n, dtype=int)
        p = (rng.random(n) >= true_acc).astype(int)
        ci = bootstrap_ci(y, p, _accuracy, n_resamples=400, random_state=r)
        hits += ci.low <= true_acc <= ci.high
    coverage = hits / reps
    # Binomial sd of the coverage estimate at 0.95 with 300 reps is 0.0126;
    # allow three of them below and above.
    assert 0.912 <= coverage <= 0.988, coverage


def test_bootstrap_validates_arguments() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci([0, 1], [0], _accuracy)
    with pytest.raises(ValueError, match="level"):
        bootstrap_ci([0, 1], [0, 1], _accuracy, level=1.0)


def test_nested_cv_on_noise_is_not_optimistic_but_flat_search_is() -> None:
    gaps_flat, gaps_nested = [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(120, 30))
        y = rng.integers(0, 2, 120)
        grid: dict[str, list[Any]] = {
            "n_neighbors": [1, 3, 5, 9, 15, 25],
            "weights": ["uniform", "distance"],
        }
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        flat = GridSearchCV(KNeighborsClassifier(), grid, cv=cv).fit(X, y).best_score_
        res = nested_cv(
            KNeighborsClassifier(),
            grid,
            X,
            y,
            outer=cv,
            inner=StratifiedKFold(4, shuffle=True, random_state=seed),
        )
        gaps_flat.append(flat - 0.5)
        gaps_nested.append(res.mean - 0.5)
        assert len(res.best_params) == 5
        assert res.test_mask.all()
    # The naive expectation is that tuning on noise is harmless. It is not: the
    # best-of-12 search score beats the nested estimate on every seed here.
    # Individual flat scores can still land below 0.5, so the claim is paired,
    # not "above chance".
    assert all(f > n for f, n in zip(gaps_flat, gaps_nested, strict=True))
    assert abs(float(np.mean(gaps_nested))) < 0.06
    assert np.mean(gaps_flat) - np.mean(gaps_nested) > 0.02


def test_nested_cv_passes_groups_to_inner_search() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(24), 5)
    X = rng.normal(size=(120, 3)) + groups[:, None] * 0.01
    y = (X[:, 0] > 0).astype(int)
    res = nested_cv(
        LogisticRegression(),
        {"C": [0.1, 1.0]},
        X,
        y,
        outer=GroupKFold(4),
        inner=GroupKFold(3),
        groups=groups,
    )
    assert res.outer_scores.shape == (4,)
    assert res.mean > 0.85


def test_nested_cv_mask_with_partial_coverage() -> None:
    from foldcheck import TimeSeriesSplit

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] > 0).astype(int)
    res = nested_cv(
        LogisticRegression(), {"C": [1.0]}, X, y, outer=TimeSeriesSplit(4), inner=KFold(3)
    )
    assert not res.test_mask[:20].any() and res.test_mask[20:].all()


def _mean_pred(_a: NDArray[Any], b: NDArray[Any]) -> float:
    return float(np.mean(b))


def test_bca_matches_scipy_on_skewed_data() -> None:
    rng = np.random.default_rng(5)
    x = rng.lognormal(0.0, 1.0, 60)
    zeros = np.zeros_like(x)
    ours = bootstrap_ci(zeros, x, _mean_pred, n_resamples=20000, method="bca", random_state=0)
    ref = stats.bootstrap(
        (x,), np.mean, n_resamples=20000, method="BCa", random_state=1
    ).confidence_interval
    # Monte Carlo error at 20,000 resamples; the interval is about 0.9 wide.
    assert ours.low == pytest.approx(ref.low, abs=0.03)
    assert ours.high == pytest.approx(ref.high, abs=0.05)
    pct = bootstrap_ci(zeros, x, _mean_pred, n_resamples=20000, random_state=0)
    # On right-skewed data BCa moves both endpoints up relative to percentile.
    assert ours.low > pct.low and ours.high > pct.high


def test_bca_covers_at_least_as_often_as_percentile_on_skewed_mean() -> None:
    # Mean of Exponential(1) at n = 20: truth 1.0, skewed sampling distribution.
    reps, n = 400, 20
    rng = np.random.default_rng(2024)
    hits = {"percentile": 0, "bca": 0}
    for r in range(reps):
        x = rng.exponential(1.0, n)
        z = np.zeros(n)
        for m in hits:
            ci = bootstrap_ci(z, x, _mean_pred, n_resamples=500, method=m, random_state=r)  # type: ignore[arg-type]
            hits[m] += ci.low <= 1.0 <= ci.high
    pct, bca = hits["percentile"] / reps, hits["bca"] / reps
    # Measured: percentile 0.905, BCa 0.925 (nominal 0.95). Neither reaches
    # nominal at n = 20; BCa is the better of the two on the same replicates.
    assert bca >= pct
    assert (pct, bca) == (0.905, 0.925)


def test_unknown_bootstrap_method_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        bootstrap_ci([0, 1], [0, 1], _accuracy, method="basic")  # type: ignore[arg-type]


def test_corrected_t_interval_matches_hand_calculation() -> None:
    from foldcheck.inference import corrected_t_interval

    scores = np.array([0.80, 0.82, 0.78, 0.85, 0.75, 0.81, 0.79, 0.83, 0.77, 0.80])
    ci = corrected_t_interval(scores, n_train=80, n_test=20)
    mean = 0.80
    var = sum((s - mean) ** 2 for s in scores) / 9
    se = ((1 / 10 + 20 / 80) * var) ** 0.5
    tcrit = stats.t.ppf(0.975, 9)
    assert ci.estimate == pytest.approx(mean)
    assert ci.low == pytest.approx(mean - tcrit * se)
    assert ci.high == pytest.approx(mean + tcrit * se)


def test_corrected_interval_widens_by_stated_factor() -> None:
    from foldcheck.inference import corrected_t_interval

    rng = np.random.default_rng(0)
    scores = rng.normal(0.8, 0.02, size=(10, 5))
    corrected = corrected_t_interval(scores, n_train=400, n_test=100)
    naive = corrected_t_interval(scores, n_train=10**12, n_test=1)
    ratio = (corrected.high - corrected.low) / (naive.high - naive.low)
    assert ratio == pytest.approx(((1 / 50 + 1 / 4) / (1 / 50)) ** 0.5, rel=1e-6)
    assert ratio == pytest.approx(3.674, abs=1e-3)


def test_corrected_interval_validates_input() -> None:
    from foldcheck.inference import corrected_t_interval

    with pytest.raises(ValueError, match="two scores"):
        corrected_t_interval([0.5], 10, 2)
    with pytest.raises(ValueError, match="positive"):
        corrected_t_interval([0.5, 0.6], 0, 2)


def test_repeated_cv_shapes_sizes_and_interval() -> None:
    from foldcheck.inference import repeated_cv

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
    res = repeated_cv(LogisticRegression(), X, y, n_splits=5, n_repeats=4, stratify=True)
    assert res.scores.shape == (4, 5)
    assert (res.n_train, res.n_test) == (160, 40)
    ci = res.interval()
    assert ci.low < ci.estimate < ci.high
    assert 0.75 < ci.estimate < 0.95
    # Different repeats use different shuffles.
    assert not np.allclose(res.scores[0], res.scores[1])
