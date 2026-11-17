"""Uncertainty for cross-validated metrics, and nested cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as t_dist
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

Metric = Callable[[NDArray[Any], NDArray[Any]], float]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def _bca_quantiles(
    stats: NDArray[np.float64],
    theta_hat: float,
    jack: NDArray[np.float64],
    alpha: float,
) -> tuple[float, float]:
    """Bias-corrected and accelerated quantile levels (Efron 1987, eq. 2.3).

    ``z0`` measures the median bias of the bootstrap distribution, counting ties
    as half (as ``scipy.stats.bootstrap`` does); ``a`` is the acceleration,
    estimated from the jackknife values ``jack``.
    """
    nd = NormalDist()
    frac = (np.sum(stats < theta_hat) + 0.5 * np.sum(stats == theta_hat)) / stats.size
    frac = float(np.clip(frac, 1.0 / (stats.size + 1), stats.size / (stats.size + 1)))
    z0 = nd.inv_cdf(frac)
    d = jack.mean() - jack
    denom = 6.0 * float(np.sum(d**2)) ** 1.5
    a = float(np.sum(d**3)) / denom if denom > 0 else 0.0
    out = []
    for q in (alpha, 1.0 - alpha):
        zq = nd.inv_cdf(q)
        out.append(nd.cdf(z0 + (z0 + zq) / (1.0 - a * (z0 + zq))))
    return out[0], out[1]


def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: Metric,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    method: Literal["percentile", "bca"] = "percentile",
    random_state: int | None = None,
) -> Interval:
    """Bootstrap interval for ``metric(y_true, y_pred)``.

    Rows are resampled with replacement as pairs. The point estimate is the
    metric on the original data, not the mean of the resamples.

    ``method="percentile"`` takes the plain quantiles of the bootstrap
    distribution. ``method="bca"`` shifts those quantiles to correct for median
    bias and for a standard error that changes with the parameter, which costs
    ``n`` extra jackknife evaluations of ``metric``.

    BCa helps with skewed statistics but does not cure small samples. For the
    mean of an Exponential(1) at n = 20, simulated 95% coverage was 0.905 for
    the percentile interval and 0.925 for BCa over 400 replicates (pinned in
    ``tests/test_inference.py``).
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if method not in ("percentile", "bca"):
        raise ValueError(f"unknown method {method!r}")
    n = yt.shape[0]
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric(yt[idx], yp[idx])
    theta_hat = float(metric(yt, yp))
    alpha = (1.0 - level) / 2.0
    if method == "percentile":
        q_lo, q_hi = alpha, 1.0 - alpha
    else:
        keep = np.ones(n, dtype=bool)
        jack = np.empty(n)
        for i in range(n):
            keep[i] = False
            jack[i] = metric(yt[keep], yp[keep])
            keep[i] = True
        q_lo, q_hi = _bca_quantiles(stats, theta_hat, jack, alpha)
    low, high = np.quantile(stats, [q_lo, q_hi])
    return Interval(theta_hat, float(low), float(high), level)


def corrected_t_interval(
    scores: ArrayLike, n_train: int, n_test: int, *, level: float = 0.95
) -> Interval:
    """Interval for the mean of repeated cross-validation scores.

    Scores from different folds are positively correlated, because their
    training sets overlap, so the naive standard error ``s / sqrt(J)``
    understates the uncertainty. Nadeau & Bengio (2003), "Inference for the
    generalization error", *Machine Learning* 52, replace the variance factor
    ``1/J`` with ``1/J + n_test/n_train``. For 10 repeats of 5-fold CV
    (J = 50, n_test/n_train = 1/4) that widens the interval by a factor of
    sqrt((1/50 + 1/4) / (1/50)) = 3.67.

    ``scores`` may be any shape; all entries are treated as the J resamples.
    """
    s = np.asarray(scores, dtype=float).ravel()
    if s.size < 2:
        raise ValueError("need at least two scores")
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be positive")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    J = s.size
    mean = float(s.mean())
    var = float(np.var(s, ddof=1))
    se = float(np.sqrt((1.0 / J + n_test / n_train) * var))
    tcrit = float(t_dist.ppf(0.5 + level / 2.0, df=J - 1))
    return Interval(mean, mean - tcrit * se, mean + tcrit * se, level)


@dataclass(frozen=True)
class RepeatedCVResult:
    scores: NDArray[np.float64]  # shape (n_repeats, n_splits)
    n_train: int
    n_test: int

    def interval(self, level: float = 0.95) -> Interval:
        return corrected_t_interval(self.scores, self.n_train, self.n_test, level=level)


def repeated_cv(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_splits: int = 5,
    n_repeats: int = 10,
    scoring: str = "accuracy",
    stratify: bool = False,
    random_state: int = 0,
) -> RepeatedCVResult:
    """Repeated shuffled k-fold, returning the score matrix and fold sizes.

    Repeat ``r`` uses seed ``random_state + r``. ``n_train`` and ``n_test`` are
    the mean fold sizes, which is what the Nadeau-Bengio correction needs.
    """
    Xa = np.asarray(X)
    ya = np.asarray(y)
    scorer = get_scorer(scoring)
    scores = np.empty((n_repeats, n_splits))
    tr_sizes: list[int] = []
    te_sizes: list[int] = []
    for r in range(n_repeats):
        cv = (StratifiedKFold if stratify else KFold)(
            n_splits, shuffle=True, random_state=random_state + r
        )
        for k, (tr, te) in enumerate(cv.split(Xa, ya)):
            model = clone(estimator).fit(Xa[tr], ya[tr])
            scores[r, k] = scorer(model, Xa[te], ya[te])
            tr_sizes.append(tr.size)
            te_sizes.append(te.size)
    return RepeatedCVResult(
        scores, round(float(np.mean(tr_sizes))), round(float(np.mean(te_sizes)))
    )


@dataclass(frozen=True)
class NestedCVResult:
    outer_scores: NDArray[np.float64]
    best_params: list[dict[str, Any]]
    oof_pred: NDArray[Any]
    test_mask: NDArray[np.bool_]

    @property
    def mean(self) -> float:
        return float(self.outer_scores.mean())


def nested_cv(
    estimator: BaseEstimator,
    param_grid: Mapping[str, Sequence[Any]],
    X: ArrayLike,
    y: ArrayLike,
    *,
    outer: Any,
    inner: Any,
    scoring: str = "accuracy",
    groups: ArrayLike | None = None,
) -> NestedCVResult:
    """Tune with ``inner`` inside each fold of ``outer``, and score on ``outer``.

    The score of a model chosen by a search is optimistic when it is measured on
    the same folds the search used: the search picks whichever configuration got
    lucky on those folds. Running the search again inside each outer training
    fold keeps the outer test fold out of model selection entirely.

    If ``groups`` is given it is passed to both splitters, so a group splitter
    can be used at either level.

    ``oof_pred`` holds the outer-fold prediction for every row that was ever in
    a test fold; ``test_mask`` marks those rows (a forward-chaining time split
    never tests its first block).
    """
    Xa = np.asarray(X)
    ya = np.asarray(y)
    g = None if groups is None else np.asarray(groups)
    scorer = get_scorer(scoring)
    outer_kwargs: dict[str, Any] = {} if g is None else {"groups": g}
    scores: list[float] = []
    params: list[dict[str, Any]] = []
    oof: NDArray[Any] = np.empty(ya.shape[0], dtype=float)
    mask = np.zeros(ya.shape[0], dtype=bool)
    for train, test in outer.split(Xa, ya, **outer_kwargs):
        search = GridSearchCV(clone(estimator), dict(param_grid), cv=inner, scoring=scoring)
        fit_kwargs: dict[str, Any] = {} if g is None else {"groups": g[train]}
        search.fit(Xa[train], ya[train], **fit_kwargs)
        scores.append(float(scorer(search.best_estimator_, Xa[test], ya[test])))
        params.append(dict(search.best_params_))
        oof[test] = search.best_estimator_.predict(Xa[test])
        mask[test] = True
    return NestedCVResult(np.asarray(scores), params, oof, mask)
