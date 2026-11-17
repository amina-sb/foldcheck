"""Uncertainty for cross-validated metrics, and nested cross-validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV

Metric = Callable[[NDArray[Any], NDArray[Any]], float]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric: Metric,
    *,
    n_resamples: int = 1000,
    level: float = 0.95,
    random_state: int | None = None,
) -> Interval:
    """Percentile bootstrap interval for ``metric(y_true, y_pred)``.

    Rows are resampled with replacement as pairs. The point estimate is the
    metric on the original data, not the mean of the resamples.

    This is the plain percentile interval. It is appropriate for metrics that
    are roughly symmetric in distribution (accuracy, R² away from its bounds)
    at the sample sizes used here; it under-covers for heavily skewed
    statistics, for which a BCa interval is the better tool.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    n = yt.shape[0]
    rng = np.random.default_rng(random_state)
    stats = np.empty(n_resamples)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats[b] = metric(yt[idx], yp[idx])
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return Interval(float(metric(yt, yp)), float(low), float(high), level)


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
