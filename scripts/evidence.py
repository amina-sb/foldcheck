"""Reproduce the README's evidence table.

Three synthetic scenarios where the right answer is known in advance, so
the size of each protocol's error can be read straight off the table:

A. Feature selection on pure noise (labels independent of all 2,000
   features). Any accuracy above the majority-class rate is leakage. This is
   the design of Ambroise & McLachlan (2002), "Selection bias in gene
   extraction on the basis of microarray gene-expression data", PNAS 99(10).
B. A random walk predicted from its time index alone. A random split lets
   the model interpolate between neighbouring days; a forward split forces
   it to forecast, which it cannot do better than carrying the last value.
C. Hyperparameter search on noise. The best-of-grid CV score is reported
   directly (flat) or re-estimated with the search nested inside each fold.

A second table (D) measures interval coverage. The true expected accuracy of
a logistic regression trained on 160 rows is estimated from 100 trainings
scored on 100,000 held-out rows. Then, on 100 independent 200-row datasets,
10 repeats of 5-fold CV are summarised with the naive t-interval and with
the Nadeau-Bengio corrected interval, and the table counts how often each
contains the truth.

Each row gives the seed-0 out-of-fold estimate with a 95% percentile
bootstrap interval (1,000 resamples), then the mean and range over seeds
0-4. Run with ``python scripts/evidence.py``; the output is the README table
verbatim, and ``tests/test_evidence.py`` fails if the two diverge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline

from foldcheck import TimeSeriesSplit
from foldcheck.inference import bootstrap_ci, corrected_t_interval, nested_cv, repeated_cv

SEEDS = range(5)
N_BOOT = 1000

Arr = NDArray[Any]
Run = Callable[[int], tuple[Arr, Arr]]  # seed -> (y_true, oof_pred) on scored rows


def accuracy(a: Arr, b: Arr) -> float:
    return float(np.mean(a == b))


def r2(a: Arr, b: Arr) -> float:
    return float(r2_score(a, b))


# --- A: selection bias -------------------------------------------------------


def _noise(seed: int) -> tuple[Arr, Arr]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(60, 2000)), rng.integers(0, 2, 60)


def _a_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def a_outside(seed: int) -> tuple[Arr, Arr]:
    X, y = _noise(seed)
    Xs = SelectKBest(f_classif, k=20).fit_transform(X, y)  # sees every row
    return y, cross_val_predict(LogisticRegression(max_iter=1000), Xs, y, cv=_a_cv(seed))


def a_inside(seed: int) -> tuple[Arr, Arr]:
    X, y = _noise(seed)
    pipe = make_pipeline(SelectKBest(f_classif, k=20), LogisticRegression(max_iter=1000))
    return y, cross_val_predict(pipe, X, y, cv=_a_cv(seed))


def a_majority(seed: int) -> tuple[Arr, Arr]:
    X, y = _noise(seed)
    return y, cross_val_predict(DummyClassifier(), X, y, cv=_a_cv(seed))


# --- B: random vs time-ordered split ------------------------------------------


def _walk(seed: int) -> tuple[Arr, Arr]:
    rng = np.random.default_rng(seed)
    y = np.cumsum(rng.normal(size=600))
    return np.arange(600, dtype=float).reshape(-1, 1), y


def b_random(seed: int) -> tuple[Arr, Arr]:
    X, y = _walk(seed)
    cv = KFold(5, shuffle=True, random_state=seed)
    return y, cross_val_predict(KNeighborsRegressor(5), X, y, cv=cv)


def _forward(seed: int, predict: Callable[[Arr, Arr, Arr], Arr]) -> tuple[Arr, Arr]:
    X, y = _walk(seed)
    pred = np.full(y.shape, np.nan)
    for tr, te in TimeSeriesSplit(5).split(X):
        pred[te] = predict(X[tr], y[tr], X[te])
    m = ~np.isnan(pred)
    return y[m], pred[m]


def b_time(seed: int) -> tuple[Arr, Arr]:
    def knn(xtr: Arr, ytr: Arr, xte: Arr) -> Arr:
        out: Arr = KNeighborsRegressor(5).fit(xtr, ytr).predict(xte)
        return out

    return _forward(seed, knn)


def b_last_value(seed: int) -> tuple[Arr, Arr]:
    return _forward(seed, lambda _xtr, ytr, xte: np.full(xte.shape[0], ytr[-1]))


# --- C: flat vs nested tuning -------------------------------------------------

GRID: dict[str, list[Any]] = {
    "n_neighbors": [1, 3, 5, 9, 15, 25],
    "weights": ["uniform", "distance"],
}


def _c_data(seed: int) -> tuple[Arr, Arr]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(120, 30)), rng.integers(0, 2, 120)


def _c_cv(seed: int) -> StratifiedKFold:
    return StratifiedKFold(5, shuffle=True, random_state=seed)


def c_flat(seed: int) -> tuple[Arr, Arr]:
    X, y = _c_data(seed)
    best = GridSearchCV(KNeighborsClassifier(), GRID, cv=_c_cv(seed)).fit(X, y).best_params_
    # Out-of-fold predictions of the winning configuration on the folds that chose it.
    return y, cross_val_predict(KNeighborsClassifier(**best), X, y, cv=_c_cv(seed))


def c_nested(seed: int) -> tuple[Arr, Arr]:
    X, y = _c_data(seed)
    inner = StratifiedKFold(4, shuffle=True, random_state=seed)
    res = nested_cv(KNeighborsClassifier(), GRID, X, y, outer=_c_cv(seed), inner=inner)
    return y, res.oof_pred.astype(int)


def c_majority(seed: int) -> tuple[Arr, Arr]:
    X, y = _c_data(seed)
    return y, cross_val_predict(DummyClassifier(), X, y, cv=_c_cv(seed))


ROWS: list[tuple[str, str, str, Run, Callable[[Arr, Arr], float]]] = [
    ("A. 2,000 noise features", "select on all rows, then CV", "accuracy", a_outside, accuracy),
    ("", "select inside each fold", "accuracy", a_inside, accuracy),
    ("", "baseline: majority class", "accuracy", a_majority, accuracy),
    ("B. random walk from time index", "random 5-fold, kNN", "R²", b_random, r2),
    ("", "forward split, kNN", "R²", b_time, r2),
    ("", "baseline: last value, forward split", "R²", b_last_value, r2),
    ("C. tuning kNN on noise", "best-of-12 grid score (flat)", "accuracy", c_flat, accuracy),
    ("", "search nested in each fold", "accuracy", c_nested, accuracy),
    ("", "baseline: majority class", "accuracy", c_majority, accuracy),
]


def _fmt(x: float) -> str:
    # Large negative R² values are printed without decimals; they are not small errors.
    return f"{x:.2f}" if x > -10 else f"{x:.0f}"


def table() -> str:
    lines = [
        "| Scenario | Protocol | Metric | Seed 0 (95% CI) | Seeds 0-4: mean [min, max] |",
        "|---|---|---|---|---|",
    ]
    for scenario, protocol, name, run, metric in ROWS:
        y0, p0 = run(0)
        ci = bootstrap_ci(y0, p0, metric, n_resamples=N_BOOT, random_state=0)
        vals = np.array([metric(*run(s)) for s in SEEDS])
        seed0 = f"{_fmt(ci.estimate)} ({_fmt(ci.low)} to {_fmt(ci.high)})"
        spread = (
            f"{_fmt(float(vals.mean()))} [{_fmt(float(vals.min()))}, {_fmt(float(vals.max()))}]"
        )
        lines.append(f"| {scenario} | {protocol} | {name} | {seed0} | {spread} |")
    return "\n".join(lines)


# --- D: interval coverage for repeated CV ------------------------------------

_BETA = np.array([1.0, 0.5, 0.25, 0.0, 0.0])
D_REPLICATES = 100


def _d_draw(rng: np.random.Generator, n: int) -> tuple[Arr, Arr]:
    X = rng.normal(size=(n, 5))
    return X, (X @ _BETA + rng.normal(size=n) > 0).astype(int)


def d_truth() -> float:
    rng = np.random.default_rng(99)
    Xt, yt = _d_draw(rng, 100_000)
    accs = []
    for _ in range(100):
        X, y = _d_draw(rng, 160)
        accs.append(LogisticRegression().fit(X, y).score(Xt, yt))
    return float(np.mean(accs))


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def coverage_table() -> str:
    truth = d_truth()
    hits = {"naive": 0, "corrected": 0}
    widths: dict[str, list[float]] = {"naive": [], "corrected": []}
    for r in range(D_REPLICATES):
        X, y = _d_draw(np.random.default_rng(r), 200)
        res = repeated_cv(LogisticRegression(), X, y, n_splits=5, n_repeats=10, random_state=r)
        intervals = {
            # A vanishing n_test/n_train turns the correction off: s / sqrt(J).
            "naive": corrected_t_interval(res.scores, n_train=10**12, n_test=1),
            "corrected": res.interval(),
        }
        for name, ci in intervals.items():
            hits[name] += ci.low <= truth <= ci.high
            widths[name].append(ci.high - ci.low)
    labels = {
        "naive": "naive t-interval, s/sqrt(J)",
        "corrected": "Nadeau-Bengio corrected",
    }
    lines = [
        f"True expected accuracy at 160 training rows: {truth:.3f}",
        "",
        "| Interval for 10 x 5-fold CV (n = 200) | Coverage of nominal 95% "
        "(Wilson 95% CI) | Mean width |",
        "|---|---|---|",
    ]
    for name, label in labels.items():
        lo, hi = _wilson(hits[name], D_REPLICATES)
        cov = f"{hits[name]}/{D_REPLICATES} ({lo:.2f} to {hi:.2f})"
        lines.append(f"| {label} | {cov} | {np.mean(widths[name]):.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table())
    print()
    print(coverage_table())
