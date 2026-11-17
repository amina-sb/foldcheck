"""Checks that a cross-validated pipeline did not see its test rows.

Three detectors, each aimed at a different way the test fold leaks into training:

* :func:`audit_cv` refits the estimator in each fold and inspects every pipeline
  step, catching steps that were not fitted on exactly the training rows.
* :func:`detect_global_scaling` inspects the raw design matrix for the
  fingerprint of a scaler fitted on all rows *before* the matrix reached CV.
* :func:`scan_target_leakage` scores each feature on its own and flags features
  that predict the target almost perfectly, which is usually a column derived
  from the target.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

Kind = Literal["sample_count", "not_refit", "global_scaling", "target_leak"]

# Fitted attributes that summarise the training data. If one is identical in
# every fold, the step is not being refit on its fold.
_STAT_ATTRS = (
    "mean_",
    "var_",
    "scale_",
    "center_",
    "statistics_",
    "data_min_",
    "data_max_",
    "quantiles_",
    "components_",
    "scores_",
    "categories_",
)


class LeakageError(RuntimeError):
    """Raised by :meth:`AuditReport.raise_if_leaky` when any finding was recorded."""


@dataclass(frozen=True)
class Finding:
    kind: Kind
    where: str
    detail: str


@dataclass
class AuditReport:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def extend(self, other: AuditReport) -> AuditReport:
        self.findings.extend(other.findings)
        return self

    def raise_if_leaky(self) -> None:
        if self.findings:
            lines = "\n".join(f"  [{f.kind}] {f.where}: {f.detail}" for f in self.findings)
            raise LeakageError(f"{len(self.findings)} leakage finding(s):\n{lines}")

    def __str__(self) -> str:
        if self.ok:
            return "AuditReport: no findings"
        return "\n".join(f"[{f.kind}] {f.where}: {f.detail}" for f in self.findings)


def _steps(est: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Yield ``(path, estimator)`` for an estimator and every nested pipeline step."""
    yield (prefix or type(est).__name__), est
    if isinstance(est, Pipeline):
        for name, step in est.steps:
            if step is None or step == "passthrough":
                continue
            yield from _steps(step, f"{prefix}{name}" if not prefix else f"{prefix}.{name}")
    transformers = getattr(est, "transformers_", None)
    if transformers is not None:
        for name, step, _cols in transformers:
            if isinstance(step, BaseEstimator):
                yield from _steps(step, f"{prefix}.{name}" if prefix else name)


def _fingerprint(est: Any) -> tuple[bytes, ...] | None:
    parts: list[bytes] = []
    for attr in _STAT_ATTRS:
        val = getattr(est, attr, None)
        if val is None:
            continue
        try:
            arr = np.asarray(val, dtype=float)
        except (TypeError, ValueError):
            arr = np.asarray([str(v) for v in np.ravel(np.asarray(val, dtype=object))])
        parts.append(attr.encode() + arr.tobytes())
    return tuple(parts) if parts else None


def audit_cv(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    cv: Any,
    *,
    groups: ArrayLike | None = None,
) -> AuditReport:
    """Fit a clone of ``estimator`` in every fold and check what each step saw.

    Two findings are possible for a step:

    ``sample_count``
        ``n_samples_seen_`` differs from the training-fold size. The step was fitted
        on something other than the training rows (for example, it was fitted once
        on all rows and then protected from refitting).
    ``not_refit``
        Its fitted statistics are identical in every fold. With different training
        rows per fold this is very unlikely unless the step ignores its fold.
    """
    Xa = np.asarray(X)
    ya = np.asarray(y)
    split_kwargs: dict[str, Any] = {} if groups is None else {"groups": groups}
    report = AuditReport()
    prints: dict[str, list[tuple[bytes, ...] | None]] = {}
    n_folds = 0
    for train, _test in cv.split(Xa, ya, **split_kwargs):
        n_folds += 1
        fitted = clone(estimator).fit(Xa[train], ya[train])
        for path, step in _steps(fitted):
            seen = getattr(step, "n_samples_seen_", None)
            if seen is not None:
                seen_arr = np.atleast_1d(np.asarray(seen))
                if np.any(seen_arr > train.size):
                    report.findings.append(
                        Finding(
                            "sample_count",
                            path,
                            f"fold {n_folds}: saw {int(seen_arr.max())} rows, "
                            f"training fold has {train.size}",
                        )
                    )
            prints.setdefault(path, []).append(_fingerprint(step))
    if n_folds >= 2:
        for path, fps in prints.items():
            if fps[0] is not None and all(fp == fps[0] for fp in fps[1:]):
                report.findings.append(
                    Finding(
                        "not_refit",
                        path,
                        f"fitted statistics identical across all {n_folds} folds",
                    )
                )
    return report


def detect_global_scaling(
    X: ArrayLike, *, atol: float = 1e-10, column_names: Sequence[str] | None = None
) -> AuditReport:
    """Flag columns scaled exactly over all rows.

    A column whose mean is 0 and population standard deviation is 1 to within
    ``atol``, or whose minimum is 0 and maximum 1 (both to within ``atol``), has almost
    certainly been passed through a scaler fitted on the full matrix. Both tests
    use the tolerance: ``MinMaxScaler`` output misses an exact 0/1 range on
    roughly a fifth of columns because of rounding in ``(x - min) * scale``. If that
    matrix is then cross-validated, every fold's training data was scaled with
    statistics that include the test rows.

    Binary 0/1 indicator columns also have min 0 and max 1, so columns with only
    two distinct values are not flagged for min-max scaling.
    """
    Xa = np.asarray(X, dtype=float)
    if Xa.ndim != 2:
        raise ValueError("X must be two-dimensional")
    names = (
        list(column_names) if column_names is not None else [f"x{j}" for j in range(Xa.shape[1])]
    )
    report = AuditReport()
    for j in range(Xa.shape[1]):
        col = Xa[:, j]
        col = col[np.isfinite(col)]
        if col.size < 3:
            continue
        if abs(col.mean()) < atol and abs(col.std() - 1.0) < atol:
            report.findings.append(
                Finding("global_scaling", names[j], "mean 0 and std 1 over all rows")
            )
        elif abs(col.min()) < atol and abs(col.max() - 1.0) < atol and np.unique(col).size > 2:
            report.findings.append(
                Finding("global_scaling", names[j], "min 0 and max 1 over all rows")
            )
    return report


def _single_feature_scores(
    X: NDArray[np.float64],
    y: NDArray[Any],
    cv: Any,
    split_kwargs: dict[str, Any],
    classification: bool,
) -> NDArray[np.float64]:
    n, p = X.shape
    oof = np.full((n, p), np.nan)
    for train, test in cv.split(X, y, **split_kwargs):
        for j in range(p):
            xtr, xte = X[train, j : j + 1], X[test, j : j + 1]
            if classification:
                clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(xtr, y[train])
                proba = clf.predict_proba(xte)
                pos = np.flatnonzero(clf.classes_ == 1)
                oof[test, j] = proba[:, pos[0]] if pos.size else 0.0
            else:
                reg = DecisionTreeRegressor(max_depth=3, random_state=0).fit(xtr, y[train])
                oof[test, j] = reg.predict(xte)
    mask = ~np.isnan(oof[:, 0])
    scores = np.empty(p)
    for j in range(p):
        if classification:
            scores[j] = roc_auc_score(y[mask], oof[mask, j])
        else:
            scores[j] = r2_score(y[mask], oof[mask, j])
    return scores


def scan_target_leakage(
    X: ArrayLike,
    y: ArrayLike,
    cv: Any,
    *,
    groups: ArrayLike | None = None,
    threshold: float | None = None,
    column_names: Sequence[str] | None = None,
) -> tuple[AuditReport, NDArray[np.float64]]:
    """Score every feature alone and flag the ones that are too good.

    Each feature is fitted alone with a depth-3 decision tree under ``cv`` and
    scored out of fold: ROC AUC for a binary 0/1 target, R² otherwise. A depth-3
    tree catches monotone and simple non-monotone relationships without being
    able to memorise. Features at or above ``threshold`` (default 0.98 for AUC,
    0.95 for R²) are flagged.

    Returns the report and the per-feature scores, so the caller can inspect the
    whole ranking and not only what crossed the line.
    """
    Xa = np.asarray(X, dtype=float)
    ya = np.asarray(y)
    if Xa.ndim != 2:
        raise ValueError("X must be two-dimensional")
    classification = set(np.unique(ya).tolist()) <= {0, 1}
    if threshold is None:
        threshold = 0.98 if classification else 0.95
    split_kwargs: dict[str, Any] = {} if groups is None else {"groups": groups}
    scores = _single_feature_scores(Xa, ya, cv, split_kwargs, classification)
    names = (
        list(column_names) if column_names is not None else [f"x{j}" for j in range(Xa.shape[1])]
    )
    metric = "AUC" if classification else "R2"
    report = AuditReport()
    for j in np.argsort(-scores):
        if scores[j] >= threshold:
            report.findings.append(
                Finding(
                    "target_leak",
                    names[j],
                    f"alone reaches out-of-fold {metric} {scores[j]:.3f} (threshold {threshold})",
                )
            )
    return report, scores
