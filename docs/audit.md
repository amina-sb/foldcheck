# Leakage audit

## `audit_cv`: what did each step see?

`audit_cv` fits a fresh clone of the estimator in every fold and walks every
step of every `Pipeline` and `ColumnTransformer`.

| Finding | Meaning |
|---|---|
| `sample_count` | the step's `n_samples_seen_` is larger than the training fold |
| `not_refit` | the step's fitted statistics (`mean_`, `scale_`, `statistics_`, `scores_`, ...) are identical in every fold |

```python
report = audit_cv(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0))
print(report)
report.raise_if_leaky()  # LeakageError listing every finding
```

!!! note "What it cannot see"
    `audit_cv` inspects the pipeline. If `X` was already transformed before it
    reached the pipeline, the pipeline is clean and the data is not. That case
    needs `detect_global_scaling`.

## `detect_global_scaling`: was X scaled before the split?

Flags columns whose mean is 0 and standard deviation is 1, or whose range is
[0, 1], over *all* rows, to within `atol=1e-10`. Scaling fitted on a training
subset does not produce this, because the full-data statistics then differ
from (0, 1).

The tolerance is necessary. `MinMaxScaler` output misses an exact 0/1 range on
about one column in five because of rounding, and an exact-equality check
missed 220 of 1,000 scaled columns in the test suite. Columns with only two
distinct values are exempt from the range check, since a 0/1 indicator is not
evidence of scaling.

## `scan_target_leakage`: is a column a disguised target?

Each feature is fitted alone with a depth-3 decision tree and scored out of
fold: ROC AUC for a 0/1 target, R² otherwise. Anything at or above 0.98 AUC or
0.95 R² is flagged. The full score vector is returned too, because the column
just below the threshold is often the one worth reading about.

On pure noise the single-feature AUCs stay within 0.15 of 0.5 across the seeds
tested. A genuinely predictive feature, with AUC between 0.7 and 0.98 in the
test, is not flagged.
