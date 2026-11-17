# Intervals and nested CV

## `bootstrap_ci`

A paired bootstrap over `(y_true, y_pred)` rows for any metric.

```python
from foldcheck import bootstrap_ci

bootstrap_ci(y, pred, accuracy, n_resamples=2000, method="bca", random_state=0)
```

| method | checked against | coverage, mean of Exponential(1), n = 20, 400 replicates |
|---|---|---|
| `"percentile"` | `scipy.stats.bootstrap(method="percentile")` | 0.905 |
| `"bca"` | `scipy.stats.bootstrap(method="BCa")` | 0.925 |

Neither method reaches the nominal 0.95 on small skewed samples. BCa is the
better of the two here, but it is not a cure.

## `repeated_cv` and `corrected_t_interval`

Fold scores share training rows, so they are positively correlated and
`s / sqrt(J)` understates the standard error. Nadeau & Bengio (2003) replace
the variance factor `1/J` with `1/J + n_test/n_train`.

```python
from foldcheck import repeated_cv

res = repeated_cv(model, X, y, n_splits=5, n_repeats=10, stratify=True)
res.scores.shape  # (10, 5)
res.interval()  # Interval(estimate, low, high, level)
```

On simulated data with a known true accuracy, the naive interval covered 45 of
100 times and the corrected one 98 of 100 ([Evidence, D](evidence.md)).

## `nested_cv`

Runs `GridSearchCV` inside each outer training fold, so the outer test fold
never takes part in model selection. It forwards `groups` to both levels.
`test_mask` marks the rows that were ever tested, because a forward time
split never tests its first block.
