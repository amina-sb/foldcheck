# foldcheck

Validation protocols for applied machine learning: grouped and time-ordered
splits, nested cross-validation, and a leakage audit that fails a run when a
transform was fitted outside its fold.

## Why it exists

Fitting a transform before splitting, splitting a time series at random, and
reporting the score of the search that chose the model are the three
commonest ways a validation score ends up measuring the wrong thing. None of
them raise an error. This package makes the correct protocol the easy one and
checks for the wrong ones.

## Evidence

Four synthetic scenarios with a known right answer. Produced by
`python scripts/evidence.py`; a test fails if these tables and the script's
output differ. For A to C: out-of-fold estimates, with a 95% percentile bootstrap interval
(1,000 resamples) for seed 0, then the mean and range over seeds 0–4.

<!-- evidence:start -->
| Scenario | Protocol | Metric | Seed 0 (95% CI) | Seeds 0-4: mean [min, max] |
|---|---|---|---|---|
| A. 2,000 noise features | select on all rows, then CV | accuracy | 0.95 (0.88 to 1.00) | 0.89 [0.85, 0.95] |
|  | select inside each fold | accuracy | 0.48 (0.37 to 0.60) | 0.52 [0.47, 0.62] |
|  | baseline: majority class | accuracy | 0.60 (0.48 to 0.72) | 0.57 [0.50, 0.65] |
| B. random walk from time index | random 5-fold, kNN | R² | 0.99 (0.98 to 0.99) | 0.99 [0.99, 0.99] |
|  | forward split, kNN | R² | 0.58 (0.50 to 0.65) | 0.48 [0.14, 0.69] |
|  | baseline: last value, forward split | R² | 0.63 (0.54 to 0.69) | 0.53 [0.23, 0.71] |
| C. tuning kNN on noise | best-of-12 grid score (flat) | accuracy | 0.47 (0.38 to 0.56) | 0.53 [0.47, 0.57] |
|  | search nested in each fold | accuracy | 0.45 (0.36 to 0.54) | 0.49 [0.42, 0.55] |
|  | baseline: majority class | accuracy | 0.49 (0.40 to 0.58) | 0.52 [0.49, 0.53] |
<!-- evidence:end -->

What the table shows:

- **A.** With 60 samples, 2,000 noise features and labels unrelated to any of
  them, choosing the 20 "best" features on all rows before cross-validating
  reports 0.89 accuracy on average. Doing the same selection inside each fold
  gives 0.52, which is no better than predicting the majority class (0.57). The
  leak is worth 37 points. This is the design of Ambroise & McLachlan (2002),
  *PNAS* 99(10).
- **B.** A random walk predicted from its time index alone. A random split
  reports R² 0.99, because the model interpolates between neighbouring days. A
  forward split gives 0.48, which is *worse* than carrying the last value
  forward (0.53). The random split hides the fact that the model loses to a
  constant.
- **C.** Reporting the best of a 12-configuration grid on noise gives 0.53. The
  same search nested inside each fold gives 0.49. The bias is consistent (the flat
  score beats the nested one on every seed in `tests/test_inference.py`) but
  small: about 4 points, well inside a single run's interval. It only
  becomes visible across repeated runs.

**D. How wide should the interval around a CV score be?** The fold scores of
repeated cross-validation share training rows, so they are correlated, and the
textbook `s / sqrt(J)` standard error is far too small.

<!-- coverage:start -->
True expected accuracy at 160 training rows: 0.764

| Interval for 10 x 5-fold CV (n = 200) | Coverage of nominal 95% (Wilson 95% CI) | Mean width |
|---|---|---|
| naive t-interval, s/sqrt(J) | 45/100 (0.36 to 0.55) | 0.035 |
| Nadeau-Bengio corrected | 98/100 (0.93 to 0.99) | 0.129 |
<!-- coverage:end -->

The naive interval contains the true accuracy less than half the time. The
Nadeau & Bengio (2003) correction restores coverage and slightly
over-covers, which matches its known conservatism. Its interval is 3.7 times
wider, and that width is the honest one.

Each scenario is reproduced to the digit under numpy 2.2 / scikit-learn 1.7 and
numpy 2.5 / scikit-learn 1.9.

## Install

```bash
pip install git+https://github.com/amina-sb/foldcheck
# development
pip install -e ".[dev]" && pytest -q && python scripts/evidence.py
```

Documentation: <https://amina-sb.github.io/foldcheck/>

## Splitters

Both splitters follow the scikit-learn protocol and can be passed as `cv=`.

```python
from foldcheck import GroupKFold, TimeSeriesSplit

# Forward-chaining, 3-row embargo between train and test, rolling 500-row window.
cv = TimeSeriesSplit(5, gap=3, max_train_size=500)
for train, test in cv.split(X, times=df["date"]):  # raises if dates are unsorted
    ...

# Every row of a patient/user/store lands on one side of each split.
cv = GroupKFold(5, shuffle=True, random_state=0)
cross_val_score(model, X, y, cv=cv, groups=df["patient_id"])
```

`TimeSeriesSplit` refuses unsorted timestamps rather than sorting them: an
unsorted time column usually means the frame was shuffled upstream, and quietly
sorting would hide that. The `gap` exists because a trailing-window feature (a
7-day mean, a lag) computed on the last training rows already contains values
from the test period.

`GroupKFold` balances fold sizes with the same greedy rule as scikit-learn's
(its fold-size multiset is tested to match), and adds seeded shuffling of
equal-sized groups so that repeating a run under several seeds produces
different folds.

## Leakage audit

```python
from foldcheck import audit_cv, detect_global_scaling, scan_target_leakage

# 1. Refit per fold and check what every pipeline step actually saw.
audit_cv(pipe, X, y, cv=KFold(5, shuffle=True, random_state=0)).raise_if_leaky()

# 2. Was X scaled on all rows before it got here?
print(detect_global_scaling(X, column_names=feature_names))

# 3. Does any single column predict the target almost perfectly?
report, scores = scan_target_leakage(X, y, cv=KFold(5, shuffle=True, random_state=0))
```

`audit_cv` reports two things per pipeline step: `sample_count` when
`n_samples_seen_` is larger than the training fold, and `not_refit` when the
step's fitted statistics (`mean_`, `scale_`, `statistics_`, ...) are
identical in every fold. It walks into `Pipeline` and `ColumnTransformer`.

`detect_global_scaling` looks for the fingerprint of a scaler fitted before the
split: columns with mean 0 and standard deviation 1, or range [0, 1], over
*all* rows, within `atol=1e-10`. The tolerance matters. `MinMaxScaler` output
misses an exact 0/1 range on about one column in five because of rounding, and
an exact-equality check missed 220 of 1,000 scaled columns in the test suite.
Two-valued columns are exempt from the range check, since a 0/1 indicator is
not evidence of scaling.

`scan_target_leakage` fits a depth-3 tree on each feature alone and scores it
out of fold (ROC AUC for binary targets, R² otherwise). Features above 0.98 AUC
or 0.95 R² are flagged. It returns the full score vector as well, because the
feature just under the threshold is often the one worth reading about.

## Nested cross-validation and intervals

```python
from foldcheck.inference import bootstrap_ci, nested_cv

res = nested_cv(
    KNeighborsClassifier(),
    {"n_neighbors": [1, 5, 15]},
    X,
    y,
    outer=StratifiedKFold(5, shuffle=True, random_state=0),
    inner=StratifiedKFold(4, shuffle=True, random_state=0),
)
m = res.test_mask
print(bootstrap_ci(y[m], res.oof_pred[m], accuracy, random_state=0))
```

`bootstrap_ci` is a paired percentile bootstrap. It is checked against
`scipy.stats.bootstrap` and by a coverage simulation (300 replicates, nominal
95%). `method="bca"` gives the bias-corrected and accelerated interval, checked
against `scipy.stats.bootstrap(method="BCa")`. Neither method reaches nominal
coverage on small skewed samples. For the mean of an Exponential(1) at n = 20,
coverage over 400 replicates was 0.905 (percentile) and 0.925 (BCa), so BCa is
better but not a cure.

For a repeated-CV mean score, use `repeated_cv(...).interval()`, which applies
the corrected resampled t-interval (table D above).

See [ROADMAP.md](ROADMAP.md) for what is still in progress.
