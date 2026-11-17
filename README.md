# foldcheck

Validation protocols for applied machine learning: grouped and time-ordered
splits, nested cross-validation, and a leakage audit that fails a run when a
transform was fitted outside its fold.

## Install

```bash
pip install -e ".[dev]"
```

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

See [ROADMAP.md](ROADMAP.md) for what is still in progress.
