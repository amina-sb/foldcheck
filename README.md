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
for train, test in cv.split(X, times=df["date"]):   # raises if dates are unsorted
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

See [ROADMAP.md](ROADMAP.md) for what is still in progress.
