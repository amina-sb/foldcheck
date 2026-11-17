# Splitters

Both splitters implement scikit-learn's splitter protocol, so they work as
`cv=` anywhere scikit-learn accepts one.

## `TimeSeriesSplit`

Rows are cut into `n_splits + 1` contiguous blocks. Split *k* tests on block
*k + 1* and trains on everything before it, minus `gap` rows.

```python
from foldcheck import TimeSeriesSplit

cv = TimeSeriesSplit(5, gap=7, max_train_size=365)
for train, test in cv.split(X, times=df["date"]):
    ...
```

- **`gap`**: a feature built from a trailing window, such as a 7-day mean,
  computed on the last training day already contains test-period values. Set
  `gap` to at least the longest window.
- **`max_train_size`**: switches from an expanding to a rolling window.
- **`times`**: if given, it must be non-decreasing, and an unsorted column
  raises. Sorting it silently would hide whatever shuffled the frame upstream.

With `gap=0` and a row count divisible by `n_splits + 1`, the indices are
identical to `sklearn.model_selection.TimeSeriesSplit`. That is tested.

## `GroupKFold`

Every row of a group is assigned to the same test fold. Assignment is greedy:
largest group first, into the fold with the fewest rows. This is
scikit-learn's rule, and the fold-size multiset is tested to match it.
`shuffle=True` breaks ties between equal-sized groups using `random_state`, so
repeating an experiment under different seeds really changes the folds.

```python
from foldcheck import GroupKFold

cross_val_score(model, X, y, cv=GroupKFold(5, shuffle=True, random_state=0), groups=user_id)
```
