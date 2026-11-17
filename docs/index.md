# foldcheck

Validation protocols for applied machine learning: grouped and time-ordered
splits, nested cross-validation, honest intervals, and a leakage audit that
fails a run when a transform was fitted outside its fold.

```bash
pip install foldcheck        # once released; until then:
pip install git+https://github.com/amina-sb/foldcheck
```

## The three mistakes it targets

| Mistake | What it does to the score | Where it is measured |
|---|---|---|
| Fitting selection or scaling before the split | 0.89 accuracy on pure noise, against 0.52 done properly | [Evidence, A](evidence.md) |
| Random split of a time series | R² 0.99 for a model that loses to last-value carried forward | [Evidence, B](evidence.md) |
| `s / sqrt(J)` interval on CV fold scores | covers the truth 45% of the time at nominal 95% | [Evidence, D](evidence.md) |

## Thirty-second tour

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from foldcheck import GroupKFold, audit_cv, repeated_cv

pipe = make_pipeline(StandardScaler(), LogisticRegression())

# Fails loudly if any step saw rows outside its training fold.
audit_cv(pipe, X, y, cv=GroupKFold(5), groups=patient_id).raise_if_leaky()

# A CV score with an interval that accounts for overlapping training sets.
print(repeated_cv(pipe, X, y, n_splits=5, n_repeats=10).interval())
```
