# Evidence


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
  score beats the nested one on every seed in `tests/test_inference.py` in the repository) but
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

