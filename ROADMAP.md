# Roadmap

## Phase 1 — Splitters
- [ ] Time-ordered splitter with gap (embargo) between train and test
- [ ] Grouped splitter that keeps every group on one side of each split
- [ ] Property tests: no overlap, full coverage, ordering respected

## Phase 2 — Leakage audit
- [ ] Inspect fitted pipeline steps per fold and fail when a step saw rows outside the training fold
- [ ] Detect a transformer that was pre-fitted before cross-validation
- [ ] Single-feature target-leak scan

## Phase 3 — Nested cross-validation and evidence
- [ ] Nested CV returning outer-fold scores with an interval
- [ ] `scripts/evidence.py` reproducing the selection-bias demonstration, pinned by a test
- [ ] CI running lint, types, tests and evidence
