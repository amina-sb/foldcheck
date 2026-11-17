# Roadmap

## Phase 1 — Splitters
- [x] Time-ordered splitter with gap (embargo) between train and test
- [x] Grouped splitter that keeps every group on one side of each split
- [x] Property tests: no overlap, full coverage, ordering respected

## Phase 2 — Leakage audit
- [x] Inspect fitted pipeline steps per fold and fail when a step saw rows outside the training fold
- [x] Detect a matrix that was scaled on all rows before cross-validation
- [x] Single-feature target-leak scan

## Phase 3 — Nested cross-validation and evidence
- [ ] Nested CV returning outer-fold scores with an interval
- [ ] `scripts/evidence.py` reproducing the selection-bias demonstration, pinned by a test
- [ ] CI running lint, types, tests and evidence
