# Results directory

Small reproducible metrics and PNG figures may be committed after experiments
have actually run. Model checkpoints, serialized estimators, and other large
binary artifacts are ignored by Git. Artifacts under a `smoke_...` run are
pipeline checks only and must not be presented as full-dataset research results.

Phase 3 writes validation-only artifacts into run-specific directories:

- `metrics/classical/<run>/`: split audit, exact configuration, runtime/package
  metadata, candidate metrics, selected validation metrics, and timings.
- `figures/classical/<run>/`: validation confusion matrices and comparison plot.
- `models/classical/<run>/`: train-only fitted pipelines selected using
  validation macro F1. This directory is ignored by Git.

The classical training command does not load or score test images. Final test
results must be produced later through the explicitly controlled unified
evaluation workflow.

Phase 4 writes CNN validation artifacts to `metrics/cnn/<run>/` and
`figures/cnn/<run>/`. Best-validation `.pt` checkpoints are stored under the
ignored `models/cnn/<run>/` directory. Smoke CNN runs verify mechanics only;
they are not full-dataset research results. The CNN trainer refuses to create a
test dataset, and final test evaluation remains a later controlled phase.

Phase 5 uses `metrics/final/final_test_state.json` as a one-shot test-set lock.
The final run writes unified metrics and provenance to `metrics/final/<run>/`
and figures to `figures/final/<run>/`. Do not delete or bypass the state file to
repeat evaluation; any recovery requires explicit review and documentation.
