# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 32/8; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.895869 | ≥0.95 | no |
| Teacher-student KL | 0.165878 | ≤0.05 | no |
| Teacher top-1 agreement | 0.767821 | ≥0.90 | no |
| NLL delta | 0.126122 | ≤0.05 | no |
| Final hidden relative L2 | 0.198755 | ≤0.10 | no |
| Candidate cache-line fraction | 0.998580 | minimize | informational |
| Scalar projected traffic | 0.611111× dense | informational | — |
| Cache-adjusted traffic | 0.777423× dense | <1.0× | yes |

The artifact contains only router factors and sparse MLP down-adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
