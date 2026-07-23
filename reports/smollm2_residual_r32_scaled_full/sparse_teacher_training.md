# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 128/32; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.896033 | ≥0.95 | no |
| Teacher-student KL | 0.151793 | ≤0.05 | no |
| Teacher top-1 agreement | 0.780041 | ≥0.90 | no |
| NLL delta | 0.100330 | ≤0.05 | no |
| Final hidden relative L2 | 0.193165 | ≤0.10 | no |
| Candidate cache-line fraction | 0.998594 | minimize | informational |
| Scalar projected traffic | 0.611111× dense | informational | — |
| Cache-adjusted traffic | 0.791315× dense | <1.0× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
