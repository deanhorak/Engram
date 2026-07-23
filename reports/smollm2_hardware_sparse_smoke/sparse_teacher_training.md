# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 1/1; validation records: 1.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.900452 | ≥0.95 | no |
| Teacher-student KL | 0.138780 | ≤0.05 | no |
| Teacher top-1 agreement | 0.764706 | ≥0.90 | no |
| NLL delta | 0.244990 | ≤0.05 | no |
| Final hidden relative L2 | 0.191306 | ≤0.10 | no |
| Candidate cache-line fraction | 0.998360 | minimize | informational |
| Scalar projected traffic | 0.611111× dense | informational | — |
| Cache-adjusted traffic | 0.777368× dense | <1.0× | yes |

The artifact contains only router factors and sparse MLP down-adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
