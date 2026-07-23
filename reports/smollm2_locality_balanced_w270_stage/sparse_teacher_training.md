# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 16/16; validation records: 4.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.900233 | ≥0.95 | no |
| Teacher-student KL | 0.153005 | ≤0.05 | no |
| Teacher top-1 agreement | 0.712121 | ≥0.90 | no |
| NLL delta | 0.205370 | ≤0.05 | no |
| Final hidden relative L2 | 0.173985 | ≤0.10 | no |
| Candidate cache-line fraction | 0.998562 | minimize | informational |
| Scalar projected traffic | 0.611111× dense | informational | — |
| Cache-adjusted traffic | 0.777418× dense | <1.0× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
