# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 16/16; validation records: 4.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.899444 | ≥0.95 | no |
| Teacher-student KL | 0.426466 | ≤0.05 | no |
| Teacher top-1 agreement | 0.696970 | ≥0.90 | no |
| NLL delta | 0.321326 | ≤0.05 | no |
| Final hidden relative L2 | 0.224975 | ≤0.10 | no |
| Candidate cache-line fraction | 0.998576 | minimize | informational |
| Scalar projected traffic | 0.611111× dense | informational | — |
| Cache-adjusted traffic | 0.777422× dense | <1.0× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
