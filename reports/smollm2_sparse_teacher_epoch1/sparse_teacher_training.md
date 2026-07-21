# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 32/32; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.900200 | ≥0.95 | no |
| Teacher-student KL | 0.448130 | ≤0.05 | no |
| Teacher top-1 agreement | 0.720978 | ≥0.90 | no |
| NLL delta | 0.342551 | ≤0.05 | no |
| Final hidden relative L2 | 0.249742 | ≤0.10 | no |

The artifact contains only router factors and sparse MLP down-adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
