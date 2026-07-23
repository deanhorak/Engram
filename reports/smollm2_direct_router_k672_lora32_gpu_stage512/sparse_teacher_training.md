# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 512/64; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.557464 | ≥0.95 | no |
| Teacher-student KL | 1.072558 | ≤0.05 | no |
| Teacher top-1 agreement | 0.478615 | ≥0.90 | no |
| NLL delta | 1.021115 | ≤0.05 | no |
| Final hidden relative L2 | 0.423067 | ≤0.10 | no |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
