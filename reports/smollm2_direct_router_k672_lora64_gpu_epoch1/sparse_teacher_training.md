# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 2048/256; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Candidate recall | 0.564763 | ≥0.95 | no |
| Teacher-student KL | 0.842684 | ≤0.05 | no |
| Teacher top-1 agreement | 0.572301 | ≥0.90 | no |
| NLL delta | 0.768195 | ≤0.05 | no |
| Final hidden relative L2 | 0.377083 | ≤0.10 | no |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
