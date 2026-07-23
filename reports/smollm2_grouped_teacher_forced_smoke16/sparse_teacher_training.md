# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 16/8; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Teacher-oracle membership recall | 0.545149 | informational after co-adaptation | — |
| Teacher-student KL | 2.046730 | ≤0.05 | no |
| Teacher top-1 agreement | 0.321792 | ≥0.90 | no |
| NLL delta | 2.073194 | ≤0.05 | no |
| Final hidden relative L2 | 0.493959 | ≤0.10 | no |
| Projected cold MLP traffic | 0.447290× dense Q4 | ≤0.45× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
