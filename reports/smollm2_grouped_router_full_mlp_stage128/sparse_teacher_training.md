# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 128/32; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Teacher-oracle membership recall | 0.544360 | informational after co-adaptation | — |
| Teacher-student KL | 1.327406 | ≤0.05 | no |
| Teacher top-1 agreement | 0.439919 | ≥0.90 | no |
| NLL delta | 1.264622 | ≤0.05 | no |
| Final hidden relative L2 | 0.448942 | ≤0.10 | no |
| Projected cold MLP traffic | 0.447290× dense Q4 | ≤0.45× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
