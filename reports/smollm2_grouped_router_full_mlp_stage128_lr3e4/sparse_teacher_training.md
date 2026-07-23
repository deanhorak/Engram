# Sparse-teacher fine-tuning pilot

Status: **stop_before_serialization**

Training records/steps: 128/64; validation records: 16.

| Metric | Mean | Threshold | Pass |
|---|---:|---:|---|
| Teacher-oracle membership recall | 0.542253 | informational after co-adaptation | — |
| Teacher-student KL | 1.215123 | ≤0.05 | no |
| Teacher top-1 agreement | 0.464358 | ≥0.90 | no |
| NLL delta | 1.132424 | ≤0.05 | no |
| Final hidden relative L2 | 0.434646 | ≤0.10 | no |
| Projected cold MLP traffic | 0.447290× dense Q4 | ≤0.45× | yes |

The artifact contains router factors and mergeable sparse-MLP adapter tensors. It is not
eligible for package serialization unless every held-out gate check passes.
