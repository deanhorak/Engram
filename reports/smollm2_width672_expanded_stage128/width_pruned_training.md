# Progressive fixed-width MLP distillation

Decision: **continue_width_distillation_or_stop**

Target width: 672/1536; projected MLP traffic: 0.437500× dense.

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 1.177297 | ≤0.05 |
| Teacher top-1 agreement | 0.474542 | ≥0.90 |
| NLL delta | 1.055306 | ≤0.05 |
| Final hidden relative L2 | 0.426021 | ≤0.10 |
| Local MLP relative L2 | 0.705340 | diagnostic |
