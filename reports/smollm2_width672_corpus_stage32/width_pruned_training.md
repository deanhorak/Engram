# Progressive fixed-width MLP distillation

Decision: **continue_width_distillation_or_stop**

Target width: 672/1536; projected MLP traffic: 0.437500× dense.

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 1.549945 | ≤0.05 |
| Teacher top-1 agreement | 0.417515 | ≥0.90 |
| NLL delta | 1.525404 | ≤0.05 |
| Final hidden relative L2 | 0.489636 | ≤0.10 |
| Local MLP relative L2 | 0.763593 | diagnostic |
