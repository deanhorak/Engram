# Progressive fixed-width MLP distillation

Decision: **continue_width_distillation_or_stop**

Target width: 672/1536; projected MLP traffic: 0.437500× dense.

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 0.976464 | ≤0.05 |
| Teacher top-1 agreement | 0.523422 | ≥0.90 |
| NLL delta | 0.853988 | ≤0.05 |
| Final hidden relative L2 | 0.398895 | ≤0.10 |
| Local MLP relative L2 | 0.693884 | diagnostic |
