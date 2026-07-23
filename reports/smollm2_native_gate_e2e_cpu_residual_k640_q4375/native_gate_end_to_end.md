# Progressive end-to-end native-gate training

Decision: **continue_training_or_stop_before_serialization**

Device: `cpu` (the forward semantics and artifact are device-neutral).

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 0.683512 | ≤0.05 |
| Teacher top-1 agreement | 0.602851 | ≥0.90 |
| NLL delta | 0.615738 | ≤0.05 |
| Final hidden relative L2 | 0.358166 | ≤0.10 |
| Local MLP relative L2 | 0.594341 | diagnostic |
| Projected traffic | 0.442491× | ≤0.45× |

Validation executes only the target hard sparse path. A smoke run below the evidence
floor validates mechanics but cannot establish model quality.
