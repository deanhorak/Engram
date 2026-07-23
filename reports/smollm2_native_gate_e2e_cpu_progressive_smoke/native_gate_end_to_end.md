# Progressive end-to-end native-gate training

Decision: **continue_training_or_stop_before_serialization**

Device: `cpu` (the forward semantics and artifact are device-neutral).

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 1.592339 | ≤0.05 |
| Teacher top-1 agreement | 0.444444 | ≥0.90 |
| NLL delta | 1.445778 | ≤0.05 |
| Final hidden relative L2 | 0.634594 | ≤0.10 |
| Local MLP relative L2 | 0.580721 | diagnostic |
| Projected traffic | 0.430556× | ≤0.45× |

Validation executes only the target hard sparse path. A smoke run below the evidence
floor validates mechanics but cannot establish model quality.
