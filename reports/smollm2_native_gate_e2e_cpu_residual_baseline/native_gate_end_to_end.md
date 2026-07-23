# Progressive end-to-end native-gate training

Decision: **continue_training_or_stop_before_serialization**

Device: `cpu` (the forward semantics and artifact are device-neutral).

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 0.628528 | ≤0.05 |
| Teacher top-1 agreement | 0.598778 | ≥0.90 |
| NLL delta | 0.582924 | ≤0.05 |
| Final hidden relative L2 | 0.362706 | ≤0.10 |
| Local MLP relative L2 | 0.625053 | diagnostic |
| Projected traffic | 0.443866× | ≤0.45× |

Validation executes only the target hard sparse path. A smoke run below the evidence
floor validates mechanics but cannot establish model quality.
