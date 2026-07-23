# Progressive end-to-end native-gate training

Decision: **continue_training_or_stop_before_serialization**

Device: `cpu` (the forward semantics and artifact are device-neutral).

| Metric | Mean | Threshold |
|---|---:|---:|
| Teacher-student KL | 0.639921 | ≤0.05 |
| Teacher top-1 agreement | 0.604888 | ≥0.90 |
| NLL delta | 0.604494 | ≤0.05 |
| Final hidden relative L2 | 0.362814 | ≤0.10 |
| Local MLP relative L2 | 0.626408 | diagnostic |
| Projected traffic | 0.443866× | ≤0.45× |

Validation executes only the target hard sparse path. A smoke run below the evidence
floor validates mechanics but cannot establish model quality.
