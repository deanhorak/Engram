# Validation-trace robust BitNet oracle allocation

## Decision

A mean active-record budget below 25% is robustly feasible on the separate
controller validation trajectory. The proposed exact-membership oracle
schedule uses 51,681 of the available 51,840 records across 30 layers:
24.9233% mean active records.

This is a schedule-selection result, not a Milestone-2 pass. It uses exact
oracle membership and has not been evaluated on an untouched causal
confirmation corpus. In particular, neither the offset-8 confirmation result
nor any other confirmation data was read or used to fit this schedule.

## Why this allocator is more defensible

The earlier allocator minimized local MLP-output relative L2 on 32 positions
and applied an assumed depth weight. This experiment instead uses all 256
positions from the existing 16-sequence controller validation trace. For every
layer and candidate width from 15% through 40%, it:

1. reconstructs the MLP input from the normalized layer state plus attention
   output and the packaged post-attention RMSNorm;
2. computes exact BitNet record coefficients and contribution-magnitude oracle
   membership;
3. adds the oracle-sparse MLP output back to the captured post-attention
   residual;
4. compares the normalized candidate next boundary with the full-MLP next
   boundary; and
5. scores mean, token-p95, sequence-p95, and worst-sequence boundary error.

The final dynamic program minimizes the worse arm score from disjoint even and
odd eight-sequence halves. This is a teacher-forced one-step causal-sensitivity
proxy. It captures residual addition and normalization, but not downstream
perturbed rollout or a layer Jacobian.

## Result

- Robust schedule mean active fraction: **0.249233**
- Robust objective improvement over uniform 25%: **42.446%**
- Macro mean one-step boundary relative L2: **0.002111**
- Worst layer token-p95 boundary relative L2: **0.053920**
- Worst individual token boundary relative L2: **0.213305**
- Macro mean local MLP relative L2: **0.011260**
- Even/odd cross-fitted schedules:
  - exact agreement on 12 of 30 layers;
  - mean absolute allocation difference of 2.0 percentage points per layer.

The reconstruction itself is high fidelity: per-layer mean semantic-output
alignment cosine ranges from 0.99959 to 0.99987. Full reconstructed MLP
boundaries differ from captured next boundaries by 0.00366–0.01840 mean
relative L2, and the sparse-boundary metric is deliberately measured against
the reconstructed full-MLP boundary so that this trace quantization residual
is not charged to sparsity.

## Proposed frozen schedule

```text
K = [
  2765, 1210, 1210, 2765, 2765, 2074, 1210, 1383, 1556, 1728,
  2247, 2420, 2247, 2247, 2247, 2074, 1901, 1556, 1383, 1210,
  1383, 1037, 1037, 1037, 1210, 1383, 1210, 1383, 1556, 2247
]

fractions = [
  .400, .175, .175, .400, .400, .300, .175, .200, .225, .250,
  .325, .350, .325, .325, .325, .300, .275, .225, .200, .175,
  .200, .150, .150, .150, .175, .200, .175, .200, .225, .325
]
```

| Layer | Fraction | K | Boundary mean | Boundary p95 | Boundary max | Local MLP mean |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.400 | 2765 | 0.016182 | 0.053920 | 0.080935 | 0.016643 |
| 1 | 0.175 | 1210 | 0.000249 | 0.000000 | 0.019543 | 0.000377 |
| 2 | 0.175 | 1210 | 0.006090 | 0.000000 | 0.213305 | 0.014413 |
| 3 | 0.400 | 2765 | 0.003542 | 0.037089 | 0.063012 | 0.010934 |
| 4 | 0.400 | 2765 | 0.002025 | 0.021331 | 0.035040 | 0.009312 |
| 5 | 0.300 | 2074 | 0.003221 | 0.028390 | 0.052443 | 0.020492 |
| 6 | 0.175 | 1210 | 0.003056 | 0.025372 | 0.048046 | 0.035414 |
| 7 | 0.200 | 1383 | 0.000986 | 0.001646 | 0.030805 | 0.013638 |
| 8 | 0.225 | 1556 | 0.000837 | 0.005805 | 0.011350 | 0.015824 |
| 9 | 0.250 | 1728 | 0.000669 | 0.004548 | 0.008456 | 0.008952 |
| 10 | 0.325 | 2247 | 0.000171 | 0.000967 | 0.005506 | 0.002206 |
| 11 | 0.350 | 2420 | 0.000064 | 0.000000 | 0.004404 | 0.000279 |
| 12 | 0.325 | 2247 | 0.000100 | 0.000000 | 0.004752 | 0.000422 |
| 13 | 0.325 | 2247 | 0.000303 | 0.002747 | 0.008484 | 0.002769 |
| 14 | 0.325 | 2247 | 0.000710 | 0.004664 | 0.009957 | 0.004220 |
| 15 | 0.300 | 2074 | 0.001856 | 0.007717 | 0.012221 | 0.009264 |
| 16 | 0.275 | 1901 | 0.001377 | 0.005900 | 0.011051 | 0.007961 |
| 17 | 0.225 | 1556 | 0.001667 | 0.007071 | 0.010329 | 0.010163 |
| 18 | 0.200 | 1383 | 0.001613 | 0.007594 | 0.012064 | 0.008952 |
| 19 | 0.175 | 1210 | 0.003082 | 0.008535 | 0.012521 | 0.020153 |
| 20 | 0.200 | 1383 | 0.000325 | 0.002737 | 0.007879 | 0.001966 |
| 21 | 0.150 | 1037 | 0.001362 | 0.005726 | 0.012617 | 0.010330 |
| 22 | 0.150 | 1037 | 0.002306 | 0.006614 | 0.009890 | 0.019281 |
| 23 | 0.150 | 1037 | 0.001532 | 0.004770 | 0.006671 | 0.013020 |
| 24 | 0.175 | 1210 | 0.000578 | 0.003654 | 0.005106 | 0.004575 |
| 25 | 0.200 | 1383 | 0.000545 | 0.002941 | 0.004429 | 0.004660 |
| 26 | 0.175 | 1210 | 0.002528 | 0.006503 | 0.009747 | 0.021441 |
| 27 | 0.200 | 1383 | 0.002384 | 0.006040 | 0.007989 | 0.019795 |
| 28 | 0.225 | 1556 | 0.003148 | 0.006758 | 0.008605 | 0.025633 |
| 29 | 0.325 | 2247 | 0.000832 | 0.004541 | 0.008615 | 0.004708 |

## Next bounded action

Freeze this schedule before looking at another corpus. Run it exactly once on
an untouched causal confirmation set, then apply the existing KL, top-1, NLL,
and final-hidden gates. A failure means this locally robust allocation is not a
causal solution; it must not be tuned against the confirmation result.

Machine-readable evidence:
[`validation_trace_schedule.json`](validation_trace_schedule.json).
