# Correction-capsule residual sweep

Status: **reject_before_causal_intervention**

Uncorrected mean local relative L2: 0.206579

| Capsules | Rank | Corrected rel-L2 | Improvement | Hard-subset rel-L2 | Match | Correction MB/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.259230 | -0.254874 | 0.359957 | 1.000000 | 1.244 |
| 1 | 16 | 0.258594 | -0.251793 | 0.359318 | 1.000000 | 2.350 |
| 4 | 8 | 0.330307 | -0.598941 | 0.385320 | 1.000000 | 1.452 |
| 4 | 16 | 0.329991 | -0.597408 | 0.385177 | 1.000000 | 2.557 |
| 8 | 8 | 0.347229 | -0.680854 | 0.400149 | 1.000000 | 1.728 |
| 8 | 16 | 0.347004 | -0.679765 | 0.400083 | 1.000000 | 2.834 |

This trace-only local MLP screen does not measure accumulated hidden-state drift, logit KL, NLL, or realized hardware traffic.
