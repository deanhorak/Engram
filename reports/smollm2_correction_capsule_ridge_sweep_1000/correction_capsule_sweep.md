# Correction-capsule residual sweep

Status: **reject_before_causal_intervention**

Uncorrected mean local relative L2: 0.206579

| Capsules | Rank | Corrected rel-L2 | Improvement | Hard-subset rel-L2 | Match | Correction MB/token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0.271533 | -0.314430 | 0.360024 | 1.000000 | 1.244 |
| 1 | 16 | 0.271270 | -0.313155 | 0.359869 | 1.000000 | 2.350 |

This trace-only local MLP screen does not measure accumulated hidden-state drift, logit KL, NLL, or realized hardware traffic.
