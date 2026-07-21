# Correction-capsule residual sweep

Status: **reject_before_causal_intervention**

Uncorrected mean local relative L2: 0.206579

| Priority | Capsules | Rank | Corrected rel-L2 | Improvement | Hard-subset rel-L2 | Match | Correction MB/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 4 | 8 | 0.317028 | -0.534660 | 0.335352 | 0.119066 | 0.416 |
| 0.10 | 8 | 8 | 0.280919 | -0.359865 | 0.326178 | 0.114464 | 0.687 |
| 0.20 | 4 | 8 | 0.268028 | -0.297460 | 0.325361 | 0.106180 | 0.401 |
| 0.20 | 8 | 8 | 0.264073 | -0.278315 | 0.325326 | 0.113346 | 0.686 |
| 0.40 | 4 | 8 | 0.233310 | -0.129397 | 0.321886 | 0.071400 | 0.360 |
| 0.40 | 8 | 8 | 0.233687 | -0.131223 | 0.321866 | 0.089546 | 0.658 |

This trace-only local MLP screen does not measure accumulated hidden-state drift, logit KL, NLL, or realized hardware traffic.
