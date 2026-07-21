# Correction-capsule residual sweep

Status: **reject_before_causal_intervention**

Uncorrected mean local relative L2: 0.206579

| Priority | Capsules | Rank | Corrected rel-L2 | Improvement | Hard-subset rel-L2 | Match | Correction MB/token |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 4 | 8 | 0.395818 | -0.916061 | 0.386964 | 0.899014 | 1.333 |
| 0.10 | 8 | 8 | 0.400711 | -0.939746 | 0.382208 | 0.861670 | 1.565 |
| 0.20 | 4 | 8 | 0.353986 | -0.713563 | 0.374729 | 0.905457 | 1.340 |
| 0.20 | 8 | 8 | 0.360052 | -0.742930 | 0.368405 | 0.897239 | 1.607 |
| 0.40 | 4 | 8 | 0.313413 | -0.517159 | 0.361575 | 0.894214 | 1.327 |
| 0.40 | 8 | 8 | 0.318858 | -0.543517 | 0.358620 | 0.873439 | 1.579 |

This trace-only local MLP screen does not measure accumulated hidden-state drift, logit KL, NLL, or realized hardware traffic.
