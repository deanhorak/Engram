# Native-gate low-rank utility residual

Decision: **reject_low_rank_utility_residual**

Baseline local relative L2: 0.385510.

| Rank | Blend | Local rel-L2 | Oracle recall | Traffic | Pass |
|---:|---:|---:|---:|---:|---|
| 16 | 0.50 | 0.355144 | 0.616858 | 0.443866× | no |
| 8 | 0.50 | 0.356119 | 0.615400 | 0.437500× | no |
| 16 | 1.00 | 0.359897 | 0.617302 | 0.443866× | no |
| 8 | 1.00 | 0.359906 | 0.616330 | 0.437500× | no |
| 16 | 0.25 | 0.362483 | 0.607157 | 0.443866× | no |
| 8 | 0.25 | 0.363384 | 0.605948 | 0.437500× | no |

This cached trace screen is not an all-layer causal intervention.
