# Native-gate low-rank utility residual

Decision: **integrate_low_rank_utility_residual**

Baseline local relative L2: 0.385510.

| Rank | Blend | Local rel-L2 | Oracle recall | Traffic | Pass |
|---:|---:|---:|---:|---:|---|
| 23 | 0.80 | 0.335859 | 0.646780 | 0.449436× | yes |
| 23 | 0.65 | 0.337399 | 0.642888 | 0.449436× | yes |
| 16 | 0.80 | 0.338007 | 0.643131 | 0.443866× | yes |
| 16 | 0.65 | 0.339529 | 0.639397 | 0.443866× | yes |
| 23 | 0.50 | 0.340624 | 0.636462 | 0.449436× | yes |
| 8 | 0.80 | 0.342090 | 0.637050 | 0.437500× | yes |
| 16 | 0.50 | 0.342705 | 0.633429 | 0.443866× | yes |
| 8 | 0.65 | 0.343328 | 0.633767 | 0.437500× | yes |
| 8 | 0.50 | 0.346379 | 0.628467 | 0.437500× | yes |
| 23 | 0.35 | 0.347381 | 0.626983 | 0.449436× | no |
| 16 | 0.35 | 0.349086 | 0.624664 | 0.443866× | no |
| 8 | 0.35 | 0.352413 | 0.620563 | 0.437500× | no |

This cached trace screen is not an all-layer causal intervention.
Deployment selection: rank 16, blend 0.80.
