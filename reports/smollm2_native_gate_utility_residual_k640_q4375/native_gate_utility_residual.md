# Native-gate low-rank utility residual

Decision: **integrate_low_rank_utility_residual**

Baseline local relative L2: 0.366737.

| Rank | Blend | Local rel-L2 | Oracle recall | Traffic | Pass |
|---:|---:|---:|---:|---:|---|
| 23 | 0.80 | 0.328929 | 0.671661 | 0.442491× | yes |
| 23 | 1.00 | 0.329281 | 0.673147 | 0.442491× | yes |
| 23 | 0.65 | 0.329533 | 0.669095 | 0.442491× | yes |
| 16 | 0.80 | 0.330390 | 0.669259 | 0.436921× | no |
| 16 | 1.00 | 0.330866 | 0.670503 | 0.436921× | no |
| 16 | 0.65 | 0.331116 | 0.666791 | 0.436921× | no |
| 23 | 0.50 | 0.331723 | 0.664762 | 0.442491× | no |
| 16 | 0.50 | 0.333133 | 0.662655 | 0.436921× | no |

This cached trace screen is not an all-layer causal intervention.
Deployment selection: rank 23, blend 0.80.
