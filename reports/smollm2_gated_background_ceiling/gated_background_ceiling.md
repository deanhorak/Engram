# Exact top-K plus gated-background ceiling

Decision: **reject_gated_background**

Mean validation relative L2: 0.149588 → 0.145296.
Projected selected + background + rank-16 router traffic: 0.429977× dense.

| Layer | Sparse | Corrected | Residual prediction |
|---:|---:|---:|---:|
| 0 | 0.070277 | 0.068121 | 0.999867 |
| 7 | 0.233296 | 0.230187 | 0.985781 |
| 14 | 0.210239 | 0.205250 | 0.976553 |
| 21 | 0.190898 | 0.184367 | 0.966709 |
| 29 | 0.043231 | 0.038553 | 0.874181 |
