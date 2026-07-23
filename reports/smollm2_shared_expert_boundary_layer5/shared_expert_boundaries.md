# Shared-plus-coarse-expert boundary training

Decision: **reject_or_continue_boundary_optimization**

The layout always reads 128 shared records and routes 32 of 96 cache-aligned experts (640 physical records total).

| Layer | Initial rel-L2 | Best rel-L2 | Final rel-L2 | Expert recall |
|---:|---:|---:|---:|---:|
| 5 | 0.660100 | 0.640998 | 0.640998 | 0.464756 |

Mean best relative L2: 0.640998 (target <= 0.200000).
Projected cold traffic: 0.437693x dense Q4.
