# DIP dual-layout diagnostic

Status: **rejected**

Version 3 duplicates gate/up weights in both coordinate-major and record-major order. The full
30-layer package grows from 318.8 MB to 531.2 MB (1.666x). At the low q=360/C=K=512 budget, six
20-pass synthetic trials give these medians:

| Completion layout | Sparse ns/layer | Dense/sparse speed | Estimated traffic |
|---|---:|---:|---:|
| Coordinate candidate gather | 1,092,855 | 0.912x | 0.775x dense |
| Coordinate full stream | 1,019,092 | 0.982x | 0.775x dense |
| Record omitted-coordinate gather | 1,271,880 | 0.815x | 0.719x dense on synthetic input |
| Record full-row stream | 1,259,280 | 0.845x | 0.750x dense |

The benchmark's deterministic period-29 hidden vector favors record-major accounting: its omitted
coordinates occupy only 31 of 36 lines at q=360. On 35,520 untouched confirmation state/layer
pairs, the omitted coordinates occupy 35.983/36 lines on average and 98.3% touch all 36. At the
quality-valid q=432 point the mean remains 35.658 lines. Real-trace record-major traffic is
therefore about 0.750x dense at the semantically failing 512-record budget, 1.028x at
q=360/C=1024/K=768, and 1.052x at the recommended quality-valid q=432/C=896/K=768.

Version 2 coordinate-major storage remains the default. Version 3 is available only through the
explicit `--dual-layout-experimental` flag so the negative result can be reproduced.
