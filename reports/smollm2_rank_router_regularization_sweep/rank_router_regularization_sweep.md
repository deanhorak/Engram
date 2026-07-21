# Rank-router regularization sweep

Status: **reject_before_causal_intervention**

Rank: 16; top-K: 768.

| Regularization | Candidates | Mean recall | Minimum layer mean | Recall gate |
|---:|---:|---:|---:|---|
| 1000 | 1024 | 0.766147 | 0.742537 | fail |
| 1000 | 1280 | 0.892583 | 0.879158 | fail |
| 3000 | 1024 | 0.776789 | 0.748806 | fail |
| 3000 | 1280 | 0.899051 | 0.883039 | fail |
| 8000 | 1024 | 0.777852 | 0.749230 | fail |
| 8000 | 1280 | 0.900080 | 0.883185 | fail |
| 10000 | 1024 | 0.776932 | 0.748742 | fail |
| 10000 | 1280 | 0.899706 | 0.883146 | fail |
| 20000 | 1024 | 0.771948 | 0.745465 | fail |
| 20000 | 1280 | 0.897293 | 0.881143 | fail |

Packed membership cache: `/home/dean/repos/engram/work/smollm2-rank-router-membership-cache` (0 hits, 30 misses).

This recall-only screen does not measure logit KL, NLL, hidden-state drift, latency, or realized memory traffic.
