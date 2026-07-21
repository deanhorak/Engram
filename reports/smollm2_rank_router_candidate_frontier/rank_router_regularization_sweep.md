# Rank-router regularization sweep

Status: **eligible_for_causal_intervention**

Rank: 16; top-K: 768.

| Regularization | Candidates | Mean recall | Minimum layer mean | Recall gate |
|---:|---:|---:|---:|---|
| 8000 | 1280 | 0.900080 | 0.883185 | fail |
| 8000 | 1344 | 0.927592 | 0.914422 | fail |
| 8000 | 1408 | 0.953808 | 0.944537 | pass |
| 8000 | 1472 | 0.978369 | 0.973216 | pass |

Packed membership cache: `/home/dean/repos/engram/work/smollm2-rank-router-membership-cache` (30 hits, 0 misses).

This recall-only screen does not measure logit KL, NLL, hidden-state drift, latency, or realized memory traffic.
