# Native BitNet practical-router screening

Date: **2026-07-26**

Milestone 2 remains blocked, but an analytic practical-router candidate now
passes held-out recall and modeled-traffic screens at representative early,
middle, and late layers.

## Rejected learned router

A nonlinear low-rank router was trained directly against exact oracle
membership on the existing 1,024-position training trace and measured on the
separate 256-position validation trace. Increasing rank from 64 to 256 raised
recall with 1.5x candidates only from 74.33% to 77.75% on layer 14. This is far
below the 95% recall gate despite adequate traffic headroom, so the direct
membership-prediction family is stopped.

## Dynamic Input Pruning result

The stronger router selects the largest-magnitude 75% of input coordinates,
scores ternary gate/up keys in a coordinate-major index, and fetches complete
packed records for exact candidate reranking. Stable top-K ordering is required
because ReLU-squared gating creates exact-zero ties.

| Layer | Oracle K | Candidate multiplier | Mean recall | Modeled Q4 fraction |
|---:|---:|---:|---:|---:|
| 0 | 1,728 | 1.50x | 96.23% | 35.02% |
| 14 | 1,728 | 1.50x | 98.06% | 35.02% |
| 29 | 2,074 | 1.75x | 96.78% | 41.03% |

The modeled numerator includes the coordinate-major packed gate/up index and
complete packed gate/up/gain/down candidate records. It does not yet include
headers or alignment, so the late-layer configuration retains only about four
percentage points of provisional headroom.

## Next gate

The configuration must now pass an all-layer recall sweep using the
layer-adaptive oracle K schedule. If it does, the next implementation is a
native selected-record kernel that computes candidate coefficients, estimates
the coupled intermediate RMS/Q8 scales without a dense scan, reranks, and runs
the frozen all-layer causal protocol. Only the serialized kernel's complete
cold traffic and measured CPU latency can close Milestone 2.
