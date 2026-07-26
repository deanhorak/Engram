# Native BitNet practical-router screening

Date: **2026-07-26**

This file records the initial, historical router screen. It is superseded by
the all-layer native development result described below: the practical DIP
implementation now passes the qualifying development gate, while Milestone 2
remains pending its sealed one-shot final confirmation.

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

## Subsequent outcome

The all-layer sweep, adaptive-K fit, coupled RMS estimator, source-bound
coordinate index, and memory-mapped CPU selected-record kernel were completed.
On the declared 8-sequence/256-position development corpus, the frozen policy
reached KL 0.0044707, top-1 agreement 0.94921875, NLL delta +0.0013609,
final-hidden relative L2 0.0498965, 20.08072% mean active records, 40.9639%
modeled physical cold traffic, 99.95917% global candidate recall, and
99.39353% worst-layer mean recall. Python/native output and route identities
are bit-exact on six live rows per layer.

This is a development-gate pass, not the final Milestone 2 result. The exact
policy is frozen in
[`../native_bitnet_m2_2026-07-26/frozen_dip_policy.json`](../native_bitnet_m2_2026-07-26/frozen_dip_policy.json);
the independent final holdout remains sealed. Timed sparse development was
1.1565x dense, and the traffic figure is modeled cache-line accounting rather
than measured DRAM.
