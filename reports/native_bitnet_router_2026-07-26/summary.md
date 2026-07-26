# Native BitNet practical-router screening

Date: **2026-07-26**

This file records the initial, historical router screen. It is superseded by
the all-layer native development result described below: the practical DIP
implementation passed the qualifying development gate and subsequently passed
the native-BitNet semantic-memory gate by postmortem adjudication of its
consumed final attempt.

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

This development result froze the exact policy in
[`../native_bitnet_m2_2026-07-26/frozen_dip_policy.json`](../native_bitnet_m2_2026-07-26/frozen_dip_policy.json);
timed sparse development was 1.1565x dense.

The subsequent independent 8-sequence/256-position raw final report passed:
KL 0.00404129, top-1 0.98828125, NLL +0.00482893, hidden L2 0.0477494,
21.3800% mean active records, 41.1371% modeled traffic, 99.9406% global
candidate recall, and 99.3943% worst-layer mean recall. The original wrapper
still ended in error because its verifier compared canonical full-record
`input_ids` object hashes with first-33-token bare-list hashes. A separate
no-model postmortem adjudicator corrected that contract and verified the
preserved evidence, producing a semantic-gate pass-by-adjudication.

This is not a pristine runner pass: the raw report was prospectively sealed
about 13 minutes after the error rather than contemporaneously bound by the
original result. The evidence is host-bound, final scale is only 8x32, and
traffic remains modeled cache-line accounting rather than measured DRAM.
Final sparse timing was 1.1449x dense; latency was not a frozen gate. See the
[final audit summary](../native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/summary.md).
