# Native BitNet Milestone 3 attention substitution

Date: **2026-07-24**

Decision: **bounded trained-model attention pass; native optimization next.**

Development evaluation rejected an all-layer 16-token local window
(KL 0.2031, top-1 0.8594) and normalized recurrent attention (KL 2.6883,
top-1 0.1875). Per-head analysis showed that 83% of development heads placed
at least 20% of their attention mass outside the local window.

The promoted hybrid uses one exact sparse softmax over the 16 local keys and
the four highest-scoring older keys. Unlike the earlier fixed blend, local and
retrieved values share one normalization denominator.

The frozen confirmation uses records 8–15, which were not used to select the
operator, and exactly 256 prediction positions:

| Metric | Threshold | Result |
|---|---:|---:|
| KL | at most 0.05 | 0.002494 |
| Teacher top-1 agreement | at least 0.90 | 0.996094 |
| NLL delta | at most +0.05 | +0.007099 |
| Final-hidden relative L2 | at most 0.10 | 0.043498 |

All semantic checks pass. On this split, 92.5% of heads place at least 20% of
their mass outside the local window, confirming that older-token retrieval is
not optional.

The exact selector was not promoted because it scans every older key. Random
multi-table sign-LSH then reached only 58.8–65.6% oracle top-k recall. Exact
coordinate-box and centroid-radius page bounds retained 100% recall but opened
about 94% of pages, reaching 105.0% and 100.3% of dense logical traffic after
metadata.

The promoted streaming hybrid retains:

- the exact 16-token local window;
- two initial attention-sink entries;
- six online cumulative-attention heavy hitters;
- exact reranking of those eight old keys to four transferred values.

It never scans an evicted key. The fixed policy was selected on records 0–1
and run once on frozen records 8–15:

| Metric | Threshold | Result |
|---|---:|---:|
| KL | at most 0.05 | 0.014093 |
| Teacher top-1 agreement | at least 0.90 | 0.941406 |
| NLL delta | at most +0.05 | −0.006133 |
| Final-hidden relative L2 | at most 0.10 | 0.085589 |

All semantic and evidence checks pass. At the deliberately short 33-token
protocol, complete modeled logical KV traffic is 93.34% of dense. The systems
benefit is that old-context storage and reads remain fixed as context grows;
this test does not establish native latency or hardware DRAM reduction. The
current implementation is a Python evaluation reference. The next step is a
native sink/heavy-hitter cache plus exact eight-to-four rerank and a
long-context hardware benchmark.

Machine-readable evidence:
[frozen_streaming_c8_k4_confirmation.json](frozen_streaming_c8_k4_confirmation.json).
