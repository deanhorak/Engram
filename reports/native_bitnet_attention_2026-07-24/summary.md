# Native BitNet Milestone 3 attention substitution

Date: **2026-07-24**

Decision: **bounded trained-model attention pass and incremental package
integration complete; native transformer fusion next.**

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
state transition and rerank now also have a C++20 implementation and C
ABI. They pass randomized native/NumPy parity, including cache eviction and
heavy replacement. On the trained one-sequence development protocol, native
substitution reaches KL 0.00528, top-1 0.96875, NLL +0.01239, and hidden L2
0.04210.

The standalone native benchmark keeps per-layer state fixed at 249,248 bytes:

| Context | Fraction of dense logical reads |
|---:|---:|
| 33 | 87.88% |
| 128 | 31.29% |
| 512 | 8.40% |
| 2,048 | 2.14% |

Those counts are at the query-head kernel interface and elapsed time includes
ctypes plus input generation.

The stateful kernel is now installed in every layer of the compiled package.
Prompt prefill and incremental decode share persistent native state. Explicit
absolute positions drive the existing BitNet RoPE implementation, and dense
Hugging Face KV caching remains disabled. Full-sequence and uneven incremental
chunks are bit-identical for the bounded operator.

Complete 30-layer package generation, including embeddings, projections,
packed native MLPs, final normalization, vocabulary projection, and greedy
selection, gives:

| Prompt tokens | Total seconds | Positions/s | Fraction of dense attention reads |
|---:|---:|---:|---:|
| 33 | 39.06 | 0.87 | 86.55% |
| 128 | 131.97 | 0.98 | 31.07% |
| 256 | 255.64 | 1.01 | 16.35% |

All-layer native attention state remains exactly 7,477,440 bytes. Packed MLP
calls consume only 5.69–12.65 seconds of these runs, identifying Python-side
projection/orchestration and the full vocabulary path as the next systems
targets. These are modeled logical reads rather than hardware DRAM counters,
and the deterministic repeated prompt is not new semantic evidence.

A subsequent position-major C ABI processes each complete prompt segment in
one native call per layer and is bit-identical to individual steps. The
controlled 256-token result is 254.23 seconds versus 255.64 seconds before
fusion, only a 0.55% improvement. A 33-token phase profile attributes 11.60
seconds to Q/K/V projection, 7.71 to output projection, 12.62 to the vocabulary
head, 5.94 to packed MLP calls, 0.12 to native attention, and 0.06 to RoPE.
This rejects further call-loop optimization and selects packed ternary
Q/K/V/O execution as the next native boundary.

That boundary is now implemented with one shared threaded kernel over the
official four-codes-per-byte projection tensors. The controlled 33-token run
falls from 38.51 to 22.29 seconds; Q/K/V/O time falls from 19.31 to 3.01
seconds, and generated tokens are unchanged. Direct 32-position development
parity against materialized projections gives KL 0.003945, top-1 0.96875, NLL
delta −0.000369, and final-hidden L2 0.035325. Frozen confirmation passes on
records 8–15 over
256 positions: KL 0.005478, top-1 0.957031, NLL delta +0.002001, and hidden L2
0.058874. Native projection execution takes 111.38 seconds versus 256.56
seconds materialized on the same tensor, so the path is promoted. The full tied
vocabulary projection now dominates the profile at 13.00 seconds.

Generation had been computing vocabulary logits for every prompt row.
Requesting `logits_to_keep=1` preserves exact full-vocabulary greedy selection
and removes that redundancy. The 33-token run falls from 22.29 to 10.16
seconds, with vocabulary time falling from 13.00 to 0.83 seconds. The
256-token run falls from 254.23 to 20.72 seconds (91.8%) and reaches 12.40
processed positions/second with identical generated tokens, bounded state,
and logical attention reads. Approximate vocabulary indexing is not promoted:
the exact final-row head is no longer the bottleneck, while an index would add
candidate-recall risk and duplicate storage.

Machine-readable evidence:
[frozen_streaming_c8_k4_confirmation.json](frozen_streaming_c8_k4_confirmation.json).
Native scaling evidence:
[native_long_context.json](native_long_context.json).
Complete generation evidence:
[end_to_end_long_context_generation.json](end_to_end_long_context_generation.json).
Stream-fused comparison:
[end_to_end_long_context_generation_stream_fused.json](end_to_end_long_context_generation_stream_fused.json).
Phase profile:
[end_to_end_generation_phase_profile.json](end_to_end_generation_phase_profile.json).
Packed projection timing:
[end_to_end_generation_native_projections.json](end_to_end_generation_native_projections.json).
Packed projection semantic parity:
[native_projection_parity.json](native_projection_parity.json).
Frozen packed projection confirmation:
[native_projection_frozen_confirmation.json](native_projection_frozen_confirmation.json).
Exact final-logit timing:
[end_to_end_generation_last_logit.json](end_to_end_generation_last_logit.json).
Optimized 256-token generation:
[end_to_end_long_context_optimized.json](end_to_end_long_context_optimized.json).
