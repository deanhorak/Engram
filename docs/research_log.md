# Research log

## 2026-07-20 — Cognitive Executive policy scaffold

- Separated the token-level recurrent controller from an optional request-level Oracle cognitive
  executive; `.engram` packages remain independent model workers.
- Added typed goal graphs, evidence-confidence summaries, attention budgeting, non-mutating memory
  curation proposals, predictive action selection, and observable-progress monitoring.
- Added compare-and-swap event streams with deterministic replay, immutable worker generations,
  pinned action attempts, predicted-budget reservation, idempotent matched outcomes, and
  prediction-versus-outcome calibration summaries.
- Added transactional SQLite and checksummed, fsynced JSONL stores using an allow-listed JSON
  codec, plus explicit worker adapters and identity-stamped structured outcomes.
- Kept dispatch and storage mutations outside the policy layer. Durable memory, tool/model
  production adapters, output-content validation, learned predictors, and confidence calibration
  remain unimplemented research work. External side effects remain at-least-once across crashes.
- Defined separate executive safety and evaluation gates; no model-quality or autonomous-agent
  claim follows from the deterministic scaffold.

## 2026-07-18 — Milestone 1 foundation

- Started from an empty workspace with no usable Git metadata.
- Established a NumPy reference package and dependency-light C++20 build.
- Defined the SwiGLU neuron decomposition and contribution-magnitude oracle.
- Chose residual energy as the explicit threshold definition and scan every prefix because
  contribution cancellation makes error non-monotonic.
- Added deterministic random Llama-shaped weights and checksummed sharded traces.
- Added exact local Hugging Face MLP hooks without Hub access.
- Exercised those hooks on a local two-layer `LlamaForCausalLM`: 8 layer-token records had
  p95 extracted-weight/teacher MLP relative L2 error of approximately `1.58e-7`.
- Fixture pipeline measurements are recorded as negative-neutral engineering evidence only;
  they are not evidence for the Engram hypothesis.
- Background operators were intentionally deferred according to the milestone order. This
  means scientific Gate 1 is not yet complete.
- Native build succeeded. The default user-local `ctest` launcher was broken because its
  Python `cmake` module is missing; `/usr/bin/ctest` passed.
- Hardware audit found AVX but no AVX2 and no accessible performance counters. CUDA is not
  usable from PyTorch in this environment. No performance claim was made.

## 2026-07-18 — Milestone 2 semantic baseline

- Added deterministic per-dimension scalar key quantization and additive residual vector
  codebooks for values, with strict metadata and corruption validation.
- Added a brute-force joint gate/up geometry router and exact candidate-only reranking.
- Added no-background and fitted low-rank linear residual operators.
- Packed exact reference arrays and quantized variants into checksummed, mmap-readable
  semantic layer directories.
- On the random fixture with K=8 and 16 candidates, mean oracle relative L2 was 0.370,
  practical routing relative L2 was 0.627, and candidate recall was 0.680.
- The rank-4 background worsened held-out fixture error to 5.76. This is recorded as an
  overfitting failure on a tiny random calibration set, not presented as a successful model.
- End-to-end logit impact remains not run because an evaluation-only transformer substitution
  path has not yet been built.

## 2026-07-18 — Milestone 3 episodic baseline

- Added exact stable local attention with an incremental bounded cache.
- Added normalized decayed recurrent linear attention with constant-size numerator and
  normalization state.
- Added a fixed-capacity older-context ring with int8 key/value storage, brute-force candidate
  search, exact reranking, recall metrics, and logical byte counters.
- On deterministic synthetic length-128 states, mean relative L2 versus full causal attention
  was 0.974 local-only, 0.835 recurrent-only, and 0.456 for the heuristic hybrid.
- Controlled older-token retrieval/copying reached 1.0, but trained teacher attention traces
  have not run and the result is explicitly labeled synthetic pipeline validation.

## 2026-07-19 — End-to-end runtimes and later gates

- Added shared GRU controller, stage embeddings/adapters, adaptive/fixed cycles, vocabulary MIPS,
  transition caches, and correction-capsule primitives.
- Added a resumable, checksummed directory compiler and PyTorch-free Python generation. Packages
  continue to generate after the source checkpoint directory is moved away.
- Added a std-only C++20 package parser/SHA-256 verifier, NPY memory mapping, semantic reads,
  bounded hybrid episodic memory, controller, vocabulary search, transition cache, and stable CLI.
- Python/native exact-vocabulary fixture generation is token-identical in integration tests.
- Safe AVX2 dot-product dispatch is compiled, but the Ivy Bridge host selects scalar as expected.
- After switching both runtimes to the quantized-only semantic path, a fair 512-token fixture
  benchmark with transition caching bypassed measured 189.5 Python versus 9,281 native tokens/s
  and 38.6 versus 7.7 MiB peak RSS. This is not a large-model result.
- Synthetic adaptive control averaged 7.98/8 cycles, providing essentially no early-exit saving.
- A random one-layer Llama Gate 5 run produced zero target accuracy in code, reasoning, factual,
  and long-context categories plus 93.75% repetition. The small KL was caused by near-uniform
  random logits and is explicitly not treated as success.
- Replaced exact runtime semantic arrays with memory-mapped affine uint8 key codes and additive
  value codebooks in both Python and native paths; parity remains green. At that checkpoint,
  trained distillation, indexed routing, and the 10x DRAM-traffic goal remained open.
- Logical byte accounting for the tiny fixture estimates 14,432 Engram semantic-plus-vocabulary
  bytes per token versus a 3,656-byte dense-Q4 read-once payload. It also activates exactly 25%
  of semantic records. These fixture gates fail; hardware-counter DRAM traffic remains unmeasured.

## 2026-07-19 — Indexed semantic routing

- Added deterministic joint gate/up IVF construction with float32 coarse centroids and uint32
  CSR postings. Python and native runtimes score all coarse centroids, expand probes only until
  enough records are available, score quantized keys only in those postings, and exactly rerank
  the candidate SwiGLU contributions.
- Moved full CSR/permutation validation to package load so native token search does not hide an
  O(record-count) validation scan. Runtime scratch remains preallocated.
- The refreshed random-fixture Gate 2 result is negative: 18.25/32 records were proxy-scored on
  average, candidate recall fell to 0.578, sparse relative L2 was 0.693, and the background
  worsened it to 7.09. This is an indexed implementation result, not evidence of useful quality.
- At the semantic-only IVF checkpoint, a 512-token cache-bypassed runtime benchmark measured
  140.4 Python and 9,433 native
  tokens/s (38.8 and 7.9 MiB peak RSS). On this tiny fixture the IVF path estimates 21,984
  semantic-plus-vocabulary logical bytes per token, 6.01x the dense-Q4 read-once payload, because
  coarse-centroid overhead dominates. The 10x-lower target remains failed/unproven.

## 2026-07-19 — Indexed vocabulary routing

- Added deterministic normalized-embedding IVF to the compiler and both runtimes, with float32
  centroids, uint32 CSR postings, adaptive probe expansion, posted-row proxy scoring, exact
  original-embedding candidate rescoring, and a dense exact fallback.
- Python/native approximate fixture generation is token-identical. Each token proxy-scores and
  exactly rescored 32/64 vocabulary rows after 32 singleton-list probes.
- The refreshed cache-bypassed benchmark measured 160.3 Python and 10,373 native tokens/s with
  36.2 and 8.2 MiB peak RSS. Combined semantic/vocabulary logical traffic is 26,464 bytes/token,
  7.24x the dense-Q4 read-once payload on this tiny fixture; this remains a failed traffic gate.
