# Research log

## 2026-07-20 — Predictor-free Dynamic Input Pruning crosses the semantic gate

- Audited the failed sparse-teacher pilot before spending more training compute. The hard
  `argsort`/gather/scatter route prevents local-MLP, hidden-state, and logit losses from updating
  router scores; only the auxiliary membership BCE reaches the router. The trained rank-8 update
  contributes about 0.0017% of the held-out sparse MLP output norm, and its candidate sets remain
  more than 99.4% identical to the initialization. The pilot is therefore an under-trained,
  non-differentiable-routing result, not evidence that sparse distillation is exhausted.
- Reviewed primary sparse-MLP work before choosing another implementation. Post-hoc
  [CMoE](https://aclanthology.org/2026.acl-long.218/)-style coactivation groups were far outside
  the local-error budget on representative layers, while
  [LTE](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b8f10193cab43d45df9bb810637333fd-Abstract-Conference.html)'s
  soft-to-hard expert training remains a credible but substantially more expensive fallback.
  Qualcomm's [Dynamic Input Pruning](https://arxiv.org/abs/2412.01380) directly addresses the
  difficulty of predicting SwiGLU activation sparsity and motivated the predictor-free path below.
- Added DIP-inspired primitives. The published method motivates dynamic top-magnitude input
  pruning and partial activation scoring; Engram adds candidate-only completion and exact
  contribution-norm reranking. For each token and layer, retain the `q` largest absolute
  coordinates of the MLP input, use those coordinates with the source gate/up weights to
  approximate every intermediate activation, retain `C` candidates, complete the omitted gate/up
  products only for those candidates, and exactly rerank them to `K` records. No learned router or
  calibration fit is required.
- Added a trace-only 2-D sweep that reports oracle-membership recall, retained oracle score mass,
  exact-reranked local MLP error, layer minima, and the logical float32 weight-read model
  `2*I*q + 2*C*(H-q) + K*H` versus dense `3*I*H`. The evaluator uses dense NumPy operations, so
  these are projected sparse-kernel reads rather than measured DRAM traffic or latency.
- Across all 30 layers and 507 development states, `q=432` (75% of the 576 input coordinates),
  `C=1,024`, and `K=768` recovers 99.708% of oracle records and 99.890% of their contribution-score
  mass. Mean local relative L2 is 0.10438 versus 0.10402 for the full-information K=768 reference,
  at a projected 77.78% of dense MLP weight traffic.
- The all-layer development intervention passes at that point: KL 0.03164, top-1 agreement
  0.92464, NLL delta +0.03081, final-hidden relative L2 0.09208, and causal-state candidate recall
  0.99707. Identity and the matched K=768 reference also pass on 16 unique sequences and 491
  next-token positions.
- A lower-traffic causal frontier finds two additional passes. The conservative recommended point
  is `q=432/C=896/K=768`: recall 0.98993, KL 0.03386, top-1 0.91242, NLL delta +0.02616,
  final-hidden relative L2 0.09383, and 76.39% projected dense traffic (1.31x reduction). A
  `q=360/C=1,024` arm reaches 75.00% traffic but has less gate margin. The next cheaper checked
  arms fail, locating rather than guessing at the checked-grid quality/projected-traffic boundary.
- Froze `q=432/C=896/K=768` and evaluated it once on a newly written 16-sequence confirmation
  corpus with 1,184 input states and 1,168 next-token positions. Exact token-sequence hashing
  finds zero overlap with both the 32-sequence calibration set and the 16-sequence
  configuration-selection set. The confirmation arm also passes: recall 0.98971, score-mass
  recall 0.99613, KL 0.02864,
  top-1 0.91010, NLL delta +0.03261, and final-hidden relative L2 0.09048.
- The confirmation decision is now `eligible_for_selector_serialization` for the predictor-free
  arm. This crosses the declared semantic quality prerequisite; it does not establish a runtime
  speedup, broad model generalization, or the long-term 10x DRAM target. DIP-aware packed weights
  and a native sparse kernel, followed by measured traffic and replication on another model, are
  still required before trained-package compilation or a systems claim.

## 2026-07-20 — Trained-teacher MLP intervention and semantic stop gate

- Added a Hugging Face forward-hook evaluator that keeps the trained transformer intact while
  replacing selected MLP outputs under teacher forcing. It reports local MLP error, final
  normalized-hidden-state drift, teacher-to-student logit KL, top-1/top-5 agreement, target NLL/perplexity,
  candidate recall, and projected parameter/key/value traffic. Identity hooks reproduce the
  teacher exactly on the checked corpus.
- Added direct factored fitting for low-rank multi-label ridge routers. QR decompositions reduce
  the compression SVD to the calibration rank (128-by-128 here) while matching dense-fit-then-SVD
  scores in tests.
- On 16 held-out SmolLM2-135M sequences (491 next-token positions), all-layer exact magnitude
  oracles measured:
  - K=256: KL 0.648, top-1 agreement 0.605, NLL +0.668, final-hidden rel-L2 0.357;
  - K=512: KL 0.132, top-1 agreement 0.809, NLL +0.085, final-hidden rel-L2 0.179;
  - K=768: KL 0.032, top-1 agreement 0.923, NLL +0.022, final-hidden rel-L2 0.092;
  - K=1,024: KL 0.006, top-1 agreement 0.976, NLL -0.002, final-hidden rel-L2 0.041.
- Declared the progression gate before interpreting routed arms: KL at most 0.05, top-1 at least
  0.90, NLL delta at most +0.05, final-hidden rel-L2 at most 0.10, and routed candidate recall at
  least 0.95. K=768, or 50% of the 1,536 records, is the first tested magnitude-reference pass.
- Hardened that gate with metric-domain and routing-budget validation, a minimum evidence floor of
  8 unique sequences and 256 next-token positions, exact token-sequence calibration/evaluation
  separation, and a provenance-checked composite mode for arms run in stages. All 32 calibration
  and 16 evaluation sequences are unique, with zero exact token-sequence overlap between splits.
- Clarified that the full-information magnitude top-K arm is a reference, not a theoretical quality
  ceiling: vector cancellation means another K-record subset can in principle do better.
- At K=768, the flat rank-16 router reached 0.722 recall with 1,024 candidates and 0.867 with
  1,280. The 1,280-candidate arm still produced KL 0.650, top-1 0.619, and NLL +0.670.
- Implemented alternating coverage-trained overlapping postings with deterministic full record
  coverage, bounded replication, learned rank-16 group selection, deduplication, and exact local
  reranking. The checked 192-by-32 layout reached only 0.858 recall at 1,280 candidates,
  KL 0.794, top-1 0.619, and NLL +0.825. It scanned about 1,954 posting entries per layer.
- Both routed arms, trained on 128 of 1,112 available calibration states per layer, fail. The gate
  decision is `stop_before_serialization`; no router artifact was written, and attention/controller
  distillation or trained-package compilation was not started. The next semantic experiment should
  test the full expanded calibration corpus before deciding whether to change the representation
  or training objective.
- Completed that full-corpus refit on all 1,112 states per layer. The low-rank fitter now chooses
  the smaller exact ridge system: dual for fewer examples than hidden dimensions and primal for
  more examples, with an exact truncated factorization of the resulting weight matrix.
- At K=768/C=1,280, flat rank-16 recall improved from 0.867 to 0.889, but KL worsened from 0.650
  to 0.789 and NLL from +0.670 to +0.764. The overlap router improved recall from 0.858 to 0.868,
  while KL worsened from 0.794 to 1.149 and NLL from +0.825 to +1.095. Both full-corpus arms fail
  every routed-arm gate, so the decision remains `stop_before_serialization`.
- More examples alone have therefore reached diminishing returns for these configurations. Before
  another expensive causal run, screen corpus-scaled regularization with cached oracle membership;
  if that does not materially improve held-out recall, move to layer-adaptive budgets, a learned
  residual path, or sparse-teacher fine-tuning.
- Added a trace-only rank-router sweep with exact token-sequence separation and reusable per-layer
  packed-bit oracle-membership caches. It screens recall without rerunning the transformer and
  explicitly does not substitute for causal quality evaluation.
- At C=1,280, λ=8,000 is the shallow optimum: held-out dense-state recall is 0.900 versus 0.893 at
  λ=1,000. λ values 3,000, 8,000, 10,000, and 20,000 do not approach the 0.95 gate.
- A cached candidate frontier at λ=8,000 reaches 0.954 recall at C=1,408 and 0.978 at C=1,472.
  Both warranted causal checks, but C=1,408 produced KL 0.146, top-1 0.829, NLL +0.116, and
  final-hidden rel-L2 0.163. C=1,472 improved those to KL 0.085, top-1 0.866, NLL +0.055, and
  rel-L2 0.131; it still fails every causal check despite passing recall.
- C=1,472 reads 95.8% of record keys and projects to only about 1.24× key/value traffic reduction
  before router overhead. Further candidate or layer-adaptive expansion would be functionally
  near-dense, so this flat rank-16 configuration is abandoned. The next semantic work must test a
  different representation or training process, starting with learned residual correction or
  sparse-teacher fine-tuning.
- Extended correction capsules with backward-compatible affine residual prediction, parameter
  accounting, deterministic fitting, and selective application to a separate MLP output using the
  MLP input as the selector state. Added exact adaptive primal/dual low-rank residual fitting.
- Added a held-out trace screen that fits the exact dense-minus-routed residual for the λ=8,000,
  K=768/C=1,280 router. It reports overall and hard-subset error, radius coverage, correction
  traffic/MACs, and blocks causal work unless local error materially improves.
- Global 1/4/8-capsule layouts at ranks 8/16 all fail. The least harmful global arm raises mean
  local relative L2 from 0.207 to 0.259; stronger ridge regularization is worse.
- Failure-region capsules trained on the hardest 10%, 20%, or 40% of calibration states also fail.
  Tight radii reduce matches to 7–12% and correction traffic to 0.36–0.69 MB/token, but the best
  result still raises relative L2 to 0.233. No capsule arm proceeds to causal integration or
  serialization. The next major semantic experiment is sparse-teacher fine-tuning.
- Added a sparse-teacher trainer with an immutable dense teacher and frozen-base student. Each
  sparse student MLP has trainable rank-16 router factors and a rank-8 down-projection adapter;
  hard routing is supervised by oracle-membership BCE alongside normalized local-MLP,
  hidden-state, and logit-KL distillation losses.
- Added exact calibration/validation token-sequence separation, gradient clipping, a safe
  router/adapter-only safetensors artifact, and the existing intervention thresholds plus evidence
  floor as the training progression gate. A one-record real-model smoke run verifies autograd,
  artifact writing, and gate reporting.
- The first full pilot runs 32 optimizer steps and evaluates all 16 held-out sequences/491
  next-token positions. Training loss declines from 0.436 to 0.326. Held-out recall is 0.900,
  KL 0.448, top-1 agreement 0.721, NLL delta +0.343, and final-hidden rel-L2 0.250. Every routed
  check fails, so the artifact remains `stop_before_serialization`.

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
- At that checkpoint end-to-end logit impact had not run; the trained-teacher substitution path
  was added on 2026-07-20 and records the negative progression result above.

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
