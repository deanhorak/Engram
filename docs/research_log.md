# Research log

## 2026-07-24 — Bounded streaming attention passes frozen confirmation

- Rejected random multi-table sign-LSH after development recall remained
  0.588–0.656. Causal quality could survive those misses, but the selector did
  not meet the declared recall standard and did not improve short-context
  traffic.
- Implemented exact branch-and-bound page indexes with coordinate boxes and
  centroid-radius bounds. Both preserve top-k exactly, but open about 94% of
  pages. Metadata raises modeled traffic to 105.0% and 100.3% of dense
  respectively, so the page-index branch is closed.
- Implemented a bounded H2O/attention-sink-inspired streaming cache. Each head
  retains the 16-token exact local window, two initial sinks, and six online
  cumulative-attention heavy hitters. Eight old keys are exact-reranked and
  only four old values enter the joint softmax. Evicted keys are never scanned.
- The fixed C=8/K=4 policy passes frozen records 8–15 over 256 prediction
  positions: KL 0.01409, top-1 0.94141, NLL delta −0.00613, and final-hidden
  relative L2 0.08559. All semantic and evidence checks pass.
- The short 33-token protocol models at 93.34% of dense logical KV traffic;
  the advantage is bounded old-context state and reads rather than a large
  short-prompt saving. The Python reference is not a latency result. The next
  step is native cache/rerank integration and long-context hardware validation.

## 2026-07-24 — Source-independent package and Milestone 3 semantic pass

- Added the `engram-native-bitnet` version-1 compiler and validator. The
  checked package contains 332 non-MLP tensors, tokenizer/configuration assets,
  and the 318,924,544-byte phase-stream artifact. All 210 source MLP tensors
  are excluded; every file is checksummed.
- Added a CPU generation runtime that builds the transformer on empty storage,
  materializes only packaged tensors, preserves native BitNet activation
  quantization for attention projections, and installs the memory-mapped C++
  MLP kernel in every layer. Package-backed and source-backed kernel execution
  has bit-exact hidden/logit parity. Greedy generation after `The capital of
  France is` produces ` Paris.`.
- Added trained-model local, recurrent, retrieval, and hybrid attention
  substitution with native Q/K/V, RoPE, and GQA semantics. Development rejected
  local-only and recurrent-only operators.
- Froze the joint-softmax W=16/K=4 hybrid and evaluated records 8–15: KL
  0.002494, top-1 0.996094, NLL delta +0.007099, and hidden L2 0.043498 over
  256 positions. All semantic criteria pass.
- The exact selector scans every older key and models at 91.89% of dense
  logical KV traffic. Milestone 3 therefore has a semantic progression pass,
  not a systems pass. The authorized next work is indexed candidate
  generation followed by exact reranking.

## 2026-07-24 — Direct native-BitNet CPU kernel passes the frozen gate

- Added a fail-closed C++20 reader for `native_bitnet_phase_base3_v1`. It
  memory-maps the artifact, validates its headers, directory, offsets, padding,
  BF16 metadata, and canonical base-3 streams, and never constructs dense
  projection weights.
- Implemented the complete BF16 MLP path over phase streams: per-token Q8
  input, fused gate/up ternary accumulation, ReLU squared, intermediate RMS
  normalization and gain, second Q8 quantization, and the transposed ternary
  down pass. A persistent thread pool and C ABI expose it to PyTorch model
  substitution while recording bytes, scratch, rows, threads, and time for
  every layer call.
- The deterministic tiny-artifact integration test is bit-exact against the
  independent dense artifact oracle. Official layers 0/14/29 differ from
  PyTorch BF16 GEMM reduction order by 0.00982, 0.00890, and 0.00684 relative
  L2 respectively; this distinction is recorded rather than described as
  bit parity.
- Ran the sealed CPU-only protocol using the pinned model and artifact hashes,
  the pinned tokenizer with its regex compatibility fix, the first eight
  unique frozen records, and exactly 256 prediction positions. The result is
  KL 0.003710, top-1 0.960938, NLL delta +0.002237, and final-hidden relative
  L2 0.046775. Every semantic threshold passes.
- The executed artifact schedules 318,924,544 cold bytes, or 40.052694% of
  dense ideal Q4, with zero dense-weight materialization. The 30 internal MLP
  timings total 9.737 seconds for 264 rows. This is exact scheduled traffic,
  not a hardware memory-controller measurement.
- The separate low-bit-native track therefore passes the Milestone 2 gate.
  Dense-Llama conversion remains blocked. See the
  [direct-kernel confirmation](../reports/semantic_gate_native_bitnet_2026-07-24/summary.md).

## 2026-07-23 — Exact native-BitNet record feasibility pass

- Preserved the unchanged 45% dense-ideal-Q4 traffic rule instead of relaxing
  it to the 83.33% DIP frontier. Primary-source and checkpoint inspection
  selected Microsoft's natively trained `bitnet-b1.58-2B-4T` as a separate
  source track, pinned to revision
  `04c3b9ad9361b824064a1f25ea60a8be9599b127`.
- Added a metadata-only, fail-closed adapter. It accepts only
  `BitNetForCausalLM`, ReLU-squared activation, offline `AutoBitLinear`
  quantization, and the official packed dimensions. It does not add BitNet to
  the existing SiLU/SwiGLU compiler, and matching metadata alone does not
  establish native-training provenance.
- Verified the downloaded 1,178,623,988-byte safetensors checkpoint at
  SHA-256
  `8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
  The fail-closed check caught and rejected an initially transcribed
  63-character digest before any conversion result was accepted.
- The official four-trits-per-byte MLP payload is 398,546,100 bytes, or
  50.0521% of the frozen denominator, so merely choosing a low-bit source
  would not pass. Added a lossless phase-stream format with five base-3 trits
  per byte. Each 1,538-byte logical record contains gate/up rows, a transposed
  down column, and the BF16 intermediate-normalization gain. The four physical
  streams are independently fixed-stride and cache aligned.
- The independently reloaded 30-layer artifact is 318,924,544 bytes
  (40.0527% of dense Q4 and 80.0220% of the native Hugging Face payload).
  Its exact gate/up, gain, and down phase order can model one cold read per
  serialized line at the same 40.0527%. Charging all independently scattered
  logical records by touched cache lines is 41.6673%. Both calculations
  include headers, directory, scales, gains, and padding. A physically
  interleaved draft was rejected because exact normalization would have
  forced multipass line rereads.
- Validated every source two-bit digit, every canonical base-3 tail, all
  1,592,524,800 reconstructed ternary coefficients, and 207,450 BF16 values.
  Source and artifact logical streams share SHA-256
  `27243c2304a2ad1b7dc87deb15eee1572ef6ace1a6c16ecbf5c485fd33f4f89d`.
- Added a CPU BF16 parity oracle matching activation quantization,
  ReLU-squared gating, intermediate RMS normalization, and scaled ternary
  linears. Layers 0/14/29 are bit-identical, and an all-30-layer causal smoke
  substitution has zero hidden/logit error, KL 0, and top-1 agreement 1.0.
- The result advances to a direct packed CPU kernel, not to compilation. The
  parity oracle materializes dense BF16 matrices and the causal smoke input
  does not meet the frozen evidence floor. This track also does not solve
  dense-Llama conversion. See the
  [native BitNet summary](../reports/semantic_gate_native_bitnet_2026-07-23/summary.md).

## 2026-07-23 — Budget-native grouped-ternary causal stop

- Added an exact full-width grouped-ternary artifact and training path. Five
  base-3 coefficients are packed per byte; every 128-weight group has one
  non-learned FP16 scale refined by two least-squares iterations. Codes,
  scales, headers, directory entries, and alignment produce a 17,173,504-byte
  30-layer file, or 43.1353% of dense ideal Q4 with 742,400 bytes of headroom.
- Added hard-forward straight-through MLPs, a deepest-layer-first continual
  transition, confidence-weighted KL, direct final-hidden and CKA losses,
  teacher-top-1 distillation, fresh training-record offsets, optional
  attention/norm/embedding/head co-adaptation, device-neutral checkpoint
  initialization/resume, and exact binary/backbone serialization and reload
  before validation.
- The hard serialized initialization is causally catastrophic (KL 28.381,
  top-1 0, hidden L2 1.264). A 128-step deepest-first stage cuts KL to 11.023
  without enough hidden recovery. Three fresh-record 64-step screens rebalance
  geometry, add a direct final-state loss, and test CKA/top-1 distillation.
  KL reaches 6.137 and hidden L2 0.907, but top-1 remains near 5%. Co-adapting
  the tied embedding/output head raises it only to 5.50%.
- Rejected a nearby 1,344/1,536-width affine-Q2 idea before production work.
  It fits the byte limit only with a very tight no-source-ID layout, while its
  MSE-fitted layer-14 initialization has 0.713 mean relative L2; full-width Q2
  is still 0.688.
- Promoted the exact ternary student to a frozen one-million-position rung.
  An RTX 3050 accelerated training only; checkpoints and artifacts remain
  CPU-compatible. The run used 8,192 fresh sequences, 1,014,225 input
  positions, and one serialized/reloaded artifact on all 16 development
  sequences/491 next-token positions. It reaches KL 2.28436, top-1 0.31976,
  NLL +2.27704, hidden L2 0.60361, and local MLP L2 1.02107.
- The predeclared pre-3M rule required 50% closure of every remaining formal
  gap. KL and NLL close 63.37% and 62.77%, but top-1 and hidden state close
  only 31.33% and 38.29%. The exact configuration is stopped before 3M/10M.
  Prior rank-32 hidden-output and correction-capsule failures also make a
  small residual fitted into the remaining byte headroom an unsupported
  repetition rather than a new hypothesis.
- Exact metrics, progression thresholds, and artifact/report hashes are in
  the
  [budget-native summary](../reports/semantic_gate_budget_native_2026-07-23/summary.json).
  Another semantic program must change the representation or establish its
  low-bit basis at pretraining scale; otherwise the honest alternative is an
  explicit relaxation toward the quality-passing DIP traffic frontier.

## 2026-07-23 — Budget-edge representation search closed

- Tested a cache-reused recurrent compact operator before returning to
  post-hoc quantization. Four applications of one width-640 Q4 SwiGLU plus
  rank-4 cycle adapters model at 44.9293% complete cold traffic if the base
  payload remains cache-resident. After 4,096 steps, recurrence improves the
  jointly fitted compact baseline only 4.84%, from 0.323918 to 0.308254
  layer-14 mean relative L2. It misses both the 25% relative-improvement rule
  and the 0.20 local ceiling; native cache validation is therefore not opened.
- Tested projection-local RMS normalization with full-width row-scaled ternary
  weights at 41.0013% traffic. After 8,192 hard-QAT steps the best internal
  layer-14 error is 0.631323, 27.86% worse than the prior asymmetric-ternary
  frontier. The normalization hypothesis is closed for this representation.
- Adapted the linear-constrained vector-training idea from
  [LC-QAT](https://arxiv.org/abs/2606.10531) into a mixed
  Q4/ternary/ternary affine lattice with complete traffic of 44.3482%. The
  8,192-step hard-QAT run improves initialization by 46.44%, from 0.628084 to
  0.336396, but remains far above the 0.20 local ceiling. One AMP overflow was
  skipped through the scaler's standard recovery path; the exact frozen
  schedule otherwise completed.
- Screened an unrestricted 128-entry, four-weight vector codebook motivated by
  [AQLM](https://arxiv.org/abs/2401.06118) and
  [QuIP#](https://proceedings.mlr.press/v235/tseng24a.html). Its 7-bit indexes,
  FP16 codebooks/scales, metadata, and padding leave only 8,000 bytes of
  all-model headroom at 44.9799% traffic. Deterministic K-means initialization
  has layer-14 error 0.576865, above the frozen 0.55 guard, so QAT is not run.
- Reproduced the official
  [LiftQuant](https://arxiv.org/abs/2606.04050) nearest-lattice search using
  repository revision `72b3875c770e4579639931fed89dc95e4067edac` and
  checksum-verified 16-to-8 and 16-to-10 projection tensors. The best mixed
  arm assigns 2 bits/weight to `up` and 1.6 bits/weight to `gate` and `down`,
  reaching 44.4012% complete traffic. Its best initialization error is
  0.556958, narrowly above the same 0.55 guard, so QAT and formal data remain
  unopened.
- Added reusable traffic models and hard-forward training modules for all five
  representations, with unit tests covering accounting, validation, shape
  contracts, hard-code behavior, gradients, and deployment-state extraction.
  The checked
  [budget-edge summary](../reports/semantic_gate_lowbit_2026-07-23/summary.json)
  records exact metrics and scratch-report hashes.
- Closed the bounded recurrent and post-hoc representation search. This
  decision led to the budget-native grouped-ternary program documented above;
  that later program also reaches its frozen stop rule. The remaining
  alternative is a materially different low-bit/pretraining hypothesis or an
  explicit relaxation of the 45% traffic policy toward the causal
  quality-passing DIP frontier.

## 2026-07-23 — Combined status freeze and output-memory stop

- Froze the formal combined Milestone 2 gate at KL <=0.05, top-1 >=0.90,
  NLL delta <=+0.05, final-hidden relative L2 <=0.10, at least eight unique
  sequences/256 prediction positions, and complete physical cold MLP traffic
  <=45% of dense ideal Q4. Candidate recall remains >=0.95 only when an
  approximate candidate stage exists.
- Stopped the layer-adaptive compact-Q4 run at its predeclared 3,000,093
  prediction-position checkpoint. The independently serialized/reloaded
  44.933449%-traffic artifact reaches KL 0.886578, top-1 0.565945, NLL delta
  +0.883762, and final-hidden relative L2 0.424521. It fails every 80%-gap
  closure continuation rule, so the final seven million positions were not
  spent and formal development/confirmation data remained unopened.
- Screened materially different local representations after post-hoc
  quantization, shared-basis, structured-dictionary, expert, and compact
  branches failed. Conditional width-640 experts reach 0.4468 held-out local
  error; a local Taylor model reaches 0.4640; finite shared Jacobian banks
  remain above 0.22. Token-conditioned Jacobians improve exact output memory
  to 0.2360 and two regions per token to 0.2164, but require tens of GiB before
  indexing or all-layer packaging.
- Measured an exact nested LLE-32 output-memory curve at 16,384/65,536/233,005
  local prototypes: 0.490340/0.401270/0.327526 mean layer-14 relative L2.
  Reconstructing query states from 512 neighbors reaches 0.0496, while output
  interpolation remains at 0.3275; the bottleneck is nonlinear operator
  variation rather than state-neighbor coverage.
- Captured exactly 1,000,000 finite FP16 layer-14 input/output pairs from
  8,192 sequences in the authenticated 10M-position SmolLM2 pretraining
  mixture. Exact stable LLE-32 over the combined 1,233,005 prototypes improves
  only to 0.321854, or 1.73%. The frozen progression rule required <=0.28 and
  at least 10% improvement, so the 10M capture, Q4/index work, external split,
  and causal evaluation were not opened.
- Consolidated the result in [Project status](status.md) and the checked
  [machine-readable snapshot](../reports/semantic_gate_status_2026-07-23/summary.json).
  No representation currently passes quality and traffic together. The later
  budget-edge campaign above tests and closes the bounded-representation
  option. The still-later grouped-ternary section above tests and stops the
  first material budget-native rung, leaving a new representation/pretraining
  hypothesis or an explicit systems-policy relaxation.

## 2026-07-22 — Teacher-boundary width ceiling rejects uniform 672

- Added MLP-only teacher tracing with deterministic per-sequence token sampling. Full sequence
  context is executed, but only sampled MLP inputs/outputs are stored; attention boundaries are
  omitted. The training trace contains 4,096 states from 256 sequences (610 MB), while the
  sequence-disjoint validation trace contains 446 states from 16 sequences (63 MB).
- Added a checkpoint-initialized local-ceiling evaluator that trains compact gate/up/down matrices
  from cached boundaries, writes a provenance-bound safetensors artifact, and requires both 10%
  mean improvement and mean validation relative L2 <=0.15.
- A 64-step screen improves all five representative layers but only 2.7% overall. At 512 steps the
  improvement is 8.8%. The final 2,048-step convergence rung reaches 10.2% improvement, from
  0.3851 to 0.3457, but fails the absolute ceiling. Layers 7/14/21 remain at
  0.5049/0.4558/0.4497; layer 29 reaches 0.0961.
- Uniform width 672 is rejected before another causal run. The next bounded architecture should
  allocate widths by measured layer sensitivity or introduce a more expressive structured basis,
  charging the aggregate serialized bytes/traffic against the unchanged 45% limit.

## 2026-07-22 — Full-corpus fixed-width student and stop decision

- Added a contiguous fixed-width SwiGLU student, deterministic local-source corpus builder, and
  device-neutral checkpoint/resume. Parameter-only checkpoint transfer validates model identity,
  compact width, tensor names/shapes, and SHA-256 lineage while deliberately restoring neither
  optimizer state nor history.
- Built 2,048 sequence records (258,899 token positions) by round-robin tokenization of 129 local
  documentation, Python, and native-source files. Training/validation exact sequence hashes are
  disjoint. Every 1,536-wide MLP is replaced by a trainable 672-wide MLP, giving 43.75% projected
  dense MLP weight traffic with no inference router.
- Completed one CPU-only epoch with all 30 compact layers active. From the earlier 128-sequence
  pilot to the full epoch, held-out KL improves 1.5499→1.1773, top-1 0.4175→0.4745, NLL delta
  +1.5254→+1.0553, hidden L2 0.4896→0.4260, and local MLP L2 0.7636→0.7053.
- The result passes evidence, traffic, and all-layers-compact checks but fails every semantic
  quality threshold by a large margin. Improvement also slows late in the epoch; top-1 briefly
  plateaus. Additional blind epochs on width 672 are rejected. The next bounded experiment is a
  larger teacher-boundary layerwise fit to measure the compact basis's local approximation ceiling
  before spending on another causal run.

## 2026-07-22 — Milestone 2 frozen-basis stop and structured-upcycling decision

- Tested a DAgger-style rank-16 residual refit on the actual q=62.5%/K=512 hard-student states.
  The parent artifact has same-state local relative L2 0.35117; the refit reaches only 0.34983
  (0.38%). This rejects teacher-state distribution mismatch as the primary blocker. The emitted
  artifact is diagnostic only; no causal run was authorized.
- Rechecked the frozen-basis ceiling. Even the nondeployable exact-utility K=512 oracle fails the
  causal gate at KL 0.132, top-1 0.809, NLL +0.085, and hidden L2 0.179. Better selection alone
  therefore cannot pass at 512 original channels.
- Tested a traffic-neutral alternative that shifts budget from gate inputs to active channels:
  q=43.75%, K=640, and a rank-23 residual project to 44.25% dense traffic. The local screen passes
  at L2 0.32893, but the all-layer development control is mixed and still far outside the gate:
  KL 0.684, top-1 0.603, NLL +0.616, hidden L2 0.358, and local L2 0.594. It also violates the
  original K<=512 constraint. The branch is rejected rather than used to redefine the gate.
- A direct rank-16 K=640 utility predictor has only 57.7% oracle overlap and local L2 0.443. A
  q=6.25% selector followed by exact selected-gate completion remains within 44.21% traffic but has
  local L2 0.435. Both fail before causal evaluation.
- Screened fixed-width structured pruning on representative layer 14. A 640-wide SwiGLU initialized
  from the strongest global channels improves held-out local L2 from 0.519 to 0.469 after 128
  full-weight steps. Expanding 512 real states into 8,192 teacher-labeled interpolations and running
  512 steps reaches a best 0.453 before overfitting. This is not enough to justify all-layer
  replacement from the current corpus.
- The frozen-neuron and cheap-recalibration search is closed. The remaining credible Milestone 2
  path is structured sparse upcycling or width pruning with materially more real token data,
  progressive dense-to-sparse training, and full MLP adaptation. CPU inference remains mandatory;
  CUDA may accelerate training but must not affect the serialized representation or runtime.

## 2026-07-22 — Low-rank native-gate utility correction

- Added a continuous multi-output ridge fit for the residual between partial-gate log utility and
  exact up-dependent channel utility. The implementation truncates the fitted map to a low-rank
  factorization, charges every factor and bias byte to inference traffic, and evaluates on exact
  sequence-disjoint traces.
- The initial 128-state rank-8/16 screen reduced local error from 0.3855 to 0.3551 but missed the
  10% threshold. A predeclared final cheap sweep used 512 states, corpus-scaled regularization,
  ranks 8/16/23, and the largest rank below 45% traffic. Rank 23/blend 0.8 reaches 0.3359 at 44.94%;
  rank 16/blend 0.8 reaches 0.3380 at 44.39%. The lower-traffic rank 16 point is selected because it
  is within 1% of the best error. Its exact top-512 oracle recall is 0.643.
- Serialized all 30 rank-16 residuals as source-hash-bound safetensors and integrated them into the
  hard native-gate wrapper and end-to-end evaluator. The correction changes selection only; exact
  up/down computation remains restricted to 512 channels.
- The full 16-sequence/491-next-token CPU control improves KL 1.235→0.629, top-1 0.460→0.599, NLL
  delta +1.202→+0.583, hidden L2 0.508→0.363, and local L2 0.702→0.625. It passes evidence and
  44.39%-traffic checks but not the final causal thresholds.
- A matched eight-step progressive run reaches 0.640/0.605/+0.604/0.363/0.626. Only top-1 improves;
  all other primary metrics regress slightly. Longer training on the unchanged objective is
  stopped. The next bounded experiment is on-policy residual recalibration using sparse-student
  layer inputs, followed by the same untouched causal gate.

## 2026-07-22 — Low-budget representation bounds and residual/adaptive stop

- Built and benchmarked an opt-in version-3 DIP package that duplicates gate/up weights in
  coordinate-major and record-major order. It grows the 30-layer package from 318.8 MB to
  531.2 MB. At q=360/C=K=512, six-trial medians are `0.815x` dense for omitted-coordinate
  record gathers and `0.845x` for full record rows, versus `0.912x` for coordinate gathers and
  `0.982x` for coordinate streaming. Real confirmation states touch essentially every omitted
  coordinate line, so record-major traffic reaches 102.8%–105.2% of dense at the quality-valid
  budgets. Version 2 remains the default; v3 is a rejected diagnostic.
- Replaced the saturated locality relaxation with a fixed-cardinality soft top-C backward path and
  exact hard occupied-line forward value. A gradient audit measured the unweighted locality
  gradient at 269.5x smaller than the causal router gradient, with cosine 0.037. Balancing those
  norms still left occupancy at 95.86/96 lines.
- Computed a stronger oracle bound on all 15,210 expanded-validation state/layer pairs. Exact
  top-512 membership itself touches 95.858/96 contiguous 16-record lines. Even a selector with
  perfect oracle knowledge captures only 91.75% mean membership with 80 lines; 88 lines are needed
  for 96.65%. Individual-record locality is structurally incompatible with a material traffic win
  under this static layout.
- Fixed nonstandard LoRA scaling: Kaiming A initialization plus alpha=rank replaces 0.01 A values
  compounded by a second 1/r factor. Added a rank-32 hidden-output residual, included its 147,456
  bytes per token-layer in traffic, and trained on all 128 independent sequences with durable
  four-step checkpoints. The 32-step held-out result improves KL from 0.166 to 0.152, top-1 from
  0.768 to 0.780, NLL delta from +0.126 to +0.100, and hidden L2 from 0.199 to 0.193, but fails every
  causal threshold. The residual itself is only 0.18% of teacher-output norm, has cosine 0.0014
  with the missing output, and slightly worsens local error, so it is disabled by default.
- Adapter learning rates 3e-4 and 1e-3 are unstable on matched 16-record screens; 1e-4 remains the
  stable point. A layer-adaptive exact-oracle schedule was then chosen from individual-layer causal
  interventions on a separate four-sequence selection split at the same mean K=512. Its untouched
  16-sequence confirmation also fails and is slightly worse than uniform K=512: KL 0.134, top-1
  0.786, NLL +0.110, and hidden L2 0.185. Fixed-total layer adaptation is stopped.
- The low-budget post-hoc configuration is therefore blocked before serialization. The next
  architecture-level experiment must make sparsity trainable and structured—expert/block routing
  with co-trained MLP weights on materially more data—rather than continue tuning selection over
  the frozen, diffusely coactive neuron basis.
- Added a hard-forward structured-expert module and a trace-only feasibility screen. It supports a
  lossless gate/up/down permutation, contiguous physical blocks, exact selected-block execution,
  a fixed-cardinality soft backward surrogate, and inference traffic that includes the router.
  Unit tests verify dense-shadow parity, deterministic grouping, exact active counts, nonzero
  causal router gradients, and that evaluation never evaluates the dense surrogate.
- Screened three balanced coactivation layouts at exactly 512 active records on 128 calibration
  and 128 disjoint validation states per layer. The 24×64/top-8, 48×32/top-16, and 96×16/top-32
  layouts project to 33.86%, 34.38%, and 35.42% of dense weight traffic. Their full-information
  greedy-residual local relative-L2 errors are 0.547, 0.497, and 0.438; fitted-router errors are
  0.655, 0.638, and 0.624. Dense-shadow parity is below 8.6e-7 maximum relative L2.
- Static grouping of the existing SmolLM2 neuron basis therefore fails the declared 0.20 local
  pretraining screen even with a non-deployable residual-aware oracle. Full end-to-end block
  training is not justified from this initialization. The next bounded experiment is native
  gate-based channel sparsity trained through the actual sparse path, with grouped selection for
  physical locality; it must be shadow-evaluated and microbenchmarked before a long run.
- Implemented that native-gate shadow without candidate completion. At K=512, exact contribution
  selection has mean local relative L2 0.190, establishing local representational headroom. Using
  the dense SwiGLU gate itself to select channels raises error to 0.375. Retaining 62.5% and 50% of
  gate input coordinates changes it to 0.386 and 0.402, at ideal traffic of 43.06% and 38.89% of
  dense. The dominant error is therefore channel-utility prediction, not input pruning.
- Added a full-weight native-gate student wrapper. Its training forward value is the exact hard
  sparse path; a detached dense surrogate supplies soft top-K selection gradients only during
  training. Tests verify dense parity, exact q/K budgets, hard-forward parity, selection gradients
  beyond chosen records, and that evaluation never invokes the surrogate.
- Added cached-boundary layerwise pretraining with dense-shadow retention and a training-only
  hardest-negative utility-ranking warm-up. On representative layer 14, 16 causal-only steps move
  held-out local error from 0.4146 to 0.4102; utility ranking reaches 0.4098. Extending the single
  controlled arm to 64 steps with stronger dense retention reaches only 0.4040 (2.55% improvement)
  while dense-shadow error reaches 0.0339. It fails the declared 10% improvement screen.
- Cached-boundary native-gate tuning is stopped rather than expanded to all layers. The remaining
  justified semantic experiment is progressive end-to-end native-gate co-training on materially
  larger data, with the student executing its hard sparse path and attention/norm initially frozen.
  The host contains an RTX 3050, but the current execution session cannot reach the NVIDIA driver:
  NVML fails, `/dev/nvidia*` is not exposed, and CUDA-enabled PyTorch reports zero devices. CUDA is
  optional training acceleration; CPU execution, reproducibility, packaging, and inference remain
  project requirements.
- Added a device-neutral end-to-end native-gate trainer. It freezes attention, normalization, and
  embeddings; co-trains complete gate/up/down MLP weights; linearly anneals dense q/K to the target;
  and combines hard-path local, dense-shadow, hidden-state, logit-KL, and utility-ranking losses.
  Held-out evaluation forces q=62.5%/K=512 and rejects any training-only surrogate execution.
- The CPU path is operational. A one-record direct-target smoke completes all 30 layers and writes
  a provenance/traffic/gate report without a model artifact. A four-step schedule traverses
  K=1,536/1,195/853/512 and q=576/504/432/360; versus the direct jump it improves smoke KL from
  1.759 to 1.592 and hidden L2 from 0.700 to 0.635. These are execution diagnostics only.
- Added an evaluation-only control and ran the complete 16-sequence/491-next-token set. The
  untrained hard native-gate baseline has KL 1.235, top-1 0.460, NLL +1.202, hidden L2 0.508, and
  local MLP L2 0.702. An eight-step CPU stage passes evidence and 43.06%-traffic checks but changes
  those to 1.254/0.481/+1.211/0.510/0.700. Because top-1/local improve slightly while KL/NLL/hidden
  worsen, a longer run on this objective is not justified by the bounded result.
- Implemented device-neutral full-weight/optimizer checkpoints whose requested total step count can
  be extended on resume. A tiny two-layer Llama integration test checkpoints one CPU step and
  resumes to two. CUDA may accelerate the same state, but is not required for checkpoint creation,
  restoration, validation semantics, or artifacts. A broken optional sklearn/NumPy ABI is ignored
  narrowly when importing causal-LM Transformers code because sklearn is not used by this trainer.

## 2026-07-21 — Low-budget sparse-teacher full gate and stop decision

- Added padding-safe sequence batching and a provenance-checked safetensors cache for the initial
  30-layer router fit. Cache reuse reduced a two-train/two-validation rerun to 58.3 seconds. The
  complete 32/16 experiment then ran in 582.8 seconds on CPU and met the 16-sequence/491-next-token
  evidence floor.
- At `q=62.5%`, `C=K=512`, the full result fails: candidate recall 0.8959, KL 0.1659,
  top-1 agreement 0.7678, NLL delta +0.1261, and final-hidden relative L2 0.1988. Candidate lines
  occupy 95.86/96 groups; cache-adjusted traffic is 77.74% of dense rather than the 61.11% scalar
  ideal.
- Audited candidate locality without another causal run. Balanced physical permutations based on
  router features or direct candidate co-occurrence leave 94.66–95.42 lines occupied. Selecting
  exactly 32 complete 16-record lines is local by construction, but the best checked layout/score
  reaches only 0.4873 oracle recall and local relative L2 remains above 0.47. Neither approach is
  suitable for compilation.
- A cached blend sweep shows the partial-weight proxy plus low-rank router peaks near scale 0.1 at
  only 0.8965 validation recall; larger router weight degrades monotonically. The original 0.119
  blend was already near this optimum, so blend tuning cannot close the gap.
- Extended the student with mergeable rank-8 gate/up LoRA in addition to the down update and gave
  router factors a separate learning rate. Added a deterministic local-source corpus builder and
  decoupled router calibration from student training. The generated corpus has 128 sequences and
  15,991 token positions.
- A bounded 16-sequence broader-corpus stage improved KL from 0.1425 to 0.1368 and top-1 from
  0.7576 to 0.7879 on its four-sequence screen, but NLL worsened from +0.1543 to +0.1699, hidden
  error stayed near 0.17, recall stayed 0.900, and line occupancy stayed 0.9986. Scaling to all
  128 sequences is not justified on this CPU host. This individual-record locality objective is
  stopped; a future systems path should decouple partial-scan and candidate-completion layouts or
  change the sparse representation rather than add epochs to the same configuration.

## 2026-07-20 — Hardware-aware differentiable sparse-teacher training

- Replaced the pilot's causally disconnected hard route with a hard-forward, soft-backward
  estimator. Deployment semantics remain exact: retain the largest `q` input coordinates, form
  partial scores, complete only `C` candidates, rerank to `K`, and gather only the selected down
  records. A straight-through sigmoid relaxation supplies gradients from local-MLP, hidden-state,
  and logit-distillation losses to the rank-16 router.
- Kept attention, normalization, embeddings, and original MLP weights frozen. Only router factors,
  a learned blend between partial source-weight scores and router scores, and rank-8 MLP adapters
  are trainable. Oracle activations are computed under `no_grad` solely for supervision and
  candidate-recall measurement; they are not the student output.
- Added a differentiable cache-line occupancy objective and reports for candidate/active line
  counts, scalar logical traffic, cache-adjusted traffic, the `q<=62.5%` and `C/K<=512` budget,
  and the existing held-out causal gate. The recommended command defaults now select
  `q=62.5%`, `C=K=512`.
- Unit tests prove that the hard forward budgets are exact and that a causal output/locality loss
  produces nonzero router-factor gradients without membership BCE. The real-model one-record
  smoke run completed and wrote a router/adapter artifact.
- The smoke run starts at 0.9005 candidate recall. Its nominal scalar model is 61.11% of dense,
  but candidates occupy 95.84 of 96 gate/up cache-line groups on average (99.84%). Cache-line
  amplification raises projected total traffic to about 77.7% of dense; the record-major down
  reads remain sparse. The run is deliberately not promoted; full-corpus accelerator training must
  improve causal quality and line occupancy together before serialization or compilation.

## 2026-07-20 — Serialized DIP kernel exposes the systems gate

- Added a version-2 experimental DIP package with checksummed mmap arrays. Gate/up weights are
  coordinate-major for sequential partial scans; down weights are record-major for contiguous
  selected-row reads. The Python and native implementations agree on selected IDs, output sums,
  and byte accounting, and candidate completion touches only candidate records in code.
- Corrected an initial cache interpretation after checking the official implementation. Published
  DIP cache awareness reweights fine-grained choices using temporal cache state; it does not force
  input coordinates into spatial cache-line blocks. The tested block-16 variant retained only
  85.24% of oracle records and 92.07% score mass on all 35,520 fresh-confirmation layer states, so
  it is rejected.
- At `q=432/C=896/K=768`, logical scalar reads remain 76.39% of dense. Counting unique 64-byte
  lines touched by the coordinate-major completion gather gives 83.33% of dense.
- Replaced array-of-structures accumulators and near-full partial sorts with contiguous float32
  accumulators, partition selection, and record-sorted candidates. Real layer-10 selected IDs
  remain exactly equal to the Python reference. This improves the full-model candidate-gather
  kernel to `0.770x` dense.
- Since 896/1,536 candidates touch every cache line, a second kernel streams all omitted-coordinate
  rows and reranks only candidates. It is faster than gather but executes 83.33% of dense weight
  bytes. Across six alternating-order 20-pass runs over all 30 layers (305 MiB), median sparse
  time is 37.673 ms versus 32.639 ms dense, or `0.863x`. The current architecture is therefore
  rejected before default-runtime integration; a robust win requires materially smaller `C/K`,
  temporal reuse on target hardware, or a different representation.

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
- Hardware audit found AVX but no AVX2 and no accessible performance counters. The RTX 3050 is
  visible on PCI, but CUDA is not exposed to PyTorch in this execution session. No performance
  claim was made.

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
