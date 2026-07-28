# Limitations

- The original dense-Llama limitation is unchanged: no serialized
  representation jointly passes the all-layer causal gate and complete cold
  MLP traffic at or below 45% of dense ideal Q4. Its older SmolLM DIP arm is a
  quality-only pass at 83.33% cache-line traffic; the mild-width compact-Q4
  artifact is a traffic-only pass with KL 0.887, top-1 0.566, NLL +0.884,
  and hidden L2 0.425 after 3M positions.
  The full-width grouped-ternary artifact is another traffic-only pass:
  43.1353% traffic with KL 2.284, top-1 0.320, NLL +2.277, and hidden L2
  0.604 after 1,014,225 positions.
  See [Project status](status.md).
- OLMoE Q7/group-64 is the first SwiGLU-family branch here to pass the
  all-layer semantic thresholds and 8-sequence/256-position evidence floor.
  Its 5,842,733,184-byte packed artifact and direct top-8 CPU kernel now pass
  the native systems gate and are installed in an authenticated native OLMoE
  generation package. The complete token boundary includes embeddings,
  normalization, attention/cache state, vocabulary projection, and a greedy
  token loop. Python still owns tokenization and prompt text. A frozen short
  generation suite and complete 8-sequence/256-position native causal suite
  now pass, but long-form generation, chat UX, and broad language evaluation
  remain. This is a native substituted-MLP/token-runtime result, not the
  Milestone 4 controller-only architecture: source embeddings, norms,
  Q/K/V/O attention projections, and `lm_head` still execute.
- OLMoE's 22.7865% native traffic result is exact scheduled unique packed
  expert/router bytes, not a hardware-counter DRAM measurement. It excludes
  activations, attention, runtime metadata already resident in cache, and
  hardware prefetch/write traffic. The 0.816-second scalar layer result is a
  historical correctness baseline. The current 12-thread packed block decoder
  reaches 12.55–16.53 ms on representative layers, but this does not establish
  whole-system DRAM traffic or optimal CPU performance.
- The complete OLMoE native prompt smoke is deliberately small. Fixture NumPy
  and cache progression parity pass, and the authenticated package produces
  ` Paris`, but this is not a language benchmark. The short eight-prompt
  protocol agrees on 29/32 generated tokens and 7/8 prompts, yet never crosses
  W=16. The complete causal protocol does include 128 post-window positions
  and passes its frozen split-mean thresholds. It remains only eight fixed
  sequences: offset 31 alone misses all four quality thresholds, maximum KL is
  0.606769, and sustained older-context generation remains open.
- The stronger frozen 8-sequence/1,024-position sustained test shows that the
  production W16/C8/K4/S2 policy does not retain semantic quality through 128
  positions. Its evidence checks pass, but overall KL 0.143578, top-1
  0.802734, NLL delta +0.159292, and hidden L2 0.238260 fail; the 32–63,
  64–95, and 96–127 bands also fail. A separately frozen post-failure W128
  control, identical except for the local window, passes overall and every
  band at KL 0.003438, top-1 0.974609, NLL +0.001459, and hidden L2 0.041389.
  All 128 pre-intervention rows (eight sequences × positions 0–15) match
  exactly. This is strong matched evidence that bounded attention is the
  primary drift source on this corpus, not proof that it is the only possible
  source of error or that the result generalizes to other text and lengths.
- W128 is deliberately nondeployable. It uses 35,825,664 bytes of attention
  state and reads 2,164,260,864 logical bytes per 128-position sequence—100%
  of the full-context logical-KV reference—versus 6,336,512 state bytes and
  677,117,952 logical bytes (31.2863%) for W16. These are algorithmic logical
  interface bytes, not hardware-counter DRAM traffic; cache hits, prefetch,
  writes, and allocator behavior are not measured.
- The exactly matched <=45% static-policy sweep is complete and negative.
  W16/C18/K16/S2, W24/C10/K8/S2, and W30/C4/K2/S2 each read 968,753,152
  logical bytes per sequence (44.7614%) and expose 32 values per mature step.
  Every authentication, source-binding, exact-counter, replay, pre-eviction
  identity, and post-run check passed. Nevertheless, their respective overall
  KL/top-1/NLL-delta/hidden-L2 results were
  0.063887/0.867188/+0.051701/0.157717,
  0.065912/0.877930/+0.058480/0.159755, and
  0.095813/0.840820/+0.075728/0.188422. Zero arms passed, so none was selected
  and the reserved confirmation corpus remains unused. This closes static
  global W/C/K reallocation at this budget, not every possible bounded
  attention architecture. Layer/head-adaptive budgets and learned or
  distilled older-context selection remain open.
- The sweep did not alter the authenticated package. Its raw token-runtime
  intervention overrode the package's immutable W16/C8/K4 policy only inside
  the evaluator. There is no promoted attention artifact, package descriptor,
  or model-format revision. W32 is already ~50.38%, W64 is worse, and W128
  remains only the diagnostic ceiling.
- The subsequent layer-adaptive upper-bound result is also negative. The
  additive layered C ABI and Python binding accept one attention policy for
  each of the 16 layers and pass exact all-base parity with the scalar path.
  A frozen greedy search selected layers 11, 6, and 10 for W128 and left 13
  layers at W16/C8/K4/S2. The schedule used 955,957,248 logical bytes per
  sequence (44.1701%), and every evidence check passed, but its six-sequence
  result was KL 0.102321, top-1 0.845052, NLL +0.116776, and hidden L2
  0.206037. Both early bands passed; every metric failed from position 32
  onward. This closes the frozen greedy three-layer W128 path under 45%, not
  every possible interacting whole-layer combination. Milestone 2 remains
  passed for Q7, but Milestone 3 remains blocked.
- That layer schedule is not a promoted model feature. The experiment invoked
  the raw runtime; package version 1 still binds one global W16/C8/K4/S2
  policy and has no per-layer or per-head descriptor. The next prospectively
  frozen boundary is a teacher-guided mask rescuing 51 of 256 layer-head pairs
  at 44.9754% logical traffic. Rescuing 52 would require 45.2438%, above the
  declared cap, and no head-wise result or package schema exists yet.
- Interpret the layer-rescue failure narrowly. The greedy search made 45
  adaptive comparisons on only two sequences, whose positions are correlated;
  its six-sequence screen reused an already-consumed corpus. Greedy choices
  can miss interacting layer combinations, and W128 is exact only over the
  tested 128-position horizon. Finally, all traffic percentages are analytical
  native logical reads, not hardware-counter DRAM measurements.
- Package authentication currently hashes about 6.8 GB and the Q7 loader
  performs strict full-artifact structural validation at startup. This is
  fail-closed but makes cold startup materially slower than steady-state token
  execution. Structural validation is parallelized (about 16.5 to 2.29
  seconds), as are package and source-shard hashing, but the latter remain
  storage-bandwidth-heavy at 27.19 and 23.65 seconds.
- The untouched Hugging Face OLMoE teacher is much less parallel than native
  inference on this CPU. Its installed fallback loops over experts and small
  per-expert GEMMs often use one core despite a 12-thread setting. The serial
  8×33 capture took 366.14 seconds and peaked near 50.4 GB RSS. Batched teacher
  capture alone improved compute by only 1.2%. Four concurrent sequence
  forwards sharing one read-only model are byte-exact and 3.86× faster, so
  they are now the CPU default. Direct expert threading is faster than serial
  but changes BF16 rounding and remains experimental opt-in.
- The frozen native causal protocol binds the package, immutable DSO, corpus,
  teacher arrays, source config/index identities, and all six teacher shards,
  but not the Engram evaluator source itself. The executed result was manually
  audited. A disclosed, non-independent hardened replay now binds the evaluator
  inventory, recomputes inputs/targets, reauthenticates all roots, and exactly
  reproduces every semantic metric. It strengthens implementation provenance
  but is not a fresh holdout because the original result was already known.
- The separate native-BitNet artifact reconstructs exactly and its direct
  full-record memory-mapped CPU kernel passes its frozen causal gate at
  40.0527% scheduled cold traffic. This does not say anything about losslessly
  converting an already-trained dense Llama checkpoint. Official-layer output
  is numerically, not bitwise, equal to PyTorch BF16 because reduction order differs (maximum
  checked relative L2 0.00982). No hardware DRAM counter was available, and
  the retired Python package/generation path retained the source attention,
  normalization, embedding, and output tensors inside a Transformers shell.
  The newer DIP package and chat command use the complete C++ token runtime,
  but this remains a source-family-specific runtime rather than the intended
  controller-only architecture distilled free of original transformer
  operators.
- The native-BitNet DIP semantic gate passes **by postmortem adjudication**,
  not by a pristine final-runner result. On the consumed 8-sequence,
  256-position holdout, the raw evaluator reports KL 0.00404129, top-1
  0.98828125, NLL +0.00482893, hidden L2 0.0477494, 21.3800% mean active
  records, 41.1371% modeled traffic, 99.9406% global recall, and 99.3943%
  worst-layer mean recall. The original runner ended in error after evaluation
  because it compared full-record canonical-object hashes with first-33-token
  bare-list hashes. A separate no-model adjudicator corrected that verifier
  contract and checked all frozen thresholds. This supports the semantic-gate
  decision, not a claim that every generic Milestone 2 deliverable is complete.
- The original error result did not contemporaneously bind the raw evaluator
  report. That report was prospectively hash-sealed about 13 minutes later,
  before the postmortem adjudication. This is weaker evidence custody than a
  clean independently sealed one-shot result, and the consumed holdout cannot
  be rerun as a fresh final.
- The sealed holdout is a plaintext repository fixture. Avoiding inspection
  before the one-shot run is a procedural/honor-system control, supported by
  committed hashes and a fail-closed runner; it is not cryptographic secrecy
  and cannot prevent a developer with filesystem access from reading it.
- The 41.1371% final practical-DIP traffic figure is a v2 cache-line model, not
  hardware-counter DRAM measurement. The complete semantic storage is larger:
  the 318,924,544-byte base record artifact plus 216,688,448-byte coordinate
  index total 535,612,992 bytes, or 67.2659% of dense Q4.
- Traffic reduction has not produced speedup. The final sparse evaluation took
  1.1449x the dense elapsed time (295.3364 versus 257.9552 seconds), or 14.49%
  longer. Latency was measured but was not a frozen gate. Debug recall and
  parity work was outside the timed pass.
- The DIP evidence is one model, one host-bound artifact/library set, one
  development corpus, and one 8-sequence/256-position consumed final corpus.
  Six rows per layer establish implementation parity, not broad numerical or
  workload coverage. Independent-host replication, hardware counters,
  SIMD/cache tuning, and broader language-quality evaluation remain open.
- The DIP package/token integration result is deliberately small: eight
  non-holdout prompts and 32 generated tokens. It has 32/32 greedy token-ID
  agreement and 8/8 exact prompts, but this is not hidden-state/logit parity or
  an independent language benchmark. Global/maximum-prompt activity is
  21.5602%/22.5892%; global/maximum-prompt complete modeled traffic is
  41.1612%/41.2984%. Complete traffic is 30,153,074,432 bytes, including
  194,304 global-metadata bytes. The traffic remains modeled.
- The rebuilt-core integration wall time is 390.4183 seconds across first generations,
  reset replays, and per-process package authentication. This is disclosure,
  not a speed claim. Reported native counters and phase timings are first-run
  snapshots. Reset proves repeated tokens, zeroed counters, and structural
  metric parity, not hidden-state identity.
- The frozen 8×4 context is at most 14 positions, below W=16. A separate
  16/17/18/24/32 protocol now validates exact eviction, older-candidate,
  older-selection, sink, heavy-hitter, fixed-state, and reset mechanics.
  Its deterministic boundary prompt and one generated token per length do not
  validate attention quality against a dense teacher or natural tasks that
  depend on older context.
- The derived DIP package is supported only by the native token runtime.
  `chat-native-bitnet` now uses that runtime through a versioned C ABI. Python
  still owns the packaged tokenizer, template, and history, but it does not
  construct a Transformers model or dense semantic fallback. Streaming and
  persistent cross-turn native state remain unimplemented.
- `engram-bitnet-token-generate` now authenticates an exact, symlink-free
  package inventory and derives architecture, paths, bounds, attention policy,
  RoPE/RMS settings, and EOS IDs from it. This is deliberately pinned to the
  currently promoted manifest and semantic trust roots; accepting a different
  separately adjudicated model still requires a new reviewed native trust
  root and binary. The executable directly links Engram kernels and has no
  Engram shared-library dependency.
- Package authentication is currently pathname based: validation rejects
  symlinks and rechecks the inventory, but files are not yet mapped through
  already-open `O_NOFOLLOW` descriptors. An adversarial local process that can
  replace package files concurrently could therefore create a validation/use
  race. The current evidence assumes a trusted, quiescent local package
  directory.
- Bounded attention is integrated into complete package generation and avoids
  the dense Hugging Face KV cache. Q/K/V and output projections ran in PyTorch
  in the retired shell; the current DIP C++ runtime executes packed
  projections and crosses Python only once per generated request. Its frozen
  evidence is still the small 8×4 suite. At a 256-token prompt, the older path's
  attention state is fixed at 7,477,440 bytes and modeled reads are 16.35% of
  dense, yet complete processing is only about one position per second.
  Logical byte counts are algorithmic interface counts, not hardware DRAM
  events. The long-context prompt is a deterministic repeated benchmark
  string and is not additional quality evidence.
- Native packed Q/K/V/O execution passes the frozen
  8-sequence/256-position confirmation and substantially improves latency, but
  its packed decode loops are threaded without explicit AVX2 intrinsics. The
  tied 128,256-entry vocabulary head remains a materialized BF16 full scan.
  Greedy generation computes that scan for only the final prompt/decode row,
  reducing its measured time to 0.83 seconds in the controlled run. Tasks that
  require logits for every input position still pay the full head cost.
- Combined frozen and sustained-generation tests show that the inference path
  works, but the evidence remains small: 8 frozen sequences/256 positions and
  eight 16-token natural-prompt completions. It is not a broad benchmark of
  reasoning, coding, instruction following, multilingual quality, or safety.
  Mean sustained throughput is only 0.194 token/s.
- The retired shell kept attention state constant through 2,048 tokens, but
  its peak RSS rose from 2.14 GB at 512 tokens to 2.57 GB because
  PyTorch/Transformers and transient prompt tensors remained. Equivalent
  long-context RSS has not yet been measured for the new shared handle.
- `chat-native-bitnet` re-prefills the entire rendered conversation every
  turn and prints only after generation completes. It has no cross-turn cache
  reuse, token streaming, context truncation policy, sampling controls, or
  concurrent sessions. Long conversations therefore grow prefill cost until a
  future context-management policy is added. The old shell's documented
  two-turn example took 166.43 and 153.15 seconds for 32 tokens; the new DIP
  binding has only a one-turn/one-token 5.16-second smoke. Bounded attention
  memory therefore does not imply low latency or bounded total
  prompt-processing work.
- The grouped-ternary result uses an exact serialized/reloaded artifact and
  clears the evidence floor, but it is not eligible for more training under
  its frozen rule. It closes 63.37%/62.77% of the remaining KL/NLL gaps and
  only 31.33%/38.29% of the top-1/hidden gaps; all four had to reach 50%
  before 3M. Its CUDA-accelerated training is device-neutral and does not
  demonstrate CUDA-free training speed or any native ternary inference speed.
- Small follow-ups do not rescue that conclusion. Geometry reweighting,
  direct final-hidden loss, CKA, teacher-top-1 distillation, and tied
  embedding/output-head co-adaptation were screened on fresh record ranges.
  None met its complete progression rule. A cheap per-row affine-Q2
  alternative at 87.5% width measured 0.713 layer-14 relative L2 and was
  rejected before adding another artifact format.
- The last bounded representation campaign does not change that result.
  Recurrent compact Q4, projection-normalized ternary, affine
  constrained-vector quantization, an unrestricted vector codebook, and a
  LiftQuant-style lifted-binary lattice all model below 45% traffic, but none
  passes the development-only layer-14 ceiling of 0.20 mean relative L2. The
  best trained point is 0.308254 and assumes unmeasured later-cycle cache
  reuse. The codebook and lifted-binary arms stop at initialization rather
  than making unsupported extrapolations from QAT.
- Exact nonparametric output memory is not a hidden solution to this gap.
  Adding one million independent pretraining records to 233,005 local
  prototypes improves layer-14 LLE-32 error only from 0.327526 to 0.321854
  (1.73%), below both frozen progression requirements. The exact search has no
  deployable index or traffic claim.
- The Cognitive Executive currently has deterministic policy, revisioned SQLite/JSONL/in-memory
  event stores, a worker capability registry, resource accounting, matched outcome observation,
  adapter protocols, and calibration metrics. It has no durable knowledge store, strategy
  registry, production model/tool adapters, domain content validators, or learned success/cost
  predictors. Its reference session permits only one in-flight attempt.
- Executive evidence confidence is a conservative policy score, not calibrated epistemic
  probability. Existing controller-residual and vocabulary-margin confidence fields are also
  proxies and must not be interpreted as knowing whether a claim is true.
- Outcome calibration covers dispatched actions only. It cannot estimate counterfactual quality
  for rejected alternatives without controlled exploration, and information-gain measurements
  are comparable only when they share a declared validator.
- External dispatch is at-least-once across crashes: selection is durable before invocation, but a
  process can fail after a worker side effect and before recording its outcome. Production workers
  must deduplicate the stable attempt ID or recover their prior result. SQLite/JSONL durability
  does not make external side effects exactly once.

- Random fixtures still validate most of the systems pipeline, but trained SmolLM2-135M
  interventions now measure MLP replacement quality on only 16 held-out sequences (491
  next-token positions). All 16 token sequences are unique, which clears the declared evidence
  floor for rejecting these artifacts; this is not a broad model, task, or language claim.
- Bias-enabled MLP projections (`mlp_bias=true`) are rejected during inspection because the
  current extraction and semantic-record formats do not represent those biases.
- The oracle computes every activation. The practical router now uses deterministic joint-key
  IVF and avoids scanning all record keys, but still scans every coarse centroid.
- Exact all-layer top-256 and top-512 magnitude-reference substitutions fail the declared quality
  gate. Top-768 is the first tested pass and retains half of all MLP records; intermediate counts
  between 640 and 768 were not tested. Magnitude ranking is not guaranteed to be the optimal
  K-record subset.
- At top-768 with 1,280 candidates, flat rank-16 and overlapping-posting routers both fail the
  downstream quality and 95% recall gates after training on all 1,112 calibration states per
  layer. Full-corpus recall is 0.889 for the flat router and 0.868 for the overlap router; the
  latter scans about 1,667 posting entries per layer on average to form 1,280 unique candidates.
  No learned router has been serialized into the package format.
- A cached rank-16 sweep peaks at λ=8,000. Candidate counts of 1,408 and 1,472 pass the 95% recall
  screen, but both fail causal quality. The 1,472 arm reads 95.8% of record keys and still has
  KL 0.085, top-1 agreement 0.866, NLL delta +0.055, and final-hidden relative L2 0.131. Further
  candidate expansion would approach a dense scan and is not considered a viable routing result.
- In the historical dense-SmolLM track, a predictor-free, DIP-inspired
  algorithm was the first realizable semantic selector to pass that track's
  all-layer quality gate. Engram's candidate-only completion and exact
  contribution reranking extend the published DIP method. After selecting
  75%-input/896-candidate/K=768 on the development grid, a
  sequence-disjoint 16-sequence confirmation run has KL 0.029, top-1 agreement 0.910, NLL delta
  +0.033, final-hidden relative L2 0.090, and 0.990 candidate recall. This still covers only one
  135M-parameter model and a small generated corpus; another model and broader natural data are
  required before generalization.
- That dense-SmolLM DIP arm has a versioned coordinate-major experimental
  package, mmap loader, Python reference, and candidate-only native kernel.
  The optimistic scalar count is 76.4% of dense; counting
  touched 64-byte lines gives 83.3%. An alternating-order benchmark over all 30 serialized layers
  measures the best streamed kernel at 37.673 ms per complete 30-layer pass versus 32.639 ms
  dense: `0.863x`,
  or about 15.4% slower. Hardware-counter DRAM traffic, hand-written SIMD/threaded kernels,
  whole-model latency, and replication remain absent. A block-16 layout is
  not a workaround: it reduced fresh-confirmation recall to 85.2%. The path remains outside the
  default runtime and far from the long-term 10x target.
- The low-rank background is implemented, but the checked random-fixture Gate 2 experiment
  overfits badly (mean relative L2 rises from 0.693 without it to 7.09 with it).
- Real-model tracing loads the source model in CPU float32. Layer-at-a-time source execution
  and activation checkpointing remain future compiler work.
- Native-BitNet trained-teacher attention analysis and substitution now run.
  The promoted bounded hybrid retains 16 local, two sink, and six heavy-hitter
  entries, reranks eight old keys to four values, and passes frozen semantics.
  The 33-token protocol still models at 93.34% of dense KV traffic. A native
  state/cache/rerank kernel and standalone long-context logical-read benchmark
  now exist, but the kernel is not wired into incremental package generation.
  Its benchmark includes ctypes/input-generation overhead and no hardware DRAM
  counters. The shared controller remains initialized rather than distilled
  from source residual trajectories.
- Python and native generation work without source transformer tensors, but generated fixture
  token IDs are not meaningful language.
- Compiled runtimes consume quantized-only semantic arrays and scan only IVF-posted key codes.
  The tiny fixture still misses the active-fraction and logical-traffic goals, and the claimed
  10x hardware DRAM-traffic reduction is unproven.
- Correction capsules and adaptive escalation policies are operational primitives but are not
  fitted by the compiler. The research fitter now targets exact routed-read residuals and hard
  regions, but all checked global and targeted layouts worsen held-out local MLP error; no fitted
  capsule is serialized.
- The original sparse-teacher pilot has only 32 optimizer steps, no hyperparameter/seed
  replication, and fails every routed quality check. Its hard route prevented causal gradients
  from reaching the router. The replacement hardware-aware trainer fixes that graph with a
  hard-forward/soft-backward estimator and executes the requested `q<=62.5%`, `C/K<=512` sparse
  path. A complete 32-sequence/16-held-out run has now completed. At initialization, 512 candidates
  touch 95.84/96 gate/up cache-line groups, raising projected total traffic from 61.1% to about
  77.7% of dense. Neither trainer's safetensors output is a supported compiled-package input.
- The complete replacement run also fails its causal gate: recall 0.8959, KL 0.1659, top-1 0.7678,
  NLL delta +0.1261, and final-hidden relative L2 0.1988. Balanced permutations cannot cluster a
  one-third independent candidate set into materially fewer lines, while selecting complete lines
  damages recall/local reconstruction. Gate/up/down LoRA and a broader 2,048-token stage did not
  improve all held-out metrics together. More epochs on the same objective are not justified.
- Held-out exact top-512 membership itself touches 95.86 of 96 contiguous 16-record lines on
  average. Even an impossible perfect group selector that knows the oracle set in advance can
  retain only 91.75% of it with 80 lines; 88 lines are required for 96.65% mean coverage. Candidate
  locality cannot supply a material traffic win for the current static record order. The opt-in
  v3 dual layout increases package storage by 66.7% and is slower than coordinate-major completion;
  it is retained only as a rejected diagnostic.
- Corrected LoRA scaling plus a rank-32 hidden-output residual was trained on all 128 local-source
  sequences and evaluated on the full held-out set. It improves the earlier low-budget result but
  still fails: KL 0.152, top-1 0.780, NLL delta +0.100, and hidden L2 0.193. The residual correction
  has 0.18% relative norm and 0.0014 cosine with the exact missing output; it is disabled by default.
  Adapter learning rates of 3e-4 and 1e-3 are unstable on matched bounded screens.
- A same-total layer-adaptive exact-oracle schedule was selected on four disjoint sequences using
  individual-layer causal sensitivity. Its 16-sequence confirmation is slightly worse than uniform
  K=512 (KL 0.134, top-1 0.786, NLL +0.110, hidden L2 0.185). This does not prove every possible
  adaptive policy fails, but it rejects the current fixed schedule and magnitude objective.
- Static contiguous expert grouping also fails before end-to-end training. Across 24×64/top-8,
  48×32/top-16, and 96×16/top-32 layouts, a full-information greedy residual oracle has mean local
  relative-L2 error 0.547, 0.497, and 0.438 at the same 512-record budget. Learned routers are worse.
  This rejects these frozen-basis initializations, not jointly trained native channel sparsity.
- Native-gate channel sparsity has not passed either. Its exact contribution K=512 reference has
  0.190 local relative L2, but the realizable q=62.5% gate route has 0.386. A 64-step cached-trace
  layer-14 pretrain improves 0.4146 to only 0.4040 and therefore fails its 10% screening threshold.
  Full end-to-end progressive co-training and later compact-model training have run. The host's
  RTX 3050 is available for bounded training and trace capture. CUDA remains an optional
  development accelerator, not a deployment requirement; literature at much larger token budgets
  does not substitute for Engram evidence.
- The progressive end-to-end trainer now works on CPU and supports resumable device-neutral
  checkpoints, but its first full-evidence stage is deliberately tiny. Eight steps improve top-1
  agreement from 0.460 to 0.481 and local L2 from 0.702 to 0.700 while worsening KL from 1.235 to
  1.254, NLL from +1.202 to +1.211, and hidden L2 from 0.508 to 0.510. This is neither a quality pass
  nor evidence that substantially longer CPU training will succeed.
- A rank-16 utility residual materially improves the actual all-layer sparse path at 44.39%
  projected traffic: KL 0.629, top-1 0.599, NLL +0.583, and hidden L2 0.363. This is still far from
  the final 0.05/0.90/+0.05/0.10 quality thresholds. Eight progressive steps do not improve the
  result consistently. The residual was fitted on dense-teacher trace states; its behavior after
  on-policy recalibration to sparse-student state drift remains unmeasured.
- On-policy recalibration is now measured and improves its controlled same-state local error by
  only 0.38%. A traffic-neutral K=640 development arm also fails causally. Fixed-width layer-14
  distillation reduces local error only to roughly 0.45 with the original cached states. The
  subsequent all-layer 672-wide student uses 2,048 real sequences and a full epoch, yet still has
  KL 1.177, top-1 0.475, NLL delta +1.055, hidden L2 0.426, and local L2 0.705 at 43.75% traffic.
  These results rule out a cheap routing fix and the current narrow fixed-width objective. They do
  not rule out a more expressive co-adapted structured basis or substantially larger pretraining.
- A teacher-boundary fit removes causal state drift as an explanation for the fixed-width failure.
  After 2,048 local steps on 4,096 boundaries, five representative layers average 0.346 held-out
  relative L2; middle layers remain at 0.45–0.50. The screen samples five layers rather than all 30
  and uses repository-derived text rather than a web-scale pretraining distribution, so it rejects
  uniform width 672 for the current project evidence—not every possible compact architecture.
- The development Xeon E5-2695 v2 lacks AVX2. AVX2 code must use runtime dispatch and be
  executed on a Haswell-or-newer host or suitable CI runner.
- Hardware performance counters are unavailable on the development host. DRAM and energy
  claims cannot be measured here.
- Trained SmolLM2 semantic-routing and causal MLP-intervention reports are
  checked in, but no trained end-to-end compiled Gate 5 run exists.
  Learned-router artifacts and the older dense-SmolLM DIP runtime remain
  blocked. Separately, native-BitNet DIP passes its semantic gate by postmortem
  adjudication; it is not evidence that the dense-Llama compiler problem is
  solved.
  Engram downloads a model when a Hub ID is supplied explicitly; this can require substantial
  disk space, and gated models still require Hugging Face authentication and license acceptance.
- Synthetic Gate 3 mean relative L2 is 0.456 for the heuristic hybrid; retrieval/copying
  accuracy of 1.0 comes from a controlled synthetic case and must not be generalized.
- Synthetic Gate 4 adaptive control averaged 7.98/8 cycles and therefore did not demonstrate
  useful early exit.
- The native fixture benchmark is not a large-model benchmark. llama.cpp is unavailable locally,
  and no trained Hugging Face or llama.cpp comparison has run.
- Semantic and vocabulary IVF both reduce proxy-scored fixture rows from 64 to 32 per token,
  but coarse-centroid overhead makes the tiny logical-byte estimate worse, not better.
- The base Anaconda environment still has an older scikit-learn incompatible with NumPy 2.4.6,
  so Hugging Face integration tests skip there. The repository's conversion environment uses
  PyTorch 2.7.1 and Transformers 5.14.1; the local-Llama intervention tests pass when
  `LD_LIBRARY_PATH` is unset so `/opt/libtorch` does not override the wheel libraries.
