# Limitations

- The original dense-Llama limitation is unchanged: no serialized
  representation jointly passes the all-layer causal gate and complete cold
  MLP traffic at or below 45% of dense ideal Q4. DIP is a quality-only pass at 83.33% cache-line
  traffic; the mild-width compact-Q4 artifact is a traffic-only pass with
  KL 0.887, top-1 0.566, NLL +0.884, and hidden L2 0.425 after 3M positions.
  The full-width grouped-ternary artifact is another traffic-only pass:
  43.1353% traffic with KL 2.284, top-1 0.320, NLL +2.277, and hidden L2
  0.604 after 1,014,225 positions.
  See [Project status](status.md).
- The separate native-BitNet artifact reconstructs exactly and its direct
  memory-mapped CPU kernel passes the frozen causal gate at 40.0527% scheduled
  cold traffic. This does not say anything about losslessly converting an
  already-trained dense Llama checkpoint. Official-layer output is numerically,
  not bitwise, equal to PyTorch BF16 because reduction order differs (maximum
  checked relative L2 0.00982). No hardware DRAM counter was available, and
  package/generation integration is now implemented, but it retains the
  original attention, normalization, embedding, and output tensors and drives
  them through a Python Transformers shell. Only the MLP is a native C++
  kernel; this is not yet a complete native transformer runtime.
- Bounded attention is integrated into complete package generation and avoids
  the dense Hugging Face KV cache, but Q/K/V and output projections still run
  in PyTorch and every token crosses the Python/ctypes boundary at every
  layer. At a 256-token prompt, attention state is fixed at 7,477,440 bytes
  and modeled reads are 16.35% of dense, yet complete processing is only about
  one position per second. Logical byte counts are algorithmic float32
  interface counts, not hardware DRAM events. The long-context prompt is a
  deterministic repeated benchmark string and is not additional quality
  evidence.
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
- Attention state is constant through 2,048 tokens, but total process memory
  is not: peak RSS rises from 2.14 GB at 512 tokens to 2.57 GB at 2,048 because
  PyTorch/Transformers and transient prompt tensors remain.
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
- A predictor-free, DIP-inspired algorithm is the first realizable semantic selector to pass the
  all-layer gate. Engram's candidate-only completion and exact contribution reranking extend the
  published DIP method. After selecting 75%-input/896-candidate/K=768 on the development grid, a
  sequence-disjoint 16-sequence confirmation run has KL 0.029, top-1 agreement 0.910, NLL delta
  +0.033, final-hidden relative L2 0.090, and 0.990 candidate recall. This still covers only one
  135M-parameter model and a small generated corpus; another model and broader natural data are
  required before generalization.
- DIP now has a versioned coordinate-major experimental package, mmap loader, Python reference,
  and candidate-only native kernel. The optimistic scalar count is 76.4% of dense; counting
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
- Trained SmolLM2 semantic-routing and causal MLP-intervention reports are checked in, but no
  trained end-to-end compiled Gate 5 run exists. Learned-router artifacts remain blocked; the
  predictor-free DIP arm is quality-eligible and has an experimental serializer/kernel, but that
  kernel fails latency and is not integrated into the compiled runtimes. Attention/controller
  distillation and trained-package compilation remain pending.
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
