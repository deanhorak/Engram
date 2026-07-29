# Research log

## 2026-07-27 — Complete native OLMoE token boundary passes

- Added a streaming compiler for the exact 131-tensor BF16 non-MLP inventory.
  The pinned production artifact is 949,242,368 bytes and excludes every
  router/expert tensor.
- Added mapped C++ weight views and a transformer-shell-free token runtime:
  embedding, RMS norms, dense Q/K/V/O, full-width Q/K norms, absolute RoPE,
  persistent bounded attention, residuals, native Q7 experts, final norm, and
  independent `lm_head`.
- Fixture checks prove NumPy next-token parity, batch-prefill versus
  incremental-cache equivalence, cache-position advancement, and reset replay.
- Production single-position native output matches an independent serialized-
  artifact reference at token ID 21787. Stateful two-token decoding returns
  `[21787, 13]`.
- A meaningful five-token prompt, `The capital of France is`, predicts token
  7785 (` Paris`) without constructing Transformers. Prefill takes 13.6725
  seconds on 12 threads; Q7 accounts for 13.4510 seconds.
- Decision: the native token boundary passes. Next build an authenticated
  package and optimize single-row Q7 expert execution before expanding chat
  or benchmark claims.

## 2026-07-27 — OLMoE packed Q7 native systems gate passes

- Implemented `olmoe_native_groupwise_q7_v1`: canonical biased LSB-first
  seven-bit codes, BF16 group scales and routers, fixed little-endian
  directories, and cache-line-aligned layer/expert/phase blocks.
- The streaming compiler converted all 16 layers and 1,024 experts from the
  pinned checkpoint into one strictly validated 5,842,733,184-byte artifact
  without retaining a layer's dense expert set in memory.
- Added independent Python and C++ mmap readers. They reject code 127,
  malformed tails, invalid BF16 scales, inconsistent offsets, and nonzero
  padding. The native kernel never materializes a dense expert matrix.
- On the production artifact, native and independently decoded execution chose
  exactly the same top eight experts. Maximum absolute output error is
  1.63913e-7 and relative L2 is 1.94718e-6.
- One layer/state schedules 262,144 router bytes and 45,613,056 selected-expert
  bytes, totaling 45,875,200 bytes or 22.7865% of all-expert ideal Q4. The
  scalar one-thread step took 0.816 seconds; this is correctness, not a speed
  claim or hardware-counter DRAM measurement.
- Decision: the remaining native Q7 systems gate passes. Next integrate the
  artifact/kernel into a complete mapped OLMoE token-step and generation
  package; do not reopen semantic quantizer selection without new evidence.

## 2026-07-27 — OLMoE Q7 passes the full causal evidence screen

- Downloaded and payload-audited pinned OLMoE revision
  `9b0c1aa87e34a20052389dce1f0cf01da783f654`; all 3,219 local tensor shapes
  match the source contract.
- Captured the trained model's own router probabilities, top-8 expert IDs and
  weights, and exact MLP boundaries on CPU. An expanded 8×16 calibration trace
  contains 128 states for layers 0, 7, and 15.
- Group-8 Q4 passed those local layers but failed all-layer causally: KL
  0.22099, top-1 0.83333, NLL +0.16632, and hidden L2 0.19944. Q6/group-32
  improved KL/NLL/hidden but still missed top-1.
- Froze Q7/group-64 after its calibration smoke passed all quality thresholds.
  The separate 8-sequence/256-position confirmation then passed with KL
  0.00900774, top-1 0.9765625, NLL +0.00391912, and final-hidden L2 0.0460273.
- A serialization preflight found that some FP16 scales underflowed. The final
  confirmation therefore executes genuinely BF16-rounded scales, which use
  the same two-byte traffic but preserve the required exponent range.
- The complete modeled numerator is 734,003,200 bytes/token: selected packed
  Q7 codes, BF16 group scales, and all BF16 routers. This is 22.7865% of the
  3,221,225,472-byte all-expert ideal-Q4 baseline.
- Decision: the OLMoE branch clears semantic quality, traffic projection, and
  evidence count. It does not complete Milestone 2 until a serialized Q7
  artifact and direct CPU top-8 kernel reproduce the result with complete
  physical traffic accounting and no Transformers execution.

## 2026-07-27 — OLMoE source and exact expert contract pass

- Added a separate fail-closed OLMoE adapter rather than weakening the
  dense-Llama inspector. It validates the learned router and every independently
  stored expert gate/up/down tensor.
- Audited official revision
  `9b0c1aa87e34a20052389dce1f0cf01da783f654`. All 3,219 indexed names and
  shapes match. The verifier used bounded ranges for six safetensors headers
  and refused unbounded shard responses, avoiding a 27.68 GB weight download.
- The audit caught and corrected a real modeling error before weight download:
  OLMoE normalizes flattened Q/K projections, so its Q/K normalization vectors
  are width 2,048 rather than per-head width 128.
- The structural expert/router model projects 406,847,488 bytes per token
  across all layers versus 3,221,225,472 all-expert dense-Q4 bytes, or
  12.6302%. This excludes attention and physical/runtime amplification and is
  not a causal-quality pass.
- Added an exact NumPy top-k router/SwiGLU expert decomposition and deterministic
  trace schema containing router probabilities, selected IDs/weights, weighted
  expert contributions, and their exact summed output.
- Decision: advance OLMoE to trained router-trace and all-layer causal
  evaluation. Do not claim Milestone 2 or download/compile the full checkpoint
  solely from this structural result.

## 2026-07-27 — Sustained native bounded-attention mechanics pass

- Extended the streaming-attention, complete token runtime, versioned C ABI,
  and Python owner with cumulative local eviction, older-candidate scoring,
  older-value selection, sink insertion, and accepted heavy-hitter update
  counters. Existing ABI size and layout remain stable by assigning the v1
  metrics extension slots.
- Added a reproducible CPU-only confirmation at exact prompt lengths
  16/17/18/24/32. Every analytical count/bound passes, attention state stays
  at 7,477,440 bytes, and the 32-position token plus all structural counters
  replay after reset.
- The 32-position run records 480 layer evictions, 60,000 older-key scores,
  34,800 older-value selections, 1,200 sink insertions, and 5,654 accepted
  heavy-hitter updates. This exercises initial admission, full older-cache
  occupancy, replacement/rejection decisions, and bounded top-K reads.
- Rebuilt the standalone executable
  (`c6c5b05b6d8be72edd7f9e12e5e66c615859b74268143a5b2023b8dae423a15b`)
  and shared runtime
  (`4b732bebd049506e649007ce2b4fd4cd52d498a5cc121d39b2610637938ce72a`).
  The fixed 8×4 semantic-core regression still passes 32/32 tokens, 8/8
  prompts, 21.56017% activity, 41.16116% modeled traffic, and reset replay in
  390.4183 seconds.
- This closes sustained attention mechanics, not long-context semantic
  quality. The next Milestone 3 experiment needs natural tasks whose answers
  require context older than W=16 and a dense-attention comparison.

## 2026-07-27 — Authenticated native DIP chat binding

- Added a dedicated versioned C ABI around the production-pinned native DIP
  token runtime. Its constructor accepts only a package root, invokes the same
  authenticated loader as the standalone executable, and derives dimensions,
  bounds, W/C/K/S policy, thread default, semantic artifacts, controller, and
  EOS IDs from the package. Callers cannot substitute editable routing or
  attention policy.
- The DSO SHA-256 is
  `df3a4f70952cddaebff2e5198d9ddf6b5e8a25487020c40b89ec99f2c7d33f96`.
  It has SONAME `libengram_bitnet_token_runtime.so.1`, depends only on system
  libraries, hides C++ symbols, and exports six versioned C symbols. The
  rebuilt standalone executable SHA-256 is
  `29526c9838ea484d8a21887dafeaba99a57348e7377e0de4138e0631dde10fad`.
- Added `NativeBitNetDIPTokenRuntime`, a narrow Python `ctypes` owner for that
  handle. `chat-native-bitnet` now uses it without constructing a Transformers
  model shell, Torch model tensors, original decoder layers, or dense semantic
  fallback. Python retains the authenticated tokenizer, packaged chat
  template, and conversation history. Each turn resets the native handle and
  re-prefills the full rendered history.
- A raw-token comparison generated token `9906` (`Hello`) in both the
  standalone executable and the Python/C ABI path, with identical 30 semantic
  calls, 240 semantic rows, 361,598 selected records, and 2,624,024,064
  modeled semantic bytes. Reset on the same mapped handle reproduced the token
  and every non-timing structural counter.
- The real interactive CLI generated `Hello` from a 17-token rendered prompt,
  crossing W=16 and reporting 7,477,440 attention-state bytes. This is a
  boundary smoke, not sustained older-memory validation.
- Reconfirmed the rebuilt core on the fixed eight-prompt/four-token protocol:
  32/32 greedy tokens, 8/8 exact prompts, 21.56017% global activity, 41.16116%
  complete modeled traffic, and exact reset replay. The 12-thread run took
  397.3352 seconds. The next boundary is a sustained long-context protocol
  with explicit eviction, older-candidate, sink, and heavy-hitter evidence.

## 2026-07-26 — Adjudicated DIP enters the native token runtime

- Added an immutable-source semantic-memory promotion step. It verifies the
  frozen policy (`c572754e…3768e`), passing adjudication
  (`ebb5ca95…a5cc`), base records (`4fcf598a…ab55`), and v2 coordinate index
  (`b98ce4e4…0e15`), then atomically creates
  `work/native_bitnet/model.engram-bitnet-dip`. It never modifies the package
  bound into the frozen policy.
- The derived manifest records
  `native_bitnet_dynamic_input_pruning_v2` as its MLP mode, authenticates the
  semantic source/policy/adjudication chain, declares all 30 MLPs substituted,
  and forbids a dense fallback. The v2 index's authenticated layer headers are
  the runtime policy.
- Replaced the complete C++ token runtime's dense semantic object with the DIP
  kernel. Each layer now directly executes native attention, semantic-input
  normalization, DIP, and semantic-output acceptance. It reports semantic
  calls, rows, selected records, separate kernel/global-metadata modeled
  traffic, and timing.
- Hardened the standalone binary around a compiled deployment trust root. It
  authenticates exact manifest SHA
  `707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926`,
  rejects symlinks and inventory drift, checks all semantic provenance hashes,
  and derives architecture, attention, bounds, RoPE/RMS, paths, and EOS IDs
  (`128001`, `128009`) from authenticated package files. The executable SHA is
  `0f6cf41c9c14dc3e05a8cad7a01f4f9909bd355f4e27f9296d6c1e15ba91dea4`;
  it directly links the kernels and has no Engram shared-library dependency.
- Ran a fixed non-holdout eight-prompt/four-token integration confirmation.
  All 32 greedy tokens and all 8 complete continuations match the frozen native
  controller-stage reference. Global/maximum-prompt mean activity is
  0.2156017260/0.2258916324. Complete traffic is 30,153,074,432 modeled bytes,
  including 194,304 global-metadata bytes; global/maximum-prompt fractions are
  0.4116115605/0.4129835480. Every backend, position, traffic-recomputation,
  stage/semantic-call, row, token-budget, and reset-replay check passes.
- The 12-thread suite took 395.3581 seconds across first generations, reset
  replays, and per-process package authentication. This is a correctness pass,
  not a speed claim; traffic remains modeled rather than measured DRAM.
  Reported native counters and phase timings are first-generation snapshots.
  Reset proves repeated tokens, zeroed counters, and structural metric parity,
  not hidden-state identity. Token agreement likewise is not hidden/logit
  parity.
- The longest processed context is 14 positions, below W=16. Eviction and
  older-context retrieval are not exercised by this integration suite.
- The current Python `chat-native-bitnet` shell does not implement DIP and now
  rejects the derived package rather than silently falling back. The next
  boundary is a native-runtime C/Python handle for packaged tokenization/chat
  orchestration.

## 2026-07-26 — Native BitNet semantic gate passes by postmortem adjudication

- Consumed the one authorized independent 8-sequence/256-position holdout with
  the frozen native-BitNet DIP policy and host-bound artifact/library hashes.
  The CPU-only raw evaluator substituted all 30 MLPs, reloaded the serialized
  index, used no dense fallback, and completed its separate untimed
  teacher-recall diagnostics.
- The raw report passes every frozen primitive threshold: KL 0.0040412880,
  top-1 agreement 0.98828125, NLL delta +0.0048289299, final-hidden relative
  L2 0.0477494113, mean active fraction 0.2138000677, modeled physical cold
  traffic 0.4113713394, global micro candidate recall 0.9994058295, and
  worst-layer mean recall 0.9939428640.
- The original one-shot wrapper nevertheless recorded
  `final_holdout_consumed_with_error`. Its post-evaluation verifier compared
  the protocol's full token sequences hashed with the canonical `input_ids`
  object envelope against the evaluator's first 33 scored tokens hashed as
  bare lists. This was a verifier-contract defect, not a failed quality,
  activity, traffic, recall, parity, or execution measurement.
- Preserved the error result and marker without rewriting them. A separate
  no-model/no-evaluator postmortem adjudicator reconstructed the two historical
  hash schemas, verified full-sequence identities and scored-prefix lengths,
  and checked the frozen authorization, implementation, artifacts, raw
  primitives, and evaluator attestations. Its decision is
  `milestone_2_semantic_gate_passed_by_postmortem_adjudication`.
- Evidence custody is imperfect and explicitly retained as a limitation. The
  raw evaluator report was prospectively hash-sealed about 13 minutes after
  the original error; it was not contemporaneously bound by that result. The
  checked-in holdout was plaintext and separation was procedural, not
  cryptographic.
- The final sparse pass took 295.3364 seconds versus 257.9552 seconds dense,
  or 1.1449x dense. Latency was not a frozen upper-bound gate, and this is not
  a speedup. The 41.1371% traffic result is deterministic cache-line modeling,
  not a hardware-counter DRAM measurement.
- Decision: the native-BitNet practical semantic-memory gate passes by
  adjudication. This does not solve dense-Llama conversion, establish
  independent-host replication, or declare every integration item in
  Milestone 2 complete.

## 2026-07-26 — Native BitNet practical DIP passes development and freezes

- Implemented a CPU-only, memory-mapped native DIP kernel and source-bound v2
  coordinate index. The sparse pass accepts live BF16 MLP boundaries, performs
  native Q8 input quantization, scans the 1,920 largest-magnitude input
  coordinates, exactly completes per-layer candidates, estimates the shared
  intermediate RMS, and reads only token-adaptive selected down rows. It never
  calls the dense full-record MLP during sparse inference.
- Froze `q=1920`, `minK=346`, energy target 1.0, and explicit per-layer C and
  Kmax schedules. The token's K is its positive candidate-utility count clipped
  to `[346,Kmax]`, not a fixed density. Layers other than 9 use candidate-ratio
  RMS. Layer 9 uses corrected-proxy RMS and an eight-record
  top-proxy-raw-square audit inside its unchanged candidate union.
- The live native-BF16 8-sequence/256-position development run passes every
  frozen threshold: KL 0.0044707, top-1 0.94921875, NLL delta +0.0013609,
  final-hidden relative L2 0.0498965, mean active fraction 0.2008072, modeled
  physical traffic 0.409639, global candidate recall 0.9995917, and
  worst-layer mean recall 0.9939353.
- Candidate recall uses a fixed router-independent dense-teacher top-K schedule
  in a separate untimed diagnostic pass; adaptive K is not its denominator.
  Six rows in every layer have bit-exact Python/native input-coordinate,
  candidate, selected-record, selected-count, and BF16-output parity.
- The 216,688,448-byte index plus 318,924,544-byte base records occupy
  535,612,992 bytes, or 67.2659% of dense Q4. The 40.9639% per-token result is
  modeled cache-line traffic, not measured DRAM.
- The sparse end-to-end development evaluation takes 1.1565x the dense elapsed
  time, so the gate pass is not a speedup claim.
- Froze the policy, artifact, tokenizer, library, protocol, report, and parity
  bindings for one sealed final confirmation. At this development checkpoint,
  that advanced Milestone 2 from “no practical router” to a
  development-qualified route awaiting confirmation; the later adjudication
  above records the final semantic-gate decision.
- The independent holdout is a plaintext repository fixture. Its non-use
  before the one-shot run is procedural and honor-system-based, backed by a
  fail-closed runner and committed hashes rather than cryptographic secrecy.

## 2026-07-26 — BitNet oracle semantic ceiling passes

- Corrected the milestone accounting: lossless full-record BitNet execution is
  a systems substrate, not a Milestone-2 practical-routing pass.
- Added an exact additive-record oracle for the trained BitNet teacher and a
  direct CPU top-K kernel. Full-width oracle execution is bit-exact with the
  existing packed kernel in fixture tests.
- The first fixed 25% frozen run passed KL, top-1, and NLL but narrowly missed
  final-hidden relative L2 at 0.10448.
- A development-only all-layer sweep moved budget among layers while preserving
  a strict aggregate ceiling. The chosen 15–35% schedule averages 24.8375%.
- On the untouched 8-sequence/256-position protocol the adaptive oracle passes:
  KL 0.02543, top-1 0.94531, NLL delta +0.02386, and hidden L2 0.09205.
- At this historical checkpoint, Milestone 2 remained blocked at practical
  selection. The native DIP development pass recorded above subsequently
  supplied the missing practical route.

## 2026-07-26 — Analytic BitNet router crosses representative recall screen

- A nonlinear rank-64 membership router recalled 74.33% of layer-14 oracle
  records with 1.5x candidates. Rank 256 improved only to 77.75%; this learned
  family is rejected before causal work.
- A BitNet-specific Dynamic Input Pruning router instead approximates the
  teacher gate/up dot products from the largest input coordinates. The
  coordinate-major ternary index is charged at five trits per byte, and every
  candidate is charged as a complete gate/up/gain/down record.
- With 75% input coordinates, held-out mean recall reaches 96.23% at layer 0,
  98.06% at layer 14, and 96.78% at layer 29. Modeled traffic is respectively
  about 35%, 35%, and 41% of dense ideal Q4.
- Fixed a recall-harness defect: exact-zero oracle ties now use stable ordering
  rather than arbitrary `argpartition` membership.
- At this historical checkpoint these were representative-layer screens, not
  a Milestone-2 pass. The later all-layer native development run recorded
  above completed the listed work and froze the one-shot-final policy.

## 2026-07-25 — Package-owned native controller boundary

- Added authenticated schema-v3 controller installation to native BitNet
  packages. The installer validates dimensions, exact-residual mode, zero
  correction scales, all existing package hashes, and refuses to overwrite a
  different controller.
- Added a float32 C ABI residual/RMS kernel to `libengram_bitnet.so`. It returns
  both normalized state and the relative RMS carried into the next stage.
  C++ and NumPy parity tests pass.
- Controller-driven generation now defaults to the package manifest's
  controller when no external path is supplied and uses the native residual
  kernel for every stage.
- The installed working package generates `[12366, 13, 12366, 374]`
  (` Paris. Paris is`) with the package-owned controller, nine correct cache
  positions, native controller mode, and zero decoder-layer calls.
- Added first-class install and generation CLI commands. Remaining work is
  moving normalization/operator orchestration and the final head out of the
  Torch module shell into a complete native C++ generation runtime.

## 2026-07-25 — Incremental controller generation matches exactly

- Added `ControllerDrivenBitNet`, an explicit 30-stage package loop that calls
  stage normalization, persistent native attention, native packed MLP, and the
  schema-v3 controller without invoking a decoder layer's `forward` method.
- The runtime carries normalized width-2,560 state plus one scalar residual RMS
  per token. This preserves the operator-output scale needed by residual
  addition while leaving the controller vector normalized.
- A real-package development prompt produced identical four-token output
  `[12366, 13, 12366, 374]` (` Paris. Paris is`) in the decoder-reference and
  controller arms. Nine prompt/decode cache positions advanced exactly and
  decoder-layer calls remained zero.
- Added a fixed prompt-suite evaluator with predeclared requirements: eight
  prompts, 32 reference tokens, at least 90% weighted token agreement, at
  least 75% exact-prompt parity, all cache positions correct, and no decoder
  layer calls.
- The frozen suite passes every check with 100% token agreement and 100% exact
  prompt parity. Controller arithmetic averages 0.0427 seconds per prompt,
  while complete controller-driven execution averages 22.581 seconds. Maximum
  reported controller state is 112,684 bytes.
- The next boundary is package-native installation and execution. Python/Torch
  still orchestrates stage modules, the controller directory is external to
  the package manifest, and the residual/RMS loop is not yet a native C++
  kernel.

## 2026-07-25 — Compiled operators pass controller replay

- Corrected operator provenance: controller traces are captured from
  `NativeBitNetRuntime`, which replaces all source MLP modules with the direct
  packed CPU phase-stream kernel. Their semantic outputs were already
  compiled; dense attention was the remaining teacher operator.
- Added `evaluate-native-bitnet-controller`, which runs a dense-attention
  package baseline and a native W16/C8/K4/S2 attention candidate, captures
  only compiled MLP/attention outputs, replays them through the schema-v3
  controller outside the decoder residual scaffold, and applies the package
  final norm and language-model head.
- A 2-sequence/32-position development run passed every quality check; only
  the deliberately undersized sample checks failed. The configuration then
  advanced unchanged to the sealed split.
- On eight unique sequences and 256 positions at record offset 8, every frozen
  check passes. Controller replay versus the dense-attention baseline has KL
  0.011125, top-1 agreement 0.957031, NLL delta -0.008285, and final hidden
  relative L2 0.075893.
- Replay versus the compiled candidate has hidden relative L2 0.006810 and
  terminal trajectory NMSE 0.000026666. The controller adds only 0.255 seconds
  for 8 x 33 x 30 stage transitions, versus 116.27 seconds for the compiled
  operator model pass.
- This passes compiled-operator/controller integration at the replay boundary.
  Incremental generation remains: controller state must directly dispatch the
  native operators without running decoder layers to obtain their outputs.

## 2026-07-25 — Exact operator residual passes the controller gate

- A controlled rank-4 stage input-adapter experiment on the unchanged
  1,024/256-position traces reduced protected terminal normalized MSE only
  from 0.159440 to 0.157431. The 1.26% gain rejects stage-conditioned input
  alignment as the main controller limitation.
- Re-examined the teacher and trace contracts. Each layer output is exactly
  the incoming residual plus its captured attention and MLP outputs, followed
  by the next boundary's RMS normalization. The learned controller had been
  compressing this known additive operation through rank 128.
- Added a backward-compatible schema-v3 controller that preserves semantic
  and episodic residual additions exactly and uses the shared factorized
  network only as an optional correction. Schema v1/v2 artifacts retain their
  original layouts and metadata.
- On the frozen protected validation trace, the zero-correction artifact
  reaches terminal normalized MSE 0.000020801, mean hidden normalized MSE
  0.000017685, and cosine loss 0.000008841. It passes the fixed 0.0225 gate
  with 1,081.7 times margin and reloads from NumPy within 5.72e-6 of Torch.
- The PyTorch-free CPU hot path skips all factorized matrices when correction
  scales are zero. A 256-state, 30-stage batch takes median 0.1847 seconds,
  or 41,575.9 stage transitions/s.
- This opens native-attention substitution, not end-to-end qualification.
  Semantic outputs already come from the packaged CPU MLP kernel, while
  attention outputs remain dense in this trace. The next experiment replaces
  attention and replays both operator outputs through the controller.

## 2026-07-25 — 1,024-position controller scale rung rejects blind scaling

- Froze a larger protected protocol before execution: 64 training sequences
  and 16 validation sequences, 16 positions each, batch-four CPU teacher
  capture, a fresh rank-128/adapter-rank-4 controller, 1,000 CUDA steps, and
  the unchanged terminal normalized MSE gate of 0.0225.
- Captured 1,024 training positions in 16 checksummed shards and 256
  validation positions in four shards. Capture took 1,044.2 and 276.2 seconds;
  the dataset hashes are distinct.
- The CUDA fit took 235.1 seconds. Protected terminal normalized MSE improves
  from the prior 0.245010 to 0.159440, cosine loss averaged across stages from
  0.333417 to 0.272803, and total loss from 1.156465 to 0.931534. CPU reload
  parity passes at 5.90e-6.
- The fixed substitution gate still fails by 7.1 times. A crude two-point
  scaling fit has exponent about 0.207 and would imply roughly 13.4 million
  positions to reach 0.0225 if that slope held. This extrapolation is only a
  diagnostic; it makes another blind capture rung unjustified.
- Stagewise self-fed NMSE is 1.077929/0.848588/0.679043/0.530822/0.419096/
  0.293215/0.159440 at stages 1/5/10/15/20/25/30. Exact later teacher
  operator outputs steadily repair the initial mismatch. The present
  controller adapts recurrent state per stage but sends every stage's token,
  semantic, and episodic vectors through one shared rank-128 input projection.
- The next justified architecture experiment is a small stage-conditioned
  low-rank input adapter into the shared bottleneck. It directly tests the
  diagnosed alignment failure while adding only bounded CPU-resident
  parameters. Compiled semantic/episodic substitution remains sealed.

## 2026-07-25 — Batched capture and protected controller scaling

- After a host reboot, the loaded NVIDIA module and userspace library both
  report 580.173.02 and PyTorch again sees the RTX 3050. Repeating the exact
  500-step rank-128 experiment on CUDA yields terminal validation MSE 0.246530
  and cosine loss 0.335160, close to the CPU run's 0.245010 and 0.333417.
  Training takes 112.6 rather than 131.8 seconds, and the exported artifact
  reloads on CPU within 5.36e-6 maximum absolute error. Optimizer device is
  therefore not the current quality limiter.
- Added multi-sequence padded teacher forwards, per-record deterministic
  cropping, batch-level checksummed shards, progress reports, record offsets,
  and restart support. Resumption verifies all prior shard hashes and skips
  captured sample IDs. An orphan shard without a manifest is rejected instead
  of overwritten.
- A two-sequence batch is bit-identical to the original single-sequence path
  for token IDs, embeddings, all 31 residual states, all 30 MLP outputs, and
  all 30 attention outputs.
- Captured 128 training positions from eight sequences and 64 protected
  validation positions from four sequences. Dataset hashes differ and every
  shard passes checksum validation.
- The rank-128, adapter-rank-4 controller first trained for 500 steps on CPU
  while the host had a temporary NVIDIA kernel/userspace mismatch. The
  post-reboot CUDA reproduction above confirms the result.
- Protected terminal normalized MSE improved from 1.998608 to 0.245010,
  cosine loss from 0.973363 to 0.333417, and total loss from 4.292672 to
  1.156465. This substantially improves the earlier 8/8-position result
  (terminal MSE 0.522350) but remains outside a defensible substitution range.
- Added artifact-based continuation and a zero-teacher-forcing mode. A
  500-step lower-rate continuation improves training terminal error from
  0.060627 to 0.029324 but regresses protected validation to 0.260050 and
  fails the development gate. More optimization on this narrow corpus is
  stopped.
- The retained artifact reloads without Torch, matches the trained operator
  within 7.45e-6 maximum absolute error, and runs 64 states through 30 NumPy
  CPU cycles at 79.7 states/s. The next justified work is broader,
  sequence-diverse trajectory capture after the host CUDA stack is restored;
  compiled semantic/episodic substitution remains sealed.

## 2026-07-24 — CUDA-assisted shared-controller distillation begins

- Replaced the impractical trained-model controller target with a factorized
  residual-gated recurrence. The original width-2,560 FP64 GRU fixture would
  store about 629 MB in its two large kernels. The checked rank-128,
  adapter-rank-4 controller stores 10,649,720 FP32 bytes while sharing its core
  across all 30 depth stages.
- Added durable BitNet trajectory capture from the qualified packaged CPU
  teacher. Traces keep token embeddings, layer states, MLP outputs, attention
  outputs, source/dataset hashes, and split identity. Teacher residual values
  exceed FP16 range, so the final contract records per-token RMS-normalized
  states and operator outputs divided by the incoming residual RMS.
- Added CUDA optimization with intermediate-state, transition-delta, cosine,
  and terminal rollout losses; scheduled teacher forcing; gradient clipping;
  protected validation checks; FP32 NumPy serialization; and independent
  PyTorch-free CPU parity.
- A protected eight-training/eight-validation-position micro-run reduced
  validation loss from 4.306441 to 1.881813, terminal normalized MSE from
  2.003824 to 0.522350, and cosine loss from 0.977434 to 0.532363. CPU reload
  passed at 7.15e-6 maximum absolute error. This is a development/infrastructure
  pass only; exact teacher operator outputs are still inputs.
- The staged objective follows the progressive-granularity lesson in
  [MOHAWK](https://arxiv.org/abs/2408.10189), while depth-shared parameters and
  explicit stage identity follow the recurrent-depth motivation of
  [Universal Transformers](https://arxiv.org/abs/1807.03819). The next run
  must scale protected data, then replace teacher operator outputs with the
  compiled semantic-memory and episodic-attention outputs during rollout.

## 2026-07-24 — Whole-model Q-Sparse campaign and CPU deployment boundary

- Restored host access to the RTX 3050 and made the architecture boundary
  explicit: CUDA may accelerate training and distillation, while serialized
  artifacts and inference remain CPU-only.
- Added an all-layer exact Q-Sparse trainer. It applies hard top-K to the input
  shared by gate/up and to the down input, uses an identity STE only during
  training, freezes the dense teacher, supports causal hidden/logit/local
  distillation and next-token continuation, saves resumable device-neutral
  checkpoints, and independently reloads every candidate on CPU.
- Added an authenticated 128-sequence tail holdout from the pinned 10M-token
  pretraining-mixture corpus. It contains 15,559 prediction positions, is
  disjoint from the 81,647-record training prefix by exact token-sequence
  hash, and remains separate from confirmation.
- The uniform 43.967%-traffic baseline reaches KL 0.742 and top-1 0.615 on
  the selection set. A downstream-KL single-layer sensitivity fit produces
  a fixed schedule at exactly 45% ideal traffic, with `q <= 360/576` and
  `K <= 512` everywhere. It generalizes to the unseen holdout at KL 0.457,
  top-1 0.669, NLL delta +0.474, and hidden L2 0.328.
- Same-input teacher targets correct an initially harmful local objective.
  MLP-only training is nearly flat. Verified attention/normalization
  co-adaptation is the best arm but after 128 batch-four steps reaches only
  KL 0.452, top-1 0.671, NLL +0.458, and hidden L2 0.327.
- Full-model label-only continuation at batch eight, a token-adaptive
  concentration policy, and a traffic-charged rank-24 correction were also
  measured. Label-only continuation is flat; the adaptive policy violates
  traffic and collapses quality; the residual loses to spending its bytes on
  sparse reads. None opens confirmation.
- The dense-source Q-Sparse scale ladder is stopped on this corpus. Published
  continuation uses a far larger token budget, and the measured host-scale
  slopes do not support extrapolation to the unchanged gate.
- Historical correction (2026-07-26): the direct BitNet phase-stream kernel
  executes every record and therefore is not a qualifying Milestone 2 semantic
  router. Controller work built valuable systems infrastructure but did not
  close the semantic-memory gate.

## 2026-07-24 — Exact activation-sparse dense-source screens

- Researched ProSparse, CATS, Q-Sparse, and transformer-to-SSM distillation
  against the existing failed router/codec ledger. The actionable distinction
  is that activation-sparse training changes the teacher computation instead
  of trying to predict diffuse source-neuron utility after training.
- Added a leakage-safe exact-gate screen. It fits per-layer CATS/FATReLU
  thresholds on calibration traces and evaluates disjoint development
  boundaries. A full gate scan plus active up/down reads costs
  `(1 + 2a) / 3` of dense MLP weights, so the 45% ideal-Q4 gate requires at
  most 17.5% active records before metadata.
- Zero-shot CATS at 82.5% target sparsity reaches 17.39% actual activity and
  44.93% ideal traffic, but mean local relative L2 is 0.511. FATReLU is worse
  at 0.591. Thresholding the unchanged teacher is rejected.
- Added hard-forward, soft-backward boundary training with a dense warmup,
  sine-squared threshold ramp, continuation artifacts, and exact traffic
  reporting. On layer 14, 1,024 progressive plus 4,096 fixed-budget updates
  lower threshold-gate error from 0.615 to 0.470 at 43.48% ideal traffic. The
  curve flattens far above the 0.18 screen.
- Added Q-Sparse-style exact top-K execution. Gate and up read the same
  activation-selected input coordinates; down reads only selected
  intermediate coordinates. No router or candidate recall is involved.
  The selected `q=282/576`, `k=522/1,536` point costs 43.967% of dense ideal
  Q4 before metadata.
- On the same held-out layer-14 boundary set, that Q-Sparse point starts at
  0.343 local error. A 1,024-step progressive stage followed by 4,096
  fixed-budget updates reaches a best 0.3228 and ends at 0.3233. This is
  stronger than exact gate thresholding but still fails the 0.18 progression
  screen, so no causal or confirmation corpus was opened.
- The local retrofit is rejected. The whole-model hypothesis and accelerator
  scale ladder were subsequently implemented and are reported in the section
  above.

## 2026-07-24 — Chat-template-aware native CLI

- Added `chat-native-bitnet`, which loads the optimized package once, keeps
  structured system/user/assistant history, renders it with the tokenizer's
  packaged chat template, and re-prefills a fresh bounded cache each turn.
- Added `/history`, `/reset`, `/quit`, `/exit`, EOF, unknown-command, and
  interrupt behavior. Unit tests verify that the second rendered prompt
  contains the first assistant answer and that reset removes prior turns.
- A real package smoke test generated `The capital` for a two-token France
  question turn in 10.50 seconds, reported 7,477,440 attention-state bytes,
  displayed history, reset to empty history, and exited cleanly.
- A longer observed session generated a 32-token poem in 166.43 seconds. On
  the next turn, `awesome!` elicited a contextually appropriate acknowledgment
  and another poem in 153.15 seconds. Both turns reported the same
  7,477,440-byte native attention state. This is direct evidence that packaged
  chat-template history reaches generation across turns; it is not a broad
  quality evaluation.
- Persistent cross-turn cache reuse and token streaming remain explicitly
  deferred. Current single-row CPU decode is too slow for an interactive user
  experience despite the working interface.

## 2026-07-24 — Complete inference stack passes first behavioral validation

- The complete optimized stack passes frozen records 8–15 over 256 positions:
  KL 0.01315, top-1 0.92969, NLL delta +0.00365, and hidden L2 0.08436.
- Added EOS-aware generation and a JSONL prompt-suite evaluator. Eight natural
  prompts generate 16 tokens each without collapse or identical-token runs.
  Outputs are recognizably coherent, with weaknesses on code and
  testing-strategy prompts. Mean throughput is 0.194 token/s.
- Actual-stack one-segment and split-prompt final logits are bit-identical.
  Cache reset reproduces the same tokens and stable counters. EOS termination
  is covered by a unit test.
- Complete prefill succeeds at 512/2,048 tokens in 24.10/81.77 seconds.
  Attention state stays at 7,477,440 bytes; modeled reads are 8.40%/2.14% of
  dense. Peak process RSS is 2.14/2.57 GB.
- A 33-token prefill takes 4.65 seconds; seven decode model steps take 38.26
  seconds (5.47 seconds/step). The engine works but is not interactive.

## 2026-07-24 — Exact last-logit generation removes vocabulary bottleneck

- Audited the existing vocabulary IVF path. It requires a duplicate float32
  normalized embedding matrix (about 1.31 GB for this model) plus an index.
  A zero-metadata partial-coordinate screen can recover the development greedy
  token, but approximate search is unnecessary for the actual generation API.
- BitNet already accepts `logits_to_keep`; package generation had used the
  default zero and projected all prompt positions even though greedy decoding
  consumes only the final one. Both dense-cache and bounded-cache generation
  now request exactly one logit row.
- With packed native projections, the 33-token run falls from 22.29 to 10.16
  seconds. Vocabulary projection falls from 13.00 to 0.83 seconds, with
  identical generated tokens.
- At 256 prompt tokens plus two decode tokens, total time falls from the prior
  254.23-second stream-fused run to 20.72 seconds (91.8%). Processing rises
  from 1.01 to 12.40 positions/second. Attention state remains 7,477,440 bytes
  and modeled attention reads remain 16.35% of dense.
- Approximate vocabulary indexing is stopped for greedy generation. The exact
  head no longer dominates; the packed MLP now consumes 13.07 of 20.72
  seconds.

## 2026-07-24 — Packed native attention projections pass development screen

- Added a shared threaded C++ kernel for the official BitNet
  four-output-codes-per-byte layout. It performs the existing per-row Q8
  activation quantization and ternary projection without expanding weights to
  BF16. A deterministic fixture matches the materialized BitLinear within the
  declared numerical tolerance.
- A real layer-0 Q projection at 33 rows runs in 0.0116 seconds versus 0.2443
  seconds materialized (21.0×), with relative L2 0.000524.
- Replacing all 120 packaged Q/K/V/O modules preserves the expected ` Paris.`
  generation. At 33 prompt tokens plus two decode tokens, projection time
  falls from 19.31 to 3.01 seconds and total time from 38.51 to 22.29 seconds,
  a 42.1% end-to-end reduction.
- Direct comparison to the materialized-projection model on 32 trained
  next-token positions reaches KL 0.003945, top-1 0.96875, NLL delta
  −0.000369, and final-hidden relative L2 0.035325.
- Frozen records 8–15 then pass over 256 next-token positions: KL 0.005478,
  top-1 0.957031, NLL delta +0.002001, and hidden L2 0.058874. Native
  projection execution takes 111.38 seconds versus 256.56 seconds
  materialized on the same batch. The path is promoted.
- The tied full-vocabulary projection now consumes 13.00 seconds and is the
  dominant measured phase.

## 2026-07-24 — Stream-call fusion rejected; projection phases isolated

- Added a position-major native stream ABI so an entire prompt segment crosses
  the C boundary once per layer instead of once per token. A 23-token stream is
  bit-identical to 23 individual steps and reports identical accumulated
  traffic.
- The controlled 256-token package run improves only from 255.64 to 254.23
  seconds (0.55%) with identical tokens, state, and logical reads. Python call
  count is therefore not the material bottleneck.
- Phase timing on a complete 33-token prompt plus two-token decode attributes
  11.60 seconds to Q/K/V projections, 7.71 seconds to attention output
  normalization/projection, 12.62 seconds to the vocabulary projection, 5.94
  seconds to packed native MLP calls, 0.12 seconds to bounded native attention,
  and 0.06 seconds to RoPE. These measured phases explain 98.8% of the
  38.51-second total.
- Cache-policy tuning and further ABI-call fusion are stopped. The next native
  boundary must execute the packaged ternary Q/K/V/O projections without
  materializing BF16 matrices; the tied vocabulary path is the next independent
  target.

## 2026-07-24 — Stateful attention enters complete package generation

- Replaced each package transformer's dense attention module with a persistent
  W=16/C=8/K=4 native cache. Prompt prefill and incremental decode now share
  that state, while the model applies normal BitNet RoPE using explicit,
  monotonically increasing absolute positions. Hugging Face `DynamicCache`
  allocation is disabled.
- Full-sequence and uneven 6/4/1-token chunk execution are bit-identical for
  the same bounded operator. The runtime rejects skipped or repeated positions
  and batch-size changes without reset.
- Complete 30-layer package generation was measured at 33, 128, and 256 prompt
  tokens plus two greedy tokens. All-layer attention state remains exactly
  7,477,440 bytes. Logical attention reads fall from 86.55% to 31.07% and
  16.35% of the dense query-head counterfactual.
- Total elapsed time is 39.06, 131.97, and 255.64 seconds, about
  0.87–1.01 processed positions/second. Packed MLP calls account for only
  5.69, 8.97, and 12.65 seconds. Native Q/K/V/output projection, removal of
  per-token Python/ctypes crossings, and a bounded vocabulary path are now
  higher-value systems work than further tuning the isolated cache.
- The report contains modeled interface bytes, not hardware DRAM counters, and
  the repeated benchmark prompt is not semantic-quality evidence.

## 2026-07-24 — Native bounded attention kernel and long-context crossover

- Added a stateful C++20 W/C/K attention kernel and C ABI implementing the
  exact local ring, sink retention, cumulative-attention heavy replacement,
  candidate-key rerank, selected-value softmax, reset, and bounded metrics.
- Found and fixed an eviction semantic edge case: when the heavy cache is
  full, an incoming token below the current minimum must be discarded rather
  than forcibly replacing a heavier entry.
- Native C++, ctypes/NumPy, and transformer replacement tests agree across
  randomized multi-head eviction sequences. All 15 native tests pass.
- Trained one-sequence native substitution reaches KL 0.00528, top-1 0.96875,
  NLL +0.01239, and hidden L2 0.04210. Its 34.28-second transformer run is
  close to the 34.79-second dense baseline and faster than the 47.51-second
  Python-cache reference on the same input.
- A standalone W=16/C=8/K=4 benchmark keeps per-layer state fixed at 249,248
  bytes. Logical reads are 87.88%, 31.29%, 8.40%, and 2.14% of dense at
  lengths 33, 128, 512, and 2,048. Counts are at the query-head kernel
  interface; it motivated the complete generation benchmark above.

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
- Historical correction (2026-07-26): the low-bit-native track passes causal
  quality and serialized cold-byte checks, but executes every MLP record. It
  does not pass routed semantic-memory Milestone 2. Dense-Llama conversion
  also remains blocked. See the
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

## 2026-07-26 — Native controller shell boundary

- Added C ABI implementations for BF16 embedding lookup, BitNet-ordered
  RMSNorm, default RoPE, exact residual/RMS advancement, and threaded
  tied-vocabulary argmax.
- Removed Torch embedding, 92 RMSNorm calls, RoPE construction/application,
  and full-logit materialization from controller-driven package generation.
- Preserved the `The capital of France is` smoke sequence exactly while
  reducing its four-token runtime from about 21.3 to 18.6 seconds.
- The unchanged eight-prompt/32-token protocol passes with 96.875% weighted
  token agreement, 87.5% exact prompts, exact cache positions, and zero
  decoder-layer calls. One final-token BF16 near-tie differs because the
  scalar native vocabulary dot product and PyTorch/oneDNN accumulate in
  different orders.
- Full validation passes: 451 Python tests and 15 native tests. The remaining
  shell is stage orchestration and Torch tensor views around already-native
  projections, MLP, and attention kernels; the next boundary is one C++
  package-runtime handle for the complete 30-stage loop.
- Added the first persistent stage-runtime handle. It owns normalized state,
  residual RMS, attention and post-attention workspaces; emits BF16 normalized
  operator inputs; accepts BF16 attention/semantic outputs; and rejects
  out-of-order calls. Real package generation is token-identical, controller
  bookkeeping falls to 11.4 ms on the four-token smoke prompt, and validation
  advances to 452 Python plus 16 native tests.
- Fused the semantic half of a stage: native post-attention normalization,
  direct packed MLP execution, metric preservation, residual insertion, and
  state renormalization now occur in one C call. The real four-token output is
  unchanged, elapsed time is 18.15 seconds, and corrected controller/shell
  overhead is 10.4 ms. Attention projection/cache dispatch remains the next
  boundary.
- Fused the attention half and complete depth loop. A descriptor-driven C++
  runner now executes all packed projections, RoPE, persistent bounded
  attention, MLPs, and exact residual transitions for 30 stages in one call.
  The frozen 8-prompt/32-token protocol passes at 96.875% weighted agreement,
  87.5% exact prompts, correct cache positions, and zero decoder-layer calls.
  Milestone 4 orchestration is closed; native package loading and per-token
  generation control are the next Milestone 6 boundary.
- Added a strict mmap safetensors reader and native BitNet non-MLP weight
  binder. The real 780,054,616-byte file validates at 332 tensors; the binder
  resolves 30 layers, a 128,256x2,560 tied embedding, every norm vector, and
  all 120 packed attention projections. Mapped projection registration avoids
  the previous approximately 120 MB copy and reports zero copied projection
  bytes.
- Assembled `NativeBitNetTokenRuntime` from the mapped non-MLP weights, packed
  MLP artifact, zero-correction controller scales, persistent attention
  caches, one-call stage runner, final norm, and vocabulary argmax. The
  standalone C++ token CLI reproduces `12366 13 12366 374` for the fixed
  six-token prompt, processes 9 positions/120 stages, rejects out-of-vocabulary
  input, and has no Python/Torch/Transformers dynamic dependency.

## 2026-07-27 — Authenticated OLMoE package and single-row Q7 boundary

- Added the atomic `engram-native-olmoe-q7` version-1 package compiler. It
  dereferences Hugging Face snapshot inputs into a regular-file package,
  records the exact seven-file inventory, and publishes only after full
  validation.
- Added package-only generation with an externally supplied manifest SHA-256.
  Validation rejects changed manifests, changed or extra files, symlinks,
  unsafe paths, and unsupported runtime policy before native weights open.
- Built the 6,795,550,536-byte production directory with authentication root
  `861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`.
  It reproduces token 7785 (` Paris`) for `The capital of France is`.
- Parallelized a one-row Q7 layer across its eight selected experts. Median
  production layer time falls from 807.5 ms at one thread to 115.7 ms at 12
  threads (6.98×), with bit-identical routes and outputs. The complete
  five-position package run spends 13.08 seconds in Q7.

## 2026-07-28 — Complete native OLMoE causal gate and CPU threading

- Froze and passed an authenticated eight-prompt package-generation protocol
  against the untouched BF16 teacher: 60/60 teacher-forced top-1 decisions,
  29/32 greedy tokens, and 7/8 exact four-token prompts.
- Reworked the canonical Q7 inner loop around eight-code/seven-byte blocks
  while preserving coefficient accumulation order. Representative production
  layers improve by 6.56×–9.24× with bit-identical routes and float outputs.
  The five-position native run falls from 13.33 to 2.17 seconds, including Q7
  time falling from 13.08 to 1.91 seconds.
- Parallelized independent 16-layer Q7 structural validation (about 16.5 to
  2.29 seconds), six-shard teacher hashing, and package inventory hashing.
  The latter two now take 23.65 and 27.19 seconds and are primarily
  storage-bandwidth limited.
- Added an additive native diagnostic ABI for the final normalized hidden
  state and all 50,304 vocabulary logits. Fixture tests compare both against an
  independent NumPy computation and reject diagnostics after reset.
- Captured an untouched CPU BF16 teacher reference for the first eight
  33-token records of `confirmation_expanded.jsonl`, then froze the package,
  immutable DSO, teacher, corpus, and all source shards before candidate
  execution.
- The complete CPU-only native package passes all frozen aggregate checks over
  256 positions: KL 0.0129809, top-1 0.960938, NLL +0.0168240, and hidden L2
  0.0620471. The independently gated 128 positions after W=16 eviction pass
  with KL 0.0106424, top-1 0.960938, NLL +0.0136896, and hidden L2 0.0752018.
  Q7 schedules 22.7865% of the all-expert ideal-Q4 bytes.
- Disclosed nonuniform tails: maximum position KL is 0.606769, and offset 31
  alone misses all four per-offset quality thresholds. The frozen contract
  gates overall and 16-position population means, not every individual
  offset.
- Diagnosed low CPU utilization in the Transformers teacher. Simple batch-8
  execution is byte-exact but only 1.2% faster. Four concurrent sequence
  forwards through one shared read-only model are also byte-exact and reduce
  teacher compute from 366.14 to 94.78 seconds (3.86×), with wall time falling
  from 389.29 to 114.30 seconds (3.41×).
- An experimental eight-worker expert scheduler reduces teacher compute to
  156.84 seconds but changes BF16 rounding (6/256 top-1 differences versus
  serial), so it remains explicit opt-in. Shared-model sequence threading is
  the new default CPU capture policy.
- Hardened subsequent evaluations to authenticate actual config/index
  contents, recompute input and target identities, bind the effective thread
  count, optionally bind evaluator source files, retain per-position rows and
  native/Q7 timing, and re-authenticate every small root after execution.
- Froze a disclosed, non-independent hardened replay protocol around that
  source inventory. The unchanged candidate reproduced every original metric,
  split, traffic value, and check exactly; all seven post-run roots passed.
  The replay separates 88.79 seconds of native execution, 72.17 seconds of Q7,
  92.14 seconds of candidate-plus-metric wall time, and 184.55 seconds for the
  complete authenticated command.
- Final validation passes: **629 Python tests** and **19 native tests**.

## 2026-07-28 — Sustained-context failure and exact-attention attribution

- Froze a prospective sustained-context gate over eight newly authored,
  distinct-domain 129-token development texts: 128 prediction positions per
  sequence and 1,024 total. The CPU-only native candidate used 12 threads, the
  authenticated OLMoE package and Q7 artifact, and bounded attention
  `W16/C8/K4/S2`.
- The run passed every authentication, replay, structural, Q7-traffic, and
  attention-traffic check but failed semantic quality. Overall results were KL
  0.143578, top-1 agreement 0.802734, target-NLL delta +0.159292, and
  final-hidden relative L2 0.238260. Offsets 0–31 passed; every four-metric
  population gate failed in bands 32–63, 64–95, and 96–127.
- W16 used 677,117,952 logical attention-read bytes per sequence, 31.2863% of
  dense attention, while Q7 remained at 22.7865% of the all-expert ideal-Q4
  reference. Its frozen protocol and result hashes are
  `82189276ed0e555c2737f4842b1d1ed625f54d9ceaa2c63fe41fe71c5c6eb599`
  and
  `673523c29b12154f98916b8ce6f203b4967842e4bcae8f5c02ad4d197aab97eb`.
- After observing that failure, froze a matched attribution diagnostic that
  changed only `W16` to `W128`, yielding exact full causal attention for the
  128 evaluated positions. Its protocol and result hashes are
  `1619cd5f3cb607a7d0e2b5cde2e61a83dba3f1615884462a30570d62c7764dd9`
  and
  `3d099ffd3121e47bdf61ed8772e5e9d08b01b8c6041e9a963b409a502808d345`.
- The W128 control passed every overall and band threshold: KL 0.00343812,
  top-1 0.974609, target-NLL delta +0.00145861, and hidden L2 0.0413892.
  All 128 positions at offsets 0–15 matched the W16 position metrics exactly,
  and all structural, deterministic replay, and post-run authentication checks
  passed.
- This attributes the frozen corpus-level drift primarily to bounded
  attention, not Q7. It does not pass the deployable attention gate: W128 used
  100% dense logical attention traffic and was explicitly a post-failure
  diagnostic.
- The next prospective boundary is an exact matched-traffic sweep under 45%:
  `W16/C18/K16/S2`, `W24/C10/K8/S2`, and `W30/C4/K2/S2` each read
  968,753,152 logical bytes per sequence (44.7614%) and expose 32 values per
  mature step. These arms isolate older retrieval versus exact locality
  without changing the corpus, Q7 path, thresholds, or other authenticated
  artifacts.
- Full protocol, metric-band, structural-counter, evidence, limitation, and
  next-experiment details are recorded in
  `reports/olmoe_q7_sustained_context_2026-07-28/summary.md`.
- Full current validation passes: **637 Python tests** and **19 native tests**.

## 2026-07-28 — Matched-budget static attention sweep closes without selection

- Froze before execution a fixed-order three-arm development sweep from source
  commit `102bda2`. The protocol binds the prior sustained failure, the W128
  diagnostic, package and source inventories, native DSO, teacher arrays,
  corpus, evaluator source, thresholds, traffic algebra, exact expected
  counters, per-position matching rules, and a no-adaptation ranking rule.
  Protocol SHA-256:
  `2853de54119f4218c165ebebfe560162f76f99b552fdfe84c803a5ca8acfcef0`.
- Held mature visible values at 32 and logical attention reads at 968,753,152
  bytes per 128-position sequence (44.7614%) while exchanging exact recent
  locality for older retrieval. The arms ran in the frozen order
  `W16/C18/K16/S2`, `W24/C10/K8/S2`, and `W30/C4/K2/S2`; all three ran even
  after the first failures.
- Every arm passed source authentication, exact pre-eviction row identity,
  analytical and observed counters, Q7 traffic, deterministic reset replay,
  and post-run authentication. Their overall
  KL/top-1/NLL-delta/hidden-L2 results were:
  `0.0638865/0.867188/+0.0517008/0.157717`,
  `0.0659123/0.877930/+0.0584798/0.159755`, and
  `0.0958134/0.840820/+0.0757284/0.188422`.
- Zero arms passed the frozen semantic gate. The result therefore contains no
  selected arm, no “best failure,” and no fresh-confirmation request. The
  reserved confirmation corpus was not evaluated. Result SHA-256:
  `813bac5b1d38af7653cf49d8c7b7ca278df8aac5402fdd28692e905bebfc7658`.
- The matched W128 ceiling remains KL 0.00343812, top-1 0.974609, NLL delta
  +0.00145861, and hidden L2 0.0413892. Together with the valid negative sweep,
  this keeps Milestone 2's Q7 semantic conclusion passed and unchanged while
  locating the active block in Milestone 3 attention.
- This was a development-only raw-runtime intervention. It overrode the
  immutable package W16/C8/K4 policy without modifying or promoting the
  package/model format. Static global W/C/K reallocation under the <=45%
  logical-read budget is closed. The next justified experiment is
  layer/head-adaptive allocation or a learned/distilled selector trained to
  retain the older context the dense teacher actually uses.
- The committed sweep source passed **653 Python tests** and **19 native
  tests**, including a real tiny-package freeze-to-three-arm native smoke test,
  before the production protocol was frozen.

## 2026-07-28 — Greedy three-layer attention rescue fails the semantic screen

- Added a backward-compatible per-layer native attention ABI and Python
  binding. The historical scalar open path remains unchanged; the additive
  layered path accepts one `W/C/K/S` policy for each of OLMoE's 16 layers and
  sums heterogeneous state, scratch, logical-read, eviction, candidate,
  selection, sink, and heavy-hitter counters exactly.
- Before production execution, commit `708782b` passed **677 Python tests**
  and **20 native tests**. The candidate DSO was then copied to an immutable
  path with SHA-256
  `fe4dfdcc7e87a3cd5e36074e07d297f838ba345c37e939eeb0d796cb39cce409`.
- Froze a deterministic greedy search over three dense-layer rescues. The
  SHA-ranked split used two already-consumed records for selection and six for
  an internal screen. Rounds evaluated all `16 + 15 + 14 = 45` candidates
  before choosing a layer, with no early stop or score adaptation. The
  protocol and evaluator-source hashes are
  `9514e90bd5d14ae01ea27185763e5a833d4f1963e6bffd0ec0c81848f35b0c3e`
  and
  `77dafe8fc1fb6ca317ad7b99d5d86122e26b94b477f5befcf6184ce14080dff0`.
- A fail-fast production parity pass proved the historical scalar DSO and the
  new all-base layered DSO exactly equal over 128 positions: tokens, normalized
  hidden states, full logits, cache positions, deterministic counter streams,
  and historical diagnostic hashes all matched.
- The frozen greedy winners were layer 11, then layer 6, then layer 10. The
  final schedule kept 13 layers at `W16/C8/K4/S2` and rescued three at
  `W128/C8/K4/S2`. It used 955,957,248 logical attention-read bytes per
  sequence, or 44.1701% of dense attention, with 11,865,728 state bytes and
  6,528 scratch bytes. Q7 remained unchanged at 22.7865% of the all-expert
  ideal-Q4 reference.
- All execution evidence passed: 45/45 candidate contracts, every exact
  round-resource check, the six-sequence population and counter checks,
  deterministic reset replay, the final traffic budget, and all 21 post-run
  authentication roots.
- Semantic quality nevertheless failed on the six-sequence internal screen.
  Overall KL/top-1/NLL-delta/hidden-L2 were
  `0.102321/0.845052/+0.116776/0.206037`. Bands 0–15 and 16–31 passed every
  metric, while all four metrics failed in bands 32–63, 64–95, and 96–127.
  The result SHA-256 is
  `97ce800bd855c1f16248cada696936c7c56acd49c02d6f1c9ce9885dc44f7c49`.
- The authenticated command took 4,443.92 seconds: 3,938.18 seconds of
  candidate sequence execution, 86.64 seconds for parity, 260.32 seconds for
  the six-sequence screen, and 42.03 seconds for reset replay. A separate
  unauthenticated operator liveness observation showed about 6.8 aggregate CPU
  cores and 59 live threads; it is systems context, not frozen result evidence.
- This valid negative result does not change the passed Milestone 2 Q7
  conclusion and does not promote a package schedule. It closes this frozen
  greedy three-layer `W128` path under the 45% attention budget, while the
  disclosed search limitation leaves interacting layer combinations
  untested. Milestone 3 remains blocked. The next prospective boundary is a
  teacher-guided fixed mask rescuing exactly 51 of 256 layer-head pairs:
  973,384,704 logical bytes (44.9754%); 52 rescues would require 45.2438% and
  violate the cap.

## 2026-07-28 — Teacher-guided 51-head rescue fails the semantic screen

- Added an additive per-head native attention ABI, strict nested Python
  binding, exact heterogeneous counter algebra, a two-phase dense-teacher
  trace/mask protocol, and a fail-closed causal screen. Commit `d4580b0`
  passed **694 Python tests** and **20 native tests** before the prospective
  artifacts were frozen.
- Froze the trace before exposing attention maps. The untouched dense BF16
  teacher captured eager float32 attention maps for only the two established
  development-selection records; no map from the six internal-screen records
  was captured. Shape, dtype, finiteness, non-negativity, causal masking,
  row normalization, source inventory, and all post-capture authentication
  checks passed. The trace protocol and metadata SHA-256 values are
  `47f2c6bd7d467130ac492e7a5dc35b05b95ac0da5e71567a508951bd9754ad05`
  and
  `0445220b967be99208ee703bfd12a421de1a0721ff51b1d6404bc3db5305e08a`.
- The 33.6 MB map array remains at
  `work/olmoe_q7/sustained_2026-07-28/headwise_trace.npz` rather than being
  duplicated in the report directory. Its SHA-256 is
  `06e72ff9e9a03b58afd2197f5f40cd4e673867ecd5c566e038ce7e3dc8e38a55`.
  The protocol, metadata, deterministic mask, screen protocol, and screen
  result JSON files are archived under
  `reports/olmoe_q7_sustained_context_2026-07-28/`.
- Ranked all 256 layer-head pairs by the dense teacher's old-key attention
  mass that an ideal four-key older selection could not retain, using only
  the two selection records and a deterministic float64/tie-break rule. The
  fixed 51-head mask file SHA-256 is
  `dbb220384bade6793950d92a88528de3039c8efdb3b1e65964d368abaac90f48`;
  its policy identity is
  `18854c256af3fc68326a2e9fa9173d943db838c116523d5c4057e3f1efe9c278`.
- Froze the screen only after the mask and immutable candidate DSO existed.
  The screen protocol SHA-256 is
  `b863a6620f269bfe1dafec023c2e9742d9605510b113174b0eadbf64dc5cc850`.
  A mandatory 128-position all-base parity run proved the old layered and new
  per-head paths exactly equal in tokens, normalized hidden states, full
  logits, cache positions, deterministic counters, and archived diagnostic
  hashes before any internal-screen output was inspected.
- Rescued exactly 51 heads with `W128/C8/K4/S2` and kept 205 at
  `W16/C8/K4/S2`. The candidate read 973,384,704 logical attention bytes per
  sequence, 44.9753872184% of the dense reference, with 12,284,864 state bytes
  and 107,136 scratch bytes. A 52nd rescued head would raise the fraction to
  45.2437999637% and violate the cap. Q7 remained unchanged at 22.7864583333%
  of the all-expert ideal-Q4 reference.
- Every execution check passed: exact resource and per-token counters,
  six-sequence population and grid, mask identity, all-base parity,
  deterministic reset replay, Q7 traffic, and all 27 post-run authentication
  roots. The complete authenticated screen took 478.54 seconds, including
  103.88 seconds for parity, 325.30 seconds for the six primary executions,
  and 45.74 seconds for replay.
- Semantic quality nevertheless failed. Overall
  KL/top-1/NLL-delta/hidden-L2 were
  `0.0737199/0.867188/+0.0534555/0.167518`. Bands 0–15 and 16–31 passed every
  metric. At 32–63, KL and NLL passed but top-1 and hidden state failed; all
  four metrics failed at 64–95 and 96–127. Result SHA-256:
  `16bc2f8c11751612023145a36ace32b44bd082b77179a3c5753cb081424daa06`.
- This is an authenticated development failure, not a systems failure or
  fresh holdout. No package policy was promoted and no fresh eight-sequence
  confirmation was run. Milestone 2's Q7 conclusion remains passed;
  Milestone 3 remains blocked. Dense attention mass alone did not identify a
  passing static mask, so the next justified direction at that point was
  causal/value-sensitive allocation or a dynamic teacher-distilled head
  allocator.

## 2026-07-28: causal/value-sensitive head-gate result and revised boundary

- Froze the exact-native-forward, straight-through causal-head evaluator at
  source commit `483c62f` and trained exactly 51 rescued heads with two IHT
  steps on selection records 0 and 1.
- M1 improved the maximum/mean training composite from
  `7.8671169/6.9172161` to `4.7559915/4.3284769` without a per-record
  regression. The CPU-only fit took 6,930.099 seconds.
- The complete native Q7 screen passed execution, authentication, replay, and
  the 44.9753872184% resource contract, but failed semantic quality:
  KL/top-1/NLL-delta/hidden-L2 were
  `0.0791321/0.864583/+0.0811990/0.1826472`. No fresh confirmation or
  promotion occurred.
- The result closes the tested two-record natural-prose causal/value
  objective, not every static selector. Retrieval-specific supervision was
  absent. The next experiment is Q7-aware training on a new synthetic
  retrieval corpus with the loss concentrated on answer positions, motivated
  by [DuoAttention](https://arxiv.org/abs/2410.10819). If it cannot pass at
  the exact budget, the next allocation class is a causally committed,
  prefix-conditioned policy, informed by
  [Ada-KV](https://arxiv.org/abs/2407.11550) and
  [Task-KV](https://arxiv.org/abs/2501.15113).
- Before a larger fit, the measured 115.5-minute CPU bottleneck warrants a
  parity-gated deterministic expert-parallel proxy. Native support extraction
  is a secondary optimization and does not change semantic evidence.
