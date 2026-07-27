# Project status

Snapshot date: **2026-07-27**

Engram is an operational research prototype, not a general quality-preserving
dense-Llama compiler. The repository can inspect and trace a Llama-compatible
teacher, decompose SwiGLU MLPs, run routing and compression experiments, and
execute fixture packages in Python and C++. The native-BitNet track now also
compiles and validates a source-independent trained-model package and performs
real greedy generation with its direct C++ MLP kernel.

**The native-BitNet Milestone 2 semantic-memory gate has passed by postmortem
adjudication.** The low-bit-native track has a practical, CPU-only native DIP
implementation, not just the earlier dense-membership oracle. Its consumed
8-sequence/256-position final attempt substituted all 30 MLPs with no dense
fallback. The preserved raw report passes quality, activity, modeled physical
cold-traffic, and candidate recall.

The final raw result is KL 0.00404129, top-1 0.98828125, NLL delta
+0.00482893, final-hidden relative L2 0.0477494, 21.38001% mean active records,
41.13713% modeled traffic, 99.94058% global candidate recall, and 99.39429%
worst-layer mean recall. End-to-end sparse evaluation took 1.1449x the dense
elapsed time, so the kernel is 14.49% slower; latency was disclosed but was
not part of the frozen semantic gate. The dense-Llama conversion track remains
blocked.

### New OLMoE feasibility branch

The repository now has a separate `olmoe_sparse_expert_v1` source adapter. It
does not route OLMoE through the dense-Llama inspector. The official
`allenai/OLMoE-1B-7B-0125` revision
`9b0c1aa87e34a20052389dce1f0cf01da783f654` passed config, complete-name, and
exact remote-header shape validation: 3,219 required tensors, 3,219 present,
no missing or unexpected names, and no shape errors. The bounded verifier read
only safetensors header byte ranges and rejected unbounded responses; the
27.68 GB checkpoint payload was not downloaded by this audit.

The native topology is promising for Milestone 2: 64 addressable experts per
layer and top-8 learned routing yield a 12.5% active-expert fraction. Selected
Q4 expert weights plus BF16 router matrices initially projected to 12.6302% of
the all-expert Q4 baseline. Trained traces then showed that post-training Q4
errors compound causally and fail. A frozen Q7/group-64 candidate subsequently
passed an all-layer 8-sequence/256-position confirmation: KL 0.00900774, top-1
0.9765625, NLL delta +0.00391912, and final-hidden relative L2 0.0460273.
Selected Q7 codes, BF16 scales, and BF16 routers project to 22.7865% of the
all-expert ideal-Q4 baseline.

This clears the OLMoE semantic quality/evidence screen, but not the final
Milestone 2 systems gate. The intervention executed decoded Q7 weights inside
Transformers; it did not read a serialized packed artifact through the
CPU-only Engram runtime, and its traffic is modeled rather than measured or
cache-line-accounted. The next implementation is the immutable packed Q7
expert format and direct top-8 CPU kernel, followed by parity and the same
frozen causal protocol.

The qualification is not a pristine runner pass. After the evaluator
completed, the original wrapper marked the consumed attempt `error`: the
verifier compared the protocol's frozen full-record hashes, made with the
canonical `input_ids` object envelope, against raw-evaluator hashes of the
first 33 scored tokens made from bare lists. A separate no-model postmortem
adjudicator verified the corrected identities, the frozen bindings, and all
primitive measurements. The raw report was prospectively sealed about 13
minutes after the error, rather than being contemporaneously hash-bound by
the original result. See the
[final evidence summary](../reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/summary.md).

That decision is now integrated into a derived native package and the complete
C++ token-step path. The installer authenticates the frozen policy
(`c572754e…3768e`), adjudication (`ebb5ca95…a5cc`), base artifact
(`4fcf598a…ab55`), and v2 index (`b98ce4e4…0e15`), and never mutates the
policy-bound source package. `NativeBitNetTokenRuntime` is DIP-only and
fail-closed: every layer directly executes attention, normalized semantic
input extraction, DIP, and semantic-output acceptance, with no dense semantic
backend or fallback.

The promoted manifest SHA is `707bbe06…26926`; the rebuilt standalone
executable SHA is `c6c5b05b…a15b`; and the versioned chat-runtime DSO SHA is
`4b732beb…72a`. The executable and shared ABI both authenticate the exact,
symlink-free inventory and all semantic trust roots before model mapping and
derive architecture and EOS (including `128009`) from packaged files. The
standalone executable has no Engram shared-library dependency. The chat DSO
depends only on system libraries and exports six versioned C symbols.

The fixed non-holdout eight-prompt/32-token integration confirmation has now
passed with 32/32 greedy token-ID agreement and 8/8 exact prompts. Global mean
activity is 21.56017%, with a 22.58916% maximum prompt mean. Complete modeled
cold traffic is 30,153,074,432 bytes, including 194,304 global-metadata bytes;
its global mean is 41.16116% of dense ideal Q4 and its maximum prompt mean is
41.29835%. All absolute-position, stage/semantic-call, semantic-row, backend,
traffic-recomputation, generated-budget, and reset-replay checks passed on
CPU.

The rebuilt-core confirmation took 390.4183 seconds across first runs, reset
replays, and per-process package authentication; native counters/timings are
first-run snapshots. Exact means greedy tokens, not hidden or logit parity.
Reset proves token replay, zeroed counters, and structural metric parity, not
hidden-state identity. The frozen suite still stops at 14 positions. A real
interactive chat smoke processed 17 prompt tokens and crossed W=16, but it
does not establish sustained older-memory quality. A separate 16/17/18/24/32
position protocol now proves exact eviction, older-candidate, older-selection,
sink, heavy-hitter, fixed-state, and reset mechanics. At 32 positions it
records 480 evictions, 60,000 older keys scored, 34,800 older values selected,
1,200 sink insertions, and 5,654 accepted heavy-hitter updates while state
remains 7,477,440 bytes. This is integration correctness, not speed or dense-
teacher long-context quality evidence. See the
[native attention report](../reports/native_bitnet_dip_attention_confirmation_2026-07-27/summary.md).

The frozen practical-routing policy is
[machine-readable](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json).
The strongest earlier full-record systems evidence is the
[2026-07-24 direct-kernel confirmation](../reports/semantic_gate_native_bitnet_2026-07-24/kernel_confirmation.json);
the prior cross-track snapshot remains the
[2026-07-23 semantic-gate summary](../reports/semantic_gate_status_2026-07-23/summary.json).
Large corpora, checkpoints, and scratch experiments remain under ignored
`work/` paths and are not source-control artifacts.

## Milestone 2 ledger

Native BitNet can proceed beyond Milestone 2. OLMoE now passes the semantic
quality/evidence screen but remains at the serialized-runtime boundary, while
generic dense-Llama conversion remains blocked:

| Milestone 2 deliverable | Native BitNet | OLMoE Q7 | Generic dense Llama |
|---|---|---|---|
| Background operators | Exact packaged residual; learned correction is zero | Native top-8 mixture requires no added residual in the passing simulation | Current fitted background hurts held-out quality |
| Semantic key/value package | Complete ternary records plus authenticated DIP-v2 index | 64 source experts/layer are addressable; packed Q7 artifact remains | Format/runtime exist; no qualifying trained artifact |
| Practical routing | **Passed**, all 30 MLPs, no fallback | **Causal screen passed** with learned top-8 routing | **Blocked** |
| Quantization | Native packed ternary | Q7/group-64 with BF16 scales passes decoded-weight confirmation | Experimental product/additive codecs exist |
| Python runtime | Persistent native DIP handle | Tracing and intervention exist; packed runtime missing | Research runtime exists |
| End-to-end substituted MLPs | Native evaluation, generation, and chat passed | 8 sequences/256 positions pass in Transformers simulation | No gate-passing compilation candidate |

These are track-specific results. Neither successful source track erases the
original dense-source failures. OLMoE is not complete until packed
serialization and CPU-only native execution reproduce its passing simulation.

## Semantic gate definition and outcome

A candidate must use one serialized and independently reloaded artifact and
pass all of the following on an all-layer causal evaluation:

| Requirement | Threshold |
|---|---:|
| Teacher-to-student KL | at most 0.05 nat/token |
| Teacher top-1 agreement | at least 0.90 |
| Target NLL delta | at most +0.05 nat/token |
| Final normalized hidden-state relative L2 | at most 0.10 |
| Evidence | at least 8 unique sequences and 256 prediction positions |
| Complete physical cold MLP traffic | at most 45% of dense ideal Q4 |
| Mean selected records | at most 25% of the 6,912 records |
| Candidate recall | global micro and every layer mean at least 0.95 |

Configuration selection used development-only data. The consumed final
confirmation used the identical frozen artifact on a sequence-disjoint corpus
that was not used for fitting or selection. The checked-in holdout is
plaintext, so this is procedural/honor-system separation, not cryptographic
secrecy. The passing decision is by the postmortem adjudication described
above, not by the original runner result.

## Strongest measured frontiers

No row in the original dense-Llama track passes both sides of the gate. The
native BitNet row now passes direct packed execution, the evidence floor, all
semantic thresholds, and exact scheduled cold-byte accounting.

| Representation | Quality result | Systems result | Decision |
|---|---|---|---|
| Native BitNet DIP, frozen practical policy | Final 8-sequence/256-position raw result: KL 0.00404129, top-1 0.98828125, NLL +0.00482893, hidden L2 0.0477494; global recall 0.9994058, worst-layer mean 0.9939429 | 21.3800% mean active records; 41.1371% modeled physical cold traffic; CPU-only native kernel; 1.1449x dense elapsed time | **Semantic gate passed by postmortem adjudication; original wrapper ended in error** |
| Native BitNet layer-adaptive exact-membership oracle | Historical oracle result: KL 0.02543, top-1 0.94531, NLL +0.02386, hidden L2 0.09205 | 15–35% per layer, 24.8375% mean selected down records; dense gate/up coefficient scan remains | **Oracle ceiling passed** and motivated practical DIP |
| Native BitNet phase-stream base-3 records | Frozen 8-sequence/256-position result: KL 0.00371, top-1 0.96094, NLL +0.00224, hidden L2 0.04678 | Direct memory-mapped CPU kernel; 318,924,544 scheduled cold bytes, 40.0527% of dense Q4; all MLP records execute | **Systems substrate only**; not routed semantic memory or a Milestone 2 pass |
| Exact float magnitude oracle, K=768 | KL 0.0336, top-1 0.9124, NLL +0.0153, hidden L2 0.0953 | More than 4x the dense-Q4 payload before a practical selector | Semantic capacity exists, but this is not deployable |
| Predictor-free DIP, q=432/C=896/K=768 | Untouched confirmation: recall 0.9897, KL 0.0286, top-1 0.9101, NLL +0.0326, hidden L2 0.0905 | 76.39% scalar traffic, 83.33% cache-line traffic; native kernel is 0.863x dense throughput | Quality pass, systems fail |
| Mild layer-adaptive compact Q4 student | At 3,000,093 training positions: KL 0.8866, top-1 0.5659, NLL +0.8838, hidden L2 0.4245 | Serialized/reloaded artifact is 44.9334% of dense ideal Q4 | Traffic pass, quality fail; stopped at the frozen 3M rule |
| Budget-native full-width grouped ternary | At 1,014,225 training positions: KL 2.2844, top-1 0.3198, NLL +2.2770, hidden L2 0.6036 | Serialized/reloaded artifact is 43.1353% of dense ideal Q4 | Traffic pass, quality fail; top-1/hidden miss frozen 50%-gap-closure rule |
| Exact nonparametric output memory | Layer-14 LLE-32 error 0.3275 with 233,005 local records; 0.3219 after adding 1,000,000 pretraining records | Exact search and FP16 values are not a deployable traffic result | Only 1.73% improvement; density-scaling branch closed |
| Mixed affine LC-VQ | Development-only layer-14 hard-QAT error 0.3364 after 8,192 steps | Complete modeled cold traffic is 44.3482% | Traffic pass, local-quality fail; no causal run |
| Fully sparse top-K activation path | Best unseen all-layer result after causal schedule fitting and verified attention/norm co-adaptation: KL 0.4517, top-1 0.6714, NLL +0.4585, hidden L2 0.3272 | Fixed per-layer q/K schedule is exactly 45% ideal traffic before metadata; every artifact reloads and executes on CPU | Whole-model hypothesis tested and stopped; far from every semantic threshold |

The native-BitNet DIP result is the first tested practical mechanism to clear
the complete final quality, recall, mean-activity, and modeled physical
traffic gate. Its status is pass-by-adjudication, with the evidence-integrity
caveats above. This does not promote it into the generic dense-Llama compiler.
The older SmolLM DIP quality result in the table is a separate dense-source
experiment that failed systems traffic and latency.

### Native-BitNet oracle semantic ceiling

The corrected Milestone-2 restart first tested the actual BitNet teacher rather
than treating lossless full-record execution as semantic routing. A new direct
CPU oracle ranks additive records after the teacher's Q8 activation,
ReLU-squared gate/up product, intermediate RMS normalization, gain, and second
Q8 quantization. Fixed 25% selection passed 32-position development but missed
the frozen final-hidden limit at 0.10448.

A development-only all-layer sweep then allocated 15–35% per layer while
holding the exact aggregate below 25%. The selected schedule averages 24.8375%
and passes the untouched frozen protocol: KL 0.02543, top-1 0.94531, NLL delta
+0.02386, and final-hidden relative L2 0.09205. The report is
[here](../reports/native_bitnet_oracle_2026-07-26/summary.md).

This establishes semantic concentration and the target membership schedule.
By itself it did not close Gate 2 because selection still consulted all dense
gate/up coefficients.

The subsequent practical-router screen was positive at representative depths.
A direct nonlinear rank-256 membership predictor is rejected at only 77.75%
recall with 1.5x candidates. BitNet-specific Dynamic Input Pruning instead
scores ternary gate/up keys from the largest 75% of input coordinates and
reaches 96.23%, 98.06%, and 96.78% recall at layers 0, 14, and 29. Its
provisional modeled traffic was about 35–41% of dense Q4. See the historical
[router screen](../reports/native_bitnet_router_2026-07-26/summary.md).

### Native-BitNet practical DIP development pass

The all-layer follow-up is now implemented as a memory-mapped C++ CPU kernel.
It accepts live BF16 MLP inputs, applies native Q8 quantization, keeps the
largest `q=1920` of 2,560 coordinates, and scans packed coordinate-major
ternary gate/up streams across all 6,912 records. It exactly completes the
frozen per-layer candidate set, estimates the coupled RMS, computes exact
candidate utilities, and reads only the down rows selected for that token.
The adaptive count is the number of nonzero candidate utilities, clipped to
`minK=346` and each layer's `Kmax`.

```text
C    = [4224,5504,4224,4224,4224,4224,4224,4224,4480,4480,
        4736,4992,4480,4992,4992,4736,4992,4992,5248,4736,
        3456,5248,5248,5248,4992,3968,3200,4992,4224,4992]
Kmax = [4224,1705,4224,4224,4224,4224,4224,4224,3753,3753,
        3241,2729,3753,2729,2729,3241,2729,2729,2217,3241,
        3456,2217,2217,2217,2729,3968,3200,2729,4224,2729]
```

Layers other than 9 use the candidate-ratio RMS estimate. Layer 9 uses a
corrected-proxy estimate and reserves 8 records inside `C=4480` for a
top-proxy-raw-square audit; this is not extra candidate traffic. The v2
coordinate index is source-hash-bound, independently reloaded, and stores the
complete RMS and q/C/K policy. Its 216,688,448 bytes plus the 318,924,544-byte
base record artifact total 535,612,992 bytes, or 67.2659% of the dense-Q4
reference as stored data. Storage size is disclosed separately from per-token
cold traffic.

On the declared development corpus, the live native BF16 substitution passes
all frozen thresholds:

| Measure | Development result | Threshold |
|---|---:|---:|
| KL | 0.0044707 | <= 0.05 |
| Top-1 agreement | 0.94921875 | >= 0.90 |
| NLL delta | +0.0013609 | <= +0.05 |
| Final-hidden relative L2 | 0.0498965 | <= 0.10 |
| Mean active fraction | 0.2008072 | <= 0.25 |
| Modeled physical cold traffic | 0.409639 | <= 0.45 |
| Global micro candidate recall | 0.9995917 | >= 0.95 |
| Worst-layer mean recall | 0.9939353 | >= 0.95 |

The candidate recall denominator is a fixed, router-independent per-layer
dense-teacher top-K schedule, not the adaptive selected count. A separate
untimed diagnostic pass computes that reference; the timed sparse pass makes
no dense full-record calls. Python/native route fields and BF16 outputs are
bit-exact for six rows in all 30 layers. The end-to-end development sparse
pass takes 1.1565x the dense elapsed time, so it is not a speedup. The traffic
fraction is modeled from touched 64-byte lines and metadata, not measured
DRAM.

The identical [frozen policy](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json)
then produced the following raw final-holdout measurements:

| Measure | Final raw result | Threshold |
|---|---:|---:|
| KL | 0.0040412880 | <= 0.05 |
| Top-1 agreement | 0.98828125 | >= 0.90 |
| NLL delta | +0.0048289299 | <= +0.05 |
| Final-hidden relative L2 | 0.0477494113 | <= 0.10 |
| Mean active fraction | 0.2138000677 | <= 0.25 |
| Modeled physical cold traffic | 0.4113713394 | <= 0.45 |
| Global micro candidate recall | 0.9994058295 | >= 0.95 |
| Worst-layer mean recall | 0.9939428640 | >= 0.95 |

The slow path took 295.3364 seconds versus 257.9552 seconds dense, or 1.1449x
dense. This is measured CPU latency but was not an upper-bound gate. The
semantic decision is a postmortem adjudication because the original runner's
full-record/object-versus-33-token/list hash verifier defect fired after the
raw evaluator completed. No model or evaluator was executed during
adjudication. The host-bound binaries and artifacts, delayed prospective
evidence seal, modeled rather than measured DRAM traffic, and small 8x32
confirmation scale remain material limitations.

## What the latest experiments changed

### Exact activation-sparse training paths

The latest dense-source experiment removes approximate routing from the
critical path. CATS-style execution reads the full gate matrix, thresholds
its activation exactly, and reads up/down records only for nonzero gates. Its
ideal traffic fraction is `(1 + 2a) / 3`, where `a` is active-record
fraction. At the traffic boundary, zero-shot layer-local error is 0.511 and a
progressive plus fixed-budget boundary fit reaches only 0.470.

The stronger Q-Sparse-style path selects already-resident activation
coordinates directly. It reads `q` input columns of both gate and up and `k`
input columns of down, for ideal traffic `(2q + k) / 3`. The selected integer
point uses 282 of 576 input coordinates and 522 of 1,536 intermediate
coordinates, or 43.967% before metadata. It has no router, candidate stage, or
recall gate. On the representative layer-14 development boundary set, its
error improves from 0.3426 to a best 0.3228 and then plateaus.

Both local screens fail the unchanged 0.18 progression ceiling. The distinct
whole-model Q-Sparse hypothesis was then tested with CUDA used only for
training. A causal single-layer sensitivity fit improved the fixed all-layer
baseline from KL 0.742 to 0.457 at exactly 45% ideal traffic, but verified
attention/normalization co-adaptation moved the unseen result only to KL
0.452, top-1 0.671, NLL +0.458, and hidden L2 0.327. Label-only full-model
continuation, per-token concentration budgets, and a traffic-charged rank-24
residual did not improve the frontier. Every artifact independently reloaded
and executed on CPU; confirmation remained sealed. The dense-source
activation-sparse branch is therefore stopped at the available scale.

### Lossless native-BitNet semantic records

The selected new program starts from Microsoft's natively trained
`bitnet-b1.58-2B-4T` rather than post-hoc quantizing SmolLM2. It is pinned to
revision `04c3b9ad9361b824064a1f25ea60a8be9599b127`; the checked
1,178,623,988-byte safetensors file has SHA-256
`8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
Engram keeps this source family outside the generic SiLU/SwiGLU compiler:
BitNet uses ReLU-squared gating, per-token activation quantization, and an
intermediate RMS normalization that couples channel magnitudes.

The official checkpoint's four-trits-per-byte MLP payload is 50.0521% of the
frozen dense-Q4 denominator, so changing source models alone does not pass.
The new format losslessly stores five base-3 trits per byte. Each channel has
a 1,538-byte logical payload: a 512-byte gate key, 512-byte up key, 512-byte
transposed down value, and a BF16 normalization gain. Those fields live in
four independently fixed-stride phase streams rather than one contiguous
record. Three BF16 projection scales remain layer-global. The complete
30-layer file is
318,924,544 bytes or 40.0527%, with 39,393,536 bytes of headroom. Its logical
source and reconstructed streams have the same SHA-256.

A CPU BF16 evaluation oracle reproduces the decoded artifact exactly. The new
C++ kernel instead memory-maps and executes the packed phase streams directly,
without creating dense weights. Its tiny-fixture integration test is
bit-exact. Official-layer outputs differ from PyTorch dense BF16 by at most
0.00982 relative L2 because their reduction orders differ, so the decisive
test is all-layer causal substitution. On the frozen 8-sequence/256-position
corpus it reaches KL 0.00371, top-1 0.96094, NLL +0.00224, and final-hidden L2
0.04678 while scheduling 40.0527% of dense-Q4 cold bytes. Exact details are in
the [direct-kernel result](../reports/semantic_gate_native_bitnet_2026-07-24/summary.md).

### Budget-native grouped-ternary distillation

The new path fixes the deployment representation before training: all 30 MLPs
remain width 1,536, five ternary coefficients are packed per byte, and each
128-weight group has one non-learned FP16 scale refined by two least-squares
code/scale iterations. The exact 17,173,504-byte file includes codes, scales,
headers, directory entries, and cache-line padding. It uses 43.1353% of dense
ideal-Q4 MLP traffic and leaves 742,400 bytes below the 45% limit.

Training uses an immutable dense teacher and a copied student whose float
master weights exist only for optimization. The forward path uses the exact
hard grouped-ternary decode with straight-through gradients. A
deepest-layer-first transition avoids quantizing all layers at once; later
rungs stay hard throughout. Local MLP, all-layer hidden, direct final-hidden,
confidence-weighted KL, teacher-top-1, and label losses are implemented.
Attention, normalization, and optionally the already-resident tied
embedding/output head may co-adapt without adding MLP cold bytes. Checkpoints
resume across CPU/CUDA, and validation reloads both the MLP binary and any
co-adapted backbone tensors.

The final bounded rung trained on 8,192 fresh sequences containing 1,014,225
input positions. Relative to its preceding head-coadaptation checkpoint, it
closed 63.37% of the remaining KL gap and 62.77% of the NLL gap, but only
31.33% of the top-1 gap and 38.29% of the final-hidden gap. The predeclared
rule required 50% on every metric before spending 3M or 10M positions. The
configuration therefore stops at KL 2.2844, top-1 0.3198, NLL +2.2770, and
hidden L2 0.6036 despite its valid traffic result. Exact hashes and intermediate
rungs are in the
[budget-native result](../reports/semantic_gate_budget_native_2026-07-23/summary.md).

### Compact-model distillation

The final bounded compact-Q4 program used a predeclared 10-million-position
sample of the released SmolLM2 training mixture. The student kept hidden size
576, used 11 MLPs of width 704 and 19 of width 672, executed fake-decoded Q4
weights during training, and serialized all MLP codes, scales, source IDs,
headers, and alignment.

At the frozen 3M checkpoint it had closed 78.2% of the KL gap but only 50.0%
of the top-1 gap and 56.7% of the hidden-state gap. The protocol required at
least 80% closure on every metric before spending the remaining seven million
positions. All four checks failed, so the run stopped without opening formal
development or confirmation data.

This rejects the tested compact architecture, width vector, initialization,
loss, and training budget. It does not prove that a compact model trained from
scratch or distilled on a much larger token budget cannot work.

### Low-bit and structured representations

Post-hoc low-bit representations do not provide the missing bridge:

- full-width Q3 QAT plateaus at layer-local relative L2 0.2174 while consuming
  far more than the traffic budget;
- a traffic-feasible asymmetric ternary plus rank-12 Q4 residual reaches
  0.4938;
- the best strict shared-input-basis rate-constrained artifact reaches 0.3554;
- structured shared-basis dictionaries, source-coordinate block sparsity,
  affine experts, and conditional compact experts all miss their local screens.

These results are why another small bit-allocation or router sweep is not the
default next step.

The final bounded representation campaign tested additional mechanisms at the
same byte edge:

| Screen | Complete traffic | Best layer-14 mean rel-L2 | Progression |
|---|---:|---:|---|
| Four-cycle width-640 recurrent compact Q4 | 44.9293% | 0.308254 | fail; later-cycle cache reuse is not measured |
| Projection-normalized full-width ternary | 41.0013% | 0.631323 | fail |
| Mixed affine LC-VQ | 44.3482% | 0.336396 | fail |
| Unrestricted 128-entry vector codebook | 44.9799% | 0.576865 initial | stopped at initialization guard |
| Mixed LiftQuant-style lifted-binary lattice | 44.4012% | 0.556958 initial | stopped at initialization guard |

The recurrent and affine-LC arms were trained on sequence-disjoint
development-role boundaries. The codebook and lifted-binary arms failed their
predeclared 0.55 initialization guard and were not trained. None reached the
0.20 local ceiling required to expose an expensive causal or external
evaluation. Exact metrics and scratch-report hashes are checked in under the
[budget-edge representation summary](../reports/semantic_gate_lowbit_2026-07-23/summary.json).

### Nonparametric output memory

The most recent branch asked whether the MLP could be represented by stored
input/output examples rather than compressed source weights.

The exact nested local curve was:

| Prototypes | Layer-14 LLE-32 mean relative L2 |
|---:|---:|
| 16,384 | 0.490340 |
| 65,536 | 0.401270 |
| 233,005 | 0.327526 |

Several local-linear variants exposed the same limitation:

- reconstructing a query state from 512 neighbors is accurate (0.0496
  relative L2), but interpolating the nonlinear MLP output remains at 0.3275;
- token-conditioned and two-region token-conditioned Jacobians improve to
  0.2360 and 0.2164, but require tens of GiB before an index or all-layer
  package;
- an exact per-query nearest-prototype Jacobian calculation reaches 0.1725,
  demonstrating a local capacity ceiling, but finite shared Jacobian banks
  regress above 0.22 and are not byte-feasible.

The frozen final pilot then captured exactly one million finite FP16 layer-14
input/output pairs from 8,192 independent pretraining sequences. Combining
them with all 233,005 local fitting records improved exact LLE-32 error only
from 0.327526 to 0.321854. The predeclared progression rule required error at
most 0.28 and at least 10% improvement; the measured improvement was 1.73%.
The ten-million-record capture was therefore not run.

## Milestone position

“Implemented” below means that code and tests exist. It does not mean the
scientific exit criterion has passed.

| Milestone | Implementation status | Evidence status |
|---|---|---|
| 1. Inspection, tracing, exact MLP decomposition, oracle experiment | Complete | Complete for the fixture and exercised on SmolLM2 |
| 2. Semantic package, routing, quantization, Python substitution runtime | Source-bound native-BitNet DIP index, CPU-only selected-record kernel, authenticated derived package, substitution evaluator, physical accounting, parity checks, DIP-only C++ token runtime, and packaged-tokenizer chat binding exist | **Native-BitNet path passed** by postmortem adjudication; rebuilt non-holdout token generation remains 32/32 and the real chat command uses the same no-fallback backend. Generic dense-Llama conversion and broader replication remain incomplete |
| 3. Local/recurrent/retrieval attention and hybrid episodic memory | Bounded W=16/C=8/K=4 streaming hybrid, stateful C++20 cache/rerank kernel, and incremental package integration implemented | **Frozen trained-model confirmation passes**; randomized parity, bounded-state scaling, cache-position advancement, and incremental generation pass; hardware counters remain |
| 4. Shared recurrent controller, adapters, adaptive cycles, transformer-free Python runtime | Versioned exact residual controller, authenticated package installation, persistent native stage state, and a one-call 30-stage C++ attention/semantic runner implemented | **Controller, compiled-substitution, incremental-generation, and C++ orchestration gates pass**; frozen generation reaches 96.875% token agreement, 87.5% exact prompts, correct cache positions, and zero decoder-layer calls |
| 5. Vocabulary index, transition cache, corrections, compiler, validation, generation CLI | Generic infrastructure plus native-BitNet package compiler, validator, native vocabulary argmax, and generation CLI implemented | Native-BitNet package excludes all source MLP tensors and passes source/package parity; generic vocabulary/cache/correction paths are not all active in the promoted native-BitNet runtime |
| 6. C++ runtime, scalar/AVX2 paths, mmap, parity, generation, benchmarks | Fixture runtime, memory-mapped BitNet DIP/projection kernels, streaming attention, native shell operators, authenticated C++ package mapping, manifest-derived model configuration/EOS, token-step control, greedy argmax, reset, standalone generation, and a versioned shared ABI implemented | **Partial**: model execution is native and chat uses the shared handle; tokenizer/template/history orchestration remains Python-side, sustained older-context quality is not covered, and AVX2 tuning and hardware-counter traffic remain |
| 7. Evaluation, ablations, tuning, documentation, final report | In progress | Many negative ablations exist; no successful reproducible final report |

The optional Oracle cognitive executive is a separate request-level subsystem.
Its revisioned SQLite/JSONL event stores, worker registry, dispatch adapters,
outcome observation, and calibration summaries are implemented, but they are
independent of the model-worker semantic-gate evidence.

## What is intentionally not claimed

- There is no quality-preserving `.engram` conversion of SmolLM2.
- The native BitNet result is a separate source track, not evidence that a
  dense Llama model can be losslessly repacked.
- There is no hardware-counter demonstration of DRAM traffic; the final
  41.1371% practical-DIP result is a modeled cache-line schedule, not measured
  DRAM.
- The native-BitNet practical DIP arm has passed its semantic gate by
  postmortem adjudication, not by a pristine final-runner result. This is not a
  blanket claim that all generic Milestone 2 packaging and conversion work is
  complete.
- The derived DIP package, standalone runtime, and shared chat handle pass
  their integration checks. Python still owns tokenization, template
  rendering, and history, but it no longer constructs or executes a
  Transformers model shell.
- The final holdout is plaintext in the repository. Its separation is
  procedural and honor-system-based, with a fail-closed runner; it is not
  cryptographically hidden from developers.
- Random-fixture generation is pipeline evidence, not language quality.
- CUDA was used for bounded training and trace capture, but no deployment
  format or inference path requires a GPU.
- Failure on SmolLM2-135M is not a theorem about every model or every possible
  architecture.

## Current development decision

The shared-controller stage has started with a deployable low-rank design
rather than the original dense FP64 GRU fixture. At width 2,560, the original
two dense GRU kernels would occupy about 629 MB and impose that shared-weight
traffic on every depth cycle. The new rank-128 controller has 2,662,430 FP32
parameters (10,649,720 bytes), an identity-biased residual path, per-token RMS
normalization, 30 stage embeddings, and rank-4 stage adapters.

The next protected run expanded to 128 training and 64 validation positions
from different dataset hashes. It reduced validation terminal normalized MSE
from 1.998608 to 0.245010, cosine loss from 0.973363 to 0.333417, and total
rollout loss from 4.292672 to 1.156465. A lower-rate 500-step continuation
loaded from the serialized artifact and used no teacher forcing. It improved
training terminal error from 0.060627 to 0.029324 but worsened validation
terminal error to 0.260050; that continuation is rejected. Independent NumPy
reload of the retained pre-continuation artifact matches Torch within 7.45e-6
maximum absolute error. Sixty-four states complete 30 CPU cycles at 79.7
states/s in the measured NumPy batch. Exact teacher MLP and attention outputs
are still supplied at the controller boundary, so compiled-operator
substitution remains sealed.

The controller-only prerequisite for substitution is terminal normalized MSE
at most 0.0225, corresponding to the existing hidden relative L2 limit of
0.15. A controlled rank-4 stage input adapter improves the 1,024/256-position
result only from 0.159440 to 0.157431, showing that another learned input
alignment is not the solution.

The host was subsequently rebooted with matching NVIDIA 580.173.02 kernel and
userspace components. An exact CUDA reproduction reaches protected terminal
normalized MSE 0.246530 and cosine loss 0.335160, compared with 0.245010 and
0.333417 for the CPU-optimizer run. Serialized CPU parity passes at 5.36e-6.
CUDA reduces the 500-step fit from 131.8 to 112.6 seconds but does not alter
the scientific decision; the slightly better CPU artifact remains retained.

A subsequent frozen 1,024/256-position rung trained a fresh rank-128
controller for 1,000 CUDA steps. Protected terminal normalized MSE improves
to 0.159440 and total rollout loss to 0.931534, but the artifact still fails
the 0.0225 substitution gate by 7.1 times. The artifact reloads on CPU within
5.90e-6 and processes a batch of 256 states through 30 cycles at 101.3
states/s.

The trace contract exposes a stronger architectural fact: teacher layer output
is the incoming residual plus the captured attention and MLP outputs. A new
schema-v3 controller preserves those additions exactly and reserves the shared
factorized recurrence for corrections only. With correction scales zero, the
protected terminal NMSE is 0.000020801, passing the gate with 1,081.7 times
margin. CPU reload parity is 5.72e-6, and the matrix-skipping NumPy path runs
41,575.9 stage transitions/s. Exact teacher operator outputs are still
supplied, so the next gate is compiled semantic and episodic substitution,
first independently and then jointly.

Operator provenance was then corrected: `NativeBitNetRuntime` already replaces
every MLP with the packaged direct CPU phase-stream kernel, so captured
semantic outputs were compiled. A frozen controller replay now combines those
packed semantic outputs with native W16/C8/K4/S2 attention outputs outside the
decoder residual scaffold. Across eight held-out sequences and 256 prediction
positions it passes every check: KL 0.011125, top-1 agreement 0.957031, NLL
delta -0.008285, final hidden relative L2 0.075893, controller-to-candidate
hidden L2 0.006810, and terminal trajectory NMSE 0.000026666.

The next boundary is no longer another substitution-quality experiment. It is
incremental runtime integration: controller state must directly feed the
native MLP and attention operators, advance RoPE/cache positions, persist
bounded episodic state, and produce logits without decoder-layer forwards.

That incremental boundary now passes. `ControllerDrivenBitNet` explicitly
dispatches all 30 normalized attention/MLP stages and advances schema-v3
controller state while native attention owns persistent cross-token memory.
It carries one residual RMS scalar per token so operator outputs retain their
correct relative scale. Across the fixed eight-prompt suite, all 32 generated
tokens match the bounded decoder reference, cache positions are exact, and
decoder-layer forward calls are zero. Controller arithmetic is only 0.0427
seconds per prompt versus 22.581 seconds complete execution.

Package-native installation and residual execution are now implemented. The
manifest owns and authenticates every controller tensor, generation can select
the installed controller without an external path, and the residual/RMS loop
runs through `libengram_bitnet.so`.

The next native-shell cut is also complete. BF16 embedding lookup, all 92
RMSNorm sites, default RoPE construction/application, and tied-vocabulary
greedy argmax now execute through the C ABI. The vocabulary path returns only
the selected token and no longer allocates 128,256 logits. The four-token
`The capital of France is` smoke test remains ` Paris. Paris is` and improves
from about 21.3 to 18.6 seconds. The frozen eight-prompt/32-token protocol
passes at 96.875% weighted agreement and 87.5% exact prompts, with exact cache
positions and zero decoder-layer calls. Its one fourth-token mismatch is a
BF16 near-tie caused by native scalar versus PyTorch/oneDNN accumulation order,
not a hidden-state or cache gate failure.

The remaining Milestone 4 systems boundary is an all-C++ stage orchestrator.
Python still sequences the 30 stages and creates Torch tensor views for the
already-native packed projections, MLPs, and streaming attention. The next
implementation should load the authenticated non-MLP/controller state into a
single native runtime handle and dispatch the full stage loop without Python
or Torch.

The first part of that orchestrator is now live. A persistent C++ stage-state
handle owns normalized residual state, attention contribution, post-attention
state, and physical RMS, enforces attention-before-semantic call ordering, and
produces BF16 normalized inputs for both operators. Package generation uses
this handle instead of the NumPy state loop and retains ` Paris. Paris is`;
measured controller bookkeeping falls from roughly 26–30 ms to 11.4 ms on
that smoke prompt. The complete suite passes at 452 Python and 16 native
tests. Python still invokes each attention and MLP operator, so this is the
buffer/state foundation for the all-C++ loop rather than completion of it.

The semantic half of each stage is now fused across that boundary.
`engram_bitnet_stage_semantic_bf16` asks the stage handle for its normalized
post-attention input, executes the selected packed phase-stream MLP directly,
records the existing traffic/timing metrics, and inserts the semantic result
back into normalized residual state in one native call. The smoke sequence
remains exact, completes in 18.15 seconds, and reports only 10.4 ms of
controller/orchestration overhead. Python no longer materializes semantic
inputs or outputs. Attention still crosses the Python/Torch boundary and is
the next half to fuse.

The attention half and depth loop are now fused as well. One descriptor-driven
C++ call executes input normalization, packed Q/K/V, RoPE, persistent bounded
attention, attention sub-normalization, packed O projection, packed MLP,
residual insertion, and renormalization for all 30 stages. The unchanged
eight-prompt/32-token protocol passes at 96.875% weighted agreement and 87.5%
exact prompts, with every cache position correct and zero decoder-layer calls.
Mean complete controller runtime is 16.50 seconds and measured
controller/orchestration overhead is 18.0 ms per prompt. This closes the
Milestone 4 systems-orchestration gate. Python still loads the non-MLP
safetensors into a Transformers-shaped holder and drives per-token generation;
moving package loading and generation control into the native runtime is the
next Milestone 6 boundary.

Native package loading has now advanced past its largest storage risk. A strict
read-only safetensors mapper validates the complete header, dtype/shape,
contiguous offsets, payload length, and typed views. It maps the real
780,054,616-byte non-MLP file with all 332 tensors. `NativeBitNetWeights` binds
the 128,256x2,560 tied embedding, final and per-layer norm vectors, and all 120
packed Q/K/V/O projections. The projection kernel now supports explicit
lifetime-bound mapped registration, so the complete 30-layer binding reports
zero copied projection bytes. The next loader step is to construct attention
caches/stage descriptors plus the MLP/controller handles and expose one native
token-step runtime.

That token-step runtime now exists. `NativeBitNetTokenRuntime` owns the mapped
weights, memory-mapped packed MLP artifact and DIP index, validated
zero-correction controller scales, 30 persistent streaming-attention caches,
position counter, final norm, and tied-vocabulary argmax. The standalone
`engram-bitnet-token-generate` executable accepts raw token IDs and performs
greedy generation without Python, Torch, the Python `safetensors` package,
Transformers, or an Engram shared library. It reads the packaged
`transformer/non_mlp.safetensors` file through Engram's C++ parser. Its package
preflight authenticates the promoted manifest, exact inventory, semantic trust
roots, controller, model/tokenizer configuration, attention policy, and EOS
IDs before deriving every runtime architecture parameter. Native
tokenizer/chat-template support remains outside the binary; model execution
is native from packaged token IDs to generated token IDs.

The low-bit-native hypothesis has passed source validation, exact
reconstruction, the unchanged MLP byte limit, direct CPU execution, frozen
causal confirmation, source-independent package compilation, and exact
generation parity. Its package contains a checksummed 780,054,616-byte
non-MLP tensor file plus the 318,924,544-byte packed MLP artifact, installed
controller, config, and tokenizer assets. The derived semantic package adds
the authenticated 216,688,448-byte DIP index and its promotion descriptor.

Milestone 3 development rejected local-only and recurrent-only replacements.
An exact W=16/K=4 hybrid passed semantics but scanned all older keys. Random
sign-LSH then missed too many top keys, and exact page bounds pruned too few
pages. The promoted streaming policy retains 16 local tokens, two sinks, and
six cumulative-attention heavy hitters, exact-reranking eight old keys to four
values. On frozen records 8–15 it passes every semantic threshold: KL 0.01409,
top-1 0.94141, NLL delta −0.00613, and hidden L2 0.08559 over 256 positions.
Its old-context state and reads are constant in context length. At the short
33-token test point it still models at 93.34% of dense KV traffic.

The same state machine now has a stateful C++20 implementation and C ABI.
Randomized 40-token native/NumPy parity passes through eviction and
heavy-hitter replacement. Trained development substitution also passes
quality (KL 0.00528, top-1 0.96875, NLL +0.01239, hidden L2 0.04210). The
standalone native benchmark holds state at 249,248 bytes per layer and reduces
logical reads to 31.29%/8.40%/2.14% of dense at context
128/512/2,048.

Incremental compiled-package integration is now complete. The runtime resets
one persistent native cache per layer and batch item, processes prompt tokens
in order, advances absolute positions during decode, applies normal BitNet
RoPE at those positions, and does not allocate a Hugging Face KV cache.
Full-sequence and uneven-chunk execution are bit-identical for the bounded
operator, and a position discontinuity is rejected. Complete generation at
33/128/256 prompt tokens holds all-layer attention state at 7,477,440 bytes
and uses 86.55%/31.07%/16.35% of dense logical attention reads. It processes
about 0.87/0.98/1.01 input positions per second. Only 5.69/8.97/12.65 seconds
of the corresponding 39.06/131.97/255.64 seconds occur inside packed MLP
calls, so the next systems work should move Q/K/V/output projection and cache
orchestration across the native boundary and avoid a full vocabulary
projection on every decode step. Hardware DRAM counters remain unmeasured.

A fused position-major stream ABI has since reduced prompt-time native calls
from one per token per layer to one per layer. At 256 prompt tokens it changes
elapsed time only from 255.64 to 254.23 seconds (0.55%), so call-loop overhead
is rejected as the main problem. A measured 33-token phase profile assigns
11.60 seconds to Q/K/V projections, 7.71 to output projection, 12.62 to the
vocabulary head, 5.94 to the packed MLP, 0.12 to native attention, and 0.06 to
RoPE. The next implementation should therefore reuse the existing
threaded/base-3 machinery for packed native Q/K/V/O projections, then address
the tied vocabulary head separately.

That packed projection path is now implemented. It retains the official
four-codes-per-byte tensors, shares one 12-thread native kernel across all 120
Q/K/V/O modules, and does not materialize their BF16 matrices. The 33-token
end-to-end run falls from 38.51 to 22.29 seconds, with projection time falling
from 19.31 to 3.01 seconds. A direct 32-position comparison against the
materialized-projection package has KL 0.00394, top-1 0.96875, NLL delta
−0.00037, and hidden L2 0.03532. This is a development semantic pass, not yet a
frozen confirmation. The subsequent frozen 8-sequence/256-position result
passes with KL 0.00548, top-1 0.95703, NLL delta +0.00200, and hidden L2
0.05887. Native projection execution is 111.38 seconds versus 256.56 seconds
materialized on the same confirmation tensor. The projection path is promoted;
the vocabulary head now dominates at 13.00 seconds.

The vocabulary bottleneck was primarily redundant work rather than search
quality: BitNet exposes `logits_to_keep`, but package generation had left it at
zero and projected every prompt position. Requesting the final prompt logit
only preserves exact full-vocabulary selection. The 33-token run falls from
22.29 to 10.16 seconds and vocabulary time from 13.00 to 0.83 seconds. At 256
tokens, total generation falls from 254.23 to 20.72 seconds (91.8%) with
unchanged output tokens, traffic, and bounded state. A bounded vocabulary
index is stopped for generation because it would add recall risk after the
exact head ceased to dominate. The packed MLP now consumes 13.07 of 20.72
seconds and is again the principal target.

## Complete inference validation

The optimized components now pass one combined frozen test. On records 8–15
(8 sequences, 256 positions), packed native MLPs and projections plus bounded
native attention reach KL 0.01315, top-1 0.92969, NLL delta +0.00365, and
hidden L2 0.08436. All thresholds pass.

An eight-prompt, 16-token greedy suite produces recognizable factual,
explanatory, narrative, and procedural text with no identical-token runs and
consistent 7,477,440-byte attention state. The code prompt is not a good code
completion and the testing prompt drifts into a multiple-choice format, so
this proves generation works rather than broad task quality.

Full-prompt versus split-prompt final logits are bit-identical. Reset
generation returns the same tokens and stable cache counters, and EOS
termination is unit-tested. Complete prefill succeeds at 512 and 2,048 tokens
in 24.10 and 81.77 seconds. Peak process RSS is 2.14 and 2.57 GB.

The main performance blocker is still decode speed. The older shell measured
about 5.47 seconds per decoded token; the current native-DIP chat smoke took
5.16 seconds for one generated token after startup. Removing the Transformers
model shell therefore closed an architectural dependency, not the
single-token CPU optimization problem. A dedicated single-row
MLP/projection/vocabulary path remains justified.

An interactive `chat-native-bitnet` CLI now applies the authenticated
packaged tokenizer's template to structured history and re-prefills that
complete history through the DIP-only shared handle on every turn. It
supports history display, reset, clean exit, EOF, and interrupt handling. A
real default-system `Hello` smoke rendered 17 prompt tokens, generated token
`9906` (`Hello`) in 5.16 seconds, and reported 7,477,440 attention-state
bytes. The earlier two-turn poem session remains historical evidence from the
retired Transformers model shell; it has not yet been repeated as a scripted
multi-turn DIP confirmation.

The C++ token-step runtime and its versioned shared ABI now move the
adjudicated semantic operator through both the model-core and chat boundaries.
Package derivation authenticates the policy, adjudication, base records, and
coordinate index; the runtime maps both semantic artifacts and has no dense
MLP object or fallback. The rebuilt non-holdout 8×4 confirmation matches all
reference tokens and reset structure. The standalone binary remains
self-contained; the chat DSO exports only the narrow C ABI and depends on no
other Engram library. The frozen suite stays within W=16, while the 17-token
chat smoke crosses the boundary without constituting a sustained
older-retrieval test. Persistent chat caching, streaming, separately
adjudicated trust roots, measured DRAM traffic, and performance optimization
remain later work.

CUDA remains an optional training accelerator only; the serialized format and
inference mechanism are CPU-native. Repeating IVF, candidate-count,
regularization, prototype-density, small residual, post-hoc bit allocation,
loss-reweighting, or short grouped-ternary continuation sweeps is no longer
supported by the accumulated evidence.
