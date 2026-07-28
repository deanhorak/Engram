# Architecture

For the current implementation/evidence boundary, see
[Project status](status.md). No representation has passed the original
dense-Llama conversion gate. The separate native-BitNet track now has a
practical CPU-only DIP route that passes the complete qualifying development
quality, recall, activity, and modeled-traffic gate. The same frozen route
also passes the independent final raw evaluation, and its semantic-memory gate
is recorded as passed by postmortem adjudication. The original final wrapper
ended in error on a full-record/object-versus-33-token/list hash-contract
defect after evaluation completed, so this is not a pristine runner pass.
This remains a distinct low-bit-native source path, not a dense-model
conversion result or proof that all Milestone 2 work is complete. The
adjudicated index is now installed into an authenticated derived package and
the DIP-only C++ token runtime passes its fixed non-holdout integration suite.

A third, OLMoE-specific branch now passes the semantic causal/evidence screen.
OLMoE supplies 64 independently stored SwiGLU experts per layer and a trained
top-8 router, avoiding post-hoc recovery of sparse membership from one dense
MLP. Signed Q7 weights in groups of 64 with BF16 scales pass an all-layer
8-sequence/256-position intervention at 22.7865% modeled expert/router traffic.
The immutable packed artifact and direct CPU top-eight expert kernel now pass
route/output and scheduled-byte parity. The architectural result is a
qualified native semantic substrate and authenticated, installable OLMoE
generation package. Its package-only frontend performs tokenization; the
persistent C++ object owns mapped weights, Q7 dispatch, attention caches,
residual state, full vocabulary logits, and vocabulary selection. The frozen
complete-native 8×32 causal protocol passes overall and independently on the
128 positions beyond W=16. A stronger authenticated 8×128 run then shows that
W16/C8/K4/S2 drifts after offset 31, while a matched W128 full-attention
control passes every band. That control preserves the Milestone 2 Q7 semantic
conclusion and moves the unresolved OLMoE boundary to Milestone 3 attention.
The subsequent fixed-budget sweep tested three different allocations of the
same 44.7614% logical-read budget. All evidence checks passed, but none of the
three policies passed semantic quality, so no arm was selected. Static
reallocation of one global W/C/K policy under this budget is now closed. The
next justified architecture is a layer/head-adaptive or learned/distilled
selector for older context. W128 itself reads 100% and is not deployable.
Measured whole-system traffic and optimization also remain.

Engram's target runtime combines a shared recurrent controller, fixed sparse semantic
memory derived from SwiGLU records, hybrid local/recurrent/retrieval episodic memory, an
indexed vocabulary projection, transition caching, and uncertainty-triggered corrections.
The target package must generate without loading source transformer layers.

This document describes one compiled model worker. A separate, request-level
[Oracle cognitive executive](cognitive_executive.md) may manage goals, evidence, persistent
memory policy, tools, and multiple workers above it. Those system functions are not stored in an
`.engram` package and do not run in its per-token loop.

The executive reference layer supports transactional SQLite and checksummed JSONL event streams,
versioned worker capabilities, and typed worker adapters. These deployment/session artifacts remain
outside the compiled model format.

## Implemented foundation

For a Llama SwiGLU MLP, Engram represents neuron `j` as two keys and one value:

```text
a_j(h) = SiLU(W_gate[j] h) * (W_up[j] h)
v_j    = W_down[:, j]
FFN(h) = sum_j a_j(h) v_j
```

The Python implementation verifies this identity and the native scalar kernel provides an
independent implementation. The magnitude reference uses all activations to establish a
full-information baseline before practical routing is attempted. It is not the optimal K-subset
in general because vector contributions can cancel.

## OLMoE expert decomposition

For OLMoE, the semantic record is an entire trained expert rather than one
neuron. The router computes a 64-way softmax and selects eight experts:

```text
p = softmax(W_router h)
S = top8(p)
MoE(h) = sum_{e in S} p_e W_down,e(
           SiLU(W_gate,e h) * (W_up,e h))
```

Engram captures the source probabilities, selected IDs and weights, individual
weighted expert contributions, and their exact sum. The passing Q7 simulation
keeps this learned selection unchanged and quantizes only expert matrices.
Router matrices remain BF16 and are included in the traffic numerator.

## Separate native-BitNet record track

Native BitNet is kept behind a distinct adapter because its MLP is not the
SiLU identity above. For channel `j`:

```text
z_j(h) = ReLU(W_gate[j] Q8(h))^2 * (W_up[j] Q8(h))
r(h)   = sqrt(mean_j z_j(h)^2 + epsilon)
FFN(h) = W_down Q8(gamma * z(h) / r(h))
```

`Q8` is native per-token activation quantization and `gamma` is the
intermediate `ffn_sub_norm` gain. The scalar denominator couples all
channels, so this format cannot be admitted silently to the existing
independent-SwiGLU oracle or router.

The exact low-bit artifact nevertheless makes each channel addressable as a
logical record containing its gate row, up row, transposed down column, and
BF16 `gamma`. Five ternary digits are packed in one byte; the three native
BF16 projection scales are layer-global. Physically, four cache-aligned
gate/up/gain/down streams follow the computation phases. Four base pointers
retain O(1) channel addressing while avoiding cache-line rereads around the
shared normalization. The direct CPU kernel now executes from those mapped
streams with one scheduled cold read of each serialized line and no dense
weight materialization. The new sparse DIP kernel additionally uses a
source-bound coordinate-major gate/up index and estimates the shared RMS
denominator without dense gate/up completion.

For every live BF16 input it preserves native Q8 quantization, keeps the
largest 1,920 of 2,560 coordinates, scans the corresponding packed ternary
gate/up coordinate rows, exactly completes a frozen per-layer candidate set,
and computes exact candidate utility. It then selects all nonzero candidate
utilities, clipped to a minimum of 346 and a per-layer maximum, before reading
only those down rows. Most layers estimate missing squared energy by scaling
the proxy tail with the exact/proxy candidate-energy ratio. Layer 9 instead
uses a corrected-proxy estimate and an eight-record top-proxy-raw-square audit
inside its fixed candidate union.

The v2 index authenticates q, per-layer candidate and maximum-K counts, RMS
policy, every payload, and the SHA-256 of the base record artifact. The native
loader fails closed on a policy, source, checksum, shape, alignment, padding,
or encoding mismatch. A debug path returns coordinate, candidate, and selected
record identities outside timing; an evaluation-only dense teacher path
computes fixed top-K membership for recall and is not called by sparse
inference.

Promotion into inference uses an immutable-source derivation rather than
editing the package bound into the frozen policy. The semantic-memory
installer verifies the complete source package inventory and the exact policy,
adjudication, base-artifact, and index hashes, copies the source package to a
new directory, installs the v2 index, rebuilds the file inventory, and writes
an authenticated `semantic_memory` descriptor. It refuses an in-place target,
a false adjudication, a policy/index mismatch, or a pre-existing derived
package with different inputs.

The native generation executable adds a second, independent trust boundary.
It pins the promoted manifest digest/size and the source-package, record,
index, policy, and adjudication hashes. It rejects a symlinked root, symlinked
or non-regular entries, missing/extra files, and checksum or descriptor drift
before mapping large weights or constructing worker threads. Architecture,
paths, vocabulary/context bounds, attention policy, RoPE/RMS values, and EOS
IDs—including `128009`—come from the authenticated package instead of C++ CLI
constants. The executable statically incorporates the Engram kernels and does
not depend on an Engram shared library at runtime.

The native token runtime is correspondingly fail-closed. Its only semantic
object is `NativeBitNetDIPKernel`; it cannot instantiate the former dense
`NativeBitNetKernel`. Each layer follows the direct sequence:

```text
stage state
  -> native Q/K/V/O attention and bounded cache update
  -> post-attention normalization and semantic input
  -> DIP candidate/selection/down-record execution
  -> scaled semantic acceptance into the stage state
```

After all 30 layers it applies final normalization and the tied-vocabulary
argmax, advances absolute positions, and retains the bounded attention caches
for the next token. Reset clears the position and every cache. The packaged
8-prompt/32-token integration run matches every greedy reference token and all
stage/semantic row and call counts without dense fallback. Reset replay also
matches tokens and structural counters after proving that counters zeroed.
Neither check compares hidden states or logits. All processed contexts are at
most 14 positions, so the W=16 integration suite does not exercise eviction
or older-context retrieval. A separate complete-runtime protocol at
16/17/18/24/32 positions now confirms the expected eviction, sink,
older-candidate, bounded-selection, and heavy-hitter counters with constant
state and reset replay. That closes the mechanical coverage gap but does not
compare long-context attention quality with a dense teacher.

## Semantic-memory prototype

Research-only semantic packages may retain exact reference keys/values; compiled runtime
packages store 8-bit per-dimension key codes plus additive residual value-codebooks. Every array has
dtype, shape, byte-order, alignment intent, byte count, and checksum metadata. The practical
router clusters concatenated normalized gate/up geometry into a deterministic IVF index,
scores the small centroid table, scans only selected postings, and then evaluates the exact
two-key SwiGLU expression only for candidates. A deterministic brute-force router remains for
tests. A fitted low-rank linear operator models the residual left by the sparse read.

Research-only learned routers include a direct multi-label ridge model, an equivalent direct
low-rank fit, disjoint coverage groups, and bounded-replication overlapping postings with a
low-rank group selector. None of those learned selectors passes the trained-teacher intervention
gate on SmolLM2-135M. The first tested passing magnitude reference retains 768/1,536 records,
which is already a weak sparsity result at that operating point.
Full-corpus refits use all 1,112 available calibration states per layer. Their modest recall gains
do not close either the 95% recall threshold or the much larger causal-quality gap, so they reject
these router configurations rather than every possible sparse representation.

The first realizable selector algorithm in the earlier dense-SmolLM study was
inspired by predictor-free Dynamic Input Pruning. The published method
supplies dynamic top-magnitude input pruning and partial scoring;
candidate-only exact completion and contribution-norm reranking are Engram
extensions. For each state, it selects the `q` largest absolute hidden coordinates,
reads only those gate/up columns for all `I` records, and computes a partial SwiGLU contribution
score. It keeps `C` candidates, reads their omitted gate/up coordinates to recover exact
activations, exactly reranks to `K`, and reads only those `K` down-projection columns. Its projected
weight reads per layer are:

```text
2 * I * q + 2 * C * (H - q) + K * H
```

At `H=576`, `I=1,536`, `q=432`, `C=896`, and `K=768`, the arm passes both its development grid and
a sequence-disjoint confirmation corpus, with 98.97% confirmation oracle-set recall. It projects
to 76.39% of dense MLP scalar reads. A separate version-2 experimental package stores gate/up
weights coordinate-major and down weights record-major; Python and native candidate-only readers
have exact selection parity. Cache-line amplification raises the estimate to 83.33%. The checked
optimized float32 kernel cycles all 30 layers but is still 15.4% slower than
dense, so it remains outside the compiled runtime. This is historical
dense-source evidence. The native-BitNet DIP design above uses a different
low-bit artifact, adaptive K, and RMS estimator and now passes its complete
development gate.

## Episodic-memory prototype

The episodic baseline keeps an exact causal local window, a normalized ELU+1
linear-attention state whose size depends on head dimensions rather than
sequence length, and a configurable fixed-capacity older-token ring. The new
native-BitNet substitution harness additionally runs the trained transformer's
real Q/K/V projections and RoPE while replacing attention itself. Local and
retrieved keys participate in one exact sparse softmax rather than separately
normalized reads.

The promoted native operator has an exact 16-token ring and eight retained old
entries: two immutable attention sinks and six cumulative-attention heavy
hitters. It scores all eight old keys, transfers the best four old values, and
normalizes those values jointly with the local window. Each layer and batch
item owns one persistent C++ state object. Prompt prefill may arrive as a
multi-token tensor, but the wrapper advances the state one token at a time.
Incremental calls must begin at the state's next absolute position. The model
computes RoPE from those explicit positions before Q/K cross the native
boundary, so rotating a decoded key never depends on a fabricated dense cache.
`use_cache=False` prevents Transformers from allocating its `DynamicCache`.
State is reset between independent generations and cannot silently survive a
batch-size or position discontinuity.

OLMoE uses the same cache mechanism but has not inherited native-BitNet's
quality result. Its matched 8×128 development sweep held mature visible values
and total logical reads fixed while exchanging recent locality for older
retrieval:

- `W16/C18/K16/S2`: KL 0.063887, top-1 0.867188, NLL delta
  +0.051701, and hidden L2 0.157717;
- `W24/C10/K8/S2`: KL 0.065912, top-1 0.877930, NLL delta
  +0.058480, and hidden L2 0.159755;
- `W30/C4/K2/S2`: KL 0.095813, top-1 0.840820, NLL delta
  +0.075728, and hidden L2 0.188422.

All three arms passed authentication, counter, traffic, replay, and
pre-eviction identity checks, but all failed the frozen semantic gate. The
W128 control remains the causal ceiling at KL 0.003438, top-1 0.974609, NLL
delta +0.001459, and hidden L2 0.041389. This rules out choosing a “best
failure” or merely tuning one global cache split. The next operator must
allocate memory by layer/head or learn which older states to retain from dense
teacher behavior.

## Token-level controller and output path

The compiled runtime uses a GRU-like controller whose base kernels are shared across cycles.
Stage embeddings and optional low-rank stage adapters retain stage identity. Fixed and adaptive
cycle policies expose residual/confidence histories and extra-cycle requests. Vocabulary output
uses a deterministic normalized-embedding IVF search followed by exact original-vector rescoring,
with an exact dense
fallback. A quantized-state LRU transition cache validates every hit against a configured radius.
Correction capsules provide state-selected low-rank residual updates and uncertainty-driven
requests for expanded semantic, episodic, vocabulary, or cycle budgets.

## Native runtime

The C++20 runtime verifies package SHA-256 checksums, memory-maps NPY semantic/controller arrays,
preallocates hot-path scratch, and implements scalar semantic, episodic, controller, vocabulary,
and transition-cache paths. AVX2 dot products are isolated behind safe CPU dispatch; this host
lacks AVX2 and executes the scalar fallback. Python/native greedy fixture tokens are tested for
exact parity.

## Open architectural work

The bounded dense-source representation search now includes cache-reused recurrent compact
MLPs, projection-normalized ternary projections, affine constrained-vector
quantization, unrestricted four-weight codebooks, and high-dimensional
lifted-binary lattices. Their reusable modules define hard-forward operators
and complete cold-byte models, but none passes the layer-local progression
ceiling. They are research components, not supported `.engram` package
layouts. That evidence motivated the separate pretraining-native BitNet track
described above.

That budget-native mechanism is now implemented for a full-width
grouped-ternary SwiGLU. The artifact packs five ternary weights per byte and a
non-learned FP16 scale per 128-weight group; complete 30-layer traffic is
43.1353% of dense ideal Q4. Training retains float masters only outside the
artifact, executes hard decoded weights with straight-through gradients, can
transition deepest layers first, and serializes/reloads the exact binary
before causal validation. Attention, normalization, and the tied
embedding/output head can co-adapt as replacements for already-resident
tensors. Checkpoints are device-neutral, so optional CUDA training does not
create a GPU runtime dependency.

The architecture is operational but not qualified. After 1,014,225 training
positions it remains at KL 2.2844, top-1 0.3198, NLL +2.2770, and hidden L2
0.6036. It misses the predeclared top-1 and hidden-state scale-up checks, so
the exact design is stopped before 3M. Another architecture must change the
learned representation or its pretraining origin rather than append another
small post-hoc correction to this artifact.

The semantic and vocabulary proxies use IVF in both runtimes, but still scan
every coarse centroid and their tiny-fixture centroid traffic is unfavorable.
The dense-SmolLM learned-routing branch remains blocked: full-corpus,
corpus-scaled regularization, and candidate-budget sweeps did not close its
gap, and the rank-16 arm fails causal quality while reading 95.8% of record
keys. Its older predictor-free DIP arm passed quality but failed physical
traffic and latency. The distinct native-BitNet DIP route now has the
cache-aware packed index, native sparse kernel, and live-BF16 all-layer
development evidence that were previously missing. Its final raw report also
passes, and the semantic gate is accepted by postmortem adjudication because
the original wrapper's token-hash schema check was defective. Hardware DRAM
measurement and replication beyond this host and model remain open. The final
sparse timing is 1.1449x dense, so it is not a speedup.
Global and failure-region low-rank correction capsules have been fitted, but
every tested dense-source layout worsens held-out local MLP error and is
rejected before causal integration.
The first sparse-teacher trainer keeps two model copies: an immutable dense teacher and a student
whose attention, normalization, embeddings, and original MLP tensors are frozen. Trainable
rank-16 router factors receive oracle-membership BCE supervision; rank-8 sparse down adapters
receive local MLP, hidden-state, and logit-distillation gradients. Only these tensors are written
to a safetensors experiment artifact. The first 32-step pilot fails the progression gate, and a
subsequent audit shows that its hard route prevents those causal losses from reaching the router.
The replacement hardware-aware wrapper uses an exact hard sparse forward at
`q<=62.5%`, `C/K<=512` and a sigmoid straight-through backward estimator. Causal losses can now
update router factors, and an expected cache-line-occupancy term trains candidate locality. The
dense oracle is detached supervision only. This design passes unit gradient checks, but its
complete 32/16 run fails the held-out progression gate. The trainer now batches masked sequences, caches
router initialization, separates router-calibration and student-training datasets, and trains
mergeable gate/up/down LoRA, but a broader-corpus stage does not improve every causal metric.
Candidate-set packing and whole-line selection also fail their trace screens, so this specific
individual-record locality formulation is stopped.
Corrected LoRA scaling and resumable checkpoints support a full 128-sequence follow-up. A rank-32
hidden residual improves several causal metrics slightly but has effectively zero alignment with
the omitted MLP output and remains far outside the gate, so it is disabled by default. Oracle
cache-line bounds and a sequence-disjoint layer-adaptive confirmation also fail. Further work must
co-train structured sparsity with the MLP basis; the frozen-neuron low-budget path is blocked. A
trace-only whole-block screen now rejects static 64-, 32-, and 16-record expert groupings before
end-to-end training: even the finest full-information greedy reference has 0.438 local relative
L2 at 512 active records. The next bounded architecture is native gate-based channel routing with
hardware-grouped selection and a sparse forward throughout training.
That native-gate shadow now projects to 43.06% dense traffic at q=62.5%/K=512 and confirms that
input pruning adds little error; gate-only utility prediction is the blocker. The exact hard-forward
training wrapper works, but its representative cached-boundary run improves only 2.55%. The next
test must therefore be progressive end-to-end co-training rather than layer-local fitting. That
trainer must remain functional on CPU; optional CUDA acceleration must not affect the model format,
hard sparse semantics, gate criteria, or CPU inference implementation.
The trainer is now implemented with those properties, including dense-to-sparse scheduling and
device-neutral resume checkpoints. Its eight-step CPU stage fails to improve all held-out causal
metrics together, so availability of the mechanism does not yet justify scaling the same objective.
A deployable low-rank utility residual now supplies the missing up-dependent selection signal
without reading dense up weights. Rank 16 adds 141,312 bytes of per-token-layer weight traffic,
bringing q=62.5%/K=512 to 44.39% of dense. The per-layer factors and bias are stored in safetensors,
validated against the source-model hash, and registered as fixed wrapper state. This nearly halves
causal KL but does not pass the final gate. Because the factors were fitted on teacher states, the
next architectural test is an on-policy refit using states generated by the sparse student itself.
That refit improves controlled same-state local error by only 0.38%, so it is rejected. A
traffic-neutral K=640 redistribution and exact selected-gate completion also fail. Because the
full-information K=512 oracle itself misses the causal gate, the next architecture must alter the
MLP basis: progressively distill a structured sparse or fixed-width student with full MLP updates
on substantially more real-token data. This is a training-time change; the target remains a packed,
contiguous CPU inference representation below 45% dense MLP traffic.

`engram train-width-pruned-student` implements that fixed-width test. Each student MLP owns frozen
dense buffers for progressive replacement and trainable compact gate, up, and down matrices. The
compact matrices are initialized from trace contribution rankings or deterministic weight
geometry, then optimized against teacher MLP outputs, intermediate hidden states, and logits.
Validation forces all layers into compact mode. A 672/1,536 width ratio reads 43.75% of the dense
MLP weights with contiguous matrices and requires no runtime router. Checkpoints are device-neutral;
`--resume` restores optimizer/history on the same corpus, while `--initial-checkpoint` transfers
only compact parameters to a different, provenance-checked corpus. The first full-corpus result
fails the semantic gate, so this representation is experimental and is not emitted by the package
compiler.

For the cheaper local-capacity screen, `engram trace --mlp-only --tokens-per-sequence N` executes
each complete sequence but stores only a seeded sample of exact MLP inputs/outputs. It omits all
attention tensors and records the sampling policy in the checksummed trace manifest.
`engram evaluate-width-local-ceiling` then fits selected compact layers entirely from these cached
boundaries. This separates the compact basis's approximation capacity from accumulated causal
state drift and avoids repeated teacher-transformer execution.

Later work tested the remaining compact and nonparametric alternatives. A
layer-adaptive 672/704-width Q4 student produces a complete, independently
reloaded 44.9334%-traffic artifact, but its frozen 3M-position checkpoint has
KL 0.8866, top-1 agreement 0.5659, NLL delta +0.8838, and final-hidden L2
0.4245. Exact input/output memory also fails to scale: adding one million
pretraining prototypes to 233,005 local prototypes lowers layer-14 LLE-32
error only from 0.327526 to 0.321854. These results close the tested
single-compact-MLP and prototype-density architectures; they are not compiler
inputs.

The original older-context retrieval prototype uses a linear candidate scan.
Native-BitNet now also has a bounded streaming operator: exact local context,
fixed attention sinks, and an online cumulative-attention heavy-hitter cache.
Its frozen trained-model confirmation passes without consulting evicted keys.
Hardware DRAM measurement and controller distillation remain open. The DIP
semantic gate passes by postmortem adjudication, while clean independent
replication and systems tuning remain open.
For native BitNet, the memory-mapped CPU kernel now reads the independently
fixed-stride gate/up/gain/down base-3 streams directly. The 1,538-byte figure
is a logical per-channel payload, not a contiguous physical record. The
source-family-specific package and generation runtime are now implemented and
produce exact source/package output parity. The bounded W=16/C=8/K=4 streaming
attention operator now also passes causal confirmation. The next architectural
task is complete: a stateful native sink/heavy-hitter cache and exact rerank
pass randomized parity, trained substitution, incremental package generation,
explicit RoPE/cache-position advancement, and long-context validation.
Attention state is fixed at 7,477,440 bytes across the complete 30-layer
runtime, and modeled attention reads fall to 2.14% of dense at 2,048 tokens.
These are logical interface bytes, not measured hardware DRAM traffic.

Interactive chat is now a thin orchestration layer over the authenticated
DIP token runtime. Python retains structured messages, renders the packaged
tokenizer's chat template, and encodes/decodes token IDs. A versioned C ABI
owns the mapped model, DIP index, controller scales, attention caches, and
token generation. Each turn explicitly resets the handle and re-prefills the
complete rendered history from position zero; the ABI rejects a second
generation without reset so full history cannot accidentally be appended to a
live cache.

The C constructor accepts only the package root and invokes the same
production-pinned loader as the standalone executable. Python cannot provide
model dimensions, artifact paths, routing policy, W/C/K/S attention settings,
or EOS overrides. There is no `AutoModelForCausalLM`, Torch execution, decoder
layer, or dense semantic fallback in this chat path. Persistent cross-turn
cache reuse, streaming output, context truncation, and a sustained
long-context older-memory confirmation remain open.
