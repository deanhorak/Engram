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
reallocation of one global W/C/K policy under this budget is now closed.
A following layer-adaptive upper-bound experiment added a per-layer native
runtime interface and greedily rescued layers 11, 6, and 10 with W128 while
leaving the other 13 layers at W16/C8/K4/S2. Its evidence was valid and its
44.1701% logical-read schedule stayed under budget, but quality again failed
from position 32 onward. This frozen greedy three-layer W128 path under the
45% budget is therefore closed; the result does not rule out every interacting
whole-layer combination. The subsequent prospective head-wise experiment
used dense-teacher attention mass to rescue 51 of the 256 layer-head pairs.
Its parity, replay, resource, and provenance evidence passed, and its
44.9753872184% logical-read schedule remained under budget; 52 rescues would
require 45.2438%. It materially improved the layer-rescue quality, but still
failed overall at KL 0.073720, top-1 0.867188, NLL delta +0.053456, and hidden
L2 0.167518. The fixed attention-mass heuristic is therefore closed, while
causal/value-sensitive and dynamic head allocation were the remaining
boundaries. The subsequent exact-native-forward, straight-through experiment
trained a static causal/value-sensitive 51-head mask on the two selection
records. It improved its maximum and mean training objectives without a
per-record regression, but transferred worse than the attention-mass mask on
the six reused development records: KL 0.079132, top-1 0.864583, NLL delta
+0.081199, and hidden L2 0.182647. Its 44.9753872% read schedule and all
execution evidence passed, but all overall quality thresholds failed. This
closes the tested two-record natural-prose causal/value objective, not every
static selector. The later Q7-aware synthetic retrieval selector also failed
its exact-51 development gate while its W128 control passed. Train-only K2
prefix, payload, label-plus-payload, K51, ranked K64/K96/K128/K165, and
fixed-K256 scalar-bias branches then failed without violating their systems
contracts. Finally, a same-state W128-shadow capacity screen found that ranks
2/4/8 recovered 40.05%/42.87%/46.93% of the K256 residual globally with
oracle coefficients, below its frozen 50% gate. A later per-head
scheduled-source-mass oracle made scalar mass matching much more accurate but
worsened post-`W_o` recovery to -10.89%. The final joint output-targeted
gamma screen also failed: its continuous optimistic capacity bound recovered
22.74% globally and its discrete direct arm recovered 19.98%, versus the
frozen 50% requirement. A later exact per-slot product-simplex oracle raised
the ceiling to 38.44%, but its exact-native-anchor optimistic hull still
failed the global gate with negligible certificate uncertainty. The current
regular aggregate plus eight episodic values is therefore closed. The
full-visible successor exposed the 16 local, four selected-older, and eight
episodic values separately. Its constructible C28 arm passed the frozen
train-only capacity gate at 66.54% global recovery, with no additional KV
state or reads; optimistic C29 also passed. This establishes that the values
already fetched contain sufficient same-state capacity and authorizes a
causal 28-logit selector. It does not establish that such a selector is
learnable or pass a native rollout, development, confirmation, package, or
Milestone 3 gate. W128 itself reads 100% when applied globally and is not
deployable. Measured whole-system traffic and optimization also remain.
Milestone 2 remains passed for the qualified Q7 semantic path, while
Milestone 3 remains blocked on causal bounded-attention selection.

Engram's target runtime combines a shared recurrent controller, fixed sparse semantic
memory derived from SwiGLU records, hybrid local/recurrent/retrieval episodic memory, an
indexed vocabulary projection, transition caching, and uncertainty-triggered corrections.
The target package must generate without loading source transformer layers.

## Hybrid host integration

Because the learned layer-free provider remains unqualified, the practical
deployment direction keeps a conventional quantized Transformer as the
quality anchor. `engram.runtime.hybrid` provides a separate sidecar boundary:
it loads immutable JSONL memory records, performs bounded deterministic CPU
retrieval, renders provenance-tagged reference context, and calls an
OpenAI-compatible chat completion endpoint. The host owns tokenization,
hidden states, attention, MLPs, logits, and generation. No Transformers model
shell or hidden-state contract is required from Engram in this mode.

The sidecar has explicit baseline and augmented modes so a fixed host can be
measured with and without retrieval. The current hashing encoder is only a
dependency-free lexical baseline. It should be replaced by a frozen semantic
embedding index only after a task-quality and end-to-end-cost benchmark shows
that retrieval is useful. This hybrid path is not evidence for the original
layer-free Milestone 4 claim; it is the concrete product fallback if the host
model remains necessary for quality.

Memory records may store a full retrieval document separately from an optional
concise deployment payload. The hashing index embeds the full `text`; only
`prompt_text` is injected after selection. This distinction reduces host prefill
traffic without weakening the lexical evidence available to the selector. Benchmark
prompts can also carry expected memory IDs and required answer terms. Those checks
are deliberately deterministic and task-specific, and the input hashes are recorded
so later optimization cannot silently change the evaluation set.

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
failure” or merely tuning one global cache split. It justified testing whether
memory should be allocated by layer/head or whether older-state selection must
be learned from dense-teacher behavior.

The first layer-adaptive test is also complete. An additive native C ABI,
`engram_olmoe_token_open_layered_v1`, accepts one W/C/K/S capacity policy for
each of the 16 OLMoE layers; the existing scalar open ABI remains unchanged.
The Python `OLMoENativeTokenRuntime` exposes the same mutually exclusive
`attention_policies` path. Heterogeneous state, scratch, traffic, eviction,
candidate, selected-entry, sink, and heavy-hitter metrics are summed across
layers, and an all-base layered run is bit-exact with the historical scalar
runtime.

A frozen three-round greedy search evaluated all 16 + 15 + 14 layer choices
on two selection sequences. It chose layers 11, 6, and 10 for
`W128/C8/K4/S2` and retained `W16/C8/K4/S2` on the other 13 layers. The final
schedule read 955,957,248 logical attention bytes per 128-position sequence,
or 44.1701% of dense attention. All candidate, resource, replay, and
authentication evidence passed. The six-sequence internal screen nevertheless
failed with KL 0.102321, top-1 agreement 0.845052, NLL delta +0.116776, and
hidden L2 0.206037. Positions 0–15 and 16–31 passed, while every metric failed
in each band from 32 onward. This closes the tested frozen greedy three-layer
path under the cap rather than proving that all layer-adaptive attention or
every interacting whole-layer combination is ineffective.

The experiment used the raw token runtime only. Package format version 1
continues to bind one global `W16/C8/K4/S2` policy, so the failed schedule was
not installed and no package schema was promoted.

The following prospective experiment allocated rescue at head granularity.
Its additive experimental C ABI,
`engram_olmoe_token_open_headwise_v1`, accepts a layer-major array containing
one W/C/K/S policy for each layer-head pair. It preserves the older scalar and
layered entry points. Version 1 deliberately supports only models with equal
query and key/value head counts, allowing each query head to own one
independent streaming K/V cache. The Python runtime exposes the corresponding
mutually exclusive `attention_head_policies` path.

An all-base head-wise configuration passed exact semantic parity with the
layered runtime; state, scratch, and eviction accounting also passed their
separately specified per-head contracts. Dense-teacher attention mass then
fixed a mask of 51 pairs at `W128/C8/K4/S2`, leaving 205 pairs at
`W16/C8/K4/S2`. The schedule reads 973,384,704 logical bytes per sequence,
44.975387218386625% of dense attention. A 52-pair mask would read
979,193,856 bytes, or 45.2437999637%, so 51 is the largest admissible count.
The Q7 path and its scheduled traffic are unchanged.

On the six reused internal 128-position records, all parity, structural,
resource, reset-replay, and authentication evidence passed. The 768-prediction
screen nevertheless reached only KL 0.073719930, top-1 0.8671875, NLL delta
+0.053455543, and final-hidden relative L2 0.167517818, against thresholds
0.05, 0.90, +0.05, and 0.10. Both bands through position 31 passed. At
positions 32–63, top-1 and hidden L2 failed; every metric failed in the
64–95 and 96–127 bands. W128 is full causal context only for this
128-position protocol, not an unbounded cache.

The head-wise mask improves substantially over the whole-layer rescue but is
still a development quality failure. It was not promoted into package format
version 1. This closes the fixed teacher-attention-mass heuristic, not the
head-wise runtime or all possible masks. A selector trained or calibrated
against causal/value sensitivity, or a dynamic teacher-distilled allocator,
is the next architectural boundary.

That static causal/value-sensitive selector has now also been tested. For
every layer-head pair, training mixed exact native `W16/C8/K4/S2` and
`W128/C8/K4/S2` forward outputs while using a differentiable
fixed-membership surrogate only for backward attribution. Two IHT steps
averaged complete gradients from selection records 0 and 1 and hard-projected
to exactly 51 rescued heads after each step. M1 was selected: the maximum
composite objective fell from 7.8671169 to 4.7559915 and the mean fell from
6.9172161 to 4.3284769, with no per-record regression. The CPU-only serial
BF16 proxy fit took about 115.5 minutes and passed its full evidence contract.

Offline causal attribution now has a qualified deterministic expert-parallel
backward path. The installed `grouped_mm` dispatcher still executes forward
serially through its CPU fallback; only independent frozen-expert backward
replays are distributed across 12 workers and then reduced in the fallback's
exact accumulation order. On one complete archived record, 961 expert tasks
reproduced the entire non-timing result exactly and reduced elapsed time from
1,564.347 to 809.168 seconds. This is a single-record development
qualification across separate executions, not a change to the native Q7
runtime or the unresolved bounded-attention architecture.

The decisive measurement remained the complete native Q7 screen, not the
BF16 training proxy. On the six reused development records and 768 positions,
all resource and execution evidence passed at 44.9753872184% logical
attention reads, but quality failed at KL 0.07913208059, top-1
0.8645833333, NLL delta +0.08119899696, and hidden L2 0.18264718059.
Both bands through position 31 passed; positions 32–63 failed top-1 and
hidden L2, and both later bands failed all four metrics. This is worse on
every overall metric than the prior attention-mass mask. No confirmation was
opened and no package policy was promoted.

This result closed static selection by attention mass and by the tested
two-record natural-prose causal/value objective. A later Q7-aware
retrieval-targeted selector used a new 8/8/8 synthetic passkey corpus and
answer-only supervision. Its exact-51 `M2` mask passed the training-selection
rule, but failed development KL, NLL-delta, and hidden-state gates while the
full-W128 control passed. The reserved confirmation split was not opened.

The subsequent train-only capacity branch removed selector uncertainty.
Causal K2 prefix prototypes, exact payload-only and label-plus-payload
episodic oracles, the transferred K51 mask, and ranked K64/K96/K128/K165
head prefixes all failed strict answer-loss progression while passing native
resource and replay checks. The all-head K256 payload was still the strongest
bounded representation at mean/worst answer CE 1.224460/1.327343 and
33.0305% upper-bound traffic.

A versioned logit-bias ABI then kept that K256 representation and schedule
fixed and added one shared `beta=float32(log(gamma))` to every episodic score.
Exact `beta=0` V1/V2 parity passed. All four nonzero arms
`gamma={1/2,1/4,3/16,1/8}` completed the eight-record train screen and
failed; their mean/worst CE values were 1.461414/1.669250,
1.883818/2.288258, 2.161750/2.595642, and 2.725091/3.430532.
`gamma=1/2` was replayed only as the best failed nonzero arm and remained
worse than historical `beta=0`. The immutable
[V2 report](../reports/olmoe_q7_retrieval_episodic_logit_bias_2026-07-29/summary.md)
closes shared scalar softmax calibration; it does not promote a new attention
policy.

The subsequent **same-state shadow residual** screen fixed `beta=0` as the
K256 base. At every layer/token, a non-intervening train-only W128 shadow and
the deployable branch consumed identical candidate-produced Q/K/V. If `b` is
the base output after `W_o` and `f` is the shadow output after the same
projection, the target was `f-b`. Leave-one-sequence-out ranks 2/4/8 recovered
0.400470/0.428686/0.469253 globally with oracle held-out coefficients. Every
rank passed its per-sequence, block-entry, finite, and positive-layer
conditions, but all missed the frozen 0.50 global gate. The
[archived result](../reports/olmoe_q7_retrieval_episodic_residual_capacity_2026-07-29/summary.md)
therefore closes rank-at-most-8 global per-layer output subspaces, not residual
correction in general.

No causal coefficient predictor, production correction operator, loader, or
package field was authorized. Two subsequent capacity tests have now closed
the remaining scalar-mass branch. First, an oracle selected one of eight gamma
codes independently at every record/read/layer/head coordinate to match exact
scheduled-source probability mass. Mean mass error fell from 0.04451 to
0.00848, yet global post-`W_o` residual recovery worsened to -10.89%. Scalar
probability mass is not a sufficient proxy for the value vector delivered
through the output projection.

The follow-up [joint output-targeted oracle](../reports/olmoe_q7_retrieval_episodic_joint_gamma_oracle_2026-07-30/summary.md)
optimized all 16 head codes together against the exact post-`W_o` residual,
including cross-head projection terms. Even its more permissive continuous
box relaxation had an optimistic recovery upper bound of only 22.74%
globally, with seven of eight sequences and every block entry below 25%. The
discrete direct float32 arm recovered 19.98%. All 16 layers improved, but the
frozen global, sequence, and block gates failed. This closes only the cached
same-state bounded affine `(q,d)` family at fixed K256. It does not close
episodic attention generally; the next architecture must expose additional
retrieved value directions or change the bounded memory mechanism rather than
retune one aggregate episodic mass.

The next [per-slot capacity oracle](../reports/olmoe_q7_retrieval_episodic_slot_simplex_oracle_2026-07-30/summary.md)
did expose all eight episodic values separately. A constructible convex
mixture over those values plus the regular aggregate recovered 38.44%
globally. An optimistic hull containing the exact native head output changed
that result by less than `4e-9`; its certified upper bound still missed the
50% gate. This closes that nine-direction value family while leaving one
important no-extra-read expansion: split the regular aggregate into the 16
local and four selected-older values that the current kernel already reads.
The frozen
[full-visible capacity oracle](../reports/olmoe_q7_retrieval_episodic_full_visible_simplex_oracle_2026-07-30/README.md)
then exposed those 20 regular values separately alongside all eight episodic
values. Constructible C28 recovered 0.6653937751 globally, with minimum
sequence and block-entry recoveries of 0.6447006551 and 0.6306278392 and
16/16 positive layers. Optimistic C29 recovered 0.6653865288. Both passed
their frozen qualification, replay, certificate, and authentication checks.
Nested C10/C16 results of 0.5335805245/0.6021187653 are diagnostics only.

Architecturally, this is the first evidence that the existing bounded value
set—not a larger KV store—contains enough same-state residual capacity. It
does not define the causal map from inference state to the 28 coefficients.
The next component is therefore a causal 28-logit selector whose inputs,
state, traffic, and latency are fully accounted. Package promotion and
Milestone 3 remain blocked until that selector survives native causal
rollout, development, and later confirmation.

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
