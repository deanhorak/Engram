# How Engram works

This document explains Engram for readers with a general computer science background. No prior
knowledge of neural networks or large language models is assumed. It describes the architecture
we are trying to build, how information is extracted from a trained Llama-compatible model, the
compiled representation, and how generation works without the source transformer.

Engram is a research prototype. Where the implemented baseline differs from the intended trained
system, this document says so explicitly.

For the shortest current account, start with [Project status](status.md). The
key result is that the separate native-BitNet path now combines teacher-like
behavior with the required modeled memory-traffic reduction through practical
DIP semantic memory. Its final semantic gate passed by postmortem
adjudication, and the authenticated DIP package now runs inside the complete
CPU-only C++ token-step runtime and the user-facing chat command. The original
dense-Llama conversion path still has no qualifying representation. A newer
OLMoE branch now passes semantic quality/evidence using its native learned
top-8 expert router and Q7 expert weights. Its packed serialization and direct
CPU expert kernel now pass too. A mapped BF16 companion artifact and complete
native token/generation path are installed in an authenticated package.
The frozen complete-native 8×32 causal protocol now passes both its exact
local and bounded older-context halves. A stronger 8×128 follow-up finds that
W16/C8/K4/S2 fails after offset 31; changing only W16 to full W128 attention
restores every semantic band. This vindicates the Q7 semantic-memory path but
blocks OLMoE Milestone 3 attention substitution until a bounded policy passes
below 45% logical reads. Broader language quality and performance remain.

### Why OLMoE changes the semantic problem

A dense Llama MLP was trained to use every neuron together. Engram's earlier
routers had to infer, after training, which small subset could reproduce that
dense computation. OLMoE was trained with sparsity already present: every
layer has 64 complete SwiGLU experts and a router that chooses eight for each
token. Engram can therefore preserve the teacher's learned selection rather
than guess a replacement.

The current experiment leaves the router unchanged, rounds every expert
matrix to signed Q7 in groups of 64, and uses BF16 group scales. On a separate
8-sequence/256-position confirmation it preserves the teacher closely while
projecting 22.7865% of the all-expert Q4 traffic baseline. Engram now compiles
those weights into a 5.84 GB immutable artifact and directly executes the
selected packed experts on CPU with route/output parity. The complete inference
boundary can now run: it includes embeddings, normalization, attention/cache
state, vocabulary projection, and persistent greedy generation. Python loads
the packaged tokenizer and converts text to token IDs; native C++ owns the
model computation from those IDs through logits and argmax. The package
authenticates its manifest and exact file inventory before the CPU-only runtime
opens any model state. It is installable and substantially faster than the
initial scalar boundary, but it is not yet an interactive-speed chat runtime.

## 1. What a language model does

A language model receives a sequence of **tokens** and predicts a probability for the next token.
A token is an integer representing a piece of text: perhaps a word, part of a word, punctuation,
or whitespace. A tokenizer might encode `The cat sleeps` as a short list of token IDs.

The model repeatedly performs this operation:

```text
text -> token IDs -> internal state -> score for every possible next token
     -> choose one token -> append it -> repeat
```

The scores for all possible next tokens are called **logits**. Converting logits with softmax
produces probabilities. Greedy decoding chooses the largest logit; sampling introduces controlled
randomness.

Neural networks do not operate directly on token integers. An embedding table maps each token ID
to a fixed-width vector. SmolLM2-135M, for example, uses vectors with 576 numeric components. The
network transforms these vectors, and a final vocabulary projection turns the resulting state
back into one score per token.

## 2. The relevant parts of Llama

A Llama-family transformer is a stack of similar layers. Each layer has two large operations:

1. **Self-attention** lets the current token read information from earlier token positions.
2. **The MLP**, or feed-forward network, transforms each token position independently.

Residual connections add each operation's result back to the current vector. Normalization keeps
the numeric scale stable. A simplified layer looks like this:

```text
state ──> normalize ──> attention over earlier tokens ──> add ──┐
  └─────────────────────────────────────────────────────────────┘
        ──> normalize ──> MLP ──> add ──> next-layer state
```

The full transformer executes every layer for every generated token. It also retains attention
keys and values for earlier tokens in a growing key/value cache. Model weights and this cache
create substantial memory traffic, which is often the limiting cost on a CPU.

### 2.1 The Llama SwiGLU MLP

The MLP is particularly useful for Engram because it has an exact record-like decomposition.
Given an input vector `h`, a Llama SwiGLU MLP computes:

```text
gate = W_gate h
up   = W_up h
activation = SiLU(gate) * up
output = W_down activation
```

`*` is element-wise multiplication. If the intermediate width is `I`, this can be rewritten as a
sum of `I` independent contributions:

```text
a_j(h) = SiLU(W_gate[j] h) * (W_up[j] h)
v_j    = W_down[:, j]
MLP(h) = sum over j of a_j(h) v_j
```

This gives neuron `j` a natural memory-record interpretation:

- `W_gate[j]` is the first key;
- `W_up[j]` is the second key;
- `W_down[:, j]` is the value;
- `a_j(h)` says how strongly the record applies to state `h`.

Engram does not assume that a neuron represents one human-readable fact. A record is simply an
exact algebraic contribution learned by the model. Meaning may be distributed across many such
records.

### A separate native-BitNet source

The new BitNet track is intentionally not passed through that SwiGLU
inspector. BitNet quantizes activations per token, uses ReLU-squared gating,
normalizes the entire intermediate vector, and then applies the down
projection. The normalization denominator depends on every channel, so
channels are addressable records but are not independent in the same way as
Llama SwiGLU records.

Because the BitNet basis was trained natively with ternary weights, Engram can
repack its coefficients losslessly instead of approximating dense weights.
Each record stores the channel's gate row, up row, down column, and
normalization gain. The practical DIP path now handles the shared denominator
with a frozen, per-layer proxy-energy policy: most layers calibrate unseen
energy from exact-versus-proxy candidate energy, while layer 9 uses a
corrected proxy plus an eight-record audit.

## 3. What Engram is trying to build

Engram's target runtime replaces repeated full-transformer execution with four cooperating parts:

This is the compiled **worker** architecture. The optional
[Oracle cognitive executive](cognitive_executive.md) sits above workers and handles request-level
goals, evidence, memory policy, strategy, and cost-aware action selection. It is not the shared
controller shown in the per-token loop below.

```text
input token embedding ───────────────┐
                                     v
recurrent state -> semantic memory -> shared controller -> next state
        │                 ^                 │
        └-> episodic memory ────────────────┘
                                              |
                                              v
                                     vocabulary search -> next token
```

### Semantic memory

Semantic memory contains fixed records derived from the source model's MLP weights. An index
should find a small candidate set, exact SwiGLU scoring should rerank those candidates, and only
the best values should be read and accumulated.

### Episodic memory

Semantic memory contains information fixed at conversion time. Episodic memory contains
information from the current prompt and generated sequence. Engram combines:

- an exact bounded window for recent context;
- a fixed-size recurrent summary;
- a capacity-bounded store for selected older states, with quantized retrieval.

Unlike a transformer key/value cache, these structures have configured size limits.

### Token-level shared recurrent controller

The controller owns the current fixed-width state. At each logical depth stage
it receives three vectors that will also exist in the final CPU runtime: the
current token embedding, that stage's semantic-memory read, and its
episodic/attention read. Teacher hidden states are targets; they are never
allowed to become hidden runtime inputs.

The original fixture uses a dense GRU-like transition. That is useful for
small package tests but is not a defensible trained-model design: at BitNet's
2,560 width its two FP64 kernels would occupy about 629 MB. The trainable path
therefore projects the supplied input and recurrent state into a shared
low-rank bottleneck, expands it into a residual gate and candidate, applies a
small stage embedding and low-rank stage adapter, then returns an
RMS-normalized residual update. The large factors are shared across all depth
cycles. Only the small adapters, embeddings, and one update scale vary by
stage.

The original distillation experiment proceeded in explicit stages. First, the packaged CPU BitNet
teacher records each layer state plus its exact attention and MLP outputs.
CUDA trains the recurrent controller against intermediate state, transition
delta, cosine-geometry, and terminal-state losses. Teacher forcing is reduced
until the student rolls through all 30 stages on its own state. The result is
exported as ordinary FP32 NumPy tensors and reloaded by a PyTorch-free CPU
runtime.

That learned transition missed its protected gate. The promoted schema-v3
controller instead preserves the known operator additions exactly and keeps
the factorized recurrence only as a zero-scaled optional correction. Compiled
semantic and bounded-attention outputs now run through that controller in
incremental generation, and the complete DIP-only C++ token runtime performs
the same residual/RMS sequence. This passes the current controller and
integration gates, but it does not establish the longer-term goal of
distilling away the original transformer operators.

### Vocabulary index

The final state must be compared with every possible output token. Engram uses an inverted-file
index to choose vocabulary candidates, then applies exact rescoring to those candidates. A dense
fallback remains available for validation.

## 4. How information is extracted from Llama

Conversion starts from either a local Hugging Face model directory or a Hub model ID. Hub models
are downloaded into the normal Hugging Face cache. Engram then performs the following stages.

### Stage 1: inspect and validate the checkpoint

Engram reads `config.json` and verifies that the architecture is supported, the activation is
SiLU/Swish, all dimensions are positive, and every layer contains gate, up, and down tensors with
the expected shapes. It inventories safetensor or PyTorch shards and hashes configuration and
weights. The source hash is recorded in traces and compiled packages so artifacts from different
models cannot be mixed accidentally.

For each source layer, the converter extracts:

```text
model.layers.N.mlp.gate_proj.weight
model.layers.N.mlp.up_proj.weight
model.layers.N.mlp.down_proj.weight
```

It also extracts token embeddings and the language-model output projection, and copies tokenizer
files when present.

### Stage 2: capture teacher behavior

Weights describe what the source model can compute, but not which computations matter on real
text. Engram therefore runs calibration and validation examples through the Hugging Face model.
Forward hooks capture, for every token and layer:

- the vector entering self-attention;
- the vector leaving self-attention;
- the vector entering the MLP;
- the vector leaving the MLP;
- the token ID, sample ID, and input category.

These arrays are written as independent NPY shards with shapes, dtypes, and SHA-256 checksums.
Calibration traces may influence fitting. Validation traces are held out and used only to measure
generalization.

### Stage 3: measure a full-information sparsity frontier

Before evaluating an index, Engram computes every MLP activation for selected trace states. It
ranks records by contribution magnitude:

```text
abs(a_j(h)) * length(v_j)
```

It then accumulates records in that order and measures how many are required to reconstruct 90%,
95%, or 99% of the full MLP output energy. This is an oracle because it already knows every exact
activation. It is a useful full-information reference, but not a mathematical upper bound: vector
contributions can cancel, so a different K-record subset can reconstruct the complete MLP output
or preserve downstream behavior better.

### Stage 4: build and test a practical router

The implemented production baseline normalizes each gate and up key, concatenates them, clusters
the resulting vectors, and stores cluster membership as CSR-style posting lists. At inference:

1. score the small centroid table;
2. open several promising postings;
3. rank records inside those postings with a cheap proxy;
4. compute exact SwiGLU activations for the surviving candidates;
5. read and sum only the selected values.

Recent experiments show that the baseline loses too many oracle records. Directly supervised
multi-label routing improves candidate recall, and a rank-16 factorization makes its scoring
weights much smaller. Disjoint hierarchies, coverage-trained groups, multiple representatives,
and learned overlapping postings have all been tested. None of the checked artifacts passes the
downstream quality gate, so none is a compiled default. The learned fits used 128 of 1,112
available calibration states per layer in the first comparison. Refitting on all 1,112 states
raises flat-router recall at 1,280 candidates from 86.7% to 88.9% and overlap-router recall from
85.8% to 86.8%, but neither reaches the 95% gate and downstream behavior remains far outside the
declared limits.

The subsequent screen caches the oracle memberships and sweeps ridge regularization without rerunning
the transformer. λ=8,000 is best, but reaching 95% recall requires at least 1,408 candidates.
Even 1,472 candidates—95.8% of all records—fails the causal gate. This shows that simply opening
more postings or assigning per-layer budgets cannot rescue the current rank-16 representation
while preserving the intended memory-traffic advantage.

The decisive test substitutes sparse MLP outputs inside the original trained transformer. This
keeps attention, residual connections, normalization, and the vocabulary head exact, allowing us
to measure how local MLP error changes actual next-token probabilities. On SmolLM2-135M, the
full-information magnitude top-256 reference already damages logits badly. The first tested
magnitude-reference pass is top-768, which keeps half the records. At that active count, rank-16
flat and overlapping-posting routers still miss too many records even after examining 1,280 of
1,536 candidates. Near-dense candidate expansion fixes recall but not causal quality or useful
traffic reduction. The gate therefore stops conversion
before those experimental parameters are serialized; it does not claim that all possible sparse
representations must fail.

The first passing follow-up in the earlier dense-SmolLM study avoided
predicting a hard oracle-membership label. Published Dynamic Input Pruning
(DIP) motivates pruning the current MLP input and forming partial scores.
Engram extended that idea with candidate-only exact completion and
contribution-norm reranking:

1. keep the `q` coordinates of the hidden vector with the largest absolute values;
2. multiply only those coordinates by every gate/up record to get a partial SwiGLU score;
3. keep `C` promising records;
4. evaluate the omitted input coordinates only for those candidates, making their activations
   exact;
5. rerank the exact candidates and read the `K` selected down-projection values.

This is predictor-free: it uses the source MLP weights rather than a separate learned router. In
the historical SmolLM2 study, retaining 75% of input coordinates, completing 896/1,536 candidates,
and selecting K=768 passes both the development grid and a sequence-disjoint confirmation run.
Its projected weight reads were 76.4% of a dense MLP. That was a real quality
progression result, but cache-line accounting reached 83.33% and its later
native kernel was slower than dense. That dense-source arm therefore remains
historical rather than the current Milestone 2 candidate.

### Stage 5: fit missing behavior

Sparse records omit a diffuse residual. Engram includes a low-rank linear background operator
that can be fitted to this residual. On the current small trained-model experiment it overfits and
worsens held-out mean error, so it should not be treated as solved.

The intended conversion pipeline must also distill:

- the recurrent controller from source residual trajectories;
- bounded episodic behavior from source attention traces;
- confidence and escalation policies from measured divergence;
- optional correction capsules for repeatable failure regions.

The correction-capsule experiment now fits state-selected local affine low-rank predictors to the
exact dense-minus-routed MLP residual. Global 1/4/8-capsule layouts and targeted layouts trained on
the hardest 10–40% of states were checked. The best global result worsens local relative L2 from
0.207 to 0.259; a tight targeted result applies on 7.1% of states but still worsens it to 0.233.
These experimental capsules therefore remain outside compiled packages.

The next implemented training stage copies the source model into a frozen student and replaces
each student MLP with a sparse training wrapper. Hard candidate selection uses the learned router;
an auxiliary membership loss trains its rank-16 factors, while a rank-8 update to the sparse down
projection receives local MLP, hidden-state, and output-logit distillation losses. Attention,
normalization, embeddings, and original MLP weights never update. The artifact stores only these
router and adapter tensors. The first 32-step pilot improves its training loss but fails held-out
recall and causal quality, so it is not used by inference. A gradient audit also shows that hard
candidate indices prevent the causal losses from training the router. The replacement trainer now
uses hard candidate choices in its forward result and a sigmoid straight-through mask in backward.
It also lowers the default budget to `q=62.5%`, `C=K=512` and penalizes the expected number of
64-byte line groups touched by candidates. Exact candidate-only completion and selected down-row
gathers are used for the student result; a detached dense oracle is retained only to create
supervision and measure recall. This fixes the gradient and execution design, but not the
scientific gate. The full 32/16 evaluation fails, and exact top-512 membership itself touches
almost every candidate line group. Mergeable gate/up/down LoRA, broader training
text, balanced record packing, and explicit whole-line candidates were screened afterward; none
improved recall, causal quality, and traffic together. The current low-budget sparse-teacher
artifact therefore remains experimental and is not consumed by inference. Corrected LoRA scaling,
a full 128-sequence rank-32 residual run, and a sequence-disjoint layer-adaptive magnitude schedule
also fail. The next major semantic design must learn a structured expert/block basis jointly with
the student MLP rather than select the teacher's diffuse frozen neurons post hoc.

A bounded shadow implementation tested whether a simple lossless permutation could provide that
structure without training. It packs the 1,536 records into equal contiguous blocks and selects
enough whole blocks to execute exactly 512 records. Smaller blocks improve the impossible
full-information reference, but even 96 blocks of 16 records leave 0.438 mean local relative-L2
error. The static shortcut is therefore closed. The remaining hypothesis requires the gates and
MLP weights to co-adapt while the forward pass already obeys the sparse hardware layout.

The implemented native-gate alternative uses the largest input coordinates to approximate every
gate value, selects 512 channels directly from those values, and reads only the selected up rows
and down columns. It does not complete candidate gate values or rerank with dense activations. At
q=62.5%, its ideal weight traffic is 43.06% of dense. The original dense basis reveals why training
is necessary: exact contribution selection has 0.190 local error, while gate-only selection has
0.375 and input pruning raises it only to 0.386. A cached-boundary warm-up improves a representative
layer by only 2.55%, so the next valid test must expose all layers to their own causal state drift
during progressive sparse-teacher training.

CUDA may accelerate that training experiment, but it is not part of the Engram representation.
The same hard sparse forward, checkpoints, reports, and evaluation criteria must work on CPU, and
the deployment target remains a packed CPU inference kernel with measured memory traffic.

The end-to-end trainer now follows this rule. It begins with dense MLP execution, progressively
reduces retained gate inputs and active channels, and co-trains full MLP weights while the rest of
the transformer remains frozen. Its checkpoints contain ordinary CPU tensors plus optimizer state
and can resume on another device. Validation resets every layer to q=62.5%/K=512 and disables the
soft selection and dense-shadow training paths.

Engram now also tests a router-free budget-native representation. It keeps all
1,536 SwiGLU channels but replaces each dense projection coefficient with
`-1`, `0`, or `+1` times one FP16 scale shared by a 128-weight group. Five
base-3 coefficients fit in one byte. Training retains float master weights,
but the student forward rounds them to the same hard ternary codes that will
be serialized; a straight-through gradient lets the masters adapt without
changing deployment semantics. Layers can transition deepest-first so the
whole model does not experience the low-bit shock at once.

The teacher and student receive the same token sequences. The losses compare
each MLP output, every transformer hidden boundary, the final hidden state,
the teacher probability distribution, the teacher's top token, and the actual
next token. Attention, normalization, and the already-resident
embedding/output head may co-adapt. Validation does not use float masters: it
writes the 17,173,504-byte MLP file, reloads and decodes it independently, and
then scores all layers at the hard representation. This file is 43.1353% of
dense ideal Q4 traffic.

The mechanism works but the tested model does not qualify. After 1,014,225
training positions it reaches KL 2.284, top-1 agreement 0.320, NLL delta
+2.277, and final-hidden relative L2 0.604. Its frozen rule required at least
half of every remaining quality gap to close before 3M; top-1 and hidden state
miss. The checked result therefore stops this conversion configuration rather
than treating a downward loss curve as success.

A materially different follow-up begins with Microsoft's native
`bitnet-b1.58-2B-4T`. The official two-bit MLP payload alone is 50.0521% of
the same dense-Q4 denominator, so source switching is not counted as a pass.
Engram instead repacks five trits per byte. Each channel has a 1,538-byte
logical record addressed across separate gate, up, normalization-gain, and
down phase streams. The complete 318,924,544-byte artifact is 40.0527%,
reconstructs every coefficient and BF16 scale/gain exactly, and gives
bit-identical dense-oracle outputs before changing execution order.

That result advances storage and quality together for the separate
low-bit-native source family. A direct CPU kernel memory-maps and executes the
packed phase streams without dense weights. The frozen 8-sequence,
256-position confirmation reaches KL 0.00371, 96.09% teacher top-1 agreement,
NLL delta +0.00224, and final-hidden relative L2 0.04678, so this source track
passes causal quality and cold-byte checks as a full-record systems substrate.
Because every MLP record executes, it does not pass routed semantic-memory
Milestone 2.

The next BitNet experiment tested the missing semantic premise directly. After
the full gate/up coefficient path, each normalized intermediate coefficient
multiplies one transposed-down record, giving an exact additive decomposition.
An oracle ranks those contributions and reads only its selected down records.
A layer-adaptive 15–35% schedule averages 24.84% and passes the frozen causal
gate. This means the selected semantic values are sufficient. It does not yet
mean they are cheaply addressable: that oracle still reads the complete
gate/up path to discover its membership.

The practical native-BitNet DIP path now supplies the missing addressing
mechanism. At each layer it:

1. receives the actual BF16 MLP input and applies the teacher's Q8 activation
   quantization;
2. selects the 1,920 largest-magnitude coordinates out of 2,560;
3. streams those coordinate rows from a packed ternary gate/up index to score
   all 6,912 records approximately;
4. keeps the layer's frozen candidate count `C` and computes each candidate's
   complete gate/up coefficient exactly;
5. estimates the shared RMS without completing the non-candidates;
6. quantizes the normalized candidate coefficients exactly, ranks their
   down-weighted utility, and chooses the token's nonzero count clipped to
   `[346,Kmax]`;
7. reads and accumulates only those selected packed down rows.

Most layers estimate the unseen RMS energy by multiplying proxy-tail energy
by the exact/proxy candidate-energy ratio. Layer 9 is the exception: it uses
corrected proxy energy and reserves eight slots inside its candidate union for
large raw-square proxy records as an audit. This avoids a fitted predictor and
does not add candidate traffic beyond layer 9's frozen `C`.

The source-bound v2 index, C/K schedule, RMS policy, and CPU shared library are
independently reloaded before evaluation. On eight development sequences and
256 positions, all 30 native sparse MLPs pass the causal, recall, activity, and
modeled-traffic limits with no dense fallback. Six rows per layer have
bit-exact Python/native route and BF16 output parity. That run froze the policy.

On the independent final 8-sequence/256-position holdout, the identical
CPU-only route produced KL 0.00404129, top-1 agreement 0.98828125, NLL delta
+0.00482893, final-hidden relative L2 0.0477494, 21.3800% mean active records,
41.1371% modeled traffic, 99.9406% global candidate recall, and 99.3943%
worst-layer mean recall. All thresholds pass with no dense fallback. The
semantic-memory gate is classified as **passed by postmortem adjudication**:
the original wrapper errored after evaluation because it compared a
full-record canonical object hash with a 33-token bare-list evaluator hash. A
separate no-model adjudicator corrected the contract and checked the preserved
evidence.

That distinction matters. The raw report was only prospectively sealed about
13 minutes after the original error and was not contemporaneously bound by the
runner result. The artifacts are host-bound, the confirmation is only 8x32,
and the 41.1371% number is modeled cache-line traffic rather than measured
DRAM. The final sparse pass is 1.1449x dense, and latency was not a frozen
gate. This is therefore neither a speedup nor a blanket declaration that all
Milestone 2 packaging and replication work is complete.

The qualifying semantic memory is now part of an executable derived package.
The frozen source package remains unchanged. A promotion step authenticates
the frozen policy, passing adjudication, base record artifact, and v2
coordinate index, copies the source package, and records the DIP operator as
the only allowed semantic backend. The v2 index itself contains the runtime
policy in authenticated per-layer headers; inference does not consult an
editable side configuration.

The pure C++ token runtime is correspondingly direct and fail-closed. For each
layer, it updates bounded attention, extracts the normalized MLP input,
executes DIP, and accepts the sparse semantic output into the residual state.
It never creates the full-record semantic kernel and has no dense fallback.
After all 30 layers it normalizes the state and performs the exact
tied-vocabulary argmax. A non-holdout eight-prompt test generated 32/32
greedy reference tokens and all eight four-token sequences. Global and
maximum-prompt mean activity are 21.5602% and 22.5892%. Complete modeled cold
traffic is 30,153,074,432 bytes, including 194,304 global-metadata bytes;
global and maximum-prompt mean traffic fractions are 41.1612% and 41.2984%.

Before mapping the model, the standalone binary authenticates its exact
manifest and symlink-free inventory against deployment trust roots. It derives
dimensions, head layout, context/vocabulary bounds, paths, attention settings,
RoPE/RMS values, and EOS IDs (including `128009`) from the authenticated
package. The executable links its kernel objects directly rather than loading
an Engram shared library.

The chat frontend reaches the same implementation through a small versioned C
ABI. Python supplies only authenticated tokenization, packaged chat-template
rendering, and conversation bookkeeping. The native handle authenticates and
maps the package once, executes every token step, and exposes structural
metrics. It does not construct a Transformers model, Torch decoder shell, or
dense MLP fallback. The current frontend resets the handle and re-prefills the
complete rendered conversation on every turn.

The reset replay proves identical greedy tokens, zeroed position/metric
counters, and structural metric parity; it does not compare hidden states.
Likewise, matching greedy tokens is not hidden-state or logit parity. The
longest processed context is 14 positions, so the W=16 test does not exercise
eviction or older retrieval. A separate boundary protocol at
16/17/18/24/32 positions now proves local eviction, sink preservation,
older-key scoring, bounded older-value selection, accepted heavy-hitter
updates, fixed attention-state bytes, and reset replay in the same packaged
runtime. This proves that semantic memory and bounded-attention mechanics
survive package and token-loop integration; it is still not a dense-teacher
long-context quality comparison, speed result, or broad language benchmark.

The source-family-specific package and generation boundary are now complete.
Compilation writes a 1,108,116,808-byte checksummed package containing 332
non-MLP tensors, tokenizer/configuration assets, and the packed MLP artifact;
all 210 original MLP tensors are omitted. At load time the transformer is
created on empty storage, only the packaged non-MLP tensors are materialized,
and each MLP is replaced by the memory-mapped native kernel. The loader fails
if a checksum, model identity, tensor boundary, or unmaterialized parameter is
wrong. Package-backed and source-backed kernel models produce bit-exact hidden
states and logits on the parity prompt, and greedy generation completes
without the source checkpoint.

For the native-BitNet source track, Milestone 3 substitutes attention while
preserving the trained Q/K/V
projections, rotary positions, grouped-query mapping, residual path, and
normalization. Local-only and recurrent-only candidates fail. An exact hybrid
first proved that four older values were sufficient but required a full key
scan. Random LSH and exact geometric page indexes did not retrieve or prune
well enough. The passing streaming operator instead retains two initial
attention sinks and six online heavy hitters beside the exact 16-token local
window. It exact-reranks those eight old keys to four values and never reads an
evicted key. The frozen 256-position result passes every causal threshold.
The OLMoE source track does not inherit that result: its W16/C8/K4/S2 policy
fails the authenticated 8×128 semantic test, while exact W128 passes at
nondeployable 100% reads. OLMoE therefore still needs a bounded attention
policy. Native integration and a genuinely long-context hardware benchmark
remain.

The generic dense-Llama compiler still writes initialized or heuristic
fallbacks and records that fact in its conversion report. The native-BitNet
DIP route is a separate source-family candidate: its serialized index and
native kernel pass the semantic gate by postmortem adjudication. Its
1.1449x-dense final timing leaves substantial performance work despite that
semantic result.

A later 3M-position compact-Q4 run establishes the opposite frontier: its
serialized MLP payload fits the 45% traffic budget, but its causal quality is
far from the teacher. An exact one-million-record output-memory pilot also
fails to improve layer-local error enough to justify scale-up. The later
budget-native grouped-ternary run also fits traffic but fails its causal
scale-up rule. Therefore the diagram in this document remains the intended
architecture, not a claim that the current converter has learned a
quality-preserving replacement.

## 5. The converted format

An Engram model is a directory rather than one opaque file. A representative package is:

```text
model.engram/
  manifest.json
  tokenizer/
    tokenizer.json
    metadata.json
  embeddings/
    token_embeddings.npy
  controller/
    metadata.json
    input_kernel.npy
    recurrent_kernel.npy
    bias.npy
    stage_embeddings.npy
    adapter_down.npy
    adapter_up.npy
  semantic/
    manifest.json
    layer-0000/
      metadata.npy
      quantized/
        gate_codes.npy
        gate_offsets.npy
        gate_scales.npy
        up_codes.npy
        up_offsets.npy
        up_scales.npy
        value_codes.npy
        value_codebooks.npy
        ivf/
          centroids.npy
          posting_offsets.npy
          posting_indices.npy
          metadata.json
    layer-0001/
      ...
  episodic/config.json
  vocabulary/
    embeddings.npy
    index.npy
    ivf/
      centroids.npy
      posting_offsets.npy
      token_ids.npy
      metadata.json
  transitions/config.json
  corrections/capsules.json
  metrics/
    stage_manifest.json
    conversion_report.json
```

The root manifest records:

- format and Engram versions;
- the source architecture and source-model hash;
- dimensions and runtime budgets;
- whether the package came from a synthetic fixture;
- every file's size and SHA-256 checksum;
- the assertion that source transformer layers are not required at runtime.

Compiled semantic keys use per-dimension 8-bit affine codes. Values use additive codebooks. IVF
centroids are float32 and posting IDs are uint32. NPY files are little-endian, C-order arrays that
can be validated and memory-mapped by both Python and C++. Research-only semantic builds may keep
exact float arrays for comparison; full compile packages deliberately omit those reference arrays.

This format is designed to be inspectable. It is not a pickle, executable program, or serialized
PyTorch object. A loader can validate paths, sizes, shapes, byte order, and hashes before exposing
arrays to inference code.

## 6. How inference works

Loading a package does not load Transformers or source Llama layers. The runtime verifies the
manifest, memory-maps arrays, constructs bounded memories, and initializes its recurrent state to
zero.

For each input or generated token, the current Python reference runtime performs:

1. **Embedding lookup.** Read the vector for the input token.
2. **Transition-cache lookup.** Check whether a sufficiently similar prior state/token pair has a
   validated cached transition. If so, reuse its next state and output candidate.
3. **Semantic lookup.** Query every compiled semantic layer's IVF index, exact-rerank candidates,
   decode the selected quantized values, and average the layer outputs.
4. **Episodic update.** Read recent exact context, recurrent context, and retrieved older context;
   update the bounded stores.
5. **Controller update.** Concatenate token embedding, semantic output, and episodic output. Run
   the shared controller for the configured number of cycles to produce the next state.
6. **Vocabulary search.** Probe vocabulary clusters, rescore candidate token vectors, and select
   the next token. Validation can request a full dense vocabulary scan.
7. **Cache insertion.** Store the observed state transition with a confidence derived from the
   controller residual.
8. Repeat from step 1 using the selected token.

The native C++20 runtime independently implements the same package validation and major inference
paths. It uses memory mapping, preallocated scratch buffers, and scalar kernels, with optional
AVX2 dispatch on supported CPUs.

The qualified native-BitNet track uses a different, currently more complete
path. Its packaged transformer retains the source embedding, normalization,
attention, and output tensors; executes losslessly repacked MLP and Q/K/V/O
weights through native CPU kernels; replaces the unbounded dense KV cache with
fixed local/sink/heavy-hitter attention state; and computes exact final-row
vocabulary logits. For chat, structured conversation messages are rendered
with the tokenizer's packaged chat template. The complete rendered history is
re-prefilled from position zero on every turn, generation advances absolute
RoPE/cache positions, and the decoded assistant message is saved for the next
render. This is why a follow-up such as `awesome!` can condition on an earlier
poem even though native cache state is not reused across turns.

### Current baseline versus intended inference

The current loop is real and executable, but several inputs to it are not learned well enough:

- semantic layer outputs are averaged rather than scheduled by a trained controller;
- the controller is initialized, not distilled from teacher state transitions;
- episodic mixing is heuristic, not trained to match source attention;
- correction capsules are empty;
- dense-Llama sparse semantic routing recall and downstream quality are below
  the required level;
- no dense-Llama learned rank-16 or overlapping-posting router is serialized
  because that source track fails its semantic gate;
- native-BitNet DIP has passed its final semantic decision by postmortem
  adjudication and is promoted into a derived DIP-only native token package;
  the current Python chat frontend calls that backend through its versioned
  native handle.

Consequently, successful package generation or Python/C++ parity proves that
the systems pipeline works. It does not prove that this original dense-Llama
conversion preserves the teacher model's language ability. The separate
native-BitNet package does pass its frozen substitution gate and produces
coherent short continuations, but its evidence is still limited to the
documented frozen corpus and small behavioral suite.

## 7. Example: generating one token

Suppose the prompt ends with `The capital of France is`. A conventional Llama runs this sequence
through every transformer layer and produces a high logit for a token such as ` Paris`.

The intended Engram path is:

1. tokenize the prompt and update bounded episodic state for each token;
2. use the recurrent state to retrieve semantic records associated with the present context;
3. retrieve relevant recent and older prompt information;
4. let the trained controller combine the token, semantic, and episodic signals;
5. search vocabulary embeddings and assign a high score to ` Paris`;
6. append that token and update the recurrent and episodic state.

No individual semantic record needs to literally contain the string `Paris`. The answer can arise
from a distributed combination of records, context, controller state, and vocabulary geometry—just
as behavior in the source network is distributed across tensors.

## 8. What would count as success

Engram should be considered successful only if a compiled trained model demonstrates all of the
following on held-out workloads:

- useful next-token quality relative to its source teacher;
- bounded memory growth with long context;
- materially fewer weight and state bytes read per token;
- competitive CPU latency and throughput;
- deterministic, independently validated package loading;
- reproducible measurements across more than one model and dataset.

Current results establish extraction correctness, format integrity, executable Python/native
runtimes, and a trained-model causal MLP intervention frontier. That frontier is negative for the
tested practical routers. For the magnitude reference on SmolLM2-135M, K=640 failed and K=768
passed, so the threshold lies above 640 and at or below 768 on this corpus; intermediate counts
were not tested. The quality and performance claims above remain unestablished.
