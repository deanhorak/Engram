# How Engram works

This document explains Engram for readers with a general computer science background. No prior
knowledge of neural networks or large language models is assumed. It describes the architecture
we are trying to build, how information is extracted from a trained Llama-compatible model, the
compiled representation, and how generation works without the source transformer.

Engram is a research prototype. Where the implemented baseline differs from the intended trained
system, this document says so explicitly.

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

The controller owns the current fixed-width state. It receives the token embedding, semantic
read, and episodic read, then applies a GRU-like transition. Its large kernels are shared across
cycles. Small stage embeddings and low-rank adapters can distinguish different logical stages
without storing a complete transformer block for every stage.

The target controller must be trained or distilled to reproduce useful source-model behavior.
The current compiler only initializes this controller deterministically. That is sufficient to
test packaging and execution, but not sufficient to preserve language quality.

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

The first passing follow-up avoids predicting a hard oracle-membership label. Published Dynamic
Input Pruning (DIP) motivates pruning the current MLP input and forming partial scores. Engram
extends that idea with candidate-only exact completion and contribution-norm reranking:

1. keep the `q` coordinates of the hidden vector with the largest absolute values;
2. multiply only those coordinates by every gate/up record to get a partial SwiGLU score;
3. keep `C` promising records;
4. evaluate the omitted input coordinates only for those candidates, making their activations
   exact;
5. rerank the exact candidates and read the `K` selected down-projection values.

This is predictor-free: it uses the source MLP weights rather than a separate learned router. On
the current SmolLM2 study, retaining 75% of input coordinates, completing 896/1,536 candidates,
and selecting K=768 passes both the development grid and a sequence-disjoint confirmation run.
Its projected weight reads are 76.4% of a dense MLP. That is a real quality progression result,
but only a modest traffic reduction; the cache-aware packed layout and native kernel needed to
realize it have not yet been built.

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
candidate indices prevent the causal losses from training the router, making a differentiable
soft-to-hard route a requirement before repeating that experiment.

Those training stages are open work. The present compiler writes initialized or heuristic
fallbacks and records that fact in its conversion report. DIP now supplies a passing semantic
substitution arm, so the immediate work is to serialize its packed layout and validate a native
sparse kernel before attributing later failures to attention or controller distillation.

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

### Current baseline versus intended inference

The current loop is real and executable, but several inputs to it are not learned well enough:

- semantic layer outputs are averaged rather than scheduled by a trained controller;
- the controller is initialized, not distilled from teacher state transitions;
- episodic mixing is heuristic, not trained to match source attention;
- correction capsules are empty;
- sparse semantic routing recall and downstream quality are below the required level;
- no learned rank-16 or overlapping-posting router is serialized because the semantic gate fails.

Consequently, successful package generation or Python/C++ parity proves that the systems pipeline
works. It does not prove that Engram preserves the teacher model's language ability.

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
