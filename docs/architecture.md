# Architecture

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

The first realizable selector algorithm to pass is inspired by predictor-free Dynamic Input
Pruning. The published method supplies dynamic top-magnitude input pruning and partial scoring;
candidate-only exact completion and contribution-norm reranking are Engram extensions. For each
state, it selects the `q` largest absolute hidden coordinates,
reads only those gate/up columns for all `I` records, and computes a partial SwiGLU contribution
score. It keeps `C` candidates, reads their omitted gate/up coordinates to recover exact
activations, exactly reranks to `K`, and reads only those `K` down-projection columns. Its projected
weight reads per layer are:

```text
2 * I * q + 2 * C * (H - q) + K * H
```

At `H=576`, `I=1,536`, `q=432`, `C=896`, and `K=768`, the arm passes both its development grid and
a sequence-disjoint confirmation corpus, with 98.97% confirmation oracle-set recall. It projects
to 76.39% of dense MLP weight traffic. The mathematical selector is implemented in the research
code, but the compiled format does not yet contain DIP-packed weights and the native runtime has
no gather/completion kernel. The current quality harness still executes the dense source MLP, so
it is not a speed benchmark.

## Episodic-memory prototype

The episodic baseline keeps an exact causal local window, a normalized ELU+1 linear-attention
state whose size depends on head dimensions rather than sequence length, and a configurable
fixed-capacity older-token ring. Older keys and values use per-vector int8 codes and scales;
retrieval performs quantized cosine candidate search followed by decoded exact dot-product
reranking. A heuristic hybrid combines the three reads and exposes state/read byte metrics.

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

The semantic and vocabulary proxies use IVF in both runtimes, but still scan every coarse
centroid and their tiny-fixture centroid traffic is unfavorable. Learned semantic routing remains
blocked: full-corpus, corpus-scaled regularization, and candidate-budget sweeps did not close its
gap, and the rank-16 arm fails causal quality while reading 95.8% of record keys. Predictor-free
DIP changes that quality decision, but requires a cache-aware packed layout, native sparse kernel,
measured DRAM traffic, and replication beyond one small model before compilation. Its 1.31x
projected reduction is also far from the long-term 10x systems target. Global and failure-region
low-rank correction capsules have been fitted, but every tested layout worsens held-out local MLP
error and is rejected before causal integration.
The first sparse-teacher trainer keeps two model copies: an immutable dense teacher and a student
whose attention, normalization, embeddings, and original MLP tensors are frozen. Trainable
rank-16 router factors receive oracle-membership BCE supervision; rank-8 sparse down adapters
receive local MLP, hidden-state, and logit-distillation gradients. Only these tensors are written
to a safetensors experiment artifact. The first 32-step pilot fails the progression gate, and a
subsequent audit shows that its hard route prevents those causal losses from reaching the router.
Older-context retrieval still uses a linear candidate scan. Trained attention and controller
distillation remain open; the DIP quality pass makes systems implementation and replication the
next semantic work rather than another blind learned-router fit.
