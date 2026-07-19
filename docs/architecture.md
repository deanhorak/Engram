# Architecture

Engram's target runtime combines a shared recurrent controller, fixed sparse semantic
memory derived from SwiGLU records, hybrid local/recurrent/retrieval episodic memory, an
indexed vocabulary projection, transition caching, and uncertainty-triggered corrections.
The target package must generate without loading source transformer layers.

## Implemented foundation

For a Llama SwiGLU MLP, Engram represents neuron `j` as two keys and one value:

```text
a_j(h) = SiLU(W_gate[j] h) * (W_up[j] h)
v_j    = W_down[:, j]
FFN(h) = sum_j a_j(h) v_j
```

The Python implementation verifies this identity and the native scalar kernel provides an
independent implementation. The magnitude oracle uses all activations to establish a
sparsity upper-bound baseline before practical routing is attempted.

## Semantic-memory prototype

Research-only semantic packages may retain exact reference keys/values; compiled runtime
packages store 8-bit per-dimension key codes plus additive residual value-codebooks. Every array has
dtype, shape, byte-order, alignment intent, byte count, and checksum metadata. The practical
router clusters concatenated normalized gate/up geometry into a deterministic IVF index,
scores the small centroid table, scans only selected postings, and then evaluates the exact
two-key SwiGLU expression only for candidates. A deterministic brute-force router remains for
tests. A fitted low-rank linear operator models the residual left by the sparse read.

## Episodic-memory prototype

The episodic baseline keeps an exact causal local window, a normalized ELU+1 linear-attention
state whose size depends on head dimensions rather than sequence length, and a configurable
fixed-capacity older-token ring. Older keys and values use per-vector int8 codes and scales;
retrieval performs quantized cosine candidate search followed by decoded exact dot-product
reranking. A heuristic hybrid combines the three reads and exposes state/read byte metrics.

## Controller and output path

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

The semantic and vocabulary proxies now use IVF in both runtimes, but still scan every coarse
centroid and their tiny-fixture centroid traffic is unfavorable. Older-context retrieval still
uses a linear candidate scan. Trained attention and controller distillation remain open.
