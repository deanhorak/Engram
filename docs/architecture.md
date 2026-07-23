# Architecture

For the current implementation/evidence boundary, see
[Project status](status.md). No semantic representation described here has yet
passed causal quality and the 45% physical cold-traffic limit together.

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
to 76.39% of dense MLP scalar reads. A separate version-2 experimental package stores gate/up
weights coordinate-major and down weights record-major; Python and native candidate-only readers
have exact selection parity. Cache-line amplification raises the estimate to 83.33%. The checked
optimized float32 kernel cycles all 30 layers but is still 15.4% slower than dense, so it remains
outside the compiled runtime. This isolates the current blocker as systems efficiency rather than
selector quality.

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

Older-context retrieval still uses a linear candidate scan. Trained attention and controller
distillation remain open. The DIP quality pass still needs systems replication, while the separate
low-budget research path must clear semantic quality before any compilation attempt.
