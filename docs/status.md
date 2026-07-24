# Project status

Snapshot date: **2026-07-24**

Engram is an operational research prototype, not a general quality-preserving
dense-Llama compiler. The repository can inspect and trace a Llama-compatible
teacher, decompose SwiGLU MLPs, run routing and compression experiments, and
execute fixture packages in Python and C++. The native-BitNet track now also
compiles and validates a source-independent trained-model package and performs
real greedy generation with its direct C++ MLP kernel.

The separate low-bit-native track has now passed the **Milestone 2 semantic
and serialized cold-traffic gate** with a direct CPU phase-stream kernel. The
original dense-Llama conversion track remains blocked: it still has no
representation that preserves its teacher closely enough below 45% of dense
ideal Q4. This distinction matters—the passing result starts from a model
trained natively with ternary MLP weights and is not a lossless conversion of
an arbitrary dense Llama checkpoint.

The new qualifying evidence is the
[2026-07-24 direct-kernel confirmation](../reports/semantic_gate_native_bitnet_2026-07-24/kernel_confirmation.json);
the prior cross-track snapshot remains the
[2026-07-23 semantic-gate summary](../reports/semantic_gate_status_2026-07-23/summary.json).
Large corpora, checkpoints, and scratch experiments remain under ignored
`work/` paths and are not source-control artifacts.

## The gate we are trying to pass

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
| Candidate recall | at least 0.95 when approximate candidate routing is used |

Configuration selection must use development-only data. A final confirmation
must use the identical frozen artifact on a sequence-disjoint corpus that was
not used for fitting or selection.

## Strongest measured frontiers

No row in the original dense-Llama track passes both sides of the gate. The
native BitNet row now passes direct packed execution, the evidence floor, all
semantic thresholds, and exact scheduled cold-byte accounting.

| Representation | Quality result | Systems result | Decision |
|---|---|---|---|
| Native BitNet phase-stream base-3 records | Frozen 8-sequence/256-position result: KL 0.00371, top-1 0.96094, NLL +0.00224, hidden L2 0.04678 | Direct memory-mapped CPU kernel; 318,924,544 scheduled cold bytes, 40.0527% of dense Q4; zero dense-weight materialization | **Low-bit-native Milestone 2 gate pass**; not a dense-Llama conversion pass |
| Exact float magnitude oracle, K=768 | KL 0.0336, top-1 0.9124, NLL +0.0153, hidden L2 0.0953 | More than 4x the dense-Q4 payload before a practical selector | Semantic capacity exists, but this is not deployable |
| Predictor-free DIP, q=432/C=896/K=768 | Untouched confirmation: recall 0.9897, KL 0.0286, top-1 0.9101, NLL +0.0326, hidden L2 0.0905 | 76.39% scalar traffic, 83.33% cache-line traffic; native kernel is 0.863x dense throughput | Quality pass, systems fail |
| Mild layer-adaptive compact Q4 student | At 3,000,093 training positions: KL 0.8866, top-1 0.5659, NLL +0.8838, hidden L2 0.4245 | Serialized/reloaded artifact is 44.9334% of dense ideal Q4 | Traffic pass, quality fail; stopped at the frozen 3M rule |
| Budget-native full-width grouped ternary | At 1,014,225 training positions: KL 2.2844, top-1 0.3198, NLL +2.2770, hidden L2 0.6036 | Serialized/reloaded artifact is 43.1353% of dense ideal Q4 | Traffic pass, quality fail; top-1/hidden miss frozen 50%-gap-closure rule |
| Exact nonparametric output memory | Layer-14 LLE-32 error 0.3275 with 233,005 local records; 0.3219 after adding 1,000,000 pretraining records | Exact search and FP16 values are not a deployable traffic result | Only 1.73% improvement; density-scaling branch closed |
| Mixed affine LC-VQ | Development-only layer-14 hard-QAT error 0.3364 after 8,192 steps | Complete modeled cold traffic is 44.3482% | Traffic pass, local-quality fail; no causal run |

The DIP result is the only tested practical mechanism that clears the causal
quality thresholds. It is deliberately excluded from the default compiler
because its measured traffic and native latency fail the systems objective.
The compact Q4 artifact demonstrates the opposite frontier: it fits the
physical byte budget and maps cleanly to contiguous CPU kernels, but it is not
close to the teacher.

## What the latest experiments changed

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
| 2. Semantic package, routing, quantization, Python substitution runtime | Native-BitNet phase artifact, direct CPU kernel, package compiler, validator, and generation runtime implemented | **Low-bit-native track passes** the frozen causal/cold-byte gate and exact package parity; dense-Llama track remains blocked |
| 3. Local/recurrent/retrieval attention and hybrid episodic memory | Trained-model local/recurrent/retrieval operators plus bounded sink/heavy-hitter streaming hybrid implemented | **Bounded trained-model confirmation passes** at W=16, C=8, K=4; native latency and long-context hardware traffic remain |
| 4. Shared recurrent controller, adapters, adaptive cycles, transformer-free Python runtime | Prototype implemented | Controller remains initialized rather than successfully distilled |
| 5. Vocabulary index, transition cache, corrections, compiler, validation, generation CLI | Generic infrastructure plus native-BitNet package compiler, validator, and generation CLI implemented | Native-BitNet package excludes all source MLP tensors and has exact source/package output parity |
| 6. C++ runtime, scalar/AVX2 paths, mmap, parity, generation, benchmarks | Fixture runtime plus direct memory-mapped BitNet MLP kernel implemented | Python transformer generation uses the native MLP kernel; a full C++ transformer and hardware-counter traffic remain |
| 7. Evaluation, ablations, tuning, documentation, final report | In progress | Many negative ablations exist; no successful reproducible final report |

The optional Oracle cognitive executive is a separate request-level subsystem.
Its revisioned SQLite/JSONL event stores, worker registry, dispatch adapters,
outcome observation, and calibration summaries are implemented, but they do
not resolve the model-worker semantic gate.

## What is intentionally not claimed

- There is no quality-preserving `.engram` conversion of SmolLM2.
- The native BitNet result is a separate source track, not evidence that a
  dense Llama model can be losslessly repacked.
- There is no hardware-counter demonstration of DRAM traffic; the 40.0527%
  result is an exact cache-line schedule for the serialized cold streams.
- The passing DIP quality arm is not a passing runtime.
- Random-fixture generation is pipeline evidence, not language quality.
- CUDA was used for bounded training and trace capture, but no deployment
  format or inference path requires a GPU.
- Failure on SmolLM2-135M is not a theorem about every model or every possible
  architecture.

## Current development decision

The low-bit-native hypothesis has passed source validation, exact
reconstruction, the unchanged MLP byte limit, direct CPU execution, frozen
causal confirmation, source-independent package compilation, and exact
generation parity. Its package contains a checksummed 780,054,616-byte
non-MLP tensor file plus the 318,924,544-byte packed MLP artifact and tokenizer
assets.

Milestone 3 development rejected local-only and recurrent-only replacements.
An exact W=16/K=4 hybrid passed semantics but scanned all older keys. Random
sign-LSH then missed too many top keys, and exact page bounds pruned too few
pages. The promoted streaming policy retains 16 local tokens, two sinks, and
six cumulative-attention heavy hitters, exact-reranking eight old keys to four
values. On frozen records 8–15 it passes every semantic threshold: KL 0.01409,
top-1 0.94141, NLL delta −0.00613, and hidden L2 0.08559 over 256 positions.
Its old-context state and reads are constant in context length. At the short
33-token test point it still models at 93.34% of dense KV traffic, and its
per-head implementation is Python. The next work is a native cache/rerank
kernel plus long-context logical and hardware-counter traffic validation.

CUDA remains an optional training accelerator only; the serialized format and
inference mechanism are CPU-native. Repeating IVF, candidate-count,
regularization, prototype-density, small residual, post-hoc bit allocation,
loss-reweighting, or short grouped-ternary continuation sweeps is no longer
supported by the accumulated evidence.
