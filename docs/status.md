# Project status

Snapshot date: **2026-07-23**

Engram is an operational research prototype, not yet a quality-preserving model
compiler. The repository can inspect and trace a Llama-compatible teacher,
decompose SwiGLU MLPs, run semantic-routing and compression experiments,
substitute experimental MLPs inside the teacher, build checksummed packages,
and execute fixture packages in Python and C++.

The current blocker is the **Milestone 2 combined semantic/systems gate**.
No measured representation has simultaneously preserved the teacher closely
enough and reduced complete cold MLP traffic to at most 45% of dense ideal Q4.

The authoritative machine-readable snapshot is
[the 2026-07-23 semantic-gate summary](../reports/semantic_gate_status_2026-07-23/summary.json).
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

No row in this table passes both sides of the gate.

| Representation | Quality result | Systems result | Decision |
|---|---|---|---|
| Exact float magnitude oracle, K=768 | KL 0.0336, top-1 0.9124, NLL +0.0153, hidden L2 0.0953 | More than 4x the dense-Q4 payload before a practical selector | Semantic capacity exists, but this is not deployable |
| Predictor-free DIP, q=432/C=896/K=768 | Untouched confirmation: recall 0.9897, KL 0.0286, top-1 0.9101, NLL +0.0326, hidden L2 0.0905 | 76.39% scalar traffic, 83.33% cache-line traffic; native kernel is 0.863x dense throughput | Quality pass, systems fail |
| Mild layer-adaptive compact Q4 student | At 3,000,093 training positions: KL 0.8866, top-1 0.5659, NLL +0.8838, hidden L2 0.4245 | Serialized/reloaded artifact is 44.9334% of dense ideal Q4 | Traffic pass, quality fail; stopped at the frozen 3M rule |
| Exact nonparametric output memory | Layer-14 LLE-32 error 0.3275 with 233,005 local records; 0.3219 after adding 1,000,000 pretraining records | Exact search and FP16 values are not a deployable traffic result | Only 1.73% improvement; density-scaling branch closed |

The DIP result is the only tested practical mechanism that clears the causal
quality thresholds. It is deliberately excluded from the default compiler
because its measured traffic and native latency fail the systems objective.
The compact Q4 artifact demonstrates the opposite frontier: it fits the
physical byte budget and maps cleanly to contiguous CPU kernels, but it is not
close to the teacher.

## What the latest experiments changed

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
| 2. Semantic package, routing, quantization, Python substitution runtime | Broadly implemented | **Blocked:** no representation jointly passes quality and ≤45% traffic |
| 3. Local/recurrent/retrieval attention and hybrid episodic memory | Prototype implemented | Synthetic/fixture evaluation only; no trained attention-substitution pass |
| 4. Shared recurrent controller, adapters, adaptive cycles, transformer-free Python runtime | Prototype implemented | Controller remains initialized rather than successfully distilled |
| 5. Vocabulary index, transition cache, corrections, compiler, validation, generation CLI | Infrastructure implemented | No quality-qualified trained package is eligible for full compilation |
| 6. C++ runtime, scalar/AVX2 paths, mmap, parity, generation, benchmarks | Fixture runtime implemented | DIP experimental kernel is parity-correct but slower than dense |
| 7. Evaluation, ablations, tuning, documentation, final report | In progress | Many negative ablations exist; no successful reproducible final report |

The optional Oracle cognitive executive is a separate request-level subsystem.
Its revisioned SQLite/JSONL event stores, worker registry, dispatch adapters,
outcome observation, and calibration summaries are implemented, but they do
not resolve the model-worker semantic gate.

## What is intentionally not claimed

- There is no quality-preserving `.engram` conversion of SmolLM2.
- There is no demonstrated 10x DRAM reduction.
- The passing DIP quality arm is not a passing runtime.
- Random-fixture generation is pipeline evidence, not language quality.
- CUDA was used for bounded training and trace capture, but no deployment
  format or inference path requires a GPU.
- Failure on SmolLM2-135M is not a theorem about every model or every possible
  architecture.

## Decision point after this pause

There is no active experiment while the project state is being consolidated.
The existing evidence leaves three honest choices for the next program:

1. fund a materially larger compact-model or sparse-upcycling training program
   with a representation designed for the 45% byte budget from the start;
2. relax the 45% cold-traffic requirement toward the measured DIP quality
   frontier and optimize that contiguous/native systems path; or
3. introduce a genuinely new conditional representation with a cheap
   all-record decision mechanism and a predeclared byte model.

Repeating IVF, candidate-count, regularization, prototype-density, small
residual, or post-hoc low-bit sweeps is not supported by the current evidence.

