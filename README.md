# Engram

Engram asks a practical research question: can we take knowledge and behavior learned by a
Llama-family transformer and reorganize them into a much smaller, CPU-native inference system?

A normal transformer evaluates every layer and most model weights for every token. Engram's
target design does something different:

1. A small recurrent **controller** maintains the current language-model state and allocates
   bounded internal update cycles.
2. A sparse **semantic memory** stores the useful records extracted from transformer MLPs and
   retrieves only a small relevant subset for each token.
3. A bounded **episodic memory** combines exact recent context with compressed older context.
4. A CPU-native runtime executes the compiled representation without PyTorch or the original
   transformer layers.

The intended result is not a wrapper, cache, or quantized copy of the source transformer. It is
a different inference architecture compiled from a trained model.

## Goals

- Preserve useful next-token behavior from a trained Llama-compatible teacher.
- Avoid reading the full transformer parameter set for every generated token.
- Bound working memory as context grows instead of retaining an unlimited attention cache.
- Run efficiently on ordinary CPUs with an inspectable C++20 implementation.
- Measure quality, latency, memory traffic, and failure modes honestly at every research gate.

The long-term systems target is a substantial reduction in DRAM traffic—ideally around 10x—while
retaining useful model quality. That target is a hypothesis, not a result. Engram will not claim
success from random fixtures, synthetic tasks, proxy byte counts, or a runnable compiler alone.

## Two levels of control

Engram's compiled runtime and its longer-term system architecture solve different problems. The
runtime controller is a low-level numeric mechanism inside one model worker. Above it, Engram is
developing an optional **Oracle cognitive executive** that represents goals, scopes attention,
estimates evidence confidence, proposes memory retention, selects strategies and workers, and
monitors progress under explicit cost and risk policy.

The executive produces typed decisions rather than prose and does not sit in the per-token hot
loop. Its deterministic policy, revisioned SQLite/JSONL/in-memory event stores, versioned worker
registry, resource ledger, worker-adapter boundary, outcome-observation loop, and calibration
metrics are implemented. Production model/tool adapters, content validators, deployment security,
and learned predictors are not. See
[The Oracle cognitive executive](docs/cognitive_executive.md) for its contracts, boundaries,
safety requirements, and separate research gates.

## Where the project stands

**Current decision:** the Milestone 2 implementation is broad, but its combined
semantic/systems gate is still blocked. Predictor-free DIP passes the causal
quality thresholds on an untouched confirmation corpus, but requires 83.33% of
dense MLP traffic after cache-line accounting and its native kernel is slower
than dense. A separately trained compact-Q4 student fits the physical traffic
limit at 44.9334%, but after 3,000,093 pretraining positions still has KL
0.887, top-1 agreement 56.6%, NLL delta +0.884, and final-hidden relative L2
0.425. The latest exact output-memory pilot improved layer-14 error only 1.73%
after adding one million independent prototypes, so that density-scaling path
is also closed.

No tested artifact passes quality and traffic together. The concise,
milestone-by-milestone account is [Project status](docs/status.md), with exact
machine-readable metrics in the
[2026-07-23 status snapshot](reports/semantic_gate_status_2026-07-23/summary.json).
The detailed history below is retained so negative results remain auditable.

The repository contains an end-to-end research prototype: Hugging Face model inspection and
download, exact teacher tracing, SwiGLU decomposition, sparsity oracles, semantic routing and
quantization, recurrent and retrieval memory primitives, compiled packages, and PyTorch-free
Python and native C++20 generation.

The first trained-model experiments used `HuggingFaceTB/SmolLM2-135M`:

- Retaining 90% of each MLP output's energy required 16.6% of neurons on average; retaining 99%
  required 44.8%. This suggests useful sparsity, but not extreme sparsity at high fidelity.
- The current joint-key IVF router is the main bottleneck. With 256 active neurons and 512
  candidates, candidate recall was only 40.6% and practical relative error was 0.673 versus the
  oracle's 0.335.
- An experimental trace-calibrated router improves mean top-256 recall to 61.0% with 512
  candidates, and to 66.8% while examining about 641 records. Increasing calibration coverage
  fourfold did not improve that router, which motivated learned multi-label and
  coverage-optimized follow-ups.
- A learned multi-label ridge router reaches 65.9% recall with 512 candidates and 72.2% with 640.
  This confirms that direct oracle-membership supervision helps, but its dense scoring matrix is
  too expensive for production.
- Low-rank compression preserves most of that gain: rank 16 reaches 63.3% recall with 512
  candidates using 141 KB of float32 router parameters per layer, 4.0% of the dense router. Rank
  32 reaches 64.4% using 276 KB.
- Hierarchical rank-16 group selection followed by exact local reranking was not successful. Its
  best configuration reached only 52.8% recall at 512 records, and the router-weight saving is
  small beside the selected key traffic.
- Training groups directly for oracle coverage improves hierarchical recall to 54.6%. Multiple
  representatives do not improve that result.
- A trained-teacher intervention harness now replaces MLP outputs inside the original transformer
  and measures final normalized-hidden-state drift, logit KL, top-1/top-5 agreement, and held-out
  NLL. It verifies the identity path exactly before testing sparse arms.
- The old top-256 target fails under the full-information magnitude reference: replacing all 30
  MLPs raises KL by 0.648 and NLL by 0.668 nats/token, while preserving only 60.5% of teacher
  top-1 choices. Magnitude top-K is not guaranteed to be the optimal K-record subset.
  K=768—half of every layer's 1,536 records—is the first tested active count that passes the
  declared progression thresholds (KL 0.032, top-1 92.3%, NLL +0.022, final-hidden rel-L2 0.092).
- At K=768, full-corpus refits using all 1,112 calibration states per layer still fail after
  examining 1,280 candidates. The flat rank-16 router reaches 88.9% recall, KL 0.789, and NLL
  +0.764. Coverage-trained overlapping postings reach 86.8% recall, KL 1.149, and NLL +1.095,
  while scanning about 1,667 posting entries to form 1,280 unique candidates. More calibration
  data modestly improved recall but did not close the 95% recall or downstream-quality gaps.
- A cached regularization sweep found a shallow optimum near λ=8,000. Raising the candidate
  budget to 1,408 and 1,472 clears the recall gate at 95.4% and 97.8%, but causal substitution
  still fails: the 1,472-candidate arm has KL 0.085, top-1 agreement 86.6%, NLL +0.055, and
  final-hidden rel-L2 0.131. It reads 95.8% of record keys, leaving only about a 1.24× projected
  key/value traffic reduction before router overhead. This rank-16 configuration is abandoned.
- A predictor-free, DIP-inspired path now uses the source model's own gate/up weights on the
  largest-magnitude input coordinates. Engram then exactly completes only its candidate records
  and reranks them to K=768; that completion/reranking stage is an Engram extension to the
  published DIP method. It requires no learned membership router. The recommended
  75%-input/896-candidate arm passes both the development frontier and a separate untouched
  confirmation corpus. Confirmation metrics are 99.0% candidate recall, KL 0.029, 91.0% top-1
  agreement, NLL +0.033, and final-hidden rel-L2 0.090. Its logical float32 weight-read model is
  76.4% of dense MLP traffic, a projected 1.31x reduction before indexes and cache effects.
- The selector now has an experimental version-2 coordinate-major package and a candidate-only
  native kernel. Cache-line accounting raises the same arm to 83.3% of dense bytes. After
  structure-of-arrays, partition selection, sorted gathers, and float32 parity work, the best
  30-layer streamed kernel is still about 15.4% slower than dense (`0.863x`). It is explicitly
  rejected before default-runtime integration. A spatial 16-float-block layout was
  also rejected: confirmation recall fell to 85.2%. The semantic quality pass therefore survives,
  but the current systems implementation does not pass.
- A second-generation sparse-teacher path now targets the systems failure directly. Its default
  budget is `q=62.5%`, `C=K=512`; the student evaluates all records on only the retained input
  coordinates, completes exactly 512 candidates, and reads 512 down records. Straight-through
  candidate masks let local-MLP, hidden-state, and logit losses train routing, while a cache-line
  occupancy loss penalizes scattered candidates. The first one-record SmolLM2 smoke run verifies
  this gradient path and sparse execution. It starts at 90.0% candidate recall, but its 512
  candidates touch 95.84/96 gate/up cache-line groups on average. Cache-line amplification raises
  the optimistic 61.1% scalar estimate to about 77.7% of dense traffic. This is a training starting
  point, not a quality or speed pass; a full-corpus training run must materially improve both recall
  and locality before compilation.
- The complete 32-sequence/16-held-out run at that budget fails despite meeting the evidence and
  hardware-budget checks: recall is 89.59%, KL 0.166, top-1 agreement 76.78%, NLL delta +0.126,
  and final-hidden relative L2 0.199. Candidates still touch 95.86/96 gate/up line groups.
  Balanced storage permutations reduce this only to 94.66 lines; forcing selection of 32 complete
  line groups cuts recall to at most 48.73% and raises local MLP error above 0.47.
- The trainer now supports masked sequence batching, provenance-checked router initialization
  caches, separate calibration/training corpora, and mergeable rank-8 LoRA updates for gate, up,
  and down projections. A deterministic local-source corpus builder produced 128 sequences and
  15,991 token positions. A bounded 16-sequence LoRA stage modestly improved KL/top-1 but worsened
  NLL and left hidden error, recall, and locality effectively unchanged, so it was not scaled.
- A held-out oracle bound now explains the locality failure rather than treating it as an optimizer
  mystery: exact top-512 membership already touches 95.86/96 contiguous 16-record lines. Even a
  perfect group selector limited to 80 lines can cover only 91.75% of the oracle set; reaching
  96.65% requires 88 lines. A duplicated record-major v3 package is also rejected: it grows MLP
  storage by 66.7% and its tested kernels are slower. Version 2 remains the default research
  package.
- The locality relaxation was audited and replaced with an exact-hard-value, fixed-cardinality
  soft-backward objective. Gradient diagnostics show its unweighted router gradient is about 269x
  smaller than the causal gradient, but a balanced 16-step trial still does not reduce hard line
  occupancy. Standard LoRA scaling and resumable checkpoints are now implemented. A full
  128-sequence rank-32 residual run improves KL/top-1/NLL/hidden L2 to
  0.152/0.780/+0.100/0.193, but still fails every causal threshold. The residual has essentially
  zero alignment with the missing output and is disabled by default; higher adapter learning rates
  are unstable.
- A fixed-total layer-adaptive magnitude oracle was selected from individual-layer interventions
  on a separate four-sequence split and frozen before confirmation. At the same mean K=512 it is
  slightly worse than uniform K=512 on the untouched 16-sequence set (KL 0.134, top-1 0.786, NLL
  +0.110, hidden L2 0.185). Layer adaptation is also stopped. The next justified direction is a
  co-trained structured expert/block representation, not more tuning of the frozen neuron basis.
- A new structured-expert shadow path tests that direction before expensive training. Balanced
  24×64/top-8, 48×32/top-16, and 96×16/top-32 layouts all execute exactly 512 records and preserve
  the dense all-block output to below 8.6e-7 maximum relative L2. However, even a non-deployable
  greedy-residual block oracle has mean local error 0.547/0.497/0.438, and fitted routers worsen
  those to 0.655/0.638/0.624. Static grouping is therefore stopped before end-to-end training. The
  next bounded design is co-trained native gate-based channel sparsity with hardware-aligned
  grouped selection, not a larger static expert router.
- The native-gate follow-up removes candidate completion entirely. At K=512, the exact
  contribution reference has local relative L2 0.190, while dense-gate channel selection is 0.375;
  q=62.5% input pruning moves it only to 0.386 at 43.06% ideal traffic. A hard-forward/soft-backward
  full-weight wrapper and cached-trace pretrainer are implemented, but a controlled 64-step
  representative-layer run improves held-out error only 2.55% and fails its 10% screen. This local
  pretraining path is stopped; the next credible run requires progressive end-to-end co-training
  on materially more data. The implementation remains CPU-capable; CUDA is an optional training
  accelerator, not an inference or format dependency.
- Progressive end-to-end native-gate co-training is now implemented and runs on CPU. It anneals
  dense execution to q=62.5%/K=512, co-trains full MLP weights while freezing non-MLP transformer
  components, validates through the hard path only, and supports resumable device-neutral
  checkpoints. The full-evidence untrained baseline is KL 1.235/top-1 0.460/NLL +1.202/hidden L2
  0.508. An eight-step CPU stage reaches 1.254/0.481/+1.211/0.510: mixed movement, not justification
  for a longer run on the same objective. The trainer is ready for controlled CPU slices or optional
  CUDA acceleration once a better training curriculum/data scale is justified.
- A low-rank utility-residual router now predicts the missing up-projection-dependent channel
  utility from the current hidden state. With 512 calibration states, rank 16/blend 0.8 lowers the
  trace-local error from 0.386 to 0.338 and raises exact-oracle recall to 0.643 at 44.39% projected
  dense traffic. The full all-layer hard-path control confirms that this is causal: KL falls from
  1.235 to 0.629, top-1 agreement rises from 0.460 to 0.599, NLL delta falls from +1.202 to +0.583,
  and hidden L2 falls from 0.508 to 0.363. It still misses the final quality gate. A matched
  eight-step run slightly improves top-1 but regresses the other metrics, so the next bounded
  experiment is on-policy residual recalibration on sparse-student states, not longer training.
- That on-policy screen is now complete and negative: same-state local L2 changes only
  0.35117→0.34983. A 44.25%-traffic q=43.75%/K=640 alternative also fails causally
  (KL 0.684, top-1 0.603, NLL +0.616, hidden L2 0.358). Since even the exact K=512 oracle misses
  the gate, the frozen-basis router search is closed. The remaining Milestone 2 path is structured
  sparse upcycling/width pruning with full MLP adaptation and materially more real-token data.
- The first all-layer fixed-width student now tests that path directly. It replaces every
  1,536-wide SwiGLU with a trainable 672-wide contiguous SwiGLU, freezes attention and
  normalization, and stays at 43.75% of dense MLP weight traffic. Training is CPU-capable,
  checkpointed, and supports parameter-only transfer to a fresh corpus. One full epoch over 2,048
  sequence-disjoint local-source examples improves the held-out result to KL 1.177, top-1 47.5%,
  NLL delta +1.055, hidden relative L2 0.426, and local MLP relative L2 0.705. This is a decisive
  semantic-gate failure, not a compilation candidate. More epochs on the same narrow fixed-width
  objective are stopped; the next experiment must measure the teacher-boundary local approximation
  ceiling before spending on another causal run.
- That ceiling is now measured on 4,096 sampled training boundaries and 446 sequence-disjoint
  validation boundaries. Five representative compact layers trained for 2,048 cached-boundary
  steps improve mean local L2 from 0.3851 to 0.3457, but miss the declared 0.15 ceiling. Middle
  layers remain between 0.45 and 0.50. Width 672 is therefore rejected as a uniform all-layer
  representation; the next design must allocate capacity by layer or use a more expressive
  structured basis while retaining the same aggregate traffic cap.
- The fitted rank-4 background operator worsened mean held-out error, so it is not currently a
  viable correction.

The compiler and runtimes work, but the controller is initialized rather than distilled and the
project has not demonstrated acceptable end-to-end language quality or its target memory-traffic
reduction. Learned rank-16, posting-group, residual-capsule, and first sparse-teacher artifacts
remain blocked. The new predictor-free arm is the first realizable selector algorithm to clear the
semantic quality prerequisite, and its experimental package/kernel now expose the next blocker:
measured native latency. It is intentionally not present in default `.engram` packages or the
generation runtime because the checked kernel is slower than dense. The original sparse-teacher
pilot's disconnected routing gradient is fixed in the hardware-aware trainer, but the complete
low-budget evaluation, corrected LoRA/residual run, locality bound, and layer-adaptive confirmation
all fail. These artifacts remain blocked; structured sparsity must be learned jointly with the MLP
weights before another compilation attempt. The first 672-wide jointly adapted student also fails
after complete exposure to its 2,048-sequence corpus, so compilation remains blocked.
Subsequent adaptive low-bit, structured-basis, compact-Q4, conditional-expert,
and nonparametric output-memory experiments also fail their frozen progression
rules. In particular, the serialized mild-width Q4 student passes the 45%
physical-byte check but remains far outside every causal threshold after 3M
pretraining positions; scaling exact output memory from 233,005 local records
to 1,233,005 combined records changes layer-14 error only from 0.327526 to
0.321854. No trained semantic artifact is currently eligible for default
package compilation.
See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md),
and [limitations](docs/limitations.md) for the precise design and caveats. The latest routing
measurement is documented in the [trace-calibrated recall report](reports/smollm2_calibrated_router/recall.md).
The directly supervised follow-up is in the
[multi-label routing report](reports/smollm2_multilabel_router/recall.md).
The compression frontier is measured in the
[low-rank routing report](reports/smollm2_lowrank_router/recall.md).
The hierarchical follow-up and its negative result are in the
[hierarchical routing report](reports/smollm2_hierarchical_router/recall.md).
The direct coverage and multiple-representative experiments are in the
[coverage-trained group report](reports/smollm2_coverage_groups/recall.md).
The causal intervention frontier and router decisions are summarized in the
[trained-teacher intervention decision](reports/smollm2_mlp_intervention/decision.md), with the
machine-readable arm reports linked there. A
[provenance-checked composite report](reports/smollm2_mlp_intervention_composite/mlp_intervention.json)
applies the final gate across the separately executed arms.
The static structured-expert screens are recorded for
[64-record blocks](reports/smollm2_structured_expert_shadow/structured_expert_shadow.md),
[32-record blocks](reports/smollm2_structured_expert_shadow_48x32/structured_expert_shadow.md),
and [16-record blocks](reports/smollm2_structured_expert_shadow_96x16/structured_expert_shadow.md).
The native-gate diagnosis is in the
[channel shadow report](reports/smollm2_native_gate_channel_shadow/native_gate_channel_shadow.md),
and the bounded training stop is in the
[64-step layer report](reports/smollm2_native_gate_trace_layer14_utility64/native_gate_trace_training.md).
The device-neutral end-to-end controls are the
[untrained CPU baseline](reports/smollm2_native_gate_e2e_cpu_baseline/native_gate_end_to_end.md)
and [eight-step CPU stage](reports/smollm2_native_gate_e2e_cpu_stage8/native_gate_end_to_end.md).
The passing local residual screen and device-neutral router tensors are in the
[expanded residual report](reports/smollm2_native_gate_utility_residual_expanded/native_gate_utility_residual.md).
Its all-layer controls are the
[untrained residual run](reports/smollm2_native_gate_e2e_cpu_residual_baseline/native_gate_end_to_end.md)
and [matched eight-step run](reports/smollm2_native_gate_e2e_cpu_residual_stage8/native_gate_end_to_end.md).
The cached [regularization sweep](reports/smollm2_rank_router_regularization_sweep/rank_router_regularization_sweep.md),
[candidate frontier](reports/smollm2_rank_router_candidate_frontier/rank_router_regularization_sweep.md),
and [near-dense causal check](reports/smollm2_mlp_intervention_rank16_lambda8000_frontier/mlp_intervention.md)
record why the flat rank-16 configuration is no longer being pursued.
The [global correction-capsule sweep](reports/smollm2_correction_capsule_sweep/correction_capsule_sweep.md)
and [targeted tight-radius sweep](reports/smollm2_correction_capsule_targeted_tight_sweep/correction_capsule_sweep.md)
record the negative residual-correction result.
The [sparse-teacher pilot](reports/smollm2_sparse_teacher_epoch1/sparse_teacher_training.md)
records the first trainable sparse-student result and its unchanged stop decision.
The [hardware-aware sparse-teacher smoke run](reports/smollm2_hardware_sparse_smoke/sparse_teacher_training.md)
checks the replacement gradient path and low-budget/cache-line reporting without claiming a
full-corpus result.
The [complete low-budget gate](reports/smollm2_hardware_sparse_full/sparse_teacher_training.md),
[three-projection LoRA stage](reports/smollm2_hardware_sparse_lora_stage/sparse_teacher_training.md),
and [broader-corpus stage](reports/smollm2_hardware_sparse_corpus_stage/sparse_teacher_training.md)
record the subsequent stop decision and bounded follow-ups.
The [oracle locality bound](reports/smollm2_locality_oracle_bound/oracle_line_coverage.md),
[dual-layout diagnostic](reports/smollm2_dip_dual_layout/dual_layout_benchmark.md),
[full corrected-LoRA/residual run](reports/smollm2_residual_r32_scaled_full/sparse_teacher_training.md),
and [layer-adaptive confirmation](reports/smollm2_mlp_intervention_oracle_adaptive512_causal/mlp_intervention.md)
record the final low-budget representation tests.
The predictor-free [DIP trace sweep](reports/smollm2_dip_exact_completion_sweep/dip_exact_completion_sweep.md)
and [causal frontier](reports/smollm2_mlp_intervention_dip_frontier/mlp_intervention.md)
record its development selection and measured quality/projected scalar-read frontier. The
[untouched confirmation report](reports/smollm2_mlp_intervention_dip_confirmation/mlp_intervention.md)
freezes the 75%/896 configuration and verifies zero exact sequence overlap with the selection set.
The [blocked-layout confirmation](reports/smollm2_dip_blocked_confirmation/dip_exact_completion_sweep.md)
and [native layer benchmark](reports/smollm2_dip_native_layer10/dip_native_benchmark.md) record the
subsequent negative systems results.

## How conversion and inference work

A Llama model alternates attention blocks, which move information between token positions, and
SwiGLU MLP blocks, which transform each position independently. Engram treats every MLP neuron as
a memory record with two lookup keys and one output value. It reads those tensors directly from
the Hugging Face checkpoint, records the real inputs and outputs seen at each layer, and measures
which records matter for each state. The converter then quantizes the records, builds indexes for
sparse lookup, copies tokenizer and embedding data, and writes a checksummed `.engram` directory.
The original transformer layers are not needed to load that directory.

At inference time, the runtime tokenizes the prompt and maintains one fixed-width recurrent state.
For each token it retrieves semantic records, updates bounded short- and long-context memory,
runs a shared recurrent controller for a small number of cycles, and searches the vocabulary for
the next token. The intended trained system will distill the controller and episodic mechanisms
from teacher traces. The current runnable baseline exercises this dataflow but uses an initialized
controller and heuristic episodic memory, so it is infrastructure for the research rather than a
quality-preserving conversion.

For a detailed explanation written for readers who know general computer science but not language
models, see [How Engram works](docs/how_engram_works.md). It covers the source Llama computation,
extraction process, compiled format, inference loop, and the work still required.

## Quick start

Python 3.10+, NumPy, CMake 3.20+, and a C++20 compiler are required. PyTorch,
Transformers, and safetensors are only required for real Hugging Face checkpoints.

```bash
python -m pip install -e '.[dev,conversion]'
engram create-fixture --out work/tiny-llama --seed 7
engram inspect --model work/tiny-llama --out work/inspection.json
engram trace \
  --model work/tiny-llama \
  --dataset tests/fixtures/calibration.jsonl \
  --out work/traces \
  --samples 32
engram analyze-mlp \
  --model work/tiny-llama \
  --traces work/traces \
  --out reports/generated/milestone1
engram build-semantic --model work/tiny-llama --out work/tiny.engram
engram trace --model work/tiny-llama --out work/validation-traces \
  --split validation --samples 32 --seed 18
engram evaluate-semantic --model work/tiny-llama \
  --calibration-traces work/traces \
  --validation-traces work/validation-traces \
  --out reports/generated/milestone2
engram evaluate-attention --out reports/generated/milestone3
engram evaluate-controller --out reports/generated/milestone4
engram compile --model work/tiny-llama --out work/tiny.engram
engram validate --model work/tiny.engram
engram generate --model work/tiny.engram --prompt "hello" --max-tokens 16

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/engram-inspect work/tiny.engram
./build/engram-run work/tiny.engram --prompt "hello" --max-tokens 16 --greedy
./build/engram-bench work/tiny.engram 512
```

The fixture is random and only validates the pipeline. To produce meaningful evidence,
use a trained Llama-compatible Hugging Face model. Pass either a local directory or a
Hub model ID. Hub models are downloaded automatically into the standard Hugging Face
cache and reused on subsequent commands. The current semantic format requires bias-free
SwiGLU MLP projections and rejects checkpoints with `mlp_bias=true`:

```bash
engram inspect --model HuggingFaceTB/SmolLM2-135M
engram trace \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/calibration.jsonl \
  --out work/real-traces \
  --samples 128
engram analyze-mlp \
  --model HuggingFaceTB/SmolLM2-135M \
  --traces work/real-traces \
  --out reports/generated/real-model
engram evaluate-mlp-intervention \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/held-out.jsonl \
  --out reports/generated/mlp-quality \
  --variants identity oracle \
  --top-k 256 512 768 \
  --layer-mode all
engram sweep-dip \
  --model HuggingFaceTB/SmolLM2-135M \
  --validation-traces /absolute/path/to/validation-traces \
  --out reports/generated/dip-sweep \
  --input-fractions 0.5 0.625 0.75 \
  --top-k 768 \
  --candidates 896 1024 1152
engram evaluate-mlp-intervention \
  --model HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/untouched-confirmation.jsonl \
  --out reports/generated/dip-confirmation \
  --variants identity oracle dip \
  --input-fractions 0.75 \
  --top-k 768 \
  --candidates 896 \
  --layer-mode all \
  --evaluation-role confirmation \
  --configuration-selection-traces /absolute/path/to/validation-traces
engram build-distillation-corpus \
  --model HuggingFaceTB/SmolLM2-135M \
  --input README.md docs src native tests \
  --out work/sparse-distillation.jsonl \
  --sequence-length 128 \
  --max-sequences 128
engram train-sparse-student \
  --model HuggingFaceTB/SmolLM2-135M \
  --calibration-dataset /absolute/path/to/calibration.jsonl \
  --training-dataset work/sparse-distillation.jsonl \
  --validation-dataset /absolute/path/to/held-out.jsonl \
  --calibration-traces /absolute/path/to/calibration-traces \
  --out reports/generated/hardware-sparse-student \
  --routing-mode hardware_ste \
  --input-fraction 0.625 \
  --top-k 512 \
  --candidates 512 \
  --locality-weight 0.05
```

A successful command exit is not a compilation claim. The generated intervention report applies
explicit quality gates; routed arms must pass before their parameters are eligible for
serialization. `engram gate-mlp-intervention --report PATH` reapplies the current declared
thresholds to an existing report. Supplying several `--report` paths plus `--out DIRECTORY`
creates a provenance-checked composite gate, which is useful when expensive arms were run in
stages.

For gated repositories, authenticate first with `hf auth login` or set `HF_TOKEN`.
Existing local directories continue to work without network access.
The cache location follows Hugging Face defaults and can be changed with `HF_HOME` or
`HF_HUB_CACHE`. See the [conversion pipeline](docs/conversion_pipeline.md) for model-source
resolution details and supported commands.

Dataset records may contain either `{"text": "...", "input_type": "prose"}` or
pretokenized `{"input_ids": [1, 2, 3], "input_type": "code"}`. Pretokenized input is
useful for tiny local checkpoints without tokenizer assets.

## Verification

```bash
python -m pytest -q
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
/usr/bin/ctest --test-dir build --output-on-failure
```

The explicit `/usr/bin/ctest` avoids a broken user-local Python wrapper observed on the
development host. Ordinary `ctest` is correct when it resolves to the CMake executable.

## Scientific interpretation

The “oracle” computes every SwiGLU activation and ranks neuron records by
`abs(activation_j) * ||value_j||₂`; it then scans every prefix because vector cancellation
can make reconstruction error non-monotonic. It is a strong full-information
contribution-magnitude baseline, not the mathematically optimal K-subset and not a
production router. A target of 90% means
`||full - approximation||² / ||full||² <= 0.10`.

The checked-in [fixture report](reports/milestone1_fixture/oracle_topk.md) is pipeline
evidence only. A subsequent SmolLM2-135M experiment measured trained-model sparsity and a
fitted-background ablation; the background failed to improve held-out mean error. Those pilot
corpora remain too small for a broad model-family claim.

The [Gate 2 fixture report](reports/milestone2_fixture/practical_routing.md) preserves a
negative result: joint-key IVF scored 18.25 of 32 records on average, but candidate recall
was only 0.578 and reconstruction error trailed the oracle. The low-rank background also
overfit the small random calibration set. This is instrumentation evidence, not a reason
to claim the semantic-memory hypothesis works.

The [Gate 3 synthetic report](reports/milestone3_fixture/attention_replacement.md) covers
bounded local, recurrent, and older-context retrieval memory. It is not teacher-attention
distillation evidence.

[Gate 4](reports/milestone4_fixture/controller_gate.json) is also synthetic; adaptive
execution averaged 7.98 of 8 allowed cycles, so it found essentially no compute saving.
The [runtime benchmark](reports/runtime_fixture/benchmark.md) is a tiny-fixture systems
measurement. Gate 5 becomes meaningful only after `evaluate-e2e` is run against a trained,
held-out local checkpoint.

The checked [Gate 5 random-fixture report](reports/milestone5_fixture/end_to_end_quality.md)
validates that evaluator and records a negative result: zero category target accuracy and 93.75%
repetition. Its small KL is an artifact of near-uniform random logits.

The trained-teacher MLP intervention is narrower and more diagnostic than Gate 5: it keeps the
trained transformer's attention, residual path, normalization, and vocabulary head exact while
replacing only selected MLP outputs. The checked SmolLM2 result finds that full-information
magnitude top-768 is the first tested selection that passes the declared progression thresholds,
and all learned practical routers fail. Predictor-free DIP subsequently passes with 75% of input
coordinates and 896 candidates. Experimental serialization and a native kernel now exist, but the
kernel fails its isolated latency gate and remains outside the compiled runtime. There is still no
trained controller or trained-package Gate 5 result.

## Documentation

- [Current project and milestone status](docs/status.md)
- [Architecture](docs/architecture.md)
- [How Engram works](docs/how_engram_works.md)
- [Conversion pipeline](docs/conversion_pipeline.md)
- [Model format](docs/model_format.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)
- [Research log](docs/research_log.md)
