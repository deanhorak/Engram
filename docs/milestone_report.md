# Engram milestone and gate report

Status date: 2026-08-02

## Executive result

The rank-16 compressed query/key selector followed by native exact reranking
now passes both the eight-record train screen and the independently captured
eight-record development screen. The development result is a causal CPU
replay, not a feature-only estimate:

- candidate-pool and exact-rerank membership recall: 99.8192% mean, 100% p10;
- answer-token agreement against the unmasked native runtime: 100%;
- mean hidden-state relative L2: 0.013236;
- mean logit relative L2: 0.006520;
- mean answer NLL delta: −0.003980, maximum +0.000574.

The authenticated result is
`work/olmoe_q7/retrieval_episodic_development_replay_rank16_pool6.json`,
SHA-256
`0c5cb2273f63b930148c78070da68ae57bb821969a68e2a6a038ee7ac5d04bb6`.

The selector is still evaluator-only. Native counters show mean logical
attention reads falling from 710,667,264 to 702,166,336 bytes (1.1962%); the
older-candidate scoring stage falls 7.3733%. This is a real semantic and
locality result, but not yet a convincing end-to-end speedup.

## Milestones

| Milestone | Status | Evidence and remaining work |
|---|---|---|
| 1. Repository, inspection, fixtures, teacher traces, exact MLP decomposition, oracle top-K, tests | Complete | Build system, source inspection, teacher traces, exact SwiGLU/MLP decomposition, oracle experiments, and regression reports are present. |
| 2. Semantic memory, practical routing, quantization, Python runtime, substituted-MLP evaluation | Gate passed; promotion pending | The train-to-development causal semantic gate now passes on CPU. Package promotion still requires long-context traffic/latency measurements and a protected evaluation. |
| 3. Attention analysis, local/recurrent/retrieval heads, hybrid episodic memory, attention substitution | In progress | Native W16/C8/K4 attention, episodic cache, Q/K candidate tracing, compressed selection, and exact reranking are implemented. Long-context performance and protected substitution remain. |
| 4. Shared recurrent controller and layer-free Engram runtime | Partial | Controller training, intermediate-state checks, adaptive-cycle experiments, and incremental dispatch exist. Broad generalization and fully promoted layer-free generation remain. |
| 5. Vocabulary/transition/correction artifacts, compiler, validation and CLI | Partial/usable | Native package generation, mapped weights, correction paths, validation, greedy generation, and chat CLI work. Full semantic-controller package promotion remains gated. |
| 6. Native C++ runtime, kernels, mapping, parity, generation, benchmarks | Partial/usable | CPU scalar/vector kernels, memory mapping, C ABI parity, native generation, and tests are operational. End-to-end long-context benchmarks and optimization remain. |
| 7. Comprehensive evaluation, ablations, tuning, documentation, final report | In progress | Documentation and reproducible reports are being maintained. Protected evaluation, broad model/task coverage, and final performance study remain. |

## Goals versus current reality

Engram is no longer handing inference to `llama.cpp` or a Transformers model
shell for the native path. The packaged OLMoE/BitNet route performs token
execution, Q7 expert reads, bounded attention, controller operations, and
greedy selection in the native CPU runtime. CUDA remains an optional training
and distillation accelerator; it is not required for inference.

The strongest demonstrated advantage is architectural: bounded semantic and
episodic state with authenticated causal replay and a path to CPU-only
execution. The measured traffic advantage of the current selector is small,
so it is not yet defensible to claim that Engram is faster than a highly tuned
`llama.cpp` implementation on general workloads. That comparison requires the
same model, context lengths, quantization, thread count, and output-quality
thresholds.

## Next development order

1. Benchmark the frozen selector at 512, 2,048, and longer contexts, reporting
   wall time, logical bytes, native counters, and peak memory.
2. Run a development-only rank/pool sweep (especially pool 4 versus pool 6)
   to quantify the quality/traffic frontier without refitting.
3. If the frontier remains near a 1% total-read reduction, keep the selector
   as a validated research path and prioritize fused projection/MLP kernels or
   a more aggressive learned residual selector.
4. Only after that decision, authorize protected evaluation and package-level
   integration.

## Verification

- Python: 1,112 passed, 1 skipped (CUDA unavailable on the test runner).
- Native CTest: 20/20 passed.
- Lint and `git diff --check`: passed.
