# Engram milestone and gate report

Status date: 2026-08-03

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

The required long-context CPU scaling boundary is now measured on development
record 0.  The exact same native package was run unmasked and masked at 512
and 2,048 positions (the first 128 positions remain the authenticated
selector window; later positions repeat tokens with episodic directives
disabled).  Answer quality remains 100% top-1 agreement with mean hidden/logit
relative L2 0.008138/0.003919 and NLL delta −0.001520.  At 512 positions the
masked logical-read fraction is 0.997079 (3.04 versus 3.09 tokens/s); at 2,048
it is 0.999275 (3.083 versus 3.085 tokens/s).  Peak resident set was 6.28 GiB
and did not grow with the repeated context.  These results establish bounded
state and honest scaling, but do not claim a speedup: the selector saves a
fixed 8.55 MiB of logical attention reads while the rest of the full model
continues to run.  The report is
`work/olmoe_q7/retrieval_episodic_long_context_rank16_pool6.json`, SHA-256
`fa205bd2ab4c91de27170247e7669f44c9def8bccea22d94558f8caa4b26bf71`.

The follow-up frozen development rank/pool sweep confirms the operating point:
rank-16/pool-4 falls to 95.1389% exact membership recall at 50% of the older
slots, while rank-16/pool-6 reaches 99.8192% at 75%; rank-16/pool-8 is exact
but reads all eight slots. Increasing rank to 32 or 64 changes pool-6 recall
only to 99.8867% and 99.9367%, respectively, so the extra projection cost has
no current causal justification. This is a feature-only sweep (no native
intervention); its report is
`work/olmoe_q7/retrieval_episodic_development_pool_sweep_rank16_pool6.json`,
SHA-256
`b170625c9802019a62be0c88a64799bef35bc8246befc000637dcb75e864f0ce`.

The selected policy is now serialized as a disabled-by-default evaluator
artifact.  It contains the train-only per-layer/head rank-16 key PCA basis,
the rank-16/pool-6/top-4 geometry, source hashes, and validation report hashes;
it contains no protected data.  Loading the artifact and rebuilding all
development masks reproduces the previous cross-split masks bit-for-bit
(shape `[8, 128, 16, 16, 8]`).  The local artifact is
`work/olmoe_q7/selector_policy_rank16_pool6_2026-08-03/` with policy SHA-256
`6d22205b543fd2d2ef986dd1a60b3f517b95a1c1218f1fd834c5cb3a6f16f46b` and basis
SHA-256
`1897767a6b22801faad4267c22152b5029285091e53fc3f36078ad6aa849f813`.
The loader deliberately refuses any artifact that is not evaluator-only and
disabled by default; native package defaults remain unchanged.
The native package compiler can copy this artifact under `selector/` and bind
both files into the authenticated package manifest when explicitly requested;
omitting that option preserves the previous package layout.

That opt-in path has now been exercised end to end on development record 0:
the serialized policy generated the native masked replay, preserving 100%
answer top-1 agreement while reducing logical attention reads from 710,668,288
to 702,124,544 bytes.  The tiny-package compiler/validator also authenticated
the copied policy and basis under `selector/`.  This closes the implementation
boundary without silently enabling the policy in ordinary generation.

## Milestones

| Milestone | Status | Evidence and remaining work |
|---|---|---|
| 1. Repository, inspection, fixtures, teacher traces, exact MLP decomposition, oracle top-K, tests | Complete | Build system, source inspection, teacher traces, exact SwiGLU/MLP decomposition, oracle experiments, and regression reports are present. |
| 2. Semantic memory, practical routing, quantization, Python runtime, substituted-MLP evaluation | Gate passed; bounded-runtime evidence complete | The train-to-development causal semantic gate, 512/2,048-position CPU scaling boundary, frozen pool frontier, and disabled-by-default policy artifact pass. The evaluator-only selector still needs a separately authorized protected evaluation before package promotion. |
| 3. Attention analysis, local/recurrent/retrieval heads, hybrid episodic memory, attention substitution | In progress | Native W16/C8/K4 attention, episodic cache, Q/K candidate tracing, compressed selection, exact reranking, and long-context scaling are implemented. Protected substitution and kernel-level speedup remain. |
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

1. Keep rank-16/pool-6 as the frozen research point: pool 4 is below the
   desired recall margin and higher ranks do not materially improve it.
2. If the frontier remains near a 1% total-read reduction, keep the selector
   as a validated research path and prioritize fused projection/MLP kernels or
   a more aggressive learned residual selector.
3. Only after that decision, authorize protected evaluation and package-level
   integration.

## Verification

- Python: 1,112 passed, 1 skipped (CUDA unavailable on the test runner).
- Native CTest: 20/20 passed.
- Lint and `git diff --check`: passed.
