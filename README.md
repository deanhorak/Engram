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
  fourfold did not improve it, so the next step is a learned multi-label or coverage-optimized
  partition rather than simply collecting more of the same traces.
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
  representatives do not improve that result. The remaining partition constraint is too costly;
  the next routing experiment should learn overlapping postings with a coverage objective.
- The fitted rank-4 background operator worsened mean held-out error, so it is not currently a
  viable correction.

The compiler and runtimes work, but the controller is initialized rather than distilled and the
project has not demonstrated acceptable end-to-end language quality or its target memory-traffic
reduction. The immediate research priority is improving semantic candidate recall before further
compilation claims. See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md),
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
cache and reused on subsequent commands:

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
engram compile --model HuggingFaceTB/SmolLM2-135M --out work/local.engram
engram evaluate-e2e --model work/local.engram \
  --teacher HuggingFaceTB/SmolLM2-135M \
  --dataset /absolute/path/to/held-out.jsonl \
  --out reports/generated/local-quality
```

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

## Documentation

- [Architecture](docs/architecture.md)
- [How Engram works](docs/how_engram_works.md)
- [Conversion pipeline](docs/conversion_pipeline.md)
- [Model format](docs/model_format.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)
- [Research log](docs/research_log.md)
