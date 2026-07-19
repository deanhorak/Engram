# Engram

Engram asks a practical research question: can we take knowledge and behavior learned by a
Llama-family transformer and reorganize them into a much smaller, CPU-native inference system?

A normal transformer evaluates every layer and most model weights for every token. Engram's
target design does something different:

1. A small recurrent **controller** maintains the current language-model state and decides what
   computation is needed next.
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
- The fitted rank-4 background operator worsened mean held-out error, so it is not currently a
  viable correction.

The compiler and runtimes work, but the controller is initialized rather than distilled and the
project has not demonstrated acceptable end-to-end language quality or its target memory-traffic
reduction. The immediate research priority is improving semantic candidate recall before further
compilation claims. See [architecture](docs/architecture.md), [evaluation](docs/evaluation.md),
and [limitations](docs/limitations.md) for the precise design and caveats.

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
- [Conversion pipeline](docs/conversion_pipeline.md)
- [Model format](docs/model_format.md)
- [Evaluation](docs/evaluation.md)
- [Limitations](docs/limitations.md)
- [Research log](docs/research_log.md)
