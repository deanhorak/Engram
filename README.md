# Engram

Engram is a research engineering project exploring whether a pretrained Llama-family
transformer can be compiled into a CPU-native recurrent controller plus sparse semantic
and episodic memories. The production goal is explicitly **not** a transformer wrapper.

The repository now contains an end-to-end research prototype: validated local-model inspection,
deterministic Llama-shaped fixtures, checksummed streaming traces at the exact MLP module
boundary for Hugging Face checkpoints, exact SwiGLU neuron decomposition, a full-information
contribution-magnitude oracle, indexed joint-key IVF routing, low-rank residual backgrounds,
scalar/additive quantization, inspectable compiled packages, a shared recurrent controller,
hybrid episodic memory, indexed vocabulary MIPS, transition caching, corrections, and PyTorch-free
Python and native C++20 generation. It is a runnable baseline, not evidence that the research
quality or memory-traffic goals have been met. See [limitations](docs/limitations.md).

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
use a local, trained Llama-compatible Hugging Face directory. Engram never downloads a
model:

```bash
engram inspect --model /absolute/path/to/local-model
engram trace \
  --model /absolute/path/to/local-model \
  --dataset /absolute/path/to/calibration.jsonl \
  --out work/real-traces \
  --samples 128
engram analyze-mlp \
  --model /absolute/path/to/local-model \
  --traces work/real-traces \
  --out reports/generated/real-model
engram compile --model /absolute/path/to/local-model --out work/local.engram
engram evaluate-e2e --model work/local.engram \
  --teacher /absolute/path/to/local-model \
  --dataset /absolute/path/to/held-out.jsonl \
  --out reports/generated/local-quality
```

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
evidence only. Gate 1 remains incomplete until a trained model is evaluated with and
without a fitted background operator.

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
