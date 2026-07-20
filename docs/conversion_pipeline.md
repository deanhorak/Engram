# Conversion pipeline

Compilation produces one self-contained model worker. It does not construct or train the optional
request-level Cognitive Executive, its durable memory, or its tool/model routing policy.

## Model sources

Commands that consume a source model accept either an existing local directory or a Hugging
Face Hub model ID, for example `meta-llama/Llama-3.2-1B`. Install the conversion dependencies
before working with trained checkpoints:

```bash
python -m pip install -e '.[conversion]'
```

When the argument is not an existing directory, Engram downloads the configuration, tokenizer,
and weight files with `huggingface_hub.snapshot_download`. Files are stored in the standard
Hugging Face cache (controlled by `HF_HOME` or `HF_HUB_CACHE`) and reused by later commands.
Authenticate gated repositories with `hf auth login` or `HF_TOKEN`. Absolute paths and paths
beginning with `./`, `../`, or `~` are always treated as explicit local paths; a missing one is
reported as an error instead of being interpreted as a Hub ID.

Automatic resolution applies to `inspect`, `trace`, `analyze-mlp`, `build-semantic`,
`evaluate-semantic`, `compile`, and the `evaluate-e2e --teacher` argument. Once resolved, model
loading remains local-only so all Transformers calls use the single cached snapshot. Existing
local directories never require Hub access.

## Milestone 1 artifacts

Milestone 1 supports the following resumable artifacts:

1. `engram inspect` resolves the source, validates configuration and MLP tensor names/shapes, inventories local
   weight shards, and records SHA-256 source hashes.
2. `engram trace` writes a manifest and independent NPY shards. Each field records dtype,
   shape, and checksum. The manifest remains incomplete until a clean close.
3. `engram analyze-mlp` verifies the trace/model hash, loads one source MLP layer at a time,
   computes the magnitude oracle, and writes both machine-readable and readable reports.

For real Hugging Face models, forward hooks capture the exact input and output of each MLP
module. Records are processed one prompt at a time and flushed as independent shards; the
entire activation corpus is never retained in RAM. Model loading is currently CPU float32
and is not yet layer-streamed, which is a known converter limitation.

Fixture tracing deliberately feeds deterministic synthetic residual states through the
extracted MLP weights. This tests the semantic experiment without claiming full-model
teacher behavior.

## Runnable compiler

`engram compile` validates and hashes the source, copies tokenizer assets when available,
extracts embeddings and vocabulary projection, builds quantized-only semantic records plus
deterministic joint-key semantic and normalized vocabulary IVF indexes,
initializes the shared controller, writes episodic/cache/correction policies, and seals every
file with a checksum. Repeating an identical compile verifies and reuses the package without
rewriting it; source or option drift is rejected to preserve prior artifacts.

The current fallback initializer is deliberately simple and is recorded in
`metrics/conversion_report.json`. Teacher trace capture, background fitting, attention analysis,
and end-to-end distillation are separate commands rather than automatically trained stages.
That is a scientific limitation, not an implicit success.

`engram validate` and normal Python runtime construction verify checksums before arrays are
memory-mapped, then exercise deterministic generation. `engram-inspect`
performs independent native parsing, dimensional checks, required-file checks, and SHA-256
validation. Real text tokenization is copied into `tokenizer/`; native inference accepts packed
little-endian uint32 token IDs so a Python tokenizer wrapper can remain outside neural inference.
