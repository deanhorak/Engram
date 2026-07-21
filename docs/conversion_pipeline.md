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
`evaluate-semantic`, `evaluate-mlp-intervention`, `compile`, and the `evaluate-e2e --teacher`
argument. Once resolved, model loading remains local-only so all Transformers calls use the
single cached snapshot. Existing local directories never require Hub access.

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

## Semantic progression gate

Before learned semantic parameters can enter a compiled package, run
`evaluate-mlp-intervention` on held-out text. The command substitutes identity, full-information
magnitude (the CLI's `oracle` arm), flat rank-16, overlapping-posting, or predictor-free DIP MLP
outputs inside the trained source transformer and applies the thresholds documented in
[evaluation](evaluation.md).
The dense source MLP still
executes inside this measurement harness, so its wall time is not a runtime benchmark.
For learned arms, calibration trace token sequences are compared directly with tokenized
evaluation sequences; any exact overlap blocks progression unless explicitly retained for a
diagnostic run, which still cannot pass the serialization gate.

A passing full-information magnitude reference is the standard screening trigger for spending
effort on a router, not proof that no other K-record subset could do better. The serialization gate
requires a reference measurement at the routed arm's K, but the reference need not pass if the
routed arm itself passes causal quality. A passing routed all-layer arm is required before
serialization. The checked SmolLM2 study first passes the reference among tested points at
768/1,536 active records, but both learned routed arms fail even at 1,280 candidates. Those fits
have now been repeated using all 1,112 available calibration states per layer and still fail.
Corpus-scaled regularization and candidate expansion up to 1,472/1,536 records also fail the
causal gate; that largest arm leaves too little traffic reduction to justify further expansion.
Experimental global and targeted correction capsules also worsen held-out local MLP error, so no
correction parameters are eligible for serialization.
The sparse-teacher trainer writes a separate safetensors router/adapter experiment artifact and a
gate report; it does not mutate the cached source model or compile the artifact. The first pilot
fails every routed quality check, so this artifact is likewise ineligible for package inclusion.
An audit also shows that its hard candidate selection blocks causal-loss gradients to the router,
so a future training retry must use a differentiable soft-to-hard stage.

The predictor-free DIP-inspired selector is the materially different arm that changes the quality
decision. Top-magnitude input pruning and partial scoring come from DIP; candidate-only exact
completion and contribution reranking are Engram extensions. After choosing 75% input coordinates,
896 candidates, and K=768 on the development grid, the fixed configuration passes again on a
sequence-disjoint confirmation corpus. `--evaluation-role confirmation` requires
`--configuration-selection-traces` and rejects any exact token-sequence overlap.
The current pipeline still has no DIP serializer: it needs a cache-aware gate/up layout, candidate
completion metadata, and a native kernel before attention/controller work can depend on it. The
quality pass authorizes that experimental systems implementation; it does not authorize compiling
the old learned-router artifacts or claiming a runtime speedup.

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
