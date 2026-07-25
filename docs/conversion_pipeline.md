# Conversion pipeline

Current eligibility is summarized in [Project status](status.md). Although the
fixture compiler and experimental serializers work, no trained SmolLM2
semantic artifact currently passes both the causal and physical-traffic gates,
so none is eligible to become the default compiled semantic representation.

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

Native BitNet uses a deliberately separate, fail-closed path:

```bash
engram audit-native-bitnet \
  --model microsoft/bitnet-b1.58-2B-4T \
  --out reports/native-bitnet-audit.json

engram repack-native-bitnet \
  --model microsoft/bitnet-b1.58-2B-4T \
  --out work/native-bitnet/model.bitnet-records.bin \
  --report reports/native-bitnet-repack.json

TORCHDYNAMO_DISABLE=1 engram evaluate-native-bitnet-parity \
  --model microsoft/bitnet-b1.58-2B-4T \
  --artifact work/native-bitnet/model.bitnet-records.bin \
  --artifact-sha256 4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55 \
  --out reports/native-bitnet-parity.json

TORCHDYNAMO_DISABLE=1 engram evaluate-native-bitnet-kernel \
  --model microsoft/bitnet-b1.58-2B-4T \
  --artifact work/native-bitnet/model.bitnet-records.bin \
  --artifact-sha256 4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55 \
  --dataset tests/fixtures/confirmation_expanded.jsonl \
  --out reports/native-bitnet-kernel.json \
  --library build/libengram_bitnet.so \
  --threads 12
```

The audit downloads configuration only and pins the official revision by
default. The repack command then downloads weights if absent, verifies the
official SHA-256, validates every packed two-bit code, writes five-trit
phase streams, reloads them, and compares all reconstructed values. Pass the
artifact hash emitted by that report into the parity command so canonical but
corrupted payloads cannot pass structural validation alone. These
commands do not make `bitnet` eligible for `build-semantic` or `compile`;
ReLU-squared gating and `ffn_sub_norm` violate those commands' exact-SwiGLU
contract. The parity command remains a dense BF16 correctness oracle. The
kernel command uses the separately built `libengram_bitnet.so`, memory-maps
the phase file, and executes its packed streams directly. It downloads the
pinned tokenizer assets when absent, applies the tokenizer regex compatibility
fix, and runs the frozen all-layer confirmation protocol. This qualifies the
low-bit-native MLP path for package integration; it does not make BitNet
compatible with the generic SwiGLU compiler.

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
An audit also shows that its hard candidate selection blocks causal-loss gradients to the router.
The replacement experiment now uses hard-forward/soft-backward candidate masks, a materially lower
`q<=62.5%`, `C/K<=512` budget, and a differentiable cache-line locality loss. It remains a separate
experiment artifact: only a one-record smoke run has completed, so it is not compiler input.
The later complete run and bounded LoRA/broader-corpus follow-ups also fail. Router initialization
is now cacheable and student training may use a separate JSONL corpus, but these engineering
improvements do not make the resulting tensors eligible for compilation.
The structured-expert shadow command is also deliberately pre-compilation. Its tested static
whole-block layouts fail the local feasibility screen, so they do not produce student weights or a
serializable package. A future native gate-routed student must first pass the unchanged held-out
causal thresholds through its exact hard forward path.
The native-gate shadow and cached-trace trainer are likewise pre-compilation experiments. They
remove candidate completion and meet the nominal q/K traffic envelope, but the checked layerwise
training arm fails its improvement screen. Their safetensors file is a diagnostic selected-layer
checkpoint, not a model package or compiler input.

`train-budget-native-ternary` is the first trainer here that fixes and writes
its complete low-bit MLP representation before compiler integration. The
binary stores five ternary coefficients per byte, FP16 scales, versioned
headers/directories, and alignment; validation strictly reloads it and checks
its file size against traffic accounting. Optional co-adapted backbone tensors
are stored separately in safetensors. This is still a research artifact, not a
default `semantic/` layout: the checked one-million-position SmolLM2 run passes
43.1353% traffic but fails causal quality and its pre-3M progression rule.

The end-to-end native-gate trainer can optionally write complete co-trained MLP tensors, because a
changed basis cannot be represented by router deltas alone. It also supports device-neutral
full-weight/optimizer checkpoints for time-sliced CPU training. Neither checkpoint nor final
safetensors output is a compiler input until the hard-path causal gate passes on held-out data.

The predictor-free DIP-inspired selector is the materially different arm that changes the quality
decision. Top-magnitude input pruning and partial scoring come from DIP; candidate-only exact
completion and contribution reranking are Engram extensions. After choosing 75% input coordinates,
896 candidates, and K=768 on the development grid, the fixed configuration passes again on a
sequence-disjoint confirmation corpus. `--evaluation-role confirmation` requires
`--configuration-selection-traces` and rejects any exact token-sequence overlap.
`engram build-dip-package` now writes a checksummed, mmap-friendly version-2 coordinate-major
research package, and the native benchmark implements candidate-gather and full-coordinate-stream
completion. An opt-in version-3 dual layout exists only to reproduce a rejected record-major
experiment: it adds 66.7% MLP storage and is slower than v2. Neither format is a default `.engram`
compiler input. The quality pass authorized this systems experiment; its measured latency failure
does not authorize compiling the old learned-router artifacts or claiming a runtime speedup.

The later compact-Q4 and output-memory branches do not change that compiler
decision. The compact artifact is physically valid at 44.9334% of dense ideal
Q4 but fails every causal metric at its frozen 3M-position stop. The exact
output-memory pilot fails its layer-local progression rule before Q4, indexing,
traffic accounting, or causal substitution. Both remain research evidence,
not package inputs.

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

For the qualified native-BitNet source track, the separate compiler and
runtime commands are:

```bash
engram compile-native-bitnet \
  --model microsoft/bitnet-b1.58-2B-4T \
  --artifact work/native_bitnet/model.bitnet-records.bin \
  --out work/native_bitnet/model.engram-bitnet

engram validate --model work/native_bitnet/model.engram-bitnet
engram generate-native-bitnet \
  --model work/native_bitnet/model.engram-bitnet \
  --prompt "The capital of France is" --max-tokens 2 \
  --bounded-attention \
  --native-projections \
  --library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so

engram benchmark-native-bitnet-generation \
  --model work/native_bitnet/model.engram-bitnet \
  --out reports/generated/native-generation.json \
  --lengths 33 128 256 --max-tokens 2 \
  --mlp-library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so \
  --native-projections

engram chat-native-bitnet \
  --model work/native_bitnet/model.engram-bitnet \
  --library build/libengram_bitnet.so \
  --attention-library build/libengram_attention.so \
  --threads 12 --max-tokens 32
```

The compiler copies only config/tokenizer assets and non-MLP tensors, embeds
the packed phase-stream artifact, and seals the result with checksums. The
runtime creates the transformer without initially allocating parameters,
loads the packaged non-MLP state, installs the native MLP module in every
layer, rejects any remaining unmaterialized parameter, and then performs
autoregressive generation. With `--bounded-attention`, it creates one
persistent native attention state per layer, applies RoPE from explicit
absolute positions, and keeps the Transformers dense KV cache disabled. It
does not consult the source checkpoint directory after compilation.
`--native-projections` additionally executes the packaged official Q/K/V/O
ternary tensors without expanding them to BF16 matrices.

`chat-native-bitnet` always enables packed native projections and bounded
attention. It uses the packaged tokenizer's chat template and re-prefills the
complete structured conversation on every turn. The supported session
commands are `/history`, `/reset`, `/quit`, and `/exit`. Persistent cross-turn
cache reuse and token streaming are intentionally deferred until this
re-prefill implementation has broader behavioral validation. The current
turn lifecycle is:

1. append the new user message to structured conversation history;
2. render all system, user, and assistant messages with the packaged template;
3. reset the bounded native attention states and prefill the rendered tokens
   from absolute position zero;
4. decode greedily while advancing RoPE and cache positions;
5. decode and append the assistant response to history.

A two-turn 32-token example took 166.43 and 153.15 seconds. The second answer
acknowledged the first turn before starting another poem, while both turns
reported the same 7,477,440-byte attention state.
