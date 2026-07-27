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

The older dense-SmolLM predictor-free DIP-inspired selector was the materially
different arm that changed that study's quality decision. Top-magnitude input
pruning and partial scoring come from DIP; candidate-only exact completion and
contribution reranking are Engram extensions. After choosing 75% input coordinates,
896 candidates, and K=768 on the development grid, the fixed configuration passes again on a
sequence-disjoint confirmation corpus. `--evaluation-role confirmation` requires
`--configuration-selection-traces` and rejects any exact token-sequence overlap.
`engram build-dip-package` now writes a checksummed, mmap-friendly version-2 coordinate-major
research package, and the native benchmark implements candidate-gather and full-coordinate-stream
completion. An opt-in version-3 dual layout exists only to reproduce a rejected record-major
experiment: it adds 66.7% MLP storage and is slower than v2. Neither format is a default `.engram`
compiler input. The quality pass authorized this systems experiment; its measured latency failure
does not authorize compiling the old learned-router artifacts or claiming a runtime speedup.
This format is not the newer packed native-BitNet DIP index.

## Native-BitNet practical semantic progression

The native-BitNet path starts from the checksummed five-trits-per-byte
phase-stream artifact, then builds a separate source-bound coordinate-major
index. The selected policy must specify all 30 layers explicitly: `q`, `C`,
`minK`, `Kmax`, energy target, RMS estimator, and any audit strategy. The
index embeds those fields in the layer checksum and embeds the base record
artifact SHA-256 in its global header.

Policy selection is permitted only on declared development corpora and must
use the live CPU native kernel at BF16 MLP boundaries. Float16 cached traces
may propose a schedule, but they cannot approve it because they do not
reproduce native BF16 quantization and accumulation. A qualifying development
report must show:

- all 30 MLPs substituted with no dense fallback;
- at least 8 unique sequences and 256 positions;
- KL <= 0.05, top-1 >= 0.90, NLL delta <= +0.05, and final-hidden
  relative L2 <= 0.10;
- mean active records <= 25%;
- v2 modeled cache-line traffic <= 45% of dense ideal Q4;
- global micro and every-layer candidate recall >= 95%;
- independently reloaded source-bound index; and
- Python/native bit parity on route identity, selected counts, and output.

The frozen configuration passes that development gate at KL 0.0044707,
top-1 0.94921875, NLL +0.0013609, hidden L2 0.0498965, active fraction
0.2008072, modeled traffic 0.409639, global recall 0.9995917, and worst-layer
mean recall 0.9939353. Six rows per layer have bit-exact parity.

Passing development froze every policy and artifact binding. The final runner
then consumed the independent plaintext holdout in the single authorized
model attempt, using the same commit, artifact, index, policy, tokenizer,
libraries, protocol, and dataset hashes. The raw evaluator passed every
threshold: KL 0.00404129, top-1 0.98828125, NLL +0.00482893, hidden L2
0.0477494, active fraction 0.2138001, modeled traffic 0.4113713, global recall
0.9994058, and worst-layer mean recall 0.9939429.

The original runner result is nevertheless an error, not a pass. Its
post-evaluation verifier compared full-record hashes using the canonical
`input_ids` object envelope with first-33-token evaluator hashes using a bare
list envelope. A no-model postmortem adjudicator corrected that hash contract
and verified the preserved raw report and frozen evidence, so the
native-BitNet semantic gate is **passed by adjudication**. The raw report was
prospectively sealed about 13 minutes after the error and was not
contemporaneously bound by the original result. The fixture remains
plaintext/procedurally separated, artifacts are host-bound, and broader
replication is still required. This decision does not make the native route a
generic dense-Llama compiler input or prove every Milestone 2 deliverable
complete.

### Promote the frozen native-BitNet semantic memory

The passing decision authorizes a derived package; it does not authorize
modifying the source package bound into the frozen policy. Install the
authenticated DIP v2 index with:

```bash
PYTHONPATH=src python -m engram.cli install-native-bitnet-semantic-memory \
  --model work/native_bitnet/model.engram-bitnet \
  --index work/native_bitnet/model.provisional.bitnet-dip-index.bin \
  --policy reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json \
  --adjudication reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.adjudication.json \
  --out work/native_bitnet/model.engram-bitnet-dip \
  --index-sha256 b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15 \
  --policy-sha256 c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e \
  --adjudication-sha256 ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc
```

The installer first verifies the source package's exact manifest inventory.
It then checks that the policy binds that package, the base record artifact
`4fcf598a…ab55`, and index `b98ce4e4…0e15`, and that adjudication
`ebb5ca95…a5cc` is a passing Milestone-2 decision over the same inputs. A new
directory is staged atomically; its manifest declares the v2 DIP operator,
all-layer substitution, CPU-only execution, and no dense fallback. Repeating
the command with identical inputs validates and reuses the derived package.
Conflicting inputs or an attempt to target the frozen source directory fail.

Build and confirm the native token runtime:

```bash
cmake -S . -B build-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build-runtime --target engram-bitnet-token-generate -j

PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-dip-token-generation \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --executable build-runtime/engram-bitnet-token-generate \
  --prompts tests/fixtures/inference_prompts.jsonl \
  --reference reports/controller_cpp_stage_runner_2026-07-26/frozen_8x4.json \
  --package-manifest-sha256 707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926 \
  --executable-sha256 0f6cf41c9c14dc3e05a8cad7a01f4f9909bd355f4e27f9296d6c1e15ba91dea4 \
  --out reports/native_bitnet_dip_token_runtime_2026-07-26/integrated_8x4.json \
  --max-tokens 4 --threads 12 --timeout 300
```

The C++ runtime maps the package weights, base records, v2 policy/index,
controller, and attention state itself. Its layer loop directly dispatches
attention → semantic input → DIP → semantic acceptance and never constructs a
dense semantic backend. Before mapping, the standalone executable
authenticates the exact manifest and symlink-free inventory against compiled
source/package/artifact/policy/adjudication trust roots. It derives runtime
architecture, paths, bounds, attention policy, and EOS IDs from the package
and has no Engram shared-library dependency.

The checked non-holdout 8×4 run has 32/32 greedy token matches and 8/8 exact
prompts. Global/maximum-prompt activity is 21.56017%/22.58916%, while
global/maximum-prompt complete modeled traffic is 41.16116%/41.29835%.
Complete modeled cold traffic is 30,153,074,432 bytes, including 194,304 bytes
of global metadata. Reset repeats token IDs, zeroes counters, and reproduces
structural metrics; it is not hidden-state parity. All contexts are at most 14
positions, so W=16 eviction and older retrieval are not tested.

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

These Python generation/chat commands currently target the original
`model.engram-bitnet` package. They deliberately reject
`model.engram-bitnet-dip`, because that shell does not implement the
authenticated DIP backend and must not substitute dense MLPs implicitly. The
derived package is currently driven by
`build-runtime/engram-bitnet-token-generate`; a native-runtime C/Python
binding is the next step toward DIP-backed chat.
