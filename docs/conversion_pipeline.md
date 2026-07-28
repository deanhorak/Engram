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

OLMoE also uses a distinct source adapter because its per-layer MLP is a
learned router over separately stored SwiGLU experts, not one dense Llama MLP:

```bash
PYTHONPATH=src python -m engram.cli audit-olmoe \
  --model allenai/OLMoE-1B-7B-0125 \
  --revision 9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --verify-remote-shapes \
  --out reports/olmoe_source_audit_2026-07-27/audit.json
```

Without `--verify-remote-shapes`, the command downloads only `config.json` and
`model.safetensors.index.json`. With the flag, it requests only bounded
safetensors header ranges, rejects any response that would stream a full
shard, and validates all tensor shapes. It still downloads no weight payload.
A deterministic fixture can exercise the same router/expert contract:

```bash
PYTHONPATH=src python -m engram.cli create-olmoe-fixture \
  --out work/fixtures/tiny-olmoe
PYTHONPATH=src python -m engram.cli trace-olmoe-fixture \
  --model work/fixtures/tiny-olmoe \
  --out work/traces/tiny-olmoe-router
```

The fixture proves serialization and exact decomposition only. The official
source now also passes a trained all-layer Q7/group-64 causal confirmation:

```bash
PYTHONPATH=src python -m engram.cli evaluate-olmoe-quantized-causal \
  --model work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --dataset tests/fixtures/confirmation_expanded.jsonl \
  --out reports/olmoe_q7_confirmation_2026-07-27/result.json \
  --samples 8 --max-tokens 33 --bits 7 --group-size 64 --threads 12
```

The run passes the 8-sequence/256-position causal thresholds and projects
22.7865% complete expert/router traffic relative to all-expert ideal Q4. The
systems follow-up compiles and confirms the physical artifact:

```bash
PYTHONPATH=src python -m engram.cli repack-olmoe-q7 \
  --model work/huggingface/models--allenai--OLMoE-1B-7B-0125/snapshots/9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --out work/olmoe_q7/model.engram-olmoe-q7 --group-size 64
PYTHONPATH=src python -m engram.cli evaluate-native-olmoe-q7 \
  --artifact work/olmoe_q7/model.engram-olmoe-q7 \
  --library build/libengram_olmoe_q7.so \
  --out reports/olmoe_q7_native_systems_2026-07-27/result.json
```

The 5,842,733,184-byte artifact and direct CPU top-eight kernel pass route,
output, and scheduled-byte parity. This qualifies the OLMoE MLP artifact for
package integration. The next implemented commands build and exercise that
token boundary:

```bash
PYTHONPATH=src python -m engram.cli repack-olmoe-non-mlp \
  --model "$OLMOE_SNAPSHOT" --out work/olmoe_q7/non_mlp.safetensors
PYTHONPATH=src python -m engram.cli run-native-olmoe-token \
  --config "$OLMOE_SNAPSHOT/config.json" \
  --non-mlp work/olmoe_q7/non_mlp.safetensors \
  --q7-artifact work/olmoe_q7/model.engram-olmoe-q7 \
  --library build/libengram_olmoe_token_runtime.so \
  --prompt "The capital of France is" \
  --tokenizer "$OLMOE_SNAPSHOT" --threads 12
```

This is a complete native token/generation path.
`compile-native-olmoe` now emits its symlink-free, exact-inventory package
authenticated by an externally supplied manifest SHA-256.
`generate-native-olmoe-package` validates that root before loading the
package-owned tokenizer, config, non-MLP mapping, and Q7 artifact.

The complete package is qualified against an untouched BF16 teacher in two
stages. The short generation integration remains inside W=16; the causal
protocol scores 128 exact-local and 128 post-window positions independently.
That formal 8×32 result qualifies the Q7 semantic package. A later authenticated
8×128 test fails under W16/C8/K4/S2 after offset 31, while a matched control
that changes only W16 to exact W128 attention passes every band. The conversion
and Q7 route are therefore not the sustained failure source; the remaining
OLMoE blocker is a deployable Milestone 3 attention policy below 45% logical
reads. W128 is a 100%-read diagnostic, not a package default.

The first matched-budget follow-up is also complete. It evaluated
`W16/C18/K16/S2`, `W24/C10/K8/S2`, and `W30/C4/K2/S2` in a fixed order at
exactly 968,753,152 logical read bytes per sequence (44.7614%) and 32 mature
visible values. Every arm passed its evidence contract, but their respective
overall KL/top-1/NLL-delta/hidden-L2 results were
0.063887/0.867188/+0.051701/0.157717,
0.065912/0.877930/+0.058480/0.159755, and
0.095813/0.840820/+0.075728/0.188422. None passed, so the frozen ranking rule
selected no arm and the reserved confirmation corpus was not consumed.

This was a development intervention, not a conversion-format change. The
evaluator constructed the raw native token runtime with each candidate policy
because the authenticated package intentionally remains immutable at
`W16/C8/K4/S2`. No candidate was installed into the package, and no model
format was promoted. That result justified the layer-adaptive experiment
below; any eventual attention policy must still pass development and fresh
confirmation before it can enter the package schema.

The layer-adaptive upper-bound experiment is now complete and negative. The
native library gained an additive layered-open ABI and the Python wrapper can
pass one W/C/K/S policy for each of OLMoE's 16 layers. Exact all-base parity
against the historical scalar ABI passed before candidate execution. A frozen
three-round greedy search then evaluated 45 candidates on two selection
sequences and chose layers 11, 6, and 10 for `W128/C8/K4/S2`, leaving the
other 13 at `W16/C8/K4/S2`. The resulting schedule used 955,957,248 logical
attention-read bytes per sequence, or 44.1701% of dense attention.

Every candidate/resource contract, replay check, and authentication root
passed, but the six-sequence internal screen failed semantic quality:
KL/top-1/NLL-delta/hidden-L2 were
0.102321/0.845052/+0.116776/0.206037. The 0–15 and 16–31 bands passed; all
four metrics failed in every band from position 32 onward. This valid negative
result closes the frozen greedy three-layer W128 path under the 45% budget; it
does not rule out every interacting whole-layer combination. It does not
change the passed Milestone 2 Q7 result, consume a fresh confirmation corpus,
or unblock Milestone 3.

This was again a raw-runtime intervention. The version-1 package still binds
one global `W16/C8/K4/S2` policy, and neither the selected layer schedule nor
the layered or head-wise ABIs have been promoted into its manifest schema. The
subsequent prospectively frozen teacher-attention-mass mask rescued 51 of 256
layer-head pairs at 973,384,704 logical bytes per sequence (44.9754%); a
52-head mask would require 45.2438% and is outside the budget. All execution
evidence passed, but the six-sequence internal semantic screen failed every
overall threshold, so the mask was not promoted and no fresh confirmation was
run. A causal/value-sensitive or dynamic head policy must first pass
development and then fresh confirmation before a package-format change is
justified.

The first command below is the historical sealed-reference reproduction, so
it explicitly retains the original serial sequence policy:

```bash
PYTHONPATH=src python -m engram.cli capture-olmoe-teacher-causal \
  --model "$OLMOE_SNAPSHOT" \
  --dataset tests/fixtures/confirmation_expanded.jsonl \
  --out work/olmoe_q7/teacher-causal.json \
  --arrays-out work/olmoe_q7/teacher-causal.npz \
  --sequences 8 --tokens-per-sequence 33 --threads 12 \
  --batch-size 1 --sequence-workers 1

PYTHONPATH=src python -m engram.cli evaluate-native-olmoe-causal \
  --package work/olmoe_q7/package \
  --manifest-sha256 861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db \
  --library build/libengram_olmoe_token_runtime.so \
  --dataset tests/fixtures/confirmation_expanded.jsonl \
  --teacher-reference work/olmoe_q7/teacher-causal.json \
  --teacher-arrays work/olmoe_q7/teacher-causal.npz \
  --protocol reports/olmoe_q7_native_causal_2026-07-28/frozen_protocol.json \
  --protocol-sha256 db41e8e6bd8f769acb9d7012354c8d983daa2da0790b6c0b203096c3438a3164 \
  --out reports/olmoe_q7_native_causal_2026-07-28/result.json --threads 12
```

The frozen run passes overall and on both halves. It used serial teacher
sequences (`--batch-size 1 --sequence-workers 1`). Future CPU captures default
to four concurrent, read-only sequence forwards through one shared model. On
this host that path is byte-identical and 3.86× faster in teacher compute.
Capture-only expert threading is also available experimentally, but changes
BF16 rounding and must not silently replace an existing sealed reference.
For a new, unsealed CPU capture, omit `--sequence-workers 1` (or pass
`--sequence-workers 4`) to use that safe default.

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
cmake --build build-runtime \
  --target engram-bitnet-token-generate engram_bitnet_token_runtime -j

PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-dip-token-generation \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --executable build-runtime/engram-bitnet-token-generate \
  --prompts tests/fixtures/inference_prompts.jsonl \
  --reference reports/controller_cpp_stage_runner_2026-07-26/frozen_8x4.json \
  --package-manifest-sha256 707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926 \
  --executable-sha256 c6c5b05b6d8be72edd7f9e12e5e66c615859b74268143a5b2023b8dae423a15b \
  --out reports/native_bitnet_dip_attention_confirmation_2026-07-27/frozen_8x4.json \
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
  --model work/native_bitnet/model.engram-bitnet-dip \
  --library build-runtime/libengram_bitnet_token_runtime.so \
  --threads 12 --max-tokens 32
```

The legacy generation/evaluation commands use a Transformers model shell.
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

`chat-native-bitnet` no longer uses that model shell. It opens only the
authenticated DIP package through `libengram_bitnet_token_runtime.so`, while
Python loads the packaged tokenizer locally. The package fixes packed
projections, semantic routing, and bounded attention; there are no CLI policy
overrides. The supported session commands are `/history`, `/reset`, `/quit`,
and `/exit`. The current turn lifecycle is:

1. append the new user message to structured conversation history;
2. render all system, user, and assistant messages with the packaged template;
3. reset the versioned native handle and prefill the rendered tokens from
   absolute position zero;
4. decode greedily while advancing RoPE and cache positions;
5. decode and append the assistant response to history.

A real default-system `Hello` smoke rendered 17 prompt tokens, generated
`Hello` in 5.16 seconds, and reported 7,477,440 attention-state bytes. A
same-handle reset replay reproduced the raw token and structural metrics. The
older two-turn 166.43/153.15-second transcript belongs to the retired
Transformers shell and has not yet been repeated as a scripted multi-turn DIP
confirmation. Persistent cross-turn cache reuse and token streaming remain
deferred.
