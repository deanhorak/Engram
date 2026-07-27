# Model format

Format support is ahead of model quality. The compiler can write and validate
the package described here, but no trained SmolLM2 semantic artifact is
currently eligible for the default format because none passes both causal
quality and the 45% physical cold-traffic gate. See
[Project status](status.md).

An Engram package is an inspectable directory with a versioned root `manifest.json`:

```text
model.engram/
  manifest.json
  tokenizer/
  embeddings/token_embeddings.npy
  controller/{metadata.json,*.npy}
  semantic/{manifest.json,layer-*/...}
  episodic/config.json
  vocabulary/{embeddings.npy,index.npy,ivf/...}
  transitions/config.json
  corrections/capsules.json
  metrics/{stage_manifest.json,conversion_report.json}
```

The root manifest records package/source versions and hashes, compile options, runtime policies,
source-transformer independence, and every file's byte count and SHA-256. Every NPY descriptor
also records dtype, shape, byte order, format version, Fortran-order flag, payload offset, and
actual alignment. Semantic submanifests repeat array-level metadata for independent inspection.

The installed controller artifact uses
`engram.controller.factorized_residual` schema version 3:

```text
controller/
  metadata.json
  input_down.npy
  recurrent_down.npy
  gate_up.npy
  bias.npy
  stage_embeddings.npy
  adapter_down.npy
  adapter_up.npy
  step_scale.npy
```

All tensors are little-endian-compatible FP32 NumPy arrays. Metadata records
the input/state dimensions, shared bottleneck rank, stage count, adapter rank,
parameter count, serialized bytes, per-tensor shapes, and the required
per-token RMS state normalization. The loader rejects missing/extra tensors or
metadata-derived shape disagreement. Schema 3 preserves known
attention/semantic operator additions exactly and applies the factorized
controller only as an optional correction. The authenticated controller is
embedded in `model.engram-bitnet` and copied into the derived DIP package; it
is independently runnable on CPU without Torch or CUDA.

An `.engram` package is a model worker, not a system-agent snapshot. It does not contain executive
goal graphs, durable user memory, tool credentials, worker registries, or Cognitive Executive
policy.

The older dense-SmolLM experimental DIP format defaults to version 2. Each layer stores float32
`gate_coordinates.npy` and `up_coordinates.npy` as `[hidden,records]` so selected input
coordinates are sequential scans, `down_rows.npy` as `[records,hidden]`, precomputed
`value_norms.npy`, and a native-readable `uint32[4]` `config.npy`. `metadata.json` checksums every
array. This format is deliberately not part of the default `.engram` manifest: its native kernel
is parity-correct but the best checked 30-layer implementation is still 15.4% slower than dense.
An opt-in version-3 diagnostic duplicates gate/up weights in record-major order. It increases MLP
storage by 66.7% and both tested record-major completion kernels are slower, so
`engram build-dip-package` emits v2 unless `--dual-layout-experimental` is supplied. Python loads
both versions and verifies metadata/checksums; the native benchmark loader validates the binary
configuration and array shapes but currently trusts files from the checked package boundary.

The separate experimental grouped-ternary file uses
`engram_budget_native_grouped_ternary_v1`. It has a fixed header, one canonical
directory entry per transformer layer, cache-aligned layer blocks, and three
packed projection payloads plus FP16 group scales per layer. Each byte stores
five base-3 digits mapped to `-1/0/+1`; scales use groups of 128 weights.
Loading validates version, dimensions, offsets, padding, code range, scale
finiteness, and exact file length before decode. The checked 30-layer
SmolLM2 file is 17,173,504 bytes. This format is deliberately not embedded in
the default `.engram` manifest because its trained artifact fails causal
quality.

The separate native-BitNet file uses
`native_bitnet_phase_base3_v1`. Its fixed header records layer count,
dimensions, cache-line size, logical record size, and RMS-normalization epsilon. A
canonical 32-byte directory entry locates each cache-aligned layer block.
Each layer begins with a header and three BF16 projection scales. Four
cache-aligned, fixed-stride streams follow:

```text
base3(gate rows) | base3(up rows) | BF16 norm gains | base3(down columns)
```

Channel `j` remains O(1)-addressable through four base pointers, so the
logical record is still `(gate[j], up[j], gain[j], down[:, j])`. Physical
phase segregation is necessary because the shared RMS normalization separates
gate/up computation from gain application and down projection. An interleaved
record layout would reread cache lines across those phases; the phase layout
can stream each serialized line once.

Each base-3 byte stores five digits and rejects values above 242 or nonzero
tail digits. For the pinned 2B4T model, each vector occupies 512 bytes and the
logical record is 1,538 bytes. Loading validates bounded dimensions,
header-field representability, headers, offsets, zero padding, scales, gains,
and every code stream before exposing a layer. The checked 30-layer artifact
is 318,924,544 bytes with SHA-256
`4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55`.
The direct CPU kernel memory-maps this file, validates the complete structure,
and consumes all four packed streams without materializing dense weights. The
frozen confirmation passes. Native BitNet therefore uses a separate
source-family package boundary rather than pretending this artifact is generic
SwiGLU semantic memory:

```text
model.engram-bitnet/
  manifest.json
  config/config.json
  transformer/non_mlp.safetensors
  mlp/model.bitnet-records.bin
  controller/{metadata.json,*.npy}
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  tokenizer/special_tokens_map.json
  tokenizer/generation_config.json
```

The `engram-native-bitnet` version-1 manifest pins the source revision and
source checkpoint hash, records every packaged file's byte count and SHA-256,
and declares the native MLP format. Compilation rejects an unpinned source,
an unexpected artifact digest, and any `.mlp.` tensor that crosses into
`non_mlp.safetensors`. Validation confines paths to the package, verifies all
checksums, reloads the packed MLP structure, and rechecks that tensor boundary.
The checked package contains 332 non-MLP tensors and no source MLP tensor.

## Native-BitNet DIP coordinate index

Practical native-BitNet routing adds a companion
`engram-native-bitnet-dip-index` version-2 binary. It does not replace the
318,924,544-byte base record artifact. Instead, it duplicates gate and up
weights in coordinate-major packed-base-3 order so the selected input
coordinates can be streamed across all 6,912 records, and stores the
down-column nonzero-count proxy needed for candidate utility.

The global 128-byte header records the version, endian marker, dimensions,
cache-line size, encodings, directory layout, and SHA-256 of the exact base
record artifact. Each 128-byte layer header plus directory entry authenticates
the input-coordinate count, candidate count, minimum and maximum adaptive K,
energy target, RMS estimator, audit strategy/count, payload offsets, and a
policy-plus-payload SHA-256. Gate/up coordinate rows use a 1,408-byte stride:
the canonical base-3 payload is padded to complete 64-byte lines. Load rejects
noncanonical trits or tails, nonzero padding, invalid offsets, inconsistent
policy bounds, unsupported RMS modes, checksum failure, and a base-artifact
hash mismatch.

The frozen 30-layer index is 216,688,448 bytes. Together with the base records,
the semantic MLP storage is 535,612,992 bytes, or 67.2659% of the dense-Q4
reference. That stored-size fraction must not be confused with the per-token
modeled cold-traffic result of 40.9639%: the coordinate layout duplicates data
to make sparse access possible, while one token touches only its selected
coordinate, completion, gain/norm, and down lines.

Version 2 embeds all effective policy fields. Every layer uses `q=1920`,
`minK=346`, and energy target 1.0; C and Kmax are per-layer. Layers 0–8 and
10–29 use candidate-ratio RMS with no audit. Layer 9 uses corrected-proxy RMS
and an eight-record top-proxy-raw-square audit inside its fixed candidate
union. The [frozen manifest](../reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json)
binds this index, base artifact, native libraries, package/tokenizer, protocol,
development report, and parity report. Those exact bindings were used for the
consumed final attempt. Its raw evaluator report passes every semantic,
activity, recall, and modeled-traffic threshold; the semantic gate is accepted
by postmortem adjudication because the original wrapper's
full-record/object-versus-33-token/list hash check failed after evaluation.
This evidence is host-bound and does not promote the layout into the generic
dense-Llama package format.

## Derived native-BitNet DIP package

The frozen source package is immutable evidence. Semantic promotion creates a
new directory:

```text
model.engram-bitnet-dip/
  manifest.json
  config/config.json
  transformer/non_mlp.safetensors
  mlp/model.bitnet-records.bin
  mlp/model.bitnet-dip-index.bin
  controller/{metadata.json,*.npy}
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  tokenizer/special_tokens_map.json
  tokenizer/generation_config.json
```

The installer verifies the source package's exact file inventory, frozen
policy SHA-256
`c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e`,
passing adjudication SHA-256
`ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc`,
base artifact SHA-256
`4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55`,
and index SHA-256
`b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15`.
It refuses to use the frozen source directory as its output.

The promoted derived manifest is 5,787 bytes with SHA-256
`707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926`.
The standalone native token executable has SHA-256
`29526c9838ea484d8a21887dafeaba99a57348e7377e0de4138e0631dde10fad`.
The versioned native chat/token-runtime DSO has SHA-256
`df3a4f70952cddaebff2e5198d9ddf6b5e8a25487020c40b89ec99f2c7d33f96`.
The manifest byte count/SHA and source-package/semantic provenance hashes are
compiled into the promoted native loader. The Python confirmation harness
separately pins the reviewed standalone executable SHA. The chat DSO is a
narrow deployment interface: it has SONAME
`libengram_bitnet_token_runtime.so.1`, exports six versioned C symbols, and
depends only on the system C and math libraries and ELF loader. Before model
mapping the loader
requires the exact symlink-free manifest inventory, hashes every file, and
cross-checks every descriptor. It then derives dimensions, head layout,
context and vocabulary bounds, attention policy, RoPE/RMS values, file paths,
and EOS IDs from `config/config.json`, the manifest, controller metadata, and
`tokenizer/generation_config.json`. The EOS set must uniquely include
`128001` and `128009`. The executable directly links its kernel objects and
does not require an Engram shared library.

The derived manifest adds a `semantic_memory` descriptor containing:

- operator `native_bitnet_dynamic_input_pruning_v2`;
- runtime scope `native_token_runtime`;
- index path, format, version, size, and SHA-256;
- source artifact and source package-manifest hashes;
- frozen policy and adjudication hashes;
- all-layer substitution, CPU-only, and `dense_fallback: false` declarations;
  and
- `modelled_cache_line_v2` traffic accounting.

The root `runtime.mlp_mode` repeats the operator. Validation requires exact
agreement among the descriptor, root inventory, index's internal payload hash,
base-artifact binding, dimensions, and layer count. A DIP package is therefore
not accepted by the older `NativeBitNetRuntime` Transformers-shell class.
`NativeBitNetDIPTokenRuntime` instead owns a persistent handle to the native
token DSO while Python performs only authenticated tokenization, chat-template
rendering, and history bookkeeping. This separation prevents a silent dense
fallback.

Each compiled semantic layer contains `quantized/ivf/centroids.npy` (`float32` joint gate/up
centroids), `posting_offsets.npy` (`uint32` CSR offsets), and `posting_indices.npy` (`uint32`
record IDs), plus versioned IVF metadata.

Vocabulary projection retains exact `embeddings.npy` for candidate rescoring, normalized
`index.npy` proxy rows, and `ivf/{centroids.npy,posting_offsets.npy,token_ids.npy,metadata.json}`.
The runtime scans coarse centroids and only the normalized rows in selected postings; exact
full-vocabulary scoring remains available when required.

NPY payloads are little-endian C-order arrays. The native loader accepts format versions 1 and 2,
uint8/uint32/float32/float64 payloads, validates sizes and shapes, and can expose read-only memory maps. JSON
parsing and SHA-256 verification are implemented in-tree without a runtime dependency.

`engram build-semantic` retains exact reference arrays beside quantized variants for research
comparison. Full `engram compile` packages are quantized-only: production runtimes memory-map
key codes, affine parameters, value codes, and additive codebooks without retaining original
float gate/up/value matrices. The IVF runtime index retains only coarse float32 centroids and
CSR postings, not duplicate full-precision record keys.
