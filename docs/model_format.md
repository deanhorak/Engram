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

The standalone trained-controller artifact currently uses
`engram.controller.factorized_residual` schema version 1:

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
metadata-derived shape disagreement. This artifact is independently runnable
on CPU without Torch or CUDA. It is not yet embedded in
`model.engram-bitnet`; package integration waits for corpus-scale trajectory
quality and compiled semantic/episodic substitution.

An `.engram` package is a model worker, not a system-agent snapshot. It does not contain executive
goal graphs, durable user memory, tool credentials, worker registries, or Cognitive Executive
policy.

The separate experimental DIP format defaults to version 2. Each layer stores float32
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
  transformer/non_mlp.safetensors
  mlp/model.bitnet-records.bin
  tokenizer/{tokenizer.json,tokenizer_config.json,...}
  config.json
  generation_config.json
```

The `engram-native-bitnet` version-1 manifest pins the source revision and
source checkpoint hash, records every packaged file's byte count and SHA-256,
and declares the native MLP format. Compilation rejects an unpinned source,
an unexpected artifact digest, and any `.mlp.` tensor that crosses into
`non_mlp.safetensors`. Validation confines paths to the package, verifies all
checksums, reloads the packed MLP structure, and rechecks that tensor boundary.
The checked package contains 332 non-MLP tensors and no source MLP tensor.

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
