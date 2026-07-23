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
