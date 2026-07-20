# Model format

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
