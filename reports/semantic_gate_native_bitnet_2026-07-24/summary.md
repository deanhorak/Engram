# Native BitNet direct-kernel confirmation

Date: **2026-07-24**

Decision: **the separate low-bit-native BitNet track passes the Milestone 2
semantic and serialized cold-traffic gate.** This is not a result for
post-hoc conversion of a dense Llama checkpoint.

The evaluated source is `microsoft/bitnet-b1.58-2B-4T` pinned to revision
`04c3b9ad9361b824064a1f25ea60a8be9599b127`. The official weight SHA-256 is
`8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
The independently reloaded phase-stream artifact is 318,924,544 bytes with
SHA-256
`4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55`.

## Result

The frozen protocol used the first eight unique records in
`tests/fixtures/confirmation_expanded.jsonl`, the pinned model tokenizer with
its Mistral-regex compatibility fix, 33 tokens per sequence, and therefore
exactly 256 next-token prediction positions. The dense reference and packed
candidate both ran on CPU in BF16; the RTX 3050 was not used.

| Metric | Threshold | Result |
|---|---:|---:|
| Teacher-to-student KL | at most 0.05 | 0.003710 |
| Teacher top-1 agreement | at least 0.90 | 0.960938 |
| NLL delta | at most +0.05 | +0.002237 |
| Final hidden relative L2 | at most 0.10 | 0.046775 |
| Unique sequences | at least 8 | 8 |
| Prediction positions | at least 256 | 256 |
| Complete cold MLP bytes / dense ideal Q4 | at most 45% | 40.052694% |

All frozen checks passed. The complete machine-readable evidence is
[kernel_confirmation.json](kernel_confirmation.json).

## What executed

`libengram_bitnet.so` memory-mapped the serialized artifact and consumed its
packed base-3 gate, up, gain, and transposed-down streams directly. It did not
materialize a dense weight matrix. The kernel performed per-token Q8 input
quantization, ternary gate/up accumulation, ReLU-squared activation,
intermediate RMS normalization and gain, a second Q8 quantization, and the
ternary down projection. A persistent 12-thread pool processed all 30 MLPs.

For the 264-row confirmation batch, the 30 measured MLP calls took 9.737
seconds in total, or 324.6 ms per layer, and the largest activation scratch
allocation was 34,603,008 bytes. The complete model forward with substituted
MLPs took 256.19 seconds versus 722.94 seconds for the dense-BF16 reference on
this host. That 2.82x whole-forward ratio is descriptive, not a portable
benchmark: the local PyTorch build reports no vector CPU capability and the
large vocabulary head dominates substantial time.

## Parity interpretation

The deterministic tiny artifact test is bit-exact against the independent
dense artifact oracle. On official layers 0, 14, and 29, the direct kernel is
not bit-identical to PyTorch's dense BF16 GEMM because their floating-point
reduction orders differ. Measured relative L2 is respectively 0.009819,
0.008900, and 0.006841. The all-layer causal confirmation above is the
quality-bearing parity result and passes every predeclared semantic threshold.

The 318,924,544-byte numerator is the exact cache-line schedule of the
memory-mapped serialized streams, including headers, directory, scales,
normalization gains, and padding. No hardware DRAM counter was available, so
this is exact scheduled cold traffic rather than a measured memory-controller
event count.

## Consequence

Milestone 2 now has a qualifying path when the source model is natively
low-bit BitNet. The original dense-Llama conversion branch remains blocked.
The next engineering step is to promote this kernel and artifact into an
end-to-end package/runtime boundary with generation tests; the next scientific
milestone is attention substitution, not another MLP-router search on this
source track.
