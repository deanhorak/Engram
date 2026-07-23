# Serialized DIP native benchmark

The version-2 coordinate-major native kernel has exact selected-ID parity with the float32 Python
reference on a real SmolLM2 layer, but it does **not** pass the performance gate.

The final benchmark cycles through all 30 serialized layers (305 MiB, larger than the host's
30 MiB L3) and alternates sparse-first/dense-first order across six 20-pass runs. At the fixed
confirmation setting `q=432 / C=896 / K=768`:

| Completion kernel | Sparse 30-layer pass | Dense pass | Dense / sparse |
|---|---:|---:|---:|
| Sorted candidate gather | 41.338 ms | 31.871 ms | 0.770x |
| Full omitted-coordinate stream | 37.673 ms | 32.639 ms | 0.863x |

The streamed kernel is about 15.4% slower than dense. It wins over gather because 896/1,536
candidates touch every 64-byte line: sequentially processing the complete omitted-coordinate rows
is faster than gathering candidate scalars. Consequently its executed weight count is 8,847,360
bytes per layer, 83.33% of dense. The candidate-only logical count remains 8,110,080 bytes
(76.39%), but is not the physically executed count for the fastest kernel.

This rejects integration into the default runtime. More selection micro-optimization cannot
recover a robust systems win at this traffic ratio. Further work requires a materially smaller
quality-preserving `C/K` budget, temporal cross-token cache reuse measured on target hardware, or
a different sparse representation.
