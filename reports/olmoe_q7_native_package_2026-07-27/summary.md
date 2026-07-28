# Authenticated native OLMoE package and single-row Q7 boundary

The version-1 `engram-native-olmoe-q7` package is complete. It contains the
packed expert/router artifact, mapped non-MLP tensors, configuration, and
tokenizer as seven regular files totaling 6,795,550,536 bytes. Its external
manifest authentication root is
`861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`.

The loader verifies that root and the exact symlink-free inventory before
opening native weights. Unit tests prove content tampering is rejected.
Package-only CPU generation, with no Transformers model shell, tokenizes
`The capital of France is` as `[510, 5347, 273, 6181, 310]` and predicts token
`7785` (` Paris`).

The single-row Q7 kernel now parallelizes work across the eight selected
experts. For a fixed production layer/state, median kernel time is 807,529,578
ns at one thread and 115,667,512 ns at 12 threads, a 6.98× improvement.
Selected routes and floating-point outputs are bit-identical across the two
runs. The complete five-position package smoke spends 13,082,561,317 ns in Q7
and schedules 3,670,016,000 Q7 bytes.

This closes authenticated package assembly and the first single-token
parallelism boundary. It does not establish broad generation quality or
acceptable interactive latency. The next boundary is a frozen native-vs-
teacher prompt suite, followed by packed matrix decode/SIMD and cold-start
optimization.
