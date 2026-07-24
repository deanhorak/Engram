# Native BitNet feasibility result — 2026-07-23

Decision: **advance the low-bit-native track to a direct packed CPU kernel.**

This is the first tested representation in Engram that preserves its source
MLP exactly while fitting below the unchanged 45% dense-ideal-Q4 traffic
limit. It is a separate source track, not a successful conversion of
SmolLM2 or another dense Llama checkpoint.

The source is Microsoft's native
[`bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
at pinned revision
`04c3b9ad9361b824064a1f25ea60a8be9599b127`. The downloaded
`model.safetensors` is 1,178,623,988 bytes with SHA-256
`8143ae115ed6babe5e5ada8fb8c5b769d8f417802b2db042ad98b4f7ed73975b`.
The model card and
[technical report](https://arxiv.org/abs/2504.12285) attest that the model
was trained natively with ternary weights; Engram does not infer this
provenance from quantization metadata alone.

## Exact representation

The Hugging Face checkpoint uses four two-bit ternary coefficients per byte.
That native MLP payload, including BF16 projection scales and intermediate
normalization gains, is 398,546,100 bytes across 30 layers: **50.0521%** of
the frozen dense-Q4 denominator and therefore not a traffic pass by itself.

Engram losslessly represents each FFN channel as one logical semantic record:

| Record field | Bytes |
|---|---:|
| gate row, 2,560 trits | 512 |
| up row, 2,560 trits | 512 |
| transposed down column, 2,560 trits | 512 |
| BF16 `ffn_sub_norm` gain | 2 |
| **Total** | **1,538** |

Five base-3 trits are stored per byte. Three BF16 projection scales remain
layer-global. Physically, each layer stores separate cache-aligned gate, up,
gain, and down streams in execution order. Four base pointers preserve O(1)
record addressing. This structure avoids the cache-line rereads that an
interleaved record would require across BitNet's shared RMS-normalization
phase.

Including the global header, 30-entry directory, per-layer headers, every
scale/gain, and all alignment bytes, the independently reloaded artifact is
**318,924,544 bytes**:

- **40.0527%** of dense ideal Q4;
- **80.0220%** of the original Hugging Face two-bit MLP payload;
- 39,393,536 bytes below the frozen 45% limit.

The phase-stream artifact has SHA-256
`4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55`.
Parity hashes the immutable bytes actually loaded and requires that digest.
It also verifies the official source-weight digest both before and after
reference-model loading.

The exact dense phase schedule can stream each serialized cache line once, so
its modeled cold bytes are the same **40.0527%**. For independently scattered
logical records, the four streams can touch at most 25 cache lines or 1,600
bytes per record. Charging every record independently is still only
**41.6673%** of dense ideal Q4.

The conversion validated every source two-bit code, rejected the unused code
`3`, reloaded every canonical base-3 stream, and compared all
1,592,524,800 ternary coefficients plus 207,450 BF16 scale/gain values.
Source and reconstructed logical streams share SHA-256
`27243c2304a2ad1b7dc87deb15eee1572ef6ace1a6c16ecbf5c485fd33f4f89d`.

## Computation parity

A CPU-only BF16 oracle preserves BitNet's per-token activation quantization,
ReLU-squared gate, intermediate RMS normalization, and scaled ternary
linears. Layers 0, 14, and 29 are bit-identical on two deterministic states.
Replacing all 30 reference MLPs for a bounded causal smoke input also gives:

- final hidden state: exact, maximum error `0`;
- logits: exact, maximum error `0`;
- KL divergence: `0`;
- top-1 agreement: `1.0`.

This proves source decoding, record reconstruction, MLP arithmetic, and
all-layer substitution agree. It does **not** yet count as the final physical
systems pass: the parity oracle materializes BF16 matrices instead of
executing the base-3 records directly, and the one-token causal input is not
the frozen confirmation corpus.

## Next gate

Implement a CPU kernel that consumes the cache-aligned base-3 phase streams
without materializing dense weights. The kernel must:

1. match the BF16 parity oracle;
2. retain the complete 40.0527% cold-byte accounting (and report cache-line
   reads);
3. run the frozen multi-sequence confirmation corpus against the pinned
   reference teacher, recording its deterministic packed-weight
   materialization; and
4. report latency and memory traffic separately from quality.

Only after those checks may this **low-bit-native** source track count as a
Milestone 2 combined-gate pass. The original dense-Llama conversion track
remains blocked.

Machine-readable evidence is in
[`source_audit.json`](source_audit.json),
[`repack.json`](repack.json), and [`parity.json`](parity.json).
