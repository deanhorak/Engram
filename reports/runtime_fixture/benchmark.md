# Fixture runtime benchmark

This is a systems smoke benchmark for a deterministic random 16-wide, two-layer fixture.
It is not representative of language-model quality or large-model performance.

Hardware: Intel Xeon E5-2695 v2, 24 logical CPUs. This CPU has AVX but not AVX2, so safe
runtime dispatch selected the scalar vector kernel.

| Runtime | Decode tokens/s | Peak RSS | Cycles/token | Active semantic records/token |
|---|---:|---:|---:|---:|
| Python reference | 160.3 | 36.2 MiB | 2 | 16 |
| Native C++20 | 10,373.1 | 8.2 MiB | 2 | 16 |

Both runs generated 512 tokens with transition caching bypassed. Vocabulary IVF proxy-scored
32 of 64 rows and exactly rescored those candidates. Native throughput was 64.7x the Python
reference for this tiny fixture.
The source fixture occupied 35,451 bytes and the inspectable quantized-only runtime package
106,476 bytes. Small codebooks, IVF metadata, manifests, controller state, and both token/vocabulary embeddings
dominate this deliberately tiny model, so package size reduction is not expected here.

The native accounting model reports 17,888 logical semantic bytes and 8,576 logical vocabulary
bytes per token, excluding controller and episodic accesses. A dense-Q4 lower-bound estimate for
reading each source weight once is 3,656 bytes per token, so this fixture uses about 7.24x that
payload rather than 10x less. It activates 16 of 64 semantic records (exactly 25%, not less than
25%) and proxy-scores 32 posted records after probing 32 one-record IVF lists across two layers.
Both tiny indexes' coarse-centroid traffic outweighs their avoided key rows. These are analytical
logical-byte estimates, not hardware-counter measurements of DRAM;
both target gates therefore remain failed/unproven.

Commands:

```bash
./build/engram-bench /tmp/model.engram 512
PYTHONPATH=src python -m engram.cli benchmark --model /tmp/model.engram --tokens 512
```

Elapsed time, throughput, RSS, and file sizes are measured. Hardware DRAM traffic, energy,
trained-model quality, prompt throughput, and time to first token were not measured.
