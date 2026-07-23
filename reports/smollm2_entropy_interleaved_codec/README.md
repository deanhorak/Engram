# Interleaved entropy-codec traffic measurement

This experiment replaces fixed-width packed rows with deterministic static
byte-rANS streams and measures complete physical cold reads, including global
headers, per-layer model tables, offset tables, FP16 row scales, checksums,
record framing, and 64-byte padding.

At 768 selected records per layer, the independently aligned signed-Q3
gate/up/down record layout reads **12,758,656 bytes**, or **32.0464%** of the
39,813,120-byte dense-Q4 MLP reference. It therefore leaves **5,157,248 bytes**
for the complete nonlinear router while remaining at or below the 45% traffic
gate.

The alternatives do not leave a router budget:

- Sequential signed-Q3 gate/up streams plus random down records: 45.8013%.
- Independently aligned signed-Q4 triple records: 46.5389%.

All figures come from the 30-layer local SmolLM2-135M checkpoint. The writer
then atomically published each artifact and the loader decoded and validated
every entropy stream before traffic was reported. See `summary.json` for exact
byte counts and the layer-14 cross-check.
