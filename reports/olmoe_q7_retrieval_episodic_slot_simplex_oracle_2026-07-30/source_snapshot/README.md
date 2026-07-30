# Frozen per-slot source snapshot

This directory preserves the seven repo-relative source files that implemented
the per-slot trace used by the frozen cached-V2 experiment. The live runtime
continued to evolve after capture, so it is not an authoritative copy of these
particular experiment inputs.

Each file below is byte-identical to the source digest bound in
`../v2/cached_v2_protocol.json`. The snapshot was recovered from the frozen
per-slot implementation state by removing the subsequent regular-entry trace
extension, then independently checking every SHA-256 digest. Repo-relative
paths are retained below this directory so the binding is unambiguous.

Run the following from this directory to verify the snapshot:

```bash
sha256sum --check SHA256SUMS
```

`SHA256SUMS` is limited intentionally to these seven recovered files. The
protocol remains the authority for its complete source-binding set.
