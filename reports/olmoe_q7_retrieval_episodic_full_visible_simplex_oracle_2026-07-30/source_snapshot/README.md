# Frozen full-visible experiment source snapshot

This directory contains the exact 31 repo-relative source files bound by
`../protocol.json`. Paths below this directory preserve their original
repository layout.

Verify every source file from this directory:

```bash
sha256sum --check SHA256SUMS
```

The archived protocol is authoritative for the source inventory and expected
digests. This snapshot intentionally excludes generated tensor shards and all
datasets.
