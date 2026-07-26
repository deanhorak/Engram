# Whole-model fully sparse dense-source campaign

Date: **2026-07-24**

Decision at the time: stop scaling this dense-source Q-Sparse retrofit and use
native BitNet as the next CPU substrate.

**2026-07-26 correction:** native BitNet's lossless full-record kernel is not a
Milestone 2 pass because it performs no practical semantic-memory routing.

CUDA accelerated training only. Every saved candidate used device-neutral
tensors, independently reloaded and executed its hard sparse MLP math on CPU,
and declared `cuda_required: false`. No confirmation data was opened.

## Best result

The uniform `q=282/576`, `k=522/1,536` baseline uses 43.967% of dense ideal-Q4
weight reads but reaches KL 0.742 and 61.5% teacher top-1 agreement. A
single-layer causal sensitivity sweep produced a fixed layer-adaptive schedule
at exactly 45% ideal traffic while keeping every layer at `q <= 360` and
`K <= 512`. On an authenticated unseen 128-sequence/15,559-position
development shard, that schedule reaches:

| Metric | Threshold | Untrained schedule | Best trained |
|---|---:|---:|---:|
| Teacher-student KL | <= 0.05 | 0.457355 | 0.451669 |
| Teacher top-1 agreement | >= 0.90 | 0.669387 | 0.671380 |
| NLL delta | <= +0.05 | +0.474366 | +0.458467 |
| Final-hidden relative L2 | <= 0.10 | 0.328055 | 0.327199 |
| Ideal MLP traffic before metadata | <= 0.45 | 0.450000 | 0.450000 |

The best trained arm updated all sparse MLP projections plus already-resident
attention and normalization weights for 128 batch-four steps. The held-out
slope is positive but far too small to justify extrapolating to a pass on this
host-scale corpus.

## Rejected variants

- Full-model next-token continuation without teacher forwards processed about
  130,000 tokens at batch eight. It reached KL 0.456 and did not improve the
  remaining quality gap.
- A per-token concentration policy shifted low/high coordinate counts but
  drifted to 46.25% traffic and KL 2.98. It is not stable under all-layer
  substitution.
- A rank-24 full-hidden correction used exactly 1.042% traffic, leaving 43.958%
  for sparse reads. After co-adaptation its KL was 0.490, worse than spending
  all traffic on the causal fixed schedule.

## Consequence

The experiment validates CUDA as a practical distillation accelerator and
validates the CPU-only artifact boundary, but does not pass the dense-source
semantic gate. Q-Sparse's published continuation evidence uses a dramatically
larger token budget; another short sweep would not be informative.

Engram has a strong systems substrate: its losslessly repacked, memory-mapped
native-BitNet phase-stream kernel reaches KL 0.00371, 96.09% top-1 agreement,
and 40.0527% exact scheduled cold bytes. Because it executes every record, it
does not satisfy practical routing Gate 2. It can be used as the teacher and
CPU substrate for renewed semantic-memory work. CUDA may train offline
components; packaged inference must remain CPU-only.
