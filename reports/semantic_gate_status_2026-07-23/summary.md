# Semantic-gate status — 2026-07-23

Decision: **Milestone 2 remains blocked.**

No tested representation jointly passes the all-layer causal thresholds and
the complete physical cold-traffic limit of 45% of dense ideal Q4.

| Frontier | KL | Top-1 | NLL delta | Hidden L2 | Traffic | Result |
|---|---:|---:|---:|---:|---:|---|
| DIP confirmation, q=432/C=896/K=768 | 0.02864 | 0.91010 | +0.03261 | 0.09048 | 83.33% cache-line | quality only |
| Compact Q4 at 3,000,093 positions | 0.88658 | 0.56594 | +0.88376 | 0.42452 | 44.9334% physical | traffic only |

The latest nonparametric pilot is layer-local rather than causal. Exact
LLE-32 over 233,005 local prototypes has mean relative L2 0.327526. Adding
1,000,000 independently captured pretraining prototypes lowers it to only
0.321854, a 1.73% improvement. Its frozen progression rule required at most
0.28 and at least 10% improvement, so the ten-million-prototype stage is
closed.

See [Project status](../../docs/status.md) for milestone-by-milestone context,
scope caveats, and the post-pause decision options. The adjacent
[`summary.json`](summary.json) records the exact thresholds, metrics, and
source-report hashes.

