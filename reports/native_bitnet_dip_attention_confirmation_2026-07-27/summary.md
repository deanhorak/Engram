# Native DIP bounded-attention confirmation

Date: **2026-07-27**

Status: **mechanical boundary protocol passed; semantic core reconfirmed**

The authenticated CPU-only native DIP handle was exercised at exact prompt
lengths 16, 17, 18, 24, and 32 with W=16, C=8, K=4, and two sinks. The runtime
now reports cumulative local-window eviction, older-candidate scoring,
older-value selection, sink-insertion, and accepted heavy-hitter-update
counters. Every counter matched its analytical bound, attention state remained
exactly 7,477,440 bytes, and the 32-position request replayed the same token and
all non-timing structural counters after reset.

## Results

| Prompt positions | Evictions | Older scored | Older selected | Sink insertions | Heavy-hitter updates |
|---:|---:|---:|---:|---:|---:|
| 16 | 0 | 0 | 0 | 0 | 0 |
| 17 | 30 | 600 | 600 | 600 | 0 |
| 18 | 60 | 1,800 | 1,800 | 1,200 | 0 |
| 24 | 240 | 21,600 | 15,600 | 1,200 | 3,600 |
| 32 | 480 | 60,000 | 34,800 | 1,200 | 5,654 |

At 32 positions, 5,654 accepted heavy-hitter updates lies inside the exact
policy interval [3,600, 8,400]. The lower bound is the initial filling of all
six non-sink slots per query head and layer; later candidates may replace the
current minimum or be rejected.

The current shared object SHA-256 is
`4b732bebd049506e649007ce2b4fd4cd52d498a5cc121d39b2610637938ce72a`.
The complete machine-readable result is in
[confirmation.json](confirmation.json).

## Semantic-core regression check

The instrumented standalone executable SHA-256 is
`c6c5b05b6d8be72edd7f9e12e5e66c615859b74268143a5b2023b8dae423a15b`.
It reran the fixed non-holdout eight-prompt/four-token protocol:

- 32/32 greedy token IDs and 8/8 exact continuations;
- 21.56017% global and 22.58916% maximum-prompt mean activity;
- 41.16116% global and 41.29835% maximum-prompt modeled dense-Q4 traffic;
- 30,153,074,432 complete modeled bytes; and
- all backend, row/call, position, traffic, budget, and reset checks passed.

The 12-thread run took 390.4183 seconds including reset replays and per-process
package authentication. See [frozen_8x4.json](frozen_8x4.json).

## Interpretation

This closes the mechanical uncertainty left by the earlier 14-position
integration suite: local eviction, sinks, older-key scoring, bounded top-K
value reads, heavy-hitter admission/replacement, fixed state size, and reset
replay all execute in the complete packaged runtime.

It does **not** establish long-context attention quality against a dense
teacher. The boundary prompt is deterministic systems evidence, not a language
benchmark, and logical byte/event counters are not hardware DRAM measurements.
The next semantic attention experiment should compare bounded and dense
attention on natural long-context tasks that require information older than
W=16.
