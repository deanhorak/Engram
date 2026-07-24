# Semantic-gate status — 2026-07-23

Decision: **Milestone 2 remains blocked, but the low-bit-native track has
cleared the representation and parity preconditions for a direct packed
kernel.**

No tested deployment runtime yet jointly passes the all-layer causal
confirmation thresholds and the complete physical cold-traffic limit of 45%
of dense ideal Q4. The dense-Llama conversion track remains blocked. A
separate native-BitNet track now preserves its source MLP exactly below the
traffic ceiling, but it has only completed a one-token all-layer parity smoke
test through a dense BF16 oracle; it has not yet executed the packed artifact
or run the frozen confirmation corpus.

| Frontier | KL | Top-1 | NLL delta | Hidden L2 | Traffic | Result |
|---|---:|---:|---:|---:|---:|---|
| Native BitNet exact repack | 0.00000¹ | 1.00000¹ | — | 0.00000¹ | 40.0527% serialized/modelled | representation + parity smoke only |
| DIP confirmation, q=432/C=896/K=768 | 0.02864 | 0.91010 | +0.03261 | 0.09048 | 83.33% cache-line | quality only |
| Compact Q4 at 3,000,093 positions | 0.88658 | 0.56594 | +0.88376 | 0.42452 | 44.9334% physical | traffic only |
| Budget-native ternary at 1,014,225 positions | 2.28436 | 0.31976 | +2.27704 | 0.60361 | 43.1353% physical | traffic only; stop before 3M |

¹ Exact on one bounded causal smoke token, not the required confirmation
corpus.

The native-BitNet result is the first tested Engram representation that is
both lossless relative to its source MLP and below the unchanged traffic
limit. Engram repacked the pinned official
`microsoft/bitnet-b1.58-2B-4T` MLP into cache-aligned gate/up/gain/down
streams with five ternary digits per byte. Logical records remain O(1)
addressable. The reloaded 318,924,544-byte artifact and modeled exact phase
schedule are 40.0527% of the frozen dense-Q4 denominator; charging every
logical record as an independent 25-cache-line read is 41.6673%. All
1,592,524,800 ternary coefficients and
207,450 BF16 scale/gain values reconstruct exactly. Local layers 0, 14, and
29 and a one-token, all-30-layer causal substitution are bit-identical to the
pinned reference model after deterministic packed-weight materialization on
CPU.

This is evidence for a viable low-bit-native deployment path, not a
dense-Llama conversion result and not a combined-gate pass. The current
parity oracle decodes records to dense BF16 matrices. The next bounded task is
a direct packed CPU kernel, followed by the frozen multi-sequence
confirmation evaluation with physical byte and latency measurements. See the
[native-BitNet feasibility result](../semantic_gate_native_bitnet_2026-07-23/summary.md).

The latest budget-native program trains full-width grouped-ternary MLPs
through their exact deployment representation. Its binary packs five trits per
byte and includes every FP16 group scale, header, directory, and alignment
byte. A deepest-layer-first continual transition, direct hidden/logit
distillation, optional embedding/head co-adaptation, device-neutral resume,
and independently reloaded validation are implemented.

The one-million-position rung closed 63.37% of the remaining KL gap and 62.77%
of the NLL gap, but only 31.33% of the top-1 gap and 38.29% of the hidden-state
gap. Its frozen rule required 50% on every metric before 3M. The exact
configuration is therefore stopped. See the
[budget-native summary](../semantic_gate_budget_native_2026-07-23/summary.md).

The latest nonparametric pilot is layer-local rather than causal. Exact
LLE-32 over 233,005 local prototypes has mean relative L2 0.327526. Adding
1,000,000 independently captured pretraining prototypes lowers it to only
0.321854, a 1.73% improvement. Its frozen progression rule required at most
0.28 and at least 10% improvement, so the ten-million-prototype stage is
closed.

Five later budget-edge screens also fit within 45% traffic but miss their
development-only layer-14 ceiling of 0.20:

| Local screen | Traffic | Mean rel-L2 | Result |
|---|---:|---:|---|
| Four-cycle recurrent compact Q4 | 44.9293% | 0.308254 | close; cache reuse unmeasured |
| Projection-normalized ternary | 41.0013% | 0.631323 | close |
| Mixed affine LC-VQ | 44.3482% | 0.336396 | close |
| Unrestricted 128-entry VQ | 44.9799% | 0.576865 initial | close before QAT |
| Mixed lifted-binary lattice | 44.4012% | 0.556958 initial | close before QAT |

The bounded recurrent/post-hoc representation search and the tested
one-million-position grouped-ternary continuation are therefore closed.
Engram is moving forward with the materially different, pretraining-scale
low-bit-native source track while preserving the original 45% policy. If its
direct packed kernel or formal confirmation fails, the honest fallback is to
keep Milestone 2 blocked or explicitly relax the traffic policy toward the
quality-passing DIP frontier.

See [Project status](../../docs/status.md) for milestone-by-milestone context,
scope caveats, and the post-pause decision options. The adjacent
[`summary.json`](summary.json) records the exact thresholds, metrics, and
source-report hashes. Detailed local-screen evidence is in the
[budget-edge representation summary](../semantic_gate_lowbit_2026-07-23/summary.md).
