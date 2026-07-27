# OLMoE Q7 causal confirmation

Source: `allenai/OLMoE-1B-7B-0125`

Revision: `9b0c1aa87e34a20052389dce1f0cf01da783f654`

Candidate: signed symmetric Q7, group size 64, BF16 group scales

Decision: **causal quality, projected traffic, and evidence screens pass**.

| Measurement | Result | Threshold |
|---|---:|---:|
| Teacher-to-student KL | 0.00900774 | ≤ 0.05 |
| Teacher top-1 agreement | 0.9765625 | ≥ 0.90 |
| Target NLL delta | +0.00391912 | ≤ +0.05 |
| Final-hidden relative L2 | 0.0460273 | ≤ 0.10 |
| Sequences | 8 | ≥ 8 |
| Prediction positions | 256 | ≥ 256 |
| Modeled traffic fraction | 0.227865 | ≤ 0.45 |

The modeled numerator is 734,003,200 bytes/token for selected Q7 expert codes,
BF16 group scales, and BF16 routers. The denominator is 3,221,225,472 bytes of
all-expert ideal Q4.

One position has KL 0.587149 despite the passing mean; this outlier must remain
visible in later validation.

This is not yet the final Milestone 2 systems pass. The evaluator rewrote all
6,442,450,944 expert parameters in place to decoded Q7 values and ran them
inside Hugging Face Transformers. It did not execute a packed Q7 artifact or a
native CPU expert kernel. The next gate is serialization, exact packed-kernel
parity, complete cache-line traffic accounting, and repetition of this causal
protocol without Transformers in the expert path.
