# Complete native OLMoE causal confirmation

The authenticated CPU-only OLMoE package passes the frozen
8-sequence/256-position causal protocol against the untouched BF16 teacher.
The candidate maps the packaged non-MLP weights and packed-Q7 semantic
artifact, owns all recurrent attention state, and executes token IDs through
final logits and argmax without constructing a Transformers model.

| Population | Positions | Mean KL | Top-1 agreement | Target NLL delta | Final-hidden relative L2 |
|---|---:|---:|---:|---:|---:|
| Overall | 256 | **0.012981** | **96.094%** | **+0.016824** | **0.062047** |
| Exact local context, offsets 0–15 | 128 | **0.015319** | **96.094%** | **+0.019958** | **0.048893** |
| Bounded older-context retrieval, offsets 16–31 | 128 | **0.010642** | **96.094%** | **+0.013690** | **0.075202** |
| Frozen threshold for every population | — | at most 0.05 | at least 90% | at most +0.05 | at most 0.10 |

Both halves are gated independently under the same quality thresholds. Offset
16 is the first prediction after the W=16 exact local window begins evicting
older context, so the passing post-window half directly exercises the
streaming C=8/K=4 older-context selection path.

The run schedules 187,904,819,200 packed Q7 bytes, **22.7865%** of the
all-expert ideal-Q4 reference. This is exact algorithmic accounting for
router/expert reads; it is not a hardware-counter measurement and excludes
the non-MLP and attention traffic.

All cache positions, reset boundaries, diagnostic-logit argmax checks, source
and package authentication, and post-run artifact checks pass. Candidate
execution plus metric capture across the eight sequences takes 93.37 seconds.
Cold authentication and runtime loading take 33.44 seconds; the complete
command, including teacher-shard and post-run package authentication, takes
183.57 seconds.

The protocol deliberately reports tails without turning them into new
post-hoc gates. Maximum position KL is 0.60677 and p95 KL is 0.08356. The
first top-1 divergence in execution order is sequence 0, offset 23. The
largest KL occurs at sequence 4, offset 0. Offset 31 alone has KL 0.051834,
top-1 agreement 0.75, NLL delta +0.265473, and hidden relative L2 0.124504;
the protocol gates the 16-position population mean rather than every offset.
The fixed corpus remains too small to support broad language-quality claims.

The original frozen protocol does not bind the Python evaluator source
inventory. A post-run manual audit independently rechecked the metric
aggregates, traffic, input/target alignment, cache positions, and every frozen
artifact hash and found no validity-critical defect. The evaluator was then
hardened to recompute input/target identity, authenticate config/index
contents, enforce the thread setting, retain all 256 position rows, and rehash
all roots after execution.

A separately frozen, explicitly non-independent hardened replay binds that
evaluator-source inventory and discloses that the original outcome was already
known. The unchanged candidate exactly reproduces every semantic metric,
position split, traffic result, divergence, and gate check. All seven post-run
roots pass. That replay measures 88.79 seconds inside native execution, 72.17
seconds inside Q7, 92.14 seconds for candidate execution plus metric capture,
and 184.55 seconds for the fully authenticated command.

## CPU threading follow-up

The serial untouched-teacher capture exposed a separate utilization problem:
this PyTorch build has no grouped expert GEMM, and Transformers visits active
OLMoE experts in a Python loop. Merely batching all eight sequences is
byte-exact but improves teacher compute by only 1.2%.

Four concurrent sequence forwards through one shared, read-only model preserve
the teacher arrays byte-for-byte while reducing teacher compute from 366.14 to
94.78 seconds (**3.86×**) and complete wall time from 389.29 to 114.30 seconds
(**3.41×**). Peak RSS remains effectively unchanged. This is now the default
CPU capture policy. An eight-worker expert scheduler is faster than serial
(2.33× compute) but changes BF16 rounding enough to alter 6/256 top-1
decisions, so it remains explicit experimental opt-in. See
[`threading.json`](threading.json) for the measured variants.

## Authentication

- Frozen protocol SHA-256:
  `db41e8e6bd8f769acb9d7012354c8d983daa2da0790b6c0b203096c3438a3164`
- Result SHA-256:
  `43672118ef1a69b15133fe1ae43e21851377a7f53ac9073b2c42ccfb552e89e8`
- Immutable candidate DSO SHA-256:
  `1892a830f84209f7f0b726b64e43cf19686801f2187f9f2663744baa2738931c`
- Package manifest SHA-256:
  `861e9cc472f9e1245db5d64e9253411d0b656a0f08df2f58264e9c708ed750db`
- Untouched-teacher reference SHA-256:
  `5be7849f30ae48d045e1293bf657da1da4be090e5bc19b72550a3bd8eeda47ba`
- Teacher-array SHA-256:
  `60a23e32116fb18b7eca7e3c35f4d21784fd6a17ee20243323d35cf90742f99d`
- Hardened replay protocol SHA-256:
  `94c3ff6cd0085c8df05562218f5fe674bd10cc33f66aae7dd2c18d09b0b3d1de`
- Hardened replay result SHA-256:
  `265f6ac3c1e6ef016e3557900a3278cb1b3b7d8cf020417629c757408fcb6f68`

## Decision

Promote the complete native OLMoE causal boundary. This closes the
source-specific Milestone 2 semantic/package/runtime gate for OLMoE Q7. It
does not make the generic dense-Llama conversion pass, prove long-form chat
quality, or establish measured whole-system DRAM traffic.
