# OLMoE frozen-expert backward proxy qualification

The deterministic CPU expert-backward proxy is authorized for larger
development fits. It reproduced one complete archived causal-head-gate record
bit for bit while reducing total record time from **1,564.347 seconds** to
**809.168 seconds**: a **1.933× speedup** and **48.274% wall-time reduction**.
The predeclared qualification boundary required at least a 10% reduction.

This is a development-training optimization, not a new attention-policy or
quality result. The failed 51-head causal/value-sensitive mask remains failed,
the packaged Q7 Milestone 2 result remains passed, and Milestone 3 remains
blocked on deployable bounded long-context attention.

## Exactness and execution

The qualification reran only the already consumed M0/sequence-0 record. It
used the archived record as the serial reference instead of spending another
26 minutes recomputing the same baseline. All of the following matched
exactly:

- loss and every one of the 256 gate-gradient values;
- complete non-timing record content and native-attention diagnostics;
- the diagnostic projected scores, exact 51-head mask, and flat head indices;
- record, mask, corpus, package, library, teacher, protocol, result, and source
  identities.

The proxy preserved Transformers' installed `grouped_mm` expert forward
dispatcher. On the frozen Torch 2.5.1 CPU stack, native grouped matrix
multiplication is unavailable, so that dispatcher uses its serial per-expert
fallback. This path is numerically different from the eager
`OlmoeExperts.forward` loop because it sorts routed pairs and restores top-K
order before reduction. The proxy therefore:

1. leaves the installed forward dispatcher unchanged;
2. replays independent frozen-expert backward tasks on 12 workers; and
3. reduces hidden gradients in the backend-specific order required for
   bit-exact autograd parity.

The complete record executed 16 serial expert forwards, 16 parallel
backwards, and 961 expert tasks. Recorded proxy component time was
118.631 seconds for serial expert forward, 126.922 seconds for parallel
backward tasks, and 0.386 seconds for ordered gradient reduction. All 16
layers were restored, the executor shut down, and frozen parameters retained
no gradients.

## Authenticated result

- Full result:
  [`result.json`](result.json)
- Result SHA-256:
  `837d4cadb793c191844eac1bc3f4495530cd8e98437804e719fa9375a89f4960`
- Expert proxy source SHA-256:
  `65015130e12204666963ce0b41c1763ee5f9972d657f6567c8988148d81ff3b0`
- Full-record qualifier source SHA-256:
  `1e563ec54d6115029a39848f2cd59da724046f49dd17b1d4975d94ea86e99f43`
- Transformers `integrations/moe.py` SHA-256:
  `abcdb5dc859c6a17be2a3caeb09f7ad609eeb340610d03280099099423e1ae97`
- Transformers `modeling_utils.py` SHA-256:
  `b8467e1ada952862d2e4d76632475ac2f9fa198121d0e1ce64fa393ea3773e16`

The result reauthenticated every inherited artifact after execution and
records the two Transformers files that resolve and implement expert dispatch.
The measured model-load, record, and cleanup interval took 819.384 seconds;
the later comparisons, post-run hashing, report assembly, and atomic write are
outside that timer.

## Decision

Use the proxy for larger offline causal selector fits on this authenticated
CPU fallback stack. Continue to reject native `grouped_mm` implementations
until their forward and backward order has been qualified separately. The
next semantic experiment remains a new Q7-aware, synthetic
retrieval-targeted head selector with answer-position loss and a reserved
holdout. If that distinct supervision cannot pass at the exact 51-head/read
budget, move to causally valid prefix-conditioned allocation.
