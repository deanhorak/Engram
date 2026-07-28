# OLMoE native token-runtime boundary

The first complete transformer-shell-free OLMoE token boundary passes.

Engram now maps a 949,242,368-byte BF16 non-MLP artifact and the existing
5,842,733,184-byte packed-Q7 expert artifact into one CPU-only runtime. A token
step performs embedding lookup, RMS normalization, dense Q/K/V/O projections,
Q/K normalization, absolute-position RoPE, persistent bounded attention,
residual updates, native top-eight Q7 expert execution, final normalization,
and the independent BF16 language head.

Fixture checks pass for:

- full NumPy next-token parity;
- batch-prefill versus incremental-cache equivalence;
- correct position advancement; and
- reset/replay.

On the pinned production checkpoint, the prompt `The capital of France is`
encodes as `[510, 5347, 273, 6181, 310]`. The native runtime predicts token
`7785`, which decodes to ` Paris`, without constructing a Transformers model.
The five-position prefill takes 13.6725 seconds on 12 threads. Q7 work accounts
for 13.4510 seconds and is now the dominant optimization target.

This proves the complete native token boundary and stateful greedy decoding.
It is not yet a signed/authenticated installable package, a chat interface, or
a performance pass.

Machine-readable evidence: [result.json](result.json).
