# Native BitNet package and generation integration

Date: **2026-07-24**

The qualifying phase-stream MLP artifact is now integrated into a complete,
source-independent transformer package and CPU generation runtime.

The compiled package contains:

- pinned model config and tokenizer assets;
- 332 non-MLP tensors in a 780,054,616-byte safetensors file;
- the 318,924,544-byte direct phase-stream MLP artifact;
- a checksummed seven-file manifest.

All 210 source MLP tensors are excluded. The package totals 1,108,116,808
inventoried bytes and does not need the original checkpoint directory.
`engram validate` verifies every checksum, reloads the MLP format, and rejects
any MLP tensor inside the non-MLP boundary.

The runtime creates the transformer on meta storage, materializes only
embedding, attention, normalization, and tied-head tensors, installs the
memory-mapped C++ MLP kernel, and then checks that no meta parameter remains.
Attention projections preserve BitNet's per-token Q8 activation and native
BF16 ternary scales.

On input IDs corresponding to `The capital of France is`, package-loaded and
source-backed direct-kernel models have bit-exact final hidden states and
logits. Greedy generation produced ` Paris.` with tokens `[12366, 13]`; two
generated tokens invoked 60 direct MLP calls. See
[package_parity.json](package_parity.json).

This is a Python transformer runtime with a native C++ MLP kernel. It is not
yet a complete C++ transformer implementation.

Subsequent runtime work added packed native Q/K/V/O projections, bounded
stateful attention with explicit RoPE/cache positions, final-row-only exact
vocabulary projection, and the `chat-native-bitnet` interface. The chat
command renders structured history with the tokenizer assets already sealed
in the package and re-prefills that complete history through a fresh bounded
cache on every turn.

In an observed two-turn session, the model generated a 32-token poem in
166.43 seconds, then responded contextually to `awesome!` and began a second
poem in 153.15 seconds. Both turns reported 7,477,440 attention-state bytes.
This validates the chat-history path but does not establish interactive
performance. Token streaming, persistent cross-turn cache reuse, and context
truncation are not implemented.
