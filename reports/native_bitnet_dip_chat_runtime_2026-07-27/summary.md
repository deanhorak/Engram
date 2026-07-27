# Authenticated native DIP chat binding

Date: **2026-07-27**

Status: **binding smoke passed; frozen core confirmation passed**

`chat-native-bitnet` now uses the promoted CPU-only DIP package through a
persistent, versioned C ABI. Python retains the authenticated packaged
tokenizer, chat-template rendering, and conversation bookkeeping, but it no
longer creates `AutoModelForCausalLM`, Torch model tensors, decoder layers, a
dense MLP, or a Transformers model shell.

The shared object accepts only a package root. Its native constructor invokes
the same production-pinned loader as the standalone executable and derives
model dimensions, context/vocabulary limits, thread default, attention
policy, semantic artifacts, controller, and both EOS IDs from the
authenticated package. Python cannot override artifact paths, routing policy,
attention W/C/K/S, or EOS. Each chat turn resets the native handle and
re-prefills the complete rendered history from position zero.

## Evidence

- Promoted package manifest:
  `707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926`
- Rebuilt standalone executable:
  `29526c9838ea484d8a21887dafeaba99a57348e7377e0de4138e0631dde10fad`
- Token-runtime shared object:
  `df3a4f70952cddaebff2e5198d9ddf6b5e8a25487020c40b89ec99f2c7d33f96`
- The DSO has SONAME `libengram_bitnet_token_runtime.so.1`, depends only on
  system `libm`, `libc`, and the ELF loader, and exports only six versioned C
  ABI symbols.
- The real production-package C ABI lifecycle test passes authentication,
  information retrieval, bounds checks, one-token generation, required-reset
  state transition, and deterministic structural replay.
- For the preserved no-system chat prompt token IDs, both the standalone
  executable and Python/C ABI produced token `9906` (`Hello`) with identical
  30 semantic calls, 240 semantic rows, 361,598 selected records, and
  2,624,024,064 modeled semantic bytes.
- Repeating that request on the same mapped handle after reset reproduced the
  token and all non-timing structural metrics.
- The actual interactive CLI generated `Hello` from the default system
  message plus user message `Hello`. The rendered prompt contained 17 tokens,
  so this smoke crossed W=16, reported 7,477,440 attention-state bytes, and
  took 5.16 seconds for one generated token after startup.
- The rebuilt native core also reran the fixed non-holdout 8×4 protocol:
  32/32 greedy tokens, 8/8 exact prompts, 21.56017% global mean activity,
  41.16116% modeled dense-Q4 traffic, and exact reset replay. See
  [frozen_8x4.json](frozen_8x4.json).

Machine-readable binding details are in
[chat_smoke.json](chat_smoke.json).

## What this closes

This closes the remaining chat/backend integration item for the promoted
native-BitNet Milestone 2 path: routed semantic memory is now exercised by the
user-facing chat command without an original transformer model shell or dense
semantic fallback.

It does not make Engram a generic dense-Llama compiler. The native-BitNet gate
was passed by the previously documented postmortem adjudication, while the
original dense-Llama routing track remains blocked. Product/additive
quantization exists in the generic research package/runtime, but it is not the
representation used by this separately trained ternary BitNet package.

## Remaining limitations

The frozen 8×4 suite still has a maximum context of 14 and does not exercise
attention eviction. The one-turn interactive smoke crosses W=16 but does not
measure older-memory selection quality or provide a sustained long-context
comparison. The next attention protocol must add explicit eviction,
older-candidate, sink, and heavy-hitter counters and test multiple boundary
lengths and multi-turn histories.

Chat output is returned only after a turn completes. There is no streaming,
persistent cross-turn cache reuse, sampling, context truncation, or concurrent
session API. Re-prefill cost grows with rendered history. Greedy token equality
is not hidden-state or logit parity, reset structural replay is not proof of
hidden-state identity, and modeled traffic is not measured DRAM traffic.
