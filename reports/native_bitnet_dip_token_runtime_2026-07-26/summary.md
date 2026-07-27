# Packaged native-BitNet DIP token runtime

## Decision

The adjudicated native-BitNet Dynamic Input Pruning (DIP) semantic memory is
now installed in a derived package and executed by the complete C++ token-step
runtime. The fixed, non-holdout eight-prompt confirmation passed:

- all 32 greedy token IDs matched the packaged dense-semantic reference;
- all 8 prompts had exact four-token continuations;
- global mean active-record fraction was `0.21560172604669886`, and
  the largest prompt mean was `0.22589163237311385`;
- complete modeled cold traffic was `30,153,074,432` bytes:
  `30,152,880,128` kernel bytes plus `194,304` global-metadata bytes;
- global mean modeled traffic was `0.41161156051507425` of dense ideal
  Q4, and the largest prompt mean was `0.41298354802526294`;
- all position, stage-call, semantic-call, semantic-row, backend, token-budget,
  and reset checks passed; and
- execution was CPU-only with no dense semantic fallback.

“Exact” here means exact greedy token-ID agreement. Hidden states and logits
were not compared. Reset verification proves repeated token IDs, zeroed
position/metric counters immediately after reset, and structural metric parity
between the first and replay runs; it does not prove hidden-state identity.

The machine-readable result is
[`integrated_8x4.json`](integrated_8x4.json). It promotes the DIP token runtime
to the chat-binding boundary.

## What was integrated

The installer authenticates the frozen semantic decision before constructing
a new package:

| Bound input | SHA-256 |
|---|---|
| Frozen policy | `c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e` |
| Milestone-2 adjudication | `ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc` |
| Base BitNet record artifact | `4fcf598af4346d5391ba428e32ba1629daae2768b73ab6bf872d3f9fb300ab55` |
| DIP v2 coordinate index | `b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15` |
| Derived package manifest | `707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926` |
| Native token executable | `0f6cf41c9c14dc3e05a8cad7a01f4f9909bd355f4e27f9296d6c1e15ba91dea4` |

The policy-bound source package is never modified. The installer copies it to
a derived package, installs the authenticated v2 coordinate index, adds a
`semantic_memory` manifest descriptor, rebuilds the exact file inventory, and
validates the result. The v2 index is the executable runtime policy: its
authenticated layer headers contain `q`, candidate counts, adaptive-K bounds,
RMS strategy, and audit strategy.

`NativeBitNetTokenRuntime` is now fail-closed and DIP-only. For every layer it
executes:

```text
attention
  -> normalized semantic input
  -> native_bitnet_dynamic_input_pruning_v2
  -> accept sparse semantic output
```

It neither constructs the former dense semantic backend nor provides a dense
fallback. Before constructing that runtime, the native executable
authenticates the exact manifest digest and byte count, rejects symlinks and
any unlisted or missing file, hashes the complete package inventory, and
checks the source-package, record, index, policy, and adjudication trust roots.
It derives model dimensions, head layout, context limit, RoPE/RMS values,
attention policy, paths, vocabulary bound, and EOS IDs from the authenticated
package. The generation config must include EOS IDs `128001` and `128009`.

The executable links the kernel objects directly and has no runtime dependency
on `libengram_bitnet.so`, `libengram_attention.so`, or another Engram shared
library. The C++ runtime owns attention caches, advances absolute
RoPE/cache positions, performs the final vocabulary argmax, and can reset and
replay a prompt. The Python confirmation harness separately pins the package
manifest and executable hashes and reauthenticates both after the final
process, before writing the report.

## Reproduction

Derive the runtime package without changing the frozen source package:

```bash
PYTHONPATH=src python -m engram.cli install-native-bitnet-semantic-memory \
  --model work/native_bitnet/model.engram-bitnet \
  --index work/native_bitnet/model.provisional.bitnet-dip-index.bin \
  --policy reports/native_bitnet_m2_2026-07-26/frozen_dip_policy.json \
  --adjudication reports/native_bitnet_m2_final_audit/49df50cc01c96844ab3e7015d66c8899025dad4e1f7f01a450f97677751b36f2/0fdce5a3eb2eaaec5e3b47587d7693bfdc0fed6fde53e3a40a74b8d2a76d4aa3.adjudication.json \
  --out work/native_bitnet/model.engram-bitnet-dip \
  --index-sha256 b98ce4e46c8ae67d9d92d4d13f5de3d4fe45ef2c76400bd9d50be08b2bd60e15 \
  --policy-sha256 c572754e597a760bc5ea6ba337bdaaf092e4ae1d5b5e90b6a2a14cbfbea3768e \
  --adjudication-sha256 ebb5ca9568387ffd3c5b187f8e17f3ce706aaee86f4bbe9e314bf1760a7da5cc
```

Build the native token executable:

```bash
cmake -S . -B build-runtime -DCMAKE_BUILD_TYPE=Release
cmake --build build-runtime --target engram-bitnet-token-generate -j
```

Run the non-holdout integrated confirmation:

```bash
PYTHONPATH=src python -m engram.cli \
  evaluate-native-bitnet-dip-token-generation \
  --model work/native_bitnet/model.engram-bitnet-dip \
  --executable build-runtime/engram-bitnet-token-generate \
  --prompts tests/fixtures/inference_prompts.jsonl \
  --reference reports/controller_cpp_stage_runner_2026-07-26/frozen_8x4.json \
  --package-manifest-sha256 707bbe069ef6892ce9bfe98258f3289e28af15a400922e950c4386f56dd26926 \
  --executable-sha256 0f6cf41c9c14dc3e05a8cad7a01f4f9909bd355f4e27f9296d6c1e15ba91dea4 \
  --out reports/native_bitnet_dip_token_runtime_2026-07-26/integrated_8x4.json \
  --max-tokens 4 \
  --threads 12 \
  --timeout 300
```

The recorded run took `395.3580831659783` seconds. Wall time includes each
first generation, its reset replay, and package authentication in every
prompt process. Reported semantic and attention counters/timings are the
snapshot from the first generation; the replay's structural counters are
compared against that snapshot. This is disclosure, not a speed result.

## Limits and next boundary

- Traffic is deterministic cache-line modeling, not measured DRAM traffic.
- Eight prompts and 32 generated tokens are integration evidence, not a broad
  language-quality evaluation.
- Exact greedy tokens are not hidden-state or logit parity.
- The longest processed context was 14 positions, below the exact 16-token
  local window. This suite did not exercise eviction or older-context
  retrieval.
- Reset attests token and structural-counter replay, not hidden-state identity.
- No latency or speedup gate was applied.
- `chat-native-bitnet` still uses the Python dense-semantic package path and
  deliberately rejects a DIP package rather than falling back. It must not be
  cited as using this backend.

The next boundary is a C/Python binding for the native token runtime, followed
by a chat command that tokenizes and renders the packaged template in Python
but executes every model step through the DIP-only native runtime.
