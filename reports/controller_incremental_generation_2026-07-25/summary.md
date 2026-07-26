# Controller-driven incremental generation

## Decision

The frozen incremental controller-generation gate passes. Across eight fixed
prompts and 32 greedy reference tokens, the controller-driven runtime matches
the existing bounded decoder runtime exactly while making zero decoder-layer
forward calls.

This is the first package generation path where normalized controller state
directly dispatches every native attention and MLP stage, advances absolute
RoPE positions, preserves bounded attention caches across tokens, and produces
logits through the package final norm and language-model head.

## Runtime design

For every token, the runtime carries:

- a width-2,560 normalized controller state;
- one scalar residual RMS;
- persistent W16/C8/K4/S2 native attention state.

At each of 30 stages it:

1. applies the stage input normalization to controller state;
2. calls native packed Q/K/V, RoPE, bounded attention, and packed output
   projection;
3. converts attention output into incoming-residual RMS coordinates;
4. applies post-attention normalization and the native packed MLP;
5. converts semantic output into the same coordinates;
6. advances state with the schema-v3 exact residual controller.

The scalar RMS preserves the relative magnitude of operator outputs without
restoring the original vector residual scaffold.

## Frozen protocol

| Property | Value |
|---|---:|
| Prompt suite | `tests/fixtures/inference_prompts.jsonl` |
| Prompt SHA-256 | `dd38c4ce92045d333edd572f23bad3f41f331393edfc58796a7cf2af01554fd2` |
| Prompts | 8 |
| Greedy tokens per prompt | 4 |
| Total reference tokens | 32 |
| Attention | Native W16/C8/K4/S2 |
| MLP | Native packed phase-stream kernel |
| Attention projections | Native packed |

The reference arm uses the same compiled operators and bounded cache inside
the existing decoder layer scaffold. The candidate replaces only the
stage/residual execution mechanism.

## Result

| Check | Required | Result |
|---|---:|---:|
| Weighted token agreement | >= 90% | **100%** |
| Exact prompt fraction | >= 75% | **100%** |
| Prompt count | >= 8 | **8** |
| Reference tokens | >= 32 | **32** |
| Correct cache positions | all | **all** |
| Decoder-layer forward calls | 0 | **0** |

Representative output:

```text
Prompt: The capital of France is
Reference:  Paris. Paris is
Controller: Paris. Paris is
Tokens: [12366, 13, 12366, 374]
```

All other factual, explanatory, coding, narrative, software, hardware, and
procedural prompts also match token-for-token.

## Performance and state

- Mean complete controller runtime per prompt: 22.581 seconds
- Mean controller arithmetic per prompt: 0.0427 seconds
- Maximum reported controller state: 112,684 bytes
- Controller arithmetic fraction of elapsed time: about 0.19%

The packed projections and MLP dominate runtime. Replacing the residual
scaffold with controller arithmetic does not create a material CPU bottleneck.

## Remaining boundary

The inference path is CPU-only and no decoder-layer forward runs, but Python
and Torch modules still orchestrate normalization, packed operators, and the
final head. The controller artifact is also supplied beside, rather than
inside, the native BitNet package.

Package-native controller installation and the residual/RMS kernel were
implemented next. The manifest now authenticates the controller directory,
generation can load it without an external path, and each residual step runs
through `libengram_bitnet.so`. The remaining boundary is broader: move stage
normalization, operator dispatch, and the final vocabulary head out of the
Python/Torch module shell into a complete native C++ generation runtime.
