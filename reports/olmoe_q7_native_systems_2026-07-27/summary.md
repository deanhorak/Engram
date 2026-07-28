# OLMoE packed-Q7 native systems confirmation

The remaining native Q7 systems gate passes on the pinned
`allenai/OLMoE-1B-7B-0125` artifact.

## Result

| Check | Result |
|---|---:|
| Artifact size | 5,842,733,184 bytes |
| Layers / experts / selected | 16 / 64 per layer / top 8 |
| Native route identity | exact |
| Maximum absolute output error | 1.6391277e-7 |
| Relative output L2 | 1.9471822e-6 |
| Router bytes per layer/state | 262,144 |
| Selected-expert bytes per layer/state | 45,613,056 |
| Complete scheduled bytes per layer/state | 45,875,200 |
| Fraction of all-expert ideal Q4 | 22.7864583% |
| Scalar layer latency, one thread | 0.816445 s |
| Gate | **passed** |

The immutable artifact stores canonical signed Q7/group-64 expert matrices,
executed BF16 scales, and BF16 routers. Both independent readers validate all
1,024 experts. The direct CPU kernel computes the learned router and reads only
the selected expert phases; it does not build dense matrices or construct a
Transformers model.

The traffic count is exact unique packed stream bytes scheduled by this
kernel, not a hardware-counter measurement of DRAM transactions. The next
boundary is full mapped OLMoE token-step/generation integration and performance
tuning, not another semantic quantization search.

Machine-readable evidence: [result.json](result.json).
