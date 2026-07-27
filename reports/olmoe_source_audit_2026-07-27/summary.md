# OLMoE source audit

Official source: `allenai/OLMoE-1B-7B-0125`

Revision: `9b0c1aa87e34a20052389dce1f0cf01da783f654`

Decision: **proceed to trained router trace**.

- Configuration checks: passed
- Indexed tensor names: 3,219/3,219 exact
- Remote safetensors-header shapes: 3,219/3,219 exact
- Missing, unexpected, or shape-invalid tensors: 0
- Layers / experts / selected experts: 16 / 64 / 8
- Active-expert fraction: 12.5%
- Selected Q4 experts plus BF16 routers: 406,847,488 modeled bytes
- All-expert dense-Q4 baseline: 3,221,225,472 bytes
- Structural fraction: 12.6302%

The verifier read bounded byte ranges containing six safetensors headers and
refused unbounded responses. It did not download the 27.68 GB checkpoint
payload.

This is source-structure evidence, not a Milestone 2 semantic pass. The byte
model excludes attention, activations, cache-line effects, runtime overhead,
and causal quality. The next experiment must capture trained router decisions
and expert contributions, then perform an all-layer causal and CPU-traffic
evaluation.
