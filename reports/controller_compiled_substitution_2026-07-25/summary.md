# Compiled operators through the exact residual controller

## Decision

The frozen compiled-operator/controller gate passes all eight checks. The
packaged native BitNet MLP, native packed attention projections, W16/C8/K4/S2
streaming attention, and schema-v3 exact residual controller can now advance
to incremental runtime integration.

This is stronger than the earlier controller-only result: the semantic vectors
come from the direct packed CPU phase-stream kernel and the episodic vectors
come from the bounded native attention kernel. The controller replay bypasses
the decoder layer's residual scaffold.

## Provenance correction

`NativeBitNetRuntime` replaces every source MLP module with
`NativeKernelBitNetMLP` while loading the package. Controller traces captured
from that runtime therefore already contained compiled semantic outputs. The
remaining uncompiled operator in the controller-only run was dense attention,
not both semantic and episodic processing.

The independent attribution is consequently:

1. Packed semantic + dense attention passed the exact controller transition
   test.
2. Packed semantic + native streaming attention passed the existing
   all-layer attention confirmation.
3. This experiment replays both compiled outputs through the exact residual
   controller and proves that the decoder residual scaffold can be removed.

## Frozen protocol

| Property | Value |
|---|---:|
| Dataset SHA-256 | `ddca79850dbd3d97935d969f84d22983ce76517e785febd2911a53545d9bafd4` |
| Record offset | 8 |
| Unique sequences | 8 |
| Prediction positions | 256 |
| Layers | 30 |
| Semantic operator | Native packed BitNet phase stream |
| Episodic operator | Native W16/C8/K4/S2 streaming attention |
| Controller correction | Disabled |

No thresholds or model parameters were fitted on this split.

## Results

### Controller replay versus dense-attention baseline

| Metric | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Mean KL divergence | <= 0.05 | 0.011125 | yes |
| Top-1 agreement | >= 0.90 | 0.957031 | yes |
| NLL delta | <= +0.05 | -0.008285 | yes |
| Final hidden relative L2 | <= 0.10 | 0.075893 | yes |

### Controller replay versus compiled candidate

| Metric | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Final hidden relative L2 | <= 0.01 | 0.006810 | yes |
| Terminal trajectory NMSE | <= 0.0225 | 0.000026666 | yes |
| Logit KL | diagnostic | 0.000417 | — |
| Top-1 agreement | diagnostic | 0.984375 | — |

The controller replay takes 0.255 seconds for 8 x 33 tokens x 30 stages.
The native operator model passes take 112.69 seconds with dense attention and
116.27 seconds with compiled attention. The controller boundary is not the
runtime bottleneck in this batch measurement.

## What is and is not passed

Passed:

- source MLP tensors excluded;
- semantic outputs produced by the packaged CPU MLP kernel;
- episodic outputs produced by bounded native attention;
- residual combination and stage advancement replayed outside decoder layers;
- final package norm/head applied to the replayed state;
- frozen hidden, logit, NLL, trajectory, and sample-size checks.

Not yet passed:

- incremental operator dispatch directly from controller state;
- persistent cross-token controller/attention state in one generation loop;
- generation without executing decoder layers to obtain operator outputs;
- native serialization/loading of the schema-v3 controller;
- long-context controller-integrated tokens/second and memory-traffic evidence.

The next implementation is an incremental package runtime in which controller
state directly drives the native semantic and episodic operators, with correct
RoPE/cache-position advancement and no decoder-layer forward calls.
