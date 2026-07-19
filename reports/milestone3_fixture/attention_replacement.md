# Gate 3: attention replacement

Status: **synthetic_pipeline_validation**

> Synthetic states validate the memory algorithms and metrics only. Teacher attention
> traces from a trained model have not been evaluated.

| Path | Mean rel-L2 | Median rel-L2 | p95 rel-L2 | ns/token |
|---|---:|---:|---:|---:|
| local | 0.974413 | 0.702165 | 2.804147 | 76106.882812 |
| recurrent | 0.834883 | 0.885883 | 1.055703 | 228863.015625 |
| hybrid | 0.456217 | 0.422843 | 1.094981 | 1201260.492188 |

Retrieval recall: 1.000000

Controlled long-context copying accuracy: 1.000000

Peak configured state bytes: 10496

Timing is Python wall-clock instrumentation, not a native production benchmark.
