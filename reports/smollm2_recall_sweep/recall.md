# SmolLM2-135M candidate-recall sweep

Status: **measured trained-model experiment**

This experiment isolates semantic candidate generation from background fitting and output
reconstruction. It uses 32 held-out states per layer across all 30 SmolLM2-135M layers, for 960
measurements per configuration. Each measurement asks whether 512 router candidates contain the
oracle top 256 MLP contributions.

| Index and search | Mean recall | Median recall | Mean records probed |
|---|---:|---:|---:|
| Joint, 64 clusters, 16 probes | 0.4050 | 0.3906 | 526.0 |
| Joint, 64 clusters, 32 probes | 0.4476 | 0.4141 | 765.6 |
| Joint, 128 clusters, 32 probes | 0.4200 | 0.4043 | 519.1 |
| Joint, 128 clusters, 64 probes | 0.4658 | 0.4297 | 778.1 |
| Separate gate/up, 64 clusters, 8 probes | 0.4395 | 0.4297 | 532.0 |
| Separate gate/up, 64 clusters, 16 probes | 0.4547 | 0.4355 | 610.6 |
| Separate gate/up, 64 clusters, 32 probes | 0.5144 | 0.4648 | 1116.8 |
| Separate + exact rerank, 64 clusters, 8 probes | 0.4500 | 0.4414 | 532.0 |
| Separate + exact rerank, 64 clusters, 16 probes | 0.4985 | 0.4844 | 610.6 |
| Separate + exact rerank, 64 clusters, 32 probes | 0.7801 | 0.7734 | 1116.8 |

The separate index clusters gate and up keys independently, unions the selected postings, and
then reranks the probed records. Its exact reranker restores the original key norms, applies the
SiLU gate, multiplies by the up projection, and includes the value-vector norm. These operations
reuse the two record-key dot products already required by geometric reranking.

## Interpretation

- Doubling probes helps, but returns diminish quickly relative to records scanned.
- Increasing joint clusters from 64 to 128 does not materially improve recall at comparable
  posting traffic.
- Separate gate/up postings consistently outperform joint postings.
- Exact contribution reranking is critical when the probed set is large. The normalized geometry
  proxy has a full-scan mean recall ceiling of only 0.556 for this candidate budget.
- The best measured recall, 0.780, requires probing roughly 73% of all 1,536 records. This is not
  sparse enough for the intended memory-traffic goal.

The result supports further work on learned or activation-aware coarse routing. It does not yet
justify replacing the production joint IVF index or compiling a quality candidate.
