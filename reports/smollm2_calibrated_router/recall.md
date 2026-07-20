# SmolLM2-135M trace-calibrated routing

Status: **measured experimental router; not a production index**

This experiment tests whether teacher traces can provide a better coarse routing signal than
clustering MLP keys alone. For each layer, the router clusters normalized calibration hidden
states. Each state cluster stores the records with the largest mean measured contribution,
`abs(SiLU(x gate_j) * (x up_j)) * ||value_j||`. At lookup time it selects nearby state clusters,
unions their postings, and exactly reranks that union using the query's real SwiGLU contribution.

The held-out evaluation uses all 30 layers of `HuggingFaceTB/SmolLM2-135M`, with 32 validation
states per layer (960 measurements per configuration). Recall asks how many of the oracle top 256
records occur in the returned candidate set.

## Results

| Calibration states/layer | State clusters | Records/cluster | Probes | Mean candidates available | Mean recall | Median recall |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 8 | 512 | 1 | 512.0 | 0.6020 | 0.5938 |
| 128 | 16 | 512 | 1 | 512.0 | 0.6076 | 0.5938 |
| 128 | 32 | 512 | 1 | 512.0 | **0.6095** | 0.5898 |
| 128 | 16 | 256 | 2 | 405.7 | 0.5205 | 0.5000 |
| 128 | 32 | 256 | 2 | 412.4 | 0.5361 | 0.5195 |
| 128 | 32 | 256 | 4 | 641.0 | **0.6684** | 0.6641 |
| 512 | 32 | 512 | 1 | 512.0 | 0.6090 | 0.6016 |
| 512 | 32 | 256 | 2 | 401.0 | 0.5183 | 0.5000 |
| 512 | 32 | 256 | 4 | 615.2 | 0.6370 | 0.6250 |

For context, the earlier joint-key IVF router achieved 0.4050 recall while probing about 526
records. Separate gate/up IVF with exact reranking achieved 0.4500 at about 532 records and 0.4985
at about 611 records. Trace calibration therefore provides a material improvement at comparable
candidate traffic, but is still far below a quality threshold suitable for compilation.

## Interpretation

- A single learned posting of 512 records reaches about 61% recall, roughly 16 percentage points
  above separate gate/up IVF at a similar traffic level.
- Increasing the number of state clusters from 8 to 32 barely changes recall. The coarse state
  partition is not the dominant limitation in this range.
- Expanding calibration from 128 to 512 states per layer does not improve fixed-budget recall.
  More samples alone are unlikely to solve the routing problem with this objective.
- Multiple postings improve recall, but overlap and candidate traffic grow quickly. Four probes
  reach 66.8% recall with the smaller calibration fit, still short of a viable router.

The completed follow-up changes the learning objective rather than merely scaling this design:
a multi-label router trained directly against oracle membership reaches 65.9% recall at 512
candidates. See the [multi-label routing report](../smollm2_multilabel_router/recall.md).
The experimental implementation is in `src/engram/semantic/calibrated_router.py`; it is deliberately
not serialized into compiled packages or selected by the runtime.
