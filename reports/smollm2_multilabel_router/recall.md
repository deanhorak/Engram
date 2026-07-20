# SmolLM2-135M learned multi-label routing

Status: **measured dense experimental baseline; not runtime-efficient**

This experiment trains the router against the quantity we actually want: membership in the
oracle top-256 MLP records. For each layer, 128 calibration hidden states become input examples,
and their exact contribution rankings become 1,536-dimensional binary label vectors. A
multi-output ridge model learns one linear membership score per record. The router ranks those
scores and returns a fixed candidate budget.

Evaluation uses 32 disjoint validation states from every one of the 30 layers of
`HuggingFaceTB/SmolLM2-135M`, giving 960 held-out measurements per configuration.

## Results

| Ridge regularization | Candidates | Mean recall | Median recall |
|---:|---:|---:|---:|
| 0.1 | 512 | 0.5923 | 0.5664 |
| 10 | 512 | 0.6116 | 0.5859 |
| 100 | 512 | 0.6372 | 0.6172 |
| 300 | 512 | 0.6516 | 0.6367 |
| **1,000** | **512** | **0.6590** | **0.6484** |
| 3,000 | 512 | 0.6478 | 0.6387 |
| 10,000 | 512 | 0.6147 | 0.6055 |
| 100 | 640 | 0.6997 | 0.6875 |
| 300 | 640 | 0.7132 | 0.7031 |
| **1,000** | **640** | **0.7221** | **0.7148** |
| 3,000 | 640 | 0.7143 | 0.7109 |
| 10,000 | 640 | 0.6874 | 0.6836 |

At 512 candidates, the learned router improves mean recall from 0.6095 for trace-cluster routing
and 0.4500 for similarly sized separate gate/up IVF to 0.6590. At approximately 640 candidates,
it improves the trace-cluster result from 0.6684 to 0.7221. Direct supervision therefore helps,
but the improvement is not sufficient for compilation or quality claims.

## Systems limitation

This baseline scores every record with a dense hidden-size-by-record matrix. It avoids evaluating
the original SwiGLU records, but its weight traffic is itself large: for SmolLM2 each layer needs
576 x 1,536 learned coefficients. In uncompressed form this is comparable to another full MLP
projection and does not meet Engram's sparse-memory objective. `probed_record_count` is consequently
reported as the entire record set even when only 512 or 640 IDs are returned.

The experiment establishes that oracle membership is a better training target, not that this
particular dense classifier is a viable production router. The completed
[low-rank follow-up](../smollm2_lowrank_router/recall.md) retains 63.3% recall at rank 16 while
using 4.0% of the dense parameter bytes. The next step is a hierarchical model that predicts
small posting groups before ranking records only within the selected groups.
