# SmolLM2-135M hierarchical low-rank routing

Status: **measured negative result; not suitable for runtime integration**

This experiment uses the rank-16 learned router to select balanced posting groups, then computes
the exact SwiGLU contribution score only for records in the selected groups. Records are clustered
by their learned rank-16 output embeddings using deterministic capacity-constrained cosine
k-means. Every group has equal size, so each configuration selects exactly 512 or 640 records.

The router was trained with 128 calibration states per layer and evaluated on 32 disjoint states
from all 30 layers of `HuggingFaceTB/SmolLM2-135M`, giving 960 held-out measurements per
configuration. As in earlier reports, recall measures oracle top-256 membership.

## Results

| Groups | Records/group | Probes @512 | Mean recall @512 | Mean recall @640 | Router bytes/layer | Router MACs/layer |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 64 | 8 | 0.4589 | 0.5415 | 38,496 | 9,600 |
| 48 | 32 | 16 | 0.4778 | 0.5591 | 40,128 | 9,984 |
| 96 | 16 | 32 | 0.5008 | 0.5812 | 43,392 | 10,752 |
| **192** | **8** | **64** | **0.5282** | **0.6094** | **49,920** | **12,288** |
| Flat rank 16 | 1 | 512 | 0.6328 | 0.6990 | 141,312 | 33,792 |

Posting IDs add 3,072 bytes per layer when stored as unsigned 16-bit values. The table's router
bytes use float32 factors and include the group bias, but exclude the original MLP keys and values.

## Interpretation

- Smaller groups consistently improve recall, approaching the flat router as the hierarchy
  becomes less coarse. This shows that averaging record classifiers into one group centroid loses
  important within-group score variation.
- The best hierarchical configuration loses 10.46 percentage points of recall at 512 records.
  Exact reranking improves ordering inside the selected union, but cannot recover oracle records
  whose groups were not selected.
- The apparent router-weight saving over flat rank 16 is about 91 KB per layer. However, exact
  reranking 512 records also reads roughly 1.18 MB of float16 gate/up keys per layer, excluding
  cache effects and later value-vector reads. The hierarchy therefore saves only about 7% of this
  combined routing-plus-reranking traffic while materially reducing recall.

This grouping method should not replace flat rank-16 scoring. The completed
[coverage-trained follow-up](../smollm2_coverage_groups/recall.md) improves the best hierarchical
recall to 54.6%, but remains well below flat routing. Two and four representatives per group do
not improve the direct count-target result.
