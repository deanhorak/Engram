# Dynamic Input Pruning exact-completion sweep

Status: **reject_before_causal_intervention**

Held-out evidence: 1184 states per layer from 16 unique sequences across 30 layers.

Method: retain the largest-magnitude input coordinates, use the source model's partial gate/up projections to score every intermediate record, exactly complete only the candidate records, then rerank candidates at full precision.

Published DIP motivates input pruning and partial scoring; candidate-only exact completion plus contribution-norm reranking is an Engram extension.

Top-K magnitude reference mean local relative L2: 0.106162.

| Input coordinates | Candidates | Top-K | Recall | Oracle score mass | Local rel-L2 | Projected dense traffic | Reduction | Trace screen |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 432 (0.750) | 896 | 768 | 0.852436 | 0.920728 | 0.171803 | 0.763889 | 1.309x | reject |

Lowest-traffic near-oracle trace arm: **dip_b16_q432_c896_k768**, with 0.852436 mean recall and 0.763889 projected dense weight traffic.

> The NumPy evaluator executes dense matrix products. Traffic is a logical float32 weight-read count; it excludes indexes, activations, and measured DRAM traffic. For blocked runs the serialized/native implementation uses 64-byte-aligned access units, but the trace evaluator is still not a latency measurement.

> This clean-state trace screen does not measure hidden-state drift, logit KL, target NLL, or top-1 agreement; the trained-teacher causal intervention remains decisive.
