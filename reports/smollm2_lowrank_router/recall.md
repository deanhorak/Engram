# SmolLM2-135M low-rank learned routing

Status: **measured compressed-router experiment; not integrated into the runtime**

This experiment compresses the dense multi-label ridge router using a truncated singular-value
decomposition. A rank-`r` router replaces the hidden-by-record score matrix with two factors:
`hidden_size x r` and `r x record_count`. The bias remains uncompressed. Candidate scores are
computed as `(hidden @ input_factor) @ output_factor + bias`.

The dense routers were trained independently for every layer using 128 calibration states,
oracle top-256 membership labels, and ridge regularization 1,000. Evaluation uses 32 disjoint
validation states across all 30 layers of `HuggingFaceTB/SmolLM2-135M`, or 960 measurements per
configuration.

## Recall and systems frontier

| Router | Mean recall @512 | Mean recall @640 | Float32 bytes/layer | Fraction of dense bytes | Score MACs/layer |
|---|---:|---:|---:|---:|---:|
| Rank 8 | 0.6221 | 0.6893 | 73,728 | 2.08% | 16,896 |
| **Rank 16** | **0.6328** | **0.6990** | **141,312** | **3.99%** | **33,792** |
| Rank 32 | 0.6444 | 0.7115 | 276,480 | 7.80% | 67,584 |
| Rank 64 | 0.6543 | 0.7192 | 546,816 | 15.42% | 135,168 |
| Rank 128 | 0.6590 | 0.7221 | 1,087,488 | 30.68% | 270,336 |
| Dense | 0.6590 | 0.7221 | 3,545,088 | 100.00% | 884,736 |

The rank-16 router loses only 2.61 percentage points of mean recall at 512 candidates relative to
the dense learned router, while using about 25 times fewer parameter bytes. Across 30 layers its
float32 parameters total 4.04 MiB instead of 101.43 MiB. Rank 32 halves that recall loss while
remaining below 8% of dense parameter traffic.

Rank 128 exactly matches the dense result because a centered ridge model trained on 128 examples
has effective weight rank no greater than 127. Higher tested ranks therefore add storage without
adding information.

## What “estimated memory traffic” means

The byte column counts each factor and bias coefficient once at four bytes. It estimates a
cold-weight read for one router invocation per layer. It excludes hidden-state reads, candidate-ID
writes, cache-line effects, implementation padding, and the later cost of evaluating selected
SwiGLU records. The MAC column similarly counts only the two routing matrix products. These are
model-level estimates, not measured CPU latency or hardware-counter DRAM traffic.

## Decision

Rank 16 is the best current operating point when routing overhead matters; rank 32 is the safer
quality-oriented choice. Both preserve the benefit of oracle-membership supervision at small
router cost. Neither reaches recall suitable for compilation claims: rank 16 still misses about
37% of oracle top-256 records at a 512-candidate budget.

The completed [hierarchical follow-up](../smollm2_hierarchical_router/recall.md) uses the low-rank
scores to select posting groups and exactly rerank their records. Its best 512-record configuration
reaches only 52.8% recall, so post-hoc embedding groups do not retain the flat router's quality.
