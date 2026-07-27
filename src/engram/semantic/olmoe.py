"""Exact NumPy decomposition of OLMoE's routed SwiGLU expert block."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class OLMoESparseMLPResult:
    router_probabilities: NDArray[np.float32]
    expert_indices: NDArray[np.int64]
    expert_weights: NDArray[np.float32]
    expert_contributions: NDArray[np.float32]
    output: NDArray[np.float32]


def olmoe_sparse_mlp(
    hidden_states: ArrayLike,
    router: ArrayLike,
    gate: ArrayLike,
    up: ArrayLike,
    down: ArrayLike,
    *,
    top_k: int,
    norm_topk_prob: bool = False,
) -> OLMoESparseMLPResult:
    """Execute exact top-k routing and retain each weighted expert contribution."""

    hidden = np.asarray(hidden_states, dtype=np.float32)
    router_weight = np.asarray(router, dtype=np.float32)
    gate_weight = np.asarray(gate, dtype=np.float32)
    up_weight = np.asarray(up, dtype=np.float32)
    down_weight = np.asarray(down, dtype=np.float32)
    if hidden.ndim != 2 or router_weight.ndim != 2:
        raise ValueError("hidden_states and router must be rank-2")
    experts, width = router_weight.shape
    if hidden.shape[1] != width:
        raise ValueError("router input width does not match hidden states")
    if not 0 < top_k <= experts:
        raise ValueError("top_k must be in [1, num_experts]")
    if gate_weight.ndim != 3 or up_weight.shape != gate_weight.shape:
        raise ValueError("gate and up must have matching [expert, intermediate, hidden] shapes")
    if gate_weight.shape[0] != experts or gate_weight.shape[2] != width:
        raise ValueError("gate/up expert dimensions do not match router")
    intermediate = gate_weight.shape[1]
    if down_weight.shape != (experts, width, intermediate):
        raise ValueError("down must have shape [expert, hidden, intermediate]")

    logits = hidden @ router_weight.T
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits, dtype=np.float32)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    indices = np.argsort(-probabilities, axis=1, kind="stable")[:, :top_k]
    selected_weights = np.take_along_axis(probabilities, indices, axis=1)
    if norm_topk_prob:
        selected_weights /= np.sum(selected_weights, axis=1, keepdims=True)

    contributions = np.empty(
        (hidden.shape[0], top_k, width), dtype=np.float32
    )
    for row in range(hidden.shape[0]):
        state = hidden[row]
        for position in range(top_k):
            expert = int(indices[row, position])
            gate_values = state @ gate_weight[expert].T
            gate_values = gate_values / (
                1.0 + np.exp(-gate_values, dtype=np.float32)
            )
            up_values = state @ up_weight[expert].T
            expert_output = (gate_values * up_values) @ down_weight[expert].T
            contributions[row, position] = (
                selected_weights[row, position] * expert_output
            )
    output = np.sum(contributions, axis=1, dtype=np.float32)
    return OLMoESparseMLPResult(
        router_probabilities=np.asarray(probabilities, dtype=np.float32),
        expert_indices=np.asarray(indices, dtype=np.int64),
        expert_weights=np.asarray(selected_weights, dtype=np.float32),
        expert_contributions=contributions,
        output=output,
    )
