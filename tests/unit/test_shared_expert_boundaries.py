import numpy as np
import torch

from engram.training.shared_expert_boundaries import (
    _curriculum_width,
    _expert_oracle_indices,
    _halved_expert_down,
    _select_shared_records,
    _wrap_shared_expert_mlp_class,
    shared_expert_traffic,
)


class _BaseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 12, bias=False)
        self.up_proj = torch.nn.Linear(4, 12, bias=False)
        self.down_proj = torch.nn.Linear(12, 4, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def test_default_shared_expert_traffic_is_cache_aligned_and_under_gate():
    traffic = shared_expert_traffic(576, 1536)

    assert traffic["records_per_expert"] == 16
    assert traffic["active_physical_records"] == 640
    assert traffic["shared_weight_q4_bytes"] == 110_592
    assert traffic["expert_block_q4_bytes"] == 13_824
    assert traffic["selected_expert_weight_q4_bytes"] == 442_368
    assert traffic["router_weight_q4_bytes"] == 27_648
    assert traffic["router_metadata_cache_aligned_bytes"] == 256
    assert traffic["total_bytes"] == 580_864
    assert traffic["dense_q4_bytes"] == 1_327_104
    assert np.isclose(traffic["fraction_of_dense_q4"], 0.4376929012345679)
    assert traffic["fraction_of_dense_q4"] < 0.45
    for key in (
        "shared_weight_q4_bytes",
        "expert_block_q4_bytes",
        "router_weight_q4_bytes",
        "router_metadata_cache_aligned_bytes",
    ):
        assert traffic[key] % 64 == 0


def test_shared_selection_and_down_duplication_are_deterministic():
    strength = np.asarray(
        [[1.0, 3.0, 2.0, 0.0], [1.0, 5.0, 0.0, 2.0]], dtype=np.float64
    )
    shared = _select_shared_records(strength, 2)
    np.testing.assert_array_equal(shared, np.asarray([1, 0]))

    down = np.arange(12, dtype=np.float64).reshape(3, 4)
    permutation = np.asarray([2, 0, 3, 1])
    expert_down = _halved_expert_down(down, permutation, shared)
    np.testing.assert_allclose(expert_down[0], down[:, 2])
    np.testing.assert_allclose(expert_down[1], 0.5 * down[:, 0])
    np.testing.assert_allclose(expert_down[2], down[:, 3])
    np.testing.assert_allclose(expert_down[3], 0.5 * down[:, 1])


def test_curriculum_hits_endpoints_and_rounds_expert_counts():
    assert _curriculum_width(48, 32, 0.0) == 48
    assert _curriculum_width(48, 32, 0.5) == 40
    assert _curriculum_width(48, 32, 1.0) == 32
    assert _curriculum_width(48, 32, 5.0) == 32


def test_shared_expert_wrapper_has_dense_parity_and_hard_causal_gradients():
    torch.manual_seed(37)
    base = _BaseMLP()
    source_gate = base.gate_proj.weight.detach().clone()
    source_up = base.up_proj.weight.detach().clone()
    source_down = base.down_proj.weight.detach().clone()
    permutation = torch.tensor([3, 0, 8, 7, 1, 11, 5, 2, 10, 4, 9, 6])
    shared = torch.tensor([1, 7])
    wrapper_type = _wrap_shared_expert_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        permutation,
        shared,
        experts=3,
        active_experts=2,
        temperature=0.75,
    )
    hidden = torch.randn(2, 3, 4)

    wrapper.mode = "dense_shadow"
    torch.testing.assert_close(wrapper(hidden), base(hidden), atol=1e-6, rtol=1e-5)

    # Both copies of each shared record carry exactly half of its source down vector.
    for original in shared.tolist():
        position = int(torch.nonzero(permutation == original).item())
        expert = position // wrapper.records_per_expert
        offset = position % wrapper.records_per_expert
        torch.testing.assert_close(
            wrapper.expert_down[expert, offset], 0.5 * source_down[:, original]
        )
    torch.testing.assert_close(wrapper.shared_gate, source_gate[shared])
    torch.testing.assert_close(wrapper.shared_up, source_up[shared])
    torch.testing.assert_close(wrapper.shared_down, 0.5 * source_down.T[shared])

    wrapper.mode = "hard"
    wrapper.train()
    hard = wrapper(hidden)
    assert wrapper.last_surrogate_used
    assert wrapper.last_active_experts.shape == (6, 2)
    assert all(
        torch.unique(row).numel() == 2 for row in wrapper.last_active_experts
    )
    flat = hidden.reshape(-1, 4)
    expected = wrapper._shared_output(flat) + wrapper._hard_expert_output(
        flat, wrapper.last_active_experts
    )
    torch.testing.assert_close(hard, expected.reshape(2, 3, 4))

    hard.square().mean().backward()
    assert wrapper.router.weight.grad is not None
    assert torch.count_nonzero(wrapper.router.weight.grad) > 0
    assert wrapper.shared_gate.grad is not None
    assert torch.count_nonzero(wrapper.shared_gate.grad) > 0
    assert wrapper.expert_gate.grad is not None
    assert torch.count_nonzero(wrapper.expert_gate.grad) > 0


def test_expert_oracle_returns_unique_fixed_cardinality_rankings():
    rng = np.random.default_rng(43)
    states = rng.normal(size=(7, 4))
    gate = rng.normal(size=(3, 4, 4))
    up = rng.normal(size=(3, 4, 4))
    down = rng.normal(size=(3, 4, 4))
    oracle = _expert_oracle_indices(
        states, gate, up, down, 2, batch_size=3
    )
    assert oracle.shape == (7, 2)
    assert all(len(np.unique(row)) == 2 for row in oracle)
    assert np.all((oracle >= 0) & (oracle < 3))

