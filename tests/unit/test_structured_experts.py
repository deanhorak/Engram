import numpy as np
import torch

from engram.semantic.swiglu import swiglu
from engram.training.on_policy import _utility_residual_targets
from engram.training.structured_experts import (
    LowRankUtilityResidual,
    _greedy_residual_experts,
    _wrap_native_gate_channel_mlp_class,
    _selected_channel_output,
    _wrap_structured_expert_mlp_class,
    balanced_expert_permutation,
    block_contributions,
    fit_low_rank_utility_residual,
    native_gate_channel_traffic,
    progressive_sparse_budget,
    structured_expert_traffic,
)


class _BaseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 8, bias=False)
        self.up_proj = torch.nn.Linear(4, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 4, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def test_structured_expert_traffic_has_exact_hardware_budget():
    traffic = structured_expert_traffic(576, 1536, experts=24, active_experts=8)
    assert traffic.records_per_expert == 64
    assert traffic.active_records == 512
    assert traffic.selected_weight_bytes == 3_538_944
    assert traffic.router_weight_bytes == 55_392
    assert traffic.total_weight_bytes == 3_594_336
    assert traffic.dense_weight_bytes == 10_616_832
    assert np.isclose(traffic.fraction_of_dense, 0.33855063657407407)


def test_native_gate_channel_traffic_removes_completion_read():
    traffic = native_gate_channel_traffic(
        576, 1536, input_fraction=0.625, active_records=512
    )
    assert traffic.input_coordinates == 360
    assert traffic.gate_weight_bytes == 2_211_840
    assert traffic.selected_up_down_weight_bytes == 2_359_296
    assert traffic.total_weight_bytes == 4_571_136
    assert np.isclose(traffic.fraction_of_dense, 0.4305555555555556)


def test_progressive_budget_has_dense_warmup_and_exact_target():
    assert progressive_sparse_budget(
        1536,
        target_input_fraction=0.625,
        target_top_k=512,
        step=1,
        warmup_steps=2,
        anneal_steps=4,
    ) == (1.0, 1536)
    middle_fraction, middle_k = progressive_sparse_budget(
        1536,
        target_input_fraction=0.625,
        target_top_k=512,
        step=3,
        warmup_steps=2,
        anneal_steps=4,
    )
    assert middle_fraction == 0.8125
    assert middle_k == 1024
    assert progressive_sparse_budget(
        1536,
        target_input_fraction=0.625,
        target_top_k=512,
        step=99,
        warmup_steps=2,
        anneal_steps=4,
    ) == (0.625, 512)


def test_low_rank_utility_residual_recovers_synthetic_signal():
    rng = np.random.default_rng(31)
    states = rng.normal(size=(96, 7))
    left = rng.normal(size=(7, 2))
    right = rng.normal(size=(2, 11))
    bias = rng.normal(size=(11,))
    targets = states @ left @ right + bias

    predictor = fit_low_rank_utility_residual(
        states, targets, rank=2, regularization=1e-8
    )

    assert predictor.rank == 2
    assert predictor.parameter_bytes() == (7 * 2 + 2 * 11 + 11) * 4
    assert np.mean((predictor.predict(states) - targets) ** 2) < 1e-12


def test_low_rank_utility_residual_validates_shapes_and_rank():
    states = np.ones((3, 2))
    targets = np.ones((3, 4))
    with np.testing.assert_raises(ValueError):
        fit_low_rank_utility_residual(states, targets[:2], rank=1, regularization=1.0)
    with np.testing.assert_raises(ValueError):
        fit_low_rank_utility_residual(states, targets, rank=3, regularization=1.0)


def test_on_policy_targets_reconstruct_exact_log_utility_residual():
    rng = np.random.default_rng(41)
    states = rng.normal(size=(5, 4))
    gate = rng.normal(size=(8, 4))
    up = rng.normal(size=(8, 4))
    down = rng.normal(size=(4, 8))
    targets, base_logits, partial_gate, up_values = _utility_residual_targets(
        states, gate, up, down, input_coordinates=2
    )

    exact = np.log(
        np.abs(torch.nn.functional.silu(torch.tensor(states @ gate.T)).numpy() * up_values)
        * np.linalg.norm(down, axis=0)[None, :]
        + 1e-8
    )
    unclipped = exact - base_logits
    expected = np.clip(unclipped, -8.0, 8.0)
    expected -= expected.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(targets, expected, atol=1e-12, rtol=1e-12)
    assert partial_gate.shape == (5, 8)


def test_balanced_grouping_is_deterministic_and_lossless():
    features = np.random.default_rng(7).normal(size=(12, 9))
    first = balanced_expert_permutation(features, 3, iterations=8)
    second = balanced_expert_permutation(features, 3, iterations=8)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(12))
    assert first.reshape(3, 4).shape == (3, 4)


def test_numpy_block_decomposition_sums_to_dense_swiglu():
    rng = np.random.default_rng(13)
    states = rng.normal(size=(5, 4))
    gate = rng.normal(size=(8, 4))
    up = rng.normal(size=(8, 4))
    down = rng.normal(size=(4, 8))
    permutation = np.asarray([3, 0, 7, 2, 1, 6, 4, 5])
    blocks = block_contributions(
        states, gate, up, down, permutation, experts=4
    )
    np.testing.assert_allclose(
        blocks.sum(axis=1),
        swiglu(states, gate, up, down),
        atol=1e-10,
        rtol=1e-10,
    )


def test_greedy_residual_selection_accounts_for_vector_cancellation():
    contributions = np.asarray(
        [[[10.0, 0.0], [-9.0, 0.0], [0.0, 2.0], [0.0, 1.0]]]
    )
    selected = _greedy_residual_experts(contributions, 2)
    assert selected.tolist() == [[2, 3]]


def test_selected_channel_output_matches_dense_when_all_channels_selected():
    rng = np.random.default_rng(23)
    gate_values = rng.normal(size=(3, 6))
    up_values = rng.normal(size=(3, 6))
    down = rng.normal(size=(4, 6))
    selected = np.broadcast_to(np.arange(6), (3, 6))
    output = _selected_channel_output(
        gate_values, up_values, down, selected
    )
    expected = (
        torch.nn.functional.silu(torch.tensor(gate_values)).numpy()
        * up_values
    ) @ down.T
    np.testing.assert_allclose(output, expected, atol=1e-12, rtol=1e-12)


def test_hard_block_wrapper_has_exact_forward_and_causal_router_gradient():
    torch.manual_seed(19)
    base = _BaseMLP()
    wrapper_type = _wrap_structured_expert_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        torch.tensor([3, 0, 7, 2, 1, 6, 4, 5]),
        experts=4,
        active_experts=2,
        temperature=0.75,
    )
    hidden = torch.randn(2, 3, 4)

    wrapper.mode = "dense_shadow"
    torch.testing.assert_close(wrapper(hidden), base(hidden))

    wrapper.mode = "hard"
    wrapper.train()
    hard = wrapper(hidden)
    assert wrapper.last_surrogate_used
    assert wrapper.last_active_experts.shape == (6, 2)
    assert all(
        torch.unique(row).numel() == 2 for row in wrapper.last_active_experts
    )
    flat = hidden.reshape(-1, 4)
    manual = wrapper._hard_output(flat, wrapper.last_active_experts).reshape(2, 3, 4)
    torch.testing.assert_close(hard, manual)

    loss = hard.square().mean()
    loss.backward()
    assert wrapper.router.weight.grad is not None
    assert torch.count_nonzero(wrapper.router.weight.grad) > 0
    assert wrapper.gate_blocks.grad is not None
    assert torch.count_nonzero(wrapper.gate_blocks.grad) > 0
    assert wrapper.up_blocks.grad is not None
    assert torch.count_nonzero(wrapper.up_blocks.grad) > 0
    assert wrapper.down_blocks.grad is not None
    assert torch.count_nonzero(wrapper.down_blocks.grad) > 0

    wrapper.eval()
    with torch.inference_mode():
        evaluated = wrapper(hidden)
        expected = wrapper._hard_output(
            flat, wrapper.last_active_experts
        ).reshape(2, 3, 4)
    assert not wrapper.last_surrogate_used
    torch.testing.assert_close(evaluated, expected)


def test_native_gate_wrapper_has_deployable_hard_forward_and_soft_gradient():
    torch.manual_seed(29)
    base = _BaseMLP()
    wrapper_type = _wrap_native_gate_channel_mlp_class(torch)
    wrapper = wrapper_type(
        base, top_k=3, input_fraction=0.5, temperature=0.75
    )
    hidden = torch.randn(2, 3, 4)

    wrapper.mode = "dense_shadow"
    torch.testing.assert_close(wrapper(hidden), base(hidden))

    wrapper.mode = "hard"
    wrapper.train()
    hard = wrapper(hidden)
    assert wrapper.last_surrogate_used
    assert wrapper.last_active_records.shape == (6, 3)
    assert wrapper.last_input_coordinates.shape == (6, 2)
    assert wrapper.last_oracle.shape == (6, 3)
    torch.testing.assert_close(wrapper.last_dense_output, base(hidden))
    assert all(
        torch.unique(row).numel() == 3 for row in wrapper.last_active_records
    )
    flat = hidden.reshape(-1, 4)
    partial_gate, _ = wrapper._partial_gate(flat)
    expected = wrapper._hard_output(
        flat, partial_gate, wrapper.last_active_records
    ).reshape(2, 3, 4)
    torch.testing.assert_close(hard, expected)

    loss = hard.square().mean()
    score_gradient = torch.autograd.grad(
        loss, wrapper.last_selection_logits, retain_graph=True
    )[0]
    assert torch.count_nonzero(score_gradient) > 6 * 3
    loss.backward()
    assert wrapper.gate_weight.grad is not None
    assert torch.count_nonzero(wrapper.gate_weight.grad) > 0
    assert wrapper.up_weight.grad is not None
    assert torch.count_nonzero(wrapper.up_weight.grad) > 0
    assert wrapper.down_weight.grad is not None
    assert torch.count_nonzero(wrapper.down_weight.grad) > 0

    wrapper.eval()
    wrapper.set_budget(top_k=2, input_fraction=0.75)
    assert wrapper.top_k == 2
    assert wrapper.input_coordinates == 3
    with torch.inference_mode():
        evaluated = wrapper(hidden)
        partial_gate, _ = wrapper._partial_gate(flat)
        expected = wrapper._hard_output(
            flat, partial_gate, wrapper.last_active_records
        ).reshape(2, 3, 4)
    assert not wrapper.last_surrogate_used
    assert wrapper.last_dense_output is None
    assert wrapper.last_oracle is None
    torch.testing.assert_close(evaluated, expected)


def test_native_gate_wrapper_uses_fixed_low_rank_utility_residual():
    torch.manual_seed(37)
    base = _BaseMLP()
    predictor = LowRankUtilityResidual(
        input_factors=np.ones((4, 1)),
        output_factors=np.asarray([[0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0]]),
        bias=np.zeros(8),
    )
    wrapper_type = _wrap_native_gate_channel_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        top_k=1,
        input_fraction=0.5,
        utility_residual=predictor,
        residual_blend=0.8,
    )
    wrapper.eval()
    with torch.inference_mode():
        wrapper(torch.ones(1, 4))

    assert wrapper.utility_residual_rank == 1
    assert wrapper.last_active_records.tolist() == [[3]]
    assert "residual_input_factors" in wrapper.state_dict()
