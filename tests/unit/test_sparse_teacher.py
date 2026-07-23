import numpy as np
import pytest
import torch

from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.training.sparse_teacher import (
    _batch_ids,
    _aligned_router_width,
    _cardinality_preserving_top_mask,
    _direct_router_traffic,
    _group_utility_membership,
    _hard_line_fraction_with_smooth_gradient,
    _masked_mean,
    _normalized_masked_mse,
    _progressive_hardware_schedule,
    _same_input_teacher_mlp_targets,
    _selected_layer_indices,
    _train_internal_checkpoint_split,
    _wrap_hardware_sparse_mlp_class,
    _wrap_sparse_mlp_class,
)


def test_grouped_direct_router_traffic_is_cache_line_honest():
    traffic = _direct_router_traffic(
        576,
        1536,
        candidate_count=672,
        top_k=672,
        router_rank=8,
        router_group_size=2,
    )

    assert traffic["record_group_q4_bytes"] == 1728
    assert traffic["record_group_fp16_scale_bytes"] == 12
    assert traffic["record_group_cache_aligned_bytes"] == 1792
    assert traffic["selected_group_q4_code_bytes"] == 580_608
    assert traffic["selected_group_fp16_scale_bytes"] == 4_032
    assert traffic["selected_record_q4_bytes"] == 602_112
    assert traffic["router_factor_q4_bytes"] == 5_376
    assert traffic["router_factor_fp16_scale_bytes"] == 1_552
    assert traffic["router_bias_fp16_bytes"] == 1_536
    assert traffic["router_nonlinear_scale_fp16_bytes"] == 16
    assert traffic["uint16_group_id_bytes"] == 672
    assert traffic["router_cache_aligned_bytes"] == 8_512
    assert traffic["selected_group_id_cache_aligned_bytes"] == 704
    assert traffic["total_bytes"] == 611_392
    assert traffic["dense_q4_bytes"] == 1_327_104
    assert np.isclose(
        traffic["fraction_of_dense_q4"], 611_392 / 1_327_104
    )
    assert traffic["passes_45_percent_traffic_gate"] is False


def test_sixteen_record_groups_recover_real_cache_line_headroom():
    traffic = _direct_router_traffic(
        576,
        1536,
        candidate_count=672,
        top_k=672,
        router_rank=8,
        router_group_size=16,
    )

    assert traffic["selected_groups"] == 42
    assert traffic["record_group_q4_bytes"] == 13_824
    assert traffic["record_group_fp16_scale_bytes"] == 96
    assert traffic["record_group_cache_aligned_bytes"] == 13_952
    assert traffic["router_cache_aligned_bytes"] == 3_136
    assert traffic["total_cold_bytes"] == 589_312
    assert np.isclose(traffic["fraction_of_dense_q4"], 589_312 / 1_327_104)
    assert traffic["passes_45_percent_traffic_gate"] is True


def test_router_curriculum_stays_group_aligned_and_reaches_endpoints():
    widths = [
        _aligned_router_width(1024, 672, progress, 2)
        for progress in np.linspace(0.0, 1.0, 33)
    ]

    assert widths[0] == 1024
    assert widths[-1] == 672
    assert all(width % 2 == 0 for width in widths)
    assert all(left >= right for left, right in zip(widths, widths[1:]))


def test_hardware_curriculum_has_exact_dense_warmup_and_hard_endpoint():
    schedules = [
        _progressive_hardware_schedule(
            step=step,
            dense_warmup_steps=4,
            anneal_steps=8,
            start_input_fraction=1.0,
            target_input_fraction=0.625,
            start_candidate_count=1536,
            target_candidate_count=512,
            start_top_k=1536,
            target_top_k=512,
            start_temperature=1.0,
            target_temperature=0.25,
        )
        for step in range(13)
    ]

    for schedule in schedules[:5]:
        assert schedule["input_fraction"] == 1.0
        assert schedule["candidate_count"] == 1536
        assert schedule["top_k"] == 1536
        assert schedule["deployment_endpoint"] is False
    assert schedules[-1] == {
        "progress": 1.0,
        "input_fraction": 0.625,
        "candidate_count": 512,
        "top_k": 512,
        "temperature": 0.25,
        "deployment_endpoint": True,
    }
    assert all(
        left["candidate_count"] >= right["candidate_count"]
        for left, right in zip(schedules, schedules[1:])
    )
    assert all(
        left["top_k"] >= right["top_k"]
        for left, right in zip(schedules, schedules[1:])
    )


def test_selected_layers_are_deduplicated_and_range_checked():
    assert _selected_layer_indices([25, 26, 25, 29], 30) == (25, 26, 29)
    assert _selected_layer_indices(None, 3) == (0, 1, 2)
    with pytest.raises(ValueError, match="valid hidden-layer"):
        _selected_layer_indices([30], 30)


def test_train_internal_checkpoint_split_is_deterministic_and_disjoint():
    records = [{"input_ids": [index, index + 1]} for index in range(12)]
    first = _train_internal_checkpoint_split(records, 3, 71)
    second = _train_internal_checkpoint_split(records, 3, 71)

    assert first[2:] == second[2:]
    assert len(first[0]) == 9
    assert len(first[1]) == 3
    assert set(first[2]).isdisjoint(first[3])
    assert set(first[2]).union(first[3]) == set(range(12))


def test_group_utility_membership_has_realizable_fixed_cardinality():
    states = np.array([[1.0, 1.0], [-1.0, -1.0]])
    gate = np.ones((4, 2))
    up = np.ones((4, 2))
    down = np.array([[1.0, 1.0, 10.0, 10.0], [0.0, 0.0, 0.0, 0.0]])

    membership = _group_utility_membership(
        states, gate, up, down, selected_records=2, group_size=2
    )

    assert membership.dtype == np.bool_
    assert membership.shape == (2, 2)
    np.testing.assert_array_equal(membership.sum(axis=1), np.ones(2))
    np.testing.assert_array_equal(
        membership, np.array([[False, True], [False, True]])
    )


def test_locality_relaxation_preserves_cardinality_and_shift_invariance():
    logits = torch.tensor(
        [[1.2, 0.9, 0.4, 0.1, -0.2, -0.7]], requires_grad=True
    )
    top, mask = _cardinality_preserving_top_mask(logits, 3, 0.75, torch)
    assert top.shape == (1, 3)
    torch.testing.assert_close(mask.sum(dim=1), torch.tensor([3.0]))
    hard_lines = torch.zeros(1, 3).scatter(1, top // 2, 1.0)
    line_mass = mask.reshape(1, 3, 2).sum(dim=2)
    locality = _hard_line_fraction_with_smooth_gradient(
        line_mass, hard_lines, torch
    )
    torch.testing.assert_close(locality, hard_lines.mean(dim=1))
    gradient = torch.autograd.grad(locality.sum(), logits)[0]
    assert torch.count_nonzero(gradient) > 0
    assert torch.all(torch.isfinite(gradient))
    torch.testing.assert_close(
        gradient.sum(dim=1), torch.zeros(1), atol=1e-6, rtol=0.0
    )


class _Tokenizer:
    pad_token_id = None
    eos_token_id = 9

    def __call__(self, text, return_tensors=None):
        del return_tensors
        return {"input_ids": torch.tensor([[ord(value) % 10 for value in text]])}


def test_batch_padding_and_masked_losses_ignore_padding():
    records = [{"input_ids": [1, 2, 3]}, {"text": "ab"}]
    input_ids, mask, lengths = _batch_ids(
        records, _Tokenizer(), torch, "cpu"
    )
    assert lengths == [3, 2]
    assert input_ids.tolist() == [[1, 2, 3], [7, 8, 9]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]

    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 1000.0]])
    assert _masked_mean(values, mask, torch).item() == 3.0
    reference = torch.ones(2, 3, 2)
    approximation = reference.clone()
    approximation[0, 0] = 2.0
    approximation[1, 2] = 1000.0
    assert torch.isclose(
        _normalized_masked_mse(approximation, reference, mask, torch),
        torch.tensor(0.2),
    )


class _BaseMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 6, bias=False)
        self.up_proj = torch.nn.Linear(4, 6, bias=False)
        self.down_proj = torch.nn.Linear(6, 4, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden):
        activation = self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(activation)


def test_same_input_teacher_target_uses_exact_student_mlp_input():
    torch.manual_seed(17)
    teacher_mlp = _BaseMLP()
    student_input = torch.randn(2, 3, 4)
    separate_teacher_state = student_input + 5.0

    target = _same_input_teacher_mlp_targets(
        [teacher_mlp], [student_input], torch
    )[0]

    torch.testing.assert_close(target, teacher_mlp(student_input))
    assert not torch.allclose(target, teacher_mlp(separate_teacher_state))


def test_same_input_teacher_target_stops_gradients_at_frozen_teacher_boundary():
    torch.manual_seed(19)
    teacher_mlp = _BaseMLP()
    student_input = torch.randn(2, 3, 4, requires_grad=True)
    student_scale = torch.nn.Parameter(torch.tensor(0.75))

    target = _same_input_teacher_mlp_targets(
        [teacher_mlp], [student_input], torch
    )[0]
    student_output = student_scale * student_input
    torch.nn.functional.mse_loss(student_output, target).backward()

    assert target.requires_grad is False
    assert student_input.grad is not None
    assert student_scale.grad is not None
    assert all(parameter.grad is None for parameter in teacher_mlp.parameters())


def test_sparse_student_mlp_modes_and_trainable_boundary():
    torch.manual_seed(3)
    base = _BaseMLP()
    dense = LowRankMultiLabelRouter(
        np.ones((4, 2)), np.ones((2, 6)), np.zeros(6)
    )
    wrapper_type = _wrap_sparse_mlp_class(torch)
    wrapper = wrapper_type(base, dense, top_k=2, candidates=4, adapter_rank=2)
    hidden = torch.randn(2, 3, 4)

    wrapper.mode = "identity"
    torch.testing.assert_close(wrapper(hidden), base(hidden))
    wrapper.mode = "oracle"
    oracle = wrapper(hidden)
    wrapper.mode = "trained"
    trained = wrapper(hidden)

    assert oracle.shape == trained.shape == (2, 3, 4)
    assert wrapper.last_router_logits.shape == (6, 6)
    assert wrapper.last_oracle.shape == (6, 2)
    assert wrapper.last_recall.shape == (6,)
    assert all(not parameter.requires_grad for parameter in wrapper.base.parameters())
    trainable = {
        name for name, parameter in wrapper.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "router_input",
        "router_output",
        "router_bias",
        "router_nonlinear_scale",
        "adapter_a",
        "adapter_b",
        "gate_adapter_a",
        "gate_adapter_b",
        "up_adapter_a",
        "up_adapter_b",
    }

    target = torch.zeros_like(wrapper.last_router_logits).scatter(
        1, wrapper.last_oracle, 1.0
    )
    loss = trained.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
        wrapper.last_router_logits, target
    )
    loss.backward()
    assert wrapper.router_input.grad is not None
    assert wrapper.router_nonlinear_scale.grad is not None
    assert torch.count_nonzero(wrapper.router_nonlinear_scale.grad) > 0
    assert wrapper.adapter_b.grad is not None
    assert wrapper.gate_adapter_b.grad is not None
    assert wrapper.up_adapter_b.grad is not None


def test_hard_router_full_mlp_mode_unfreezes_base_and_backpropagates():
    torch.manual_seed(7)
    base = _BaseMLP()
    router = LowRankMultiLabelRouter(
        np.random.default_rng(7).normal(size=(4, 2)),
        np.random.default_rng(8).normal(size=(2, 6)),
        np.zeros(6),
    )
    wrapper_type = _wrap_sparse_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        router,
        top_k=3,
        candidates=3,
        adapter_rank=2,
        train_full_mlp=True,
    )

    assert all(parameter.requires_grad for parameter in wrapper.base.parameters())
    output = wrapper(torch.randn(2, 3, 4))
    output.square().mean().backward()
    for parameter in wrapper.base.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_grouped_hard_router_expands_complete_pairs_and_backpropagates():
    torch.manual_seed(13)
    base = _BaseMLP()
    router = LowRankMultiLabelRouter(
        np.random.default_rng(13).normal(size=(4, 2)),
        np.random.default_rng(14).normal(size=(2, 3)),
        np.zeros(3),
    )
    wrapper_type = _wrap_sparse_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        router,
        top_k=4,
        candidates=4,
        adapter_rank=2,
        router_group_size=2,
    )
    hidden = torch.randn(2, 3, 4)

    trained = wrapper(hidden)
    assert wrapper.last_router_logits.shape == (6, 3)
    selected_groups = torch.topk(
        wrapper.last_router_logits, 2, dim=1
    ).indices
    expanded = torch.stack(
        (2 * selected_groups, 2 * selected_groups + 1), dim=-1
    ).reshape(6, 4)
    for row in expanded:
        assert torch.unique(row).numel() == 4
        assert torch.all(row.reshape(2, 2)[:, 1] == row.reshape(2, 2)[:, 0] + 1)

    activation = base.act_fn(base.gate_proj(hidden)) * base.up_proj(hidden)
    active_mask = torch.zeros_like(activation).reshape(6, 6)
    active_mask.scatter_(1, expanded, 1.0)
    expected = base.down_proj(
        (activation.reshape(6, 6) * active_mask).reshape_as(activation)
    )
    torch.testing.assert_close(trained, expected)

    flat_activation = activation.reshape(6, 6)
    value_norms = torch.linalg.vector_norm(base.down_proj.weight, dim=0)
    group_utility = (
        torch.abs(flat_activation) * value_norms.unsqueeze(0)
    ).reshape(6, 3, 2).sum(dim=2)
    expected_groups = torch.topk(group_utility, 2, dim=1, sorted=False).indices
    expected_target = torch.zeros_like(wrapper.last_router_logits).scatter(
        1, expected_groups, 1.0
    )
    torch.testing.assert_close(wrapper.last_router_target, expected_target)
    torch.testing.assert_close(
        wrapper.last_router_target.sum(dim=1), torch.full((6,), 2.0)
    )
    loss = trained.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
        wrapper.last_router_logits, wrapper.last_router_target
    )
    loss.backward()
    assert wrapper.router_input.grad is not None
    assert torch.count_nonzero(wrapper.router_input.grad) > 0
    assert wrapper.router_output.grad is not None
    assert torch.count_nonzero(wrapper.router_output.grad) > 0


def test_hardware_sparse_student_has_hard_budget_and_causal_router_gradient():
    torch.manual_seed(11)
    base = _BaseMLP()
    router = LowRankMultiLabelRouter(
        np.random.default_rng(11).normal(size=(4, 2)),
        np.random.default_rng(12).normal(size=(2, 6)),
        np.zeros(6),
    )
    wrapper_type = _wrap_hardware_sparse_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        router,
        top_k=2,
        candidates=3,
        input_fraction=0.5,
        adapter_rank=2,
        residual_rank=3,
        temperature=0.75,
        cache_line_records=2,
    )
    hidden = torch.randn(2, 3, 4, requires_grad=True)

    wrapper.mode = "identity"
    torch.testing.assert_close(wrapper(hidden), base(hidden))
    wrapper.mode = "trained"
    trained = wrapper(hidden)

    torch.testing.assert_close(wrapper.last_input, hidden)
    assert wrapper.last_input.requires_grad is False
    assert wrapper.last_candidate_ids.shape == (6, 3)
    assert wrapper.last_active.shape == (6, 2)
    assert torch.all(wrapper.last_occupied_lines <= 3)
    for row, candidate_ids in enumerate(wrapper.last_candidate_ids):
        assert torch.unique(candidate_ids).numel() == 3
        expected_lines = torch.unique(candidate_ids // 2).numel()
        assert wrapper.last_occupied_lines[row].item() == expected_lines
    assert wrapper.last_locality_loss.ndim == 0
    full_activation = base.act_fn(base.gate_proj(hidden)) * base.up_proj(hidden)
    active_mask = torch.zeros_like(full_activation).scatter(
        2, wrapper.last_active.reshape(2, 3, 2), 1.0
    )
    expected_hard = base.down_proj(full_activation * active_mask)
    torch.testing.assert_close(trained, expected_hard)
    locality_gradient = torch.autograd.grad(
        wrapper.last_locality_loss,
        wrapper.router_input,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(locality_gradient) > 0
    causal_loss = trained.square().mean()
    causal_loss.backward()
    assert wrapper.router_input.grad is not None
    assert torch.count_nonzero(wrapper.router_input.grad) > 0
    assert wrapper.router_output.grad is not None
    assert torch.count_nonzero(wrapper.router_output.grad) > 0
    assert wrapper.router_blend_logit.grad is not None
    assert torch.isfinite(wrapper.router_blend_logit.grad)
    assert wrapper.gate_adapter_b.grad is not None
    assert torch.count_nonzero(wrapper.gate_adapter_b.grad) > 0
    assert wrapper.up_adapter_b.grad is not None
    assert torch.count_nonzero(wrapper.up_adapter_b.grad) > 0
    assert wrapper.residual_b.grad is not None
    assert torch.count_nonzero(wrapper.residual_b.grad) > 0


def test_hardware_sparse_trained_mode_is_exact_at_dense_start():
    torch.manual_seed(23)
    base = _BaseMLP()
    router = LowRankMultiLabelRouter(
        np.random.default_rng(23).normal(size=(4, 2)),
        np.random.default_rng(24).normal(size=(2, 6)),
        np.zeros(6),
    )
    wrapper_type = _wrap_hardware_sparse_mlp_class(torch)
    wrapper = wrapper_type(
        base,
        router,
        top_k=6,
        candidates=6,
        input_fraction=1.0,
        adapter_rank=2,
        residual_rank=0,
        temperature=1.0,
        cache_line_records=2,
    )
    hidden = torch.randn(2, 3, 4)

    expected = base(hidden)
    actual = wrapper(hidden)

    torch.testing.assert_close(actual, expected)
    assert wrapper.last_candidate_ids.shape == (6, 6)
    assert torch.all(wrapper.last_recall == 1.0)
