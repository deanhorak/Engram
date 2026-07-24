import pytest
import torch

from engram.training.budget_native_ternary import (
    confidence_weighted_kl,
    grouped_ternary_mlp_class,
    layer_quantization_strengths_for_step,
    masked_linear_cka_loss,
    quantization_strength_for_step,
)
from engram.training.budget_native_ternary_codec import (
    grouped_ternary_decode,
    grouped_ternary_quantize,
)


class _MLP(torch.nn.Module):
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


def test_continual_quantization_schedule_is_dense_then_cosine_then_hard():
    values = [
        quantization_strength_for_step(
            step,
            total_steps=8,
            dense_warmup_steps=2,
            anneal_steps=4,
        )
        for step in range(8)
    ]

    assert values[:2] == [0.0, 0.0]
    assert values[2] == pytest.approx(0.1464466094)
    assert values[4] == pytest.approx(0.8535533906)
    assert values[5:] == [1.0, 1.0, 1.0]
    assert values == sorted(values)


def test_deepest_first_schedule_limits_each_transition_to_one_layer():
    rows = [
        layer_quantization_strengths_for_step(
            step,
            total_steps=8,
            dense_warmup_steps=1,
            anneal_steps=6,
            layer_count=3,
            transition_mode="deepest_first",
        )
        for step in range(8)
    ]

    assert rows[0] == (0.0, 0.0, 0.0)
    assert rows[1] == pytest.approx((0.0, 0.0, 0.5))
    assert rows[2] == (0.0, 0.0, 1.0)
    assert rows[3] == pytest.approx((0.0, 0.5, 1.0))
    assert rows[4] == (0.0, 1.0, 1.0)
    assert rows[5] == pytest.approx((0.5, 1.0, 1.0))
    assert rows[6:] == [(1.0, 1.0, 1.0)] * 2
    assert all(
        sum(0.0 < strength < 1.0 for strength in row) <= 1
        for row in rows
    )


def test_global_layer_schedule_matches_scalar_schedule():
    scalar = quantization_strength_for_step(
        2,
        total_steps=6,
        dense_warmup_steps=1,
        anneal_steps=4,
    )

    assert layer_quantization_strengths_for_step(
        2,
        total_steps=6,
        dense_warmup_steps=1,
        anneal_steps=4,
        layer_count=3,
        transition_mode="global",
    ) == (scalar, scalar, scalar)


def test_linear_cka_is_scale_invariant_and_has_gradient():
    torch.manual_seed(47)
    teacher = torch.randn(2, 4, 6)
    student = (teacher * 3.0).clone().requires_grad_(True)
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]]
    )

    loss = masked_linear_cka_loss(student, teacher, mask, torch)
    loss.backward()

    assert loss.item() == pytest.approx(0.0, abs=2e-7)
    assert student.grad is not None
    assert torch.all(torch.isfinite(student.grad))


def test_grouped_ternary_forward_is_hard_and_has_identity_ste():
    torch.manual_seed(20260723)
    Module = grouped_ternary_mlp_class(torch)
    module = Module(_MLP(), group_size=4)
    module.set_quantization_strength(1.0)
    hidden = torch.randn(2, 3, 4)

    output = module(hidden)
    output.square().mean().backward()

    assert output.shape == hidden.shape
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))
    zero_fractions = module.zero_fractions()
    assert set(zero_fractions) == {"gate", "up", "down"}
    assert all(0.0 <= value <= 1.0 for value in zero_fractions.values())


def test_training_quantizer_matches_deployment_group_scale_fit():
    torch.manual_seed(29)
    Module = grouped_ternary_mlp_class(torch)
    module = Module(_MLP(), group_size=4)
    projection = module.gate
    codes, scales = grouped_ternary_quantize(
        projection.master.detach().numpy(),
        group_size=4,
    )
    expected = grouped_ternary_decode(
        codes,
        scales,
        shape=tuple(projection.master.shape),
        group_size=4,
    )

    torch.testing.assert_close(
        projection.quantized_weight().detach(),
        torch.from_numpy(expected),
        rtol=0.0,
        atol=0.0,
    )


def test_dense_strength_matches_source_mlp():
    torch.manual_seed(31)
    base = _MLP()
    Module = grouped_ternary_mlp_class(torch)
    module = Module(base, group_size=4)
    module.set_quantization_strength(0.0)
    hidden = torch.randn(2, 3, 4)

    torch.testing.assert_close(module(hidden), base(hidden))


def test_installed_deployment_weights_override_master_quantization():
    torch.manual_seed(37)
    Module = grouped_ternary_mlp_class(torch)
    module = Module(_MLP(), group_size=4)
    gate = torch.full_like(module.gate.master, 0.25)
    up = torch.full_like(module.up.master, -0.5)
    down = torch.full_like(module.down.master, 0.75)
    module.install_deployment_weights(gate=gate, up=up, down=down)
    hidden = torch.randn(2, 4)

    output = module(hidden)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(hidden, gate)
        )
        * torch.nn.functional.linear(hidden, up),
        down,
    )

    assert module.deployment_loaded
    torch.testing.assert_close(output, expected)


def test_confidence_weighted_kl_is_zero_for_matching_logits_and_has_gradient():
    torch.manual_seed(41)
    teacher = torch.randn(2, 3, 7)
    student = teacher.clone().requires_grad_(True)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    matching = confidence_weighted_kl(
        student,
        teacher,
        mask,
        torch,
        temperature=1.5,
        confidence_weight=1.0,
    )
    assert matching.item() == pytest.approx(0.0, abs=1e-7)

    shifted = (teacher + torch.randn_like(teacher) * 0.2).requires_grad_(True)
    loss = confidence_weighted_kl(
        shifted,
        teacher,
        mask,
        torch,
        temperature=1.5,
        confidence_weight=0.5,
    )
    loss.backward()
    assert loss.item() > 0
    assert shifted.grad is not None
    assert torch.all(torch.isfinite(shifted.grad))
