from __future__ import annotations

import torch
from safetensors.torch import save_file

from engram.training.fully_sparse import fully_sparse_mlp_traffic
from engram.training.fully_sparse_distillation import (
    fully_sparse_mlp_class,
    progressive_fully_sparse_counts,
    validate_fully_sparse_artifact_cpu,
)


class _TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(6, 10, bias=False)
        self.up_proj = torch.nn.Linear(6, 10, bias=False)
        self.down_proj = torch.nn.Linear(10, 6, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def test_progressive_counts_reach_exact_target() -> None:
    assert progressive_fully_sparse_counts(
        6,
        10,
        target_input_count=3,
        target_intermediate_count=4,
        step=0,
        warmup_steps=1,
        anneal_steps=2,
    ) == (6, 10)
    assert progressive_fully_sparse_counts(
        6,
        10,
        target_input_count=3,
        target_intermediate_count=4,
        step=2,
        warmup_steps=1,
        anneal_steps=2,
    ) == (3, 4)


def test_hard_wrapper_matches_explicit_sparse_forward() -> None:
    torch.manual_seed(7)
    base = _TinyMLP()
    wrapper = fully_sparse_mlp_class(torch)(base, input_count=3, intermediate_count=4)
    wrapper.eval()
    hidden = torch.randn(2, 5, 6)
    actual = wrapper(hidden)
    assert wrapper.last_input is hidden
    input_indices = torch.topk(hidden.abs(), 3, dim=-1, sorted=False).indices
    sparse_hidden = torch.zeros_like(hidden).scatter(
        -1, input_indices, hidden.gather(-1, input_indices)
    )
    activation = torch.nn.functional.silu(
        torch.nn.functional.linear(sparse_hidden, wrapper.gate_weight)
    ) * torch.nn.functional.linear(sparse_hidden, wrapper.up_weight)
    activation_indices = torch.topk(activation.abs(), 4, dim=-1, sorted=False).indices
    sparse_activation = torch.zeros_like(activation).scatter(
        -1,
        activation_indices,
        activation.gather(-1, activation_indices),
    )
    expected = torch.nn.functional.linear(sparse_activation, wrapper.down_weight)
    torch.testing.assert_close(actual, expected)
    assert wrapper.last_surrogate_used is False


def test_training_ste_preserves_hard_forward() -> None:
    torch.manual_seed(11)
    base = _TinyMLP()
    wrapper = fully_sparse_mlp_class(torch)(base, input_count=3, intermediate_count=4)
    hidden = torch.randn(2, 6, requires_grad=True)
    wrapper.eval()
    expected = wrapper(hidden.detach())
    wrapper.train()
    actual = wrapper(hidden)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert wrapper.last_surrogate_used is True
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_low_rank_residual_is_zero_initialized_and_trainable() -> None:
    torch.manual_seed(12)
    base = _TinyMLP()
    wrapper = fully_sparse_mlp_class(torch)(
        base,
        input_count=3,
        intermediate_count=4,
        residual_rank=2,
    )
    wrapper.eval()
    hidden = torch.randn(2, 6)
    initial = wrapper(hidden)
    assert torch.count_nonzero(wrapper.residual_output_weight) == 0
    with torch.no_grad():
        wrapper.residual_output_weight.fill_(0.1)
    corrected = wrapper(hidden)
    assert not torch.equal(initial, corrected)


def test_traffic_target_stays_under_gate_before_metadata() -> None:
    traffic = fully_sparse_mlp_traffic(576, 1536, 282, 522)
    assert traffic["fraction_of_dense"] < 0.45
    assert traffic["candidate_recall_applicable"] is False
    assert traffic["metadata_included"] is False


def test_artifact_is_reloaded_and_executed_on_cpu(tmp_path) -> None:
    torch.manual_seed(13)
    path = tmp_path / "student.safetensors"
    save_file(
        {
            "layer_0.gate": torch.randn(10, 6),
            "layer_0.up": torch.randn(10, 6),
            "layer_0.down": torch.randn(6, 10),
            "layer_0.residual_input": torch.randn(2, 6),
            "layer_0.residual_output": torch.randn(6, 2),
        },
        path,
    )
    report = validate_fully_sparse_artifact_cpu(
        path,
        hidden_size=6,
        intermediate_size=10,
        input_count=3,
        intermediate_count=4,
    )
    assert report["passed"] is True
    assert report["execution_device"] == "cpu"
    assert report["cuda_required"] is False
    assert report["hard_input_cardinality"] == 3
    assert report["hard_intermediate_cardinality"] == 4
