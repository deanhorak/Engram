import numpy as np
import torch

from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.training.sparse_teacher import _wrap_sparse_mlp_class


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
        "adapter_a",
        "adapter_b",
    }

    target = torch.zeros_like(wrapper.last_router_logits).scatter(
        1, wrapper.last_oracle, 1.0
    )
    loss = trained.square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(
        wrapper.last_router_logits, target
    )
    loss.backward()
    assert wrapper.router_input.grad is not None
    assert wrapper.adapter_b.grad is not None
