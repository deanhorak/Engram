from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from engram.evaluation.olmoe_expert_proxy import (
    frozen_olmoe_expert_backward_proxy,
)
from engram.tracing.olmoe import _prepare_transformers_imports


torch = pytest.importorskip("torch")
try:
    _prepare_transformers_imports()
    from transformers.models.olmoe.configuration_olmoe import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeExperts as InstalledOlmoeExperts,
    )
except ImportError:
    pytest.skip(
        "the installed Transformers OLMoE implementation is unavailable",
        allow_module_level=True,
    )


class ToyModel(torch.nn.Module):
    def __init__(self, experts: torch.nn.Module) -> None:
        super().__init__()
        self.experts = experts


def _experts(
    *,
    experts: int = 16,
    hidden: int = 9,
    intermediate: int = 7,
    top_k: int = 8,
    seed: int = 4,
):
    config = OlmoeConfig(
        vocab_size=32,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        num_experts=experts,
        num_experts_per_tok=top_k,
        hidden_act="silu",
    )
    module = InstalledOlmoeExperts(config).to(dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.gate_up_proj.copy_(
            torch.randn(
                module.gate_up_proj.shape,
                dtype=torch.bfloat16,
                generator=generator,
            )
        )
        module.down_proj.copy_(
            torch.randn(
                module.down_proj.shape,
                dtype=torch.bfloat16,
                generator=generator,
            )
        )
    module.requires_grad_(False)
    return module


@pytest.fixture(autouse=True)
def deterministic_algorithms():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _inputs(
    experts,
    *,
    tokens: int = 9,
    top_k: int = 8,
    seed: int = 11,
):
    if top_k > experts.num_experts:
        raise ValueError("fixture top-k exceeds expert count")
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(
        tokens,
        experts.hidden_dim,
        dtype=torch.bfloat16,
        generator=generator,
    )
    index = torch.stack(
        [
            (torch.arange(top_k, dtype=torch.int64) + token) % experts.num_experts
            for token in range(tokens)
        ]
    )
    weights = torch.softmax(
        torch.randn(tokens, top_k, generator=generator) * 3.0,
        dim=-1,
    )
    probe = torch.randn(
        tokens,
        experts.hidden_dim,
        dtype=torch.bfloat16,
        generator=generator,
    )
    return hidden, index, weights, probe


def _forward_and_grad(
    forward,
    hidden,
    index,
    weights,
    probe,
):
    hidden_leaf = hidden.detach().clone().requires_grad_(True)
    weight_leaf = weights.detach().clone().requires_grad_(True)
    output = forward(hidden_leaf, index, weight_leaf)
    hidden_gradient, weight_gradient = torch.autograd.grad(
        output,
        (hidden_leaf, weight_leaf),
        grad_outputs=probe,
    )
    return (
        output.detach(),
        hidden_gradient.detach(),
        weight_gradient.detach(),
    )


def _local_gradient_parts(
    experts,
    hidden,
    index,
    weights,
    probe,
):
    mask = torch.nn.functional.one_hot(
        index,
        num_classes=experts.num_experts,
    ).permute(2, 1, 0)
    parts = []
    for expert_index in range(experts.num_experts):
        top_k_position, token_index = torch.where(mask[expert_index])
        if token_index.numel() == 0:
            continue
        with torch.enable_grad():
            local_hidden = hidden[token_index].detach().requires_grad_(True)
            local_weight = (
                weights[token_index, top_k_position].detach().requires_grad_(True)
            )
            gate, up = torch.nn.functional.linear(
                local_hidden,
                experts.gate_up_proj[expert_index],
            ).chunk(2, dim=-1)
            value = experts.act_fn(gate) * up
            value = torch.nn.functional.linear(
                value,
                experts.down_proj[expert_index],
            )
            contribution = (value * local_weight[:, None]).to(hidden.dtype)
            hidden_gradient, weight_gradient = torch.autograd.grad(
                contribution,
                (local_hidden, local_weight),
                grad_outputs=probe[token_index],
            )
        parts.append(
            (
                token_index,
                top_k_position,
                hidden_gradient,
                weight_gradient,
            )
        )
    return parts


def test_actual_eager_forward_and_both_gradients_are_exact_and_repeatable():
    experts = _experts()
    model = ToyModel(experts).eval()
    hidden, index, weights, probe = _inputs(experts)
    assert experts.__class__ is InstalledOlmoeExperts
    assert experts.config._experts_implementation is None
    eager_forward = experts.forward
    reference = _forward_and_grad(
        eager_forward,
        hidden,
        index,
        weights,
        probe,
    )
    assert "forward" not in vars(experts)

    with frozen_olmoe_expert_backward_proxy(model, workers=3) as stats:
        first = _forward_and_grad(
            experts,
            hidden,
            index,
            weights,
            probe,
        )
        second = _forward_and_grad(
            experts,
            hidden,
            index,
            weights,
            probe,
        )
        assert "forward" in vars(experts)
        active_experts = int(torch.unique(index).numel())
        assert stats.serial_forward_calls == 2
        assert stats.parallel_backward_calls == 2
        assert stats.expert_backward_tasks == 2 * active_experts

    for run in (first, second):
        for actual, expected in zip(run, reference, strict=True):
            assert torch.equal(actual, expected)
    assert "forward" not in vars(experts)
    assert experts.forward.__func__ is eager_forward.__func__
    assert stats.restored_layers == 1
    assert not stats.context_active
    assert stats.executor_shutdown
    assert experts.gate_up_proj.grad is None
    assert experts.down_proj.grad is None


def test_actual_grouped_fallback_forward_and_both_gradients_are_exact():
    experts = _experts()
    experts.config._experts_implementation = "grouped_mm"
    model = ToyModel(experts).eval()
    hidden, index, weights, probe = _inputs(experts)
    grouped_forward = experts.forward
    reference = _forward_and_grad(
        grouped_forward,
        hidden,
        index,
        weights,
        probe,
    )

    with frozen_olmoe_expert_backward_proxy(model, workers=3) as stats:
        actual = _forward_and_grad(
            experts,
            hidden,
            index,
            weights,
            probe,
        )

    for actual_value, expected in zip(actual, reference, strict=True):
        assert torch.equal(actual_value, expected)
    assert stats.serial_forward_calls == 1
    assert stats.parallel_backward_calls == 1
    assert "forward" not in vars(experts)
    assert experts.forward.__func__ is grouped_forward.__func__
    assert experts.gate_up_proj.grad is None
    assert experts.down_proj.grad is None


def test_hidden_gradient_matches_eager_reverse_expert_order():
    experts = _experts(seed=4)
    model = ToyModel(experts).eval()
    hidden, index, weights, probe = _inputs(experts, seed=17)
    reference = _forward_and_grad(
        experts,
        hidden,
        index,
        weights,
        probe,
    )
    parts = _local_gradient_parts(
        experts,
        hidden,
        index,
        weights,
        probe,
    )
    ascending = torch.zeros_like(hidden)
    for token_index, _top_k_position, hidden_gradient, _weight_gradient in parts:
        ascending.index_add_(0, token_index, hidden_gradient)
    assert not torch.equal(ascending, reference[1])

    with frozen_olmoe_expert_backward_proxy(model, workers=4):
        actual = _forward_and_grad(
            experts,
            hidden,
            index,
            weights,
            probe,
        )

    for actual_value, expected in zip(actual, reference, strict=True):
        assert torch.equal(actual_value, expected)


def test_router_input_gradient_includes_exact_top_k_weight_path():
    experts = _experts(
        experts=8,
        hidden=7,
        intermediate=5,
        top_k=4,
        seed=23,
    )
    model = ToyModel(experts).eval()
    generator = torch.Generator().manual_seed(29)
    hidden = torch.randn(
        6,
        experts.hidden_dim,
        dtype=torch.bfloat16,
        generator=generator,
    )
    router_weight = torch.nn.Parameter(
        torch.randn(
            experts.num_experts,
            experts.hidden_dim,
            dtype=torch.bfloat16,
            generator=generator,
        ),
        requires_grad=False,
    )
    probe = torch.randn(
        hidden.shape,
        dtype=torch.bfloat16,
        generator=generator,
    )

    def execute():
        state = hidden.detach().clone().requires_grad_(True)
        logits = torch.nn.functional.linear(state, router_weight)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        route_weights, route_index = torch.topk(probabilities, 4, dim=-1)
        output = experts(state, route_index, route_weights)
        (gradient,) = torch.autograd.grad(
            output,
            (state,),
            grad_outputs=probe,
        )
        return output.detach(), gradient.detach()

    reference_output, reference_gradient = execute()
    with frozen_olmoe_expert_backward_proxy(model, workers=3):
        actual_output, actual_gradient = execute()

    assert torch.equal(actual_output, reference_output)
    assert torch.equal(actual_gradient, reference_gradient)
    assert router_weight.grad is None


def test_forward_and_backward_must_finish_inside_proxy_context():
    experts = _experts(experts=8, top_k=4)
    model = ToyModel(experts).eval()
    hidden, index, weights, probe = _inputs(experts, top_k=4)

    with frozen_olmoe_expert_backward_proxy(model, workers=2) as stats:
        retained_forward = experts.forward
        hidden_leaf = hidden.detach().clone().requires_grad_(True)
        weight_leaf = weights.detach().clone().requires_grad_(True)
        output = experts(hidden_leaf, index, weight_leaf)

    assert not stats.context_active
    with pytest.raises(RuntimeError, match="both execute inside"):
        torch.autograd.grad(
            output,
            (hidden_leaf, weight_leaf),
            grad_outputs=probe,
        )
    with pytest.raises(RuntimeError, match="both execute inside"):
        retained_forward(hidden, index, weights)


def test_no_grad_forward_uses_no_backward_tasks_and_restores_after_error():
    experts = _experts(experts=8, top_k=4)
    model = ToyModel(experts).eval()
    hidden, index, weights, _probe = _inputs(experts, top_k=4)
    eager_forward = experts.forward

    with pytest.raises(RuntimeError, match="sentinel"):
        with frozen_olmoe_expert_backward_proxy(model, workers=2) as stats:
            with torch.no_grad():
                actual = experts(hidden, index, weights)
                expected = eager_forward(hidden, index, weights)
            assert torch.equal(actual, expected)
            assert stats.serial_forward_calls == 1
            assert stats.parallel_backward_calls == 0
            assert stats.expert_backward_tasks == 0
            raise RuntimeError("sentinel")

    assert "forward" not in vars(experts)
    assert stats.restored_layers == 1
    assert not stats.context_active
    assert stats.executor_shutdown


@pytest.mark.parametrize("workers", [True, 0, -1, 1.5])
def test_worker_count_must_be_a_positive_integer(workers):
    model = ToyModel(_experts()).eval()
    with pytest.raises(ValueError, match="positive integer"):
        with frozen_olmoe_expert_backward_proxy(model, workers=workers):
            pass


def test_proxy_requires_deterministic_algorithms():
    model = ToyModel(_experts()).eval()
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match="deterministic"):
        with frozen_olmoe_expert_backward_proxy(model, workers=2):
            pass


@pytest.mark.parametrize(
    "mutation",
    [
        "training",
        "trainable",
        "float32_weights",
        "unsupported_activation",
        "bias",
        "transposed",
        "non_eager",
        "missing_config",
        "missing_implementation",
    ],
)
def test_proxy_rejects_unsupported_actual_expert_modules(mutation):
    experts = _experts()
    model = ToyModel(experts).eval()
    if mutation == "training":
        experts.train()
    elif mutation == "trainable":
        experts.gate_up_proj.requires_grad_(True)
    elif mutation == "float32_weights":
        experts.gate_up_proj = torch.nn.Parameter(
            experts.gate_up_proj.detach().float(),
            requires_grad=False,
        )
    elif mutation == "unsupported_activation":
        experts.act_fn = torch.nn.ReLU()
    elif mutation == "bias":
        experts.has_bias = True
    elif mutation == "transposed":
        experts.is_transposed = True
    elif mutation == "non_eager":
        experts.config._experts_implementation = "batched_mm"
    elif mutation == "missing_config":
        experts.config = None
    elif mutation == "missing_implementation":
        experts.config = type("Config", (), {"hidden_act": "silu"})()

    with pytest.raises(ValueError):
        with frozen_olmoe_expert_backward_proxy(model, workers=2):
            pass


def test_proxy_rejects_preexisting_instance_forward():
    experts = _experts()
    experts.forward = experts.forward
    model = ToyModel(experts).eval()

    with pytest.raises(ValueError, match="pre-existing instance-level"):
        with frozen_olmoe_expert_backward_proxy(model, workers=2):
            pass


def test_proxy_rejects_invalid_forward_inputs():
    experts = _experts(experts=8, top_k=4)
    model = ToyModel(experts).eval()
    hidden, index, weights, _probe = _inputs(experts, top_k=4)

    with frozen_olmoe_expert_backward_proxy(model, workers=2):
        with pytest.raises(ValueError, match="hidden states"):
            experts(hidden.float(), index, weights)
        with pytest.raises(ValueError, match="top-k indices"):
            experts(hidden, index.to(torch.int32), weights)
        with pytest.raises(ValueError, match="top-k weights"):
            experts(hidden, index, weights.to(torch.float64))

        duplicate = index.clone()
        duplicate[:, 1] = duplicate[:, 0]
        with pytest.raises(ValueError, match="unique"):
            experts(hidden, duplicate, weights)

        outside = index.clone()
        outside[0, 0] = experts.num_experts
        with pytest.raises(ValueError, match="outside"):
            experts(hidden, outside, weights)

        nonfinite = weights.clone()
        nonfinite[0, 0] = float("nan")
        with pytest.raises(ValueError, match="outside"):
            experts(hidden, index, nonfinite)


def test_proxy_rejects_class_name_look_alike():
    class OlmoeExperts(torch.nn.Module):
        pass

    look_alike = OlmoeExperts().eval()
    model = ToyModel(look_alike).eval()
    with pytest.raises(ValueError, match="look-alike"):
        with frozen_olmoe_expert_backward_proxy(model, workers=2):
            pass


def test_proxy_rejects_models_without_exact_installed_experts():
    with pytest.raises(ValueError, match="no exact installed"):
        with frozen_olmoe_expert_backward_proxy(
            torch.nn.Linear(3, 2).eval(),
            workers=2,
        ):
            pass


def test_worker_exception_propagates_and_context_restores(monkeypatch):
    experts = _experts(experts=8, top_k=4)
    model = ToyModel(experts).eval()
    hidden, index, weights, probe = _inputs(experts, top_k=4)

    with frozen_olmoe_expert_backward_proxy(model, workers=2) as stats:
        hidden_leaf = hidden.detach().clone().requires_grad_(True)
        weight_leaf = weights.detach().clone().requires_grad_(True)
        output = experts(hidden_leaf, index, weight_leaf)

        def fail_map(self, function, *iterables, **kwargs):
            del self, function, iterables, kwargs
            raise RuntimeError("worker sentinel")

        monkeypatch.setattr(ThreadPoolExecutor, "map", fail_map)
        with pytest.raises(RuntimeError, match="worker sentinel"):
            torch.autograd.grad(
                output,
                (hidden_leaf, weight_leaf),
                grad_outputs=probe,
            )

    assert "forward" not in vars(experts)
    assert stats.restored_layers == 1
    assert stats.executor_shutdown
