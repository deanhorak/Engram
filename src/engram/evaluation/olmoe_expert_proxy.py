"""Deterministic CPU backward proxy for frozen OLMoE experts.

The installed Transformers OLMoE implementation can use either its eager
expert loop or the ``grouped_mm`` CPU fallback.  Their BF16 accumulation orders
are observably different, so this proxy invokes the installed forward
dispatcher verbatim and never parallelizes the forward path.

Backward is substantially more expensive and has a safe independent boundary:
expert weights are frozen, while gradients are required only for expert inputs
and router weights.  The proxy replays each active expert in a thread-local
autograd graph.  It reduces hidden-state gradients in descending expert order
for eager execution and ascending expert order for the serial ``grouped_mm``
fallback, matching the installed autograd graphs exactly.
"""

from __future__ import annotations

import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class ExpertProxyStats:
    """Execution counters for one installed expert-proxy context."""

    workers: int
    patched_layers: int = 0
    serial_forward_calls: int = 0
    serial_forward_seconds: float = 0.0
    parallel_backward_calls: int = 0
    expert_backward_tasks: int = 0
    parallel_backward_task_seconds: float = 0.0
    ordered_reduction_seconds: float = 0.0
    restored_layers: int = 0
    context_active: bool = True
    executor_shutdown: bool = False
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )

    def _record_forward(self, elapsed_seconds: float) -> None:
        with self._lock:
            self.serial_forward_calls += 1
            self.serial_forward_seconds += elapsed_seconds

    def _require_active(self) -> None:
        with self._lock:
            if not self.context_active:
                raise RuntimeError(
                    "expert proxy forward and backward must both execute "
                    "inside the proxy context"
                )

    def _deactivate(self) -> None:
        with self._lock:
            self.context_active = False

    def _mark_executor_shutdown(self) -> None:
        with self._lock:
            self.executor_shutdown = True

    def _record_backward(
        self,
        *,
        tasks: int,
        task_seconds: float,
        reduction_seconds: float,
    ) -> None:
        with self._lock:
            self.parallel_backward_calls += 1
            self.expert_backward_tasks += tasks
            self.parallel_backward_task_seconds += task_seconds
            self.ordered_reduction_seconds += reduction_seconds

    def snapshot(self) -> dict[str, int | float | bool]:
        """Return a stable JSON-compatible copy of the counters."""

        with self._lock:
            return {
                "workers": self.workers,
                "patched_layers": self.patched_layers,
                "serial_forward_calls": self.serial_forward_calls,
                "serial_forward_seconds": self.serial_forward_seconds,
                "parallel_backward_calls": self.parallel_backward_calls,
                "expert_backward_tasks": self.expert_backward_tasks,
                "parallel_backward_task_seconds": (self.parallel_backward_task_seconds),
                "ordered_reduction_seconds": self.ordered_reduction_seconds,
                "restored_layers": self.restored_layers,
                "context_active": self.context_active,
                "executor_shutdown": self.executor_shutdown,
            }


def _positive_worker_count(workers: int) -> int:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("expert proxy workers must be a positive integer")
    return workers


def _is_supported_silu(torch: Any, activation: Any) -> bool:
    if activation is torch.nn.functional.silu:
        return True
    activation_type = type(activation)
    return (
        activation_type.__module__ == "transformers.activations"
        and activation_type.__name__ == "SiLUActivation"
    )


def _validate_expert_module(
    torch: Any,
    module: Any,
    installed_experts_class: type,
) -> str:
    if module.__class__ is not installed_experts_class:
        raise ValueError(
            "expert proxy requires the exact installed Transformers OlmoeExperts class"
        )
    if module.training:
        raise ValueError("expert proxy requires eval-mode OLMoE experts")
    for name in ("num_experts", "hidden_dim", "intermediate_dim"):
        value = getattr(module, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"expert proxy module {name} is invalid")
    config = getattr(module, "config", None)
    implementation = getattr(config, "_experts_implementation", None)
    if (
        config is None
        or not hasattr(config, "_experts_implementation")
        or getattr(config, "hidden_act", None) != "silu"
        or implementation not in {None, "eager", "grouped_mm"}
        or getattr(module, "has_bias", False)
        or getattr(module, "is_transposed", False)
        or not _is_supported_silu(torch, getattr(module, "act_fn", None))
    ):
        raise ValueError("expert proxy requires supported bias-free SwiGLU experts")
    resolved_implementation = (
        "eager" if implementation in {None, "eager"} else implementation
    )
    if resolved_implementation == "grouped_mm" and (
        hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
    ):
        raise ValueError(
            "expert proxy supports grouped_mm only through the proven serial "
            "CPU fallback"
        )
    gate_up = getattr(module, "gate_up_proj", None)
    down = getattr(module, "down_proj", None)
    if not isinstance(gate_up, torch.Tensor) or not isinstance(down, torch.Tensor):
        raise ValueError("expert proxy module weights are missing")
    expected_gate_up = (
        module.num_experts,
        2 * module.intermediate_dim,
        module.hidden_dim,
    )
    expected_down = (
        module.num_experts,
        module.hidden_dim,
        module.intermediate_dim,
    )
    if tuple(gate_up.shape) != expected_gate_up or tuple(down.shape) != expected_down:
        raise ValueError("expert proxy module weight shape is invalid")
    if (
        gate_up.device.type != "cpu"
        or down.device.type != "cpu"
        or gate_up.dtype != torch.bfloat16
        or down.dtype != torch.bfloat16
    ):
        raise ValueError("expert proxy requires CPU BF16 expert weights")
    if gate_up.requires_grad or down.requires_grad:
        raise ValueError("expert proxy does not support trainable expert weights")
    return resolved_implementation


def _validate_forward_inputs(
    torch: Any,
    module: Any,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
) -> None:
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("expert proxy requires deterministic Torch algorithms")
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.ndim != 2
        or hidden_states.shape[0] <= 0
        or hidden_states.shape[1] != module.hidden_dim
        or hidden_states.device.type != "cpu"
        or hidden_states.dtype != torch.bfloat16
    ):
        raise ValueError("expert proxy hidden states are invalid")
    if (
        not isinstance(top_k_index, torch.Tensor)
        or top_k_index.ndim != 2
        or top_k_index.shape[0] != hidden_states.shape[0]
        or top_k_index.shape[1] <= 0
        or top_k_index.shape[1] > module.num_experts
        or top_k_index.device.type != "cpu"
        or top_k_index.dtype != torch.int64
        or top_k_index.requires_grad
    ):
        raise ValueError("expert proxy top-k indices are invalid")
    if (
        not isinstance(top_k_weights, torch.Tensor)
        or top_k_weights.shape != top_k_index.shape
        or top_k_weights.device.type != "cpu"
        or top_k_weights.dtype != torch.float32
    ):
        raise ValueError("expert proxy top-k weights are invalid")
    with torch.no_grad():
        if (
            not bool(torch.isfinite(top_k_weights).all())
            or bool((top_k_weights < 0.0).any())
            or bool((top_k_weights > 1.0).any())
            or bool((top_k_index < 0).any())
            or bool((top_k_index >= module.num_experts).any())
        ):
            raise ValueError("expert proxy routes are outside their domain")
        sorted_indices = torch.sort(top_k_index, dim=-1).values
        if sorted_indices.shape[1] > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError("expert proxy requires unique experts per token")


def _expert_schedule(
    torch: Any, module: Any, top_k_index: Any
) -> tuple[Any, list[int]]:
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index,
            num_classes=module.num_experts,
        ).permute(2, 1, 0)
        expert_indices = [
            int(value[0].item())
            for value in torch.greater(
                expert_mask.sum(dim=(-1, -2)),
                0,
            ).nonzero()
        ]
    return expert_mask, expert_indices


def _installed_expert_forward(
    module: Any,
    hidden_states: Any,
    top_k_index: Any,
    top_k_weights: Any,
    installed_experts_class: type,
) -> Any:
    """Invoke the installed class dispatcher, bypassing the instance patch."""

    return installed_experts_class.forward(
        module,
        hidden_states,
        top_k_index,
        top_k_weights,
    )


def _frozen_expert_autograd_function(
    torch: Any,
    installed_experts_class: type,
) -> type:
    class FrozenOLMoEExperts(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: Any,
            hidden_states: Any,
            top_k_index: Any,
            top_k_weights: Any,
            module: Any,
            executor: ThreadPoolExecutor,
            stats: ExpertProxyStats,
        ) -> Any:
            stats._require_active()
            implementation = _validate_expert_module(
                torch,
                module,
                installed_experts_class,
            )
            _validate_forward_inputs(
                torch,
                module,
                hidden_states,
                top_k_index,
                top_k_weights,
            )
            started = time.perf_counter()
            result = _installed_expert_forward(
                module,
                hidden_states,
                top_k_index,
                top_k_weights,
                installed_experts_class,
            )
            stats._record_forward(time.perf_counter() - started)
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
                ctx.save_for_backward(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    module.gate_up_proj,
                    module.down_proj,
                )
                ctx.module = module
                ctx.executor = executor
                ctx.stats = stats
                ctx.implementation = implementation
            return result

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, ...]:
            if torch.is_grad_enabled():
                raise RuntimeError(
                    "expert proxy does not support higher-order gradients"
                )
            ctx.stats._require_active()
            (
                hidden_states,
                top_k_index,
                top_k_weights,
                gate_up_proj,
                down_proj,
            ) = ctx.saved_tensors
            module = ctx.module
            need_hidden = bool(ctx.needs_input_grad[0])
            need_routes = bool(ctx.needs_input_grad[2])
            if (
                grad_output.device.type != "cpu"
                or grad_output.dtype != hidden_states.dtype
                or grad_output.shape != hidden_states.shape
            ):
                raise RuntimeError("expert proxy output gradient is invalid")
            expert_mask, expert_indices = _expert_schedule(
                torch,
                module,
                top_k_index,
            )

            def calculate(expert_idx: int) -> tuple[Any, ...]:
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                with torch.enable_grad():
                    current_state = hidden_states[token_idx].detach()
                    route = top_k_weights[token_idx, top_k_pos].detach()
                    inputs: list[Any] = []
                    if need_hidden:
                        current_state.requires_grad_(True)
                        inputs.append(current_state)
                    if need_routes:
                        route.requires_grad_(True)
                        inputs.append(route)
                    gate, up = torch.nn.functional.linear(
                        current_state,
                        gate_up_proj[expert_idx].detach(),
                    ).chunk(2, dim=-1)
                    current_hidden_states = module.act_fn(gate) * up
                    current_hidden_states = torch.nn.functional.linear(
                        current_hidden_states,
                        down_proj[expert_idx].detach(),
                    )
                    contribution = (current_hidden_states * route[:, None]).to(
                        hidden_states.dtype
                    )
                    gradients = torch.autograd.grad(
                        contribution,
                        tuple(inputs),
                        grad_outputs=grad_output[token_idx],
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )
                gradient_offset = 0
                hidden_gradient = None
                route_gradient = None
                if need_hidden:
                    hidden_gradient = gradients[gradient_offset].detach()
                    gradient_offset += 1
                if need_routes:
                    route_gradient = gradients[gradient_offset].detach()
                return (
                    expert_idx,
                    token_idx,
                    top_k_pos,
                    hidden_gradient,
                    route_gradient,
                )

            task_started = time.perf_counter()
            results = list(ctx.executor.map(calculate, expert_indices))
            task_seconds = time.perf_counter() - task_started
            reduction_started = time.perf_counter()
            hidden_gradient = torch.zeros_like(hidden_states) if need_hidden else None
            route_gradient = torch.zeros_like(top_k_weights) if need_routes else None
            assigned_routes = 0
            for (
                _expert_idx,
                token_idx,
                top_k_pos,
                _hidden_gradient,
                local_route_gradient,
            ) in results:
                if route_gradient is not None:
                    route_gradient[token_idx, top_k_pos] = local_route_gradient
                    assigned_routes += int(token_idx.numel())
            if route_gradient is not None and assigned_routes != top_k_index.numel():
                raise RuntimeError("expert proxy did not cover every routed pair")
            if hidden_gradient is not None:
                reduction_results = (
                    results if ctx.implementation == "grouped_mm" else reversed(results)
                )
                for (
                    _expert_idx,
                    token_idx,
                    _top_k_pos,
                    local_hidden_gradient,
                    _route_gradient,
                ) in reduction_results:
                    hidden_gradient.index_add_(
                        0,
                        token_idx,
                        local_hidden_gradient.to(hidden_gradient.dtype),
                    )
            reduction_seconds = time.perf_counter() - reduction_started
            ctx.stats._record_backward(
                tasks=len(results),
                task_seconds=task_seconds,
                reduction_seconds=reduction_seconds,
            )
            return (
                hidden_gradient,
                None,
                route_gradient,
                None,
                None,
                None,
            )

    return FrozenOLMoEExperts


@contextmanager
def frozen_olmoe_expert_backward_proxy(
    model: Any,
    *,
    workers: int,
) -> Iterator[ExpertProxyStats]:
    """Install an exact-forward, parallel-backward frozen-expert proxy.

    A differentiable forward and its backward must both finish before this
    context exits.  Retaining a proxied graph or bound forward beyond the
    context is rejected explicitly.
    """

    worker_count = _positive_worker_count(workers)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for the OLMoE expert proxy"
        ) from exc
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("expert proxy requires deterministic Torch algorithms")
    try:
        from engram.tracing.olmoe import _prepare_transformers_imports

        _prepare_transformers_imports()
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeExperts as InstalledOlmoeExperts,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "the installed Transformers OLMoE implementation is unavailable"
        ) from exc
    modules_method = getattr(model, "modules", None)
    if not callable(modules_method):
        raise ValueError("expert proxy model does not expose Torch modules")
    model_modules = list(modules_method())
    look_alikes = [
        module
        for module in model_modules
        if module.__class__ is not InstalledOlmoeExperts
        and (
            module.__class__.__name__ == "OlmoeExperts"
            or isinstance(module, InstalledOlmoeExperts)
        )
    ]
    if look_alikes:
        raise ValueError("expert proxy rejected an OlmoeExperts look-alike or subclass")
    modules = [
        module for module in model_modules if module.__class__ is InstalledOlmoeExperts
    ]
    if not modules:
        raise ValueError(
            "expert proxy found no exact installed Transformers OlmoeExperts modules"
        )
    for module in modules:
        _validate_expert_module(torch, module, InstalledOlmoeExperts)
        if "forward" in vars(module):
            raise ValueError(
                "expert proxy refuses a pre-existing instance-level forward"
            )

    function = _frozen_expert_autograd_function(
        torch,
        InstalledOlmoeExperts,
    )
    stats = ExpertProxyStats(workers=worker_count, patched_layers=len(modules))
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="engram-olmoe-expert-backward",
    )
    patched: list[tuple[Any, bool, Any]] = []
    try:
        for module in modules:
            had_instance_forward = "forward" in vars(module)
            previous_instance_forward = vars(module).get("forward")

            def forward(
                self: Any,
                hidden_states: Any,
                top_k_index: Any,
                top_k_weights: Any,
                *,
                _executor: ThreadPoolExecutor = executor,
                _function: type = function,
                _stats: ExpertProxyStats = stats,
            ) -> Any:
                return _function.apply(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    self,
                    _executor,
                    _stats,
                )

            module.forward = types.MethodType(forward, module)
            patched.append((module, had_instance_forward, previous_instance_forward))
        yield stats
    finally:
        stats._deactivate()
        restored_layers = 0
        try:
            for module, had_instance_forward, previous_instance_forward in patched:
                if had_instance_forward:
                    module.forward = previous_instance_forward
                else:
                    delattr(module, "forward")
                restored_layers += 1
        finally:
            stats.restored_layers = restored_layers
            executor.shutdown(wait=True)
            stats._mark_executor_shutdown()
