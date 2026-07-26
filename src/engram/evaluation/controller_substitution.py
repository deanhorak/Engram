"""Compiled-operator replay through the transformer-free controller boundary."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.evaluation.native_bitnet_attention import _attention_replacement_class
from engram.evaluation.native_bitnet_kernel import _load_frozen_sequences
from engram.evaluation.native_bitnet_parity import (
    _logit_metrics,
    _tensor_metrics,
    _torch_modules,
)
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json


def _hidden_tensor(output):
    if isinstance(output, tuple):
        output = output[0]
    if not hasattr(output, "detach") or output.ndim != 3:
        raise RuntimeError("compiled operator hook did not receive hidden states")
    return output.detach()


def replay_compiled_operator_trajectory(
    controller: FactorizedRecurrentController,
    initial_state: np.ndarray,
    semantic_outputs: np.ndarray,
    episodic_outputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay normalized compiled outputs without decoder-layer scaffolding."""

    initial = np.asarray(initial_state, dtype=np.float32)
    semantic = np.asarray(semantic_outputs, dtype=np.float32)
    episodic = np.asarray(episodic_outputs, dtype=np.float32)
    expected = (*initial.shape[:-1], controller.num_stages, controller.state_dim)
    if semantic.shape != expected or episodic.shape != expected:
        raise ValueError(
            "compiled operator outputs must have shape "
            "[..., num_stages, state_dim]"
        )
    if initial.shape[-1] != controller.state_dim:
        raise ValueError("initial state width does not match the controller")
    rms = np.sqrt(np.mean(np.square(initial), axis=-1, keepdims=True) + 1e-6)
    state = initial / rms
    token_embedding = state.copy()
    states = []
    for stage in range(controller.num_stages):
        supplied = np.concatenate(
            (token_embedding, semantic[..., stage, :], episodic[..., stage, :]),
            axis=-1,
        )
        state = controller.step(state, supplied, stage=stage)
        states.append(state)
    return state, np.stack(states, axis=-2)


def _normalized_operator_trajectory(
    layer_inputs: dict[int, Any],
    layer_outputs: dict[int, Any],
    semantic_outputs: dict[int, Any],
    episodic_outputs: dict[int, Any],
    *,
    num_stages: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    initial = layer_inputs[0].float()
    semantic = []
    episodic = []
    targets = []
    for stage in range(num_stages):
        incoming = layer_inputs[stage].float()
        incoming_rms = (
            incoming.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        )
        semantic.append(semantic_outputs[stage].float() / incoming_rms)
        episodic.append(episodic_outputs[stage].float() / incoming_rms)
        outgoing = layer_outputs[stage].float()
        outgoing_rms = (
            outgoing.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        )
        targets.append(outgoing / outgoing_rms)
    return (
        initial.numpy(),
        np.stack([value.numpy() for value in semantic], axis=-2),
        np.stack([value.numpy() for value in episodic], axis=-2),
        np.stack([value.numpy() for value in targets], axis=-2),
    )


def _trajectory_metrics(
    replay_states: np.ndarray,
    target_states: np.ndarray,
) -> dict[str, Any]:
    error = replay_states - target_states
    stage_nmse = np.mean(np.square(error), axis=tuple(range(error.ndim - 2)) + (-1,))
    return {
        "mean_stage_normalized_mse": float(np.mean(stage_nmse)),
        "maximum_stage_normalized_mse": float(np.max(stage_nmse)),
        "terminal_normalized_mse": float(stage_nmse[-1]),
        "stage_normalized_mse": [float(value) for value in stage_nmse],
    }


def evaluate_native_bitnet_controller_substitution(
    package: str | Path,
    dataset: str | Path,
    controller: str | Path,
    *,
    out: str | Path,
    library: str | Path | None = None,
    attention_library: str | Path | None = None,
    threads: int | None = None,
    native_projections: bool = True,
    sequence_count: int = 2,
    prediction_positions: int = 32,
    record_offset: int = 0,
    local_window: int = 16,
    retrieval_candidates: int = 8,
    retrieval_top_k: int = 4,
    sink_tokens: int = 2,
) -> dict[str, Any]:
    """Replay packed semantic and native episodic outputs through the controller."""

    if sequence_count <= 0 or prediction_positions <= 0:
        raise ValueError("sequence_count and prediction_positions must be positive")
    if prediction_positions % sequence_count:
        raise ValueError("prediction positions must divide evenly across sequences")
    controller_runtime = FactorizedRecurrentController.load(controller)
    if not controller_runtime.has_operator_residual:
        raise ValueError("controller must preserve the exact operator residual")
    if np.any(controller_runtime.step_scale != 0.0):
        raise ValueError("compiled substitution confirmation requires zero correction")

    predictions_per_sequence = prediction_positions // sequence_count
    tokens_per_sequence = predictions_per_sequence + 1
    texts, dataset_evidence = _load_frozen_sequences(
        dataset,
        sequence_count=sequence_count,
        record_offset=record_offset,
    )
    torch, _, functional = _torch_modules()
    with NativeBitNetRuntime(
        package,
        library=library,
        threads=threads,
        native_projections=native_projections,
    ) as runtime:
        layers = runtime.model.model.layers
        if (
            len(layers) != controller_runtime.num_stages
            or int(runtime.model.config.hidden_size) != controller_runtime.state_dim
        ):
            raise ValueError("controller and native BitNet package dimensions differ")
        encoded = []
        for index, text in enumerate(texts):
            tokens = runtime.tokenizer.encode(text, add_special_tokens=True)
            if len(tokens) < tokens_per_sequence:
                raise ValueError(f"controller sequence {index} has too few tokens")
            encoded.append([int(value) for value in tokens[:tokens_per_sequence]])
        input_ids = torch.tensor(encoded, dtype=torch.long)
        labels = input_ids[:, 1:]

        runtime.kernel.clear_metrics()
        with torch.inference_mode():
            baseline_started = time.perf_counter()
            baseline = runtime.forward(
                input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            baseline_seconds = time.perf_counter() - baseline_started
        baseline_calls = list(runtime.kernel.calls)

        Replacement = _attention_replacement_class()
        originals = {
            index: layer.self_attn for index, layer in enumerate(layers)
        }
        layer_inputs: dict[int, Any] = {}
        layer_outputs: dict[int, Any] = {}
        semantic_outputs: dict[int, Any] = {}
        episodic_outputs: dict[int, Any] = {}
        hooks = []

        def pre_hook(index: int):
            def capture(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                layer_inputs[index] = _hidden_tensor(hidden)

            return capture

        def output_hook(destination: dict[int, Any], index: int):
            def capture(_module, _args, output):
                destination[index] = _hidden_tensor(output)

            return capture

        try:
            for index, layer in enumerate(layers):
                layer.self_attn = Replacement(
                    originals[index],
                    mode="native_streaming",
                    local_window=local_window,
                    recurrent_decay=0.99,
                    retrieval_top_k=retrieval_top_k,
                    older_weight=0.5,
                    retrieval_candidates=retrieval_candidates,
                    sink_tokens=sink_tokens,
                    native_attention_library=attention_library,
                )
                hooks.append(
                    layer.register_forward_pre_hook(
                        pre_hook(index),
                        with_kwargs=True,
                    )
                )
                hooks.append(
                    layer.register_forward_hook(
                        output_hook(layer_outputs, index)
                    )
                )
                hooks.append(
                    layer.mlp.register_forward_hook(
                        output_hook(semantic_outputs, index)
                    )
                )
                hooks.append(
                    layer.self_attn.register_forward_hook(
                        output_hook(episodic_outputs, index)
                    )
                )
            runtime.kernel.clear_metrics()
            with torch.inference_mode():
                candidate_started = time.perf_counter()
                candidate = runtime.forward(
                    input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                candidate_seconds = time.perf_counter() - candidate_started
            candidate_calls = list(runtime.kernel.calls)
        finally:
            for hook in hooks:
                hook.remove()
            for index, original in originals.items():
                layers[index].self_attn = original

        expected_stages = set(range(controller_runtime.num_stages))
        for label, captured in (
            ("layer inputs", layer_inputs),
            ("layer outputs", layer_outputs),
            ("semantic outputs", semantic_outputs),
            ("episodic outputs", episodic_outputs),
        ):
            if set(captured) != expected_stages:
                raise RuntimeError(f"incomplete compiled {label} capture")

        initial, semantic, episodic, targets = _normalized_operator_trajectory(
            layer_inputs,
            layer_outputs,
            semantic_outputs,
            episodic_outputs,
            num_stages=controller_runtime.num_stages,
        )
        replay_started = time.perf_counter()
        replay_final, replay_states = replay_compiled_operator_trajectory(
            controller_runtime,
            initial,
            semantic,
            episodic,
        )
        replay_seconds = time.perf_counter() - replay_started
        trajectory = _trajectory_metrics(replay_states, targets)

        replay_tensor = torch.from_numpy(replay_final).to(
            dtype=layer_outputs[controller_runtime.num_stages - 1].dtype
        )
        with torch.inference_mode():
            replay_hidden = runtime.model.model.norm(replay_tensor)
            replay_logits = runtime.model.lm_head(replay_hidden).float()

        baseline_logits = baseline.logits[:, :-1].float()
        candidate_logits = candidate.logits[:, :-1].float()
        replay_prediction_logits = replay_logits[:, :-1]
        baseline_hidden = baseline.hidden_states[-1][:, :-1].float()
        candidate_hidden = candidate.hidden_states[-1][:, :-1].float()
        replay_prediction_hidden = replay_hidden[:, :-1].float()
        baseline_nll = functional.cross_entropy(
            baseline_logits.reshape(-1, baseline_logits.shape[-1]),
            labels.reshape(-1),
        )
        candidate_nll = functional.cross_entropy(
            candidate_logits.reshape(-1, candidate_logits.shape[-1]),
            labels.reshape(-1),
        )
        replay_nll = functional.cross_entropy(
            replay_prediction_logits.reshape(
                -1, replay_prediction_logits.shape[-1]
            ),
            labels.reshape(-1),
        )

        candidate_vs_baseline = {
            "logits": _logit_metrics(baseline_logits, candidate_logits),
            "final_hidden": _tensor_metrics(baseline_hidden, candidate_hidden),
            "nll_delta": float((candidate_nll - baseline_nll).item()),
        }
        replay_vs_candidate = {
            "logits": _logit_metrics(
                candidate_logits,
                replay_prediction_logits,
            ),
            "final_hidden": _tensor_metrics(
                candidate_hidden,
                replay_prediction_hidden,
            ),
            "nll_delta": float((replay_nll - candidate_nll).item()),
        }
        replay_vs_baseline = {
            "logits": _logit_metrics(
                baseline_logits,
                replay_prediction_logits,
            ),
            "final_hidden": _tensor_metrics(
                baseline_hidden,
                replay_prediction_hidden,
            ),
            "nll_delta": float((replay_nll - baseline_nll).item()),
        }

    thresholds = {
        "maximum_teacher_student_kl": 0.05,
        "minimum_teacher_top1_agreement": 0.9,
        "maximum_nll_delta": 0.05,
        "maximum_final_hidden_relative_l2": 0.1,
        "maximum_controller_replay_relative_l2": 0.01,
        "maximum_controller_trajectory_nmse": 0.0225,
        "minimum_unique_sequences": 8,
        "minimum_prediction_positions": 256,
    }
    checks = {
        "teacher_student_kl": (
            replay_vs_baseline["logits"]["mean_kl_divergence"]
            <= thresholds["maximum_teacher_student_kl"]
        ),
        "teacher_top1_agreement": (
            replay_vs_baseline["logits"]["top1_agreement"]
            >= thresholds["minimum_teacher_top1_agreement"]
        ),
        "nll_delta": (
            replay_vs_baseline["nll_delta"] <= thresholds["maximum_nll_delta"]
        ),
        "final_hidden_relative_l2": (
            replay_vs_baseline["final_hidden"]["relative_l2"]
            <= thresholds["maximum_final_hidden_relative_l2"]
        ),
        "controller_replay_relative_l2": (
            replay_vs_candidate["final_hidden"]["relative_l2"]
            <= thresholds["maximum_controller_replay_relative_l2"]
        ),
        "controller_trajectory_nmse": (
            trajectory["terminal_normalized_mse"]
            <= thresholds["maximum_controller_trajectory_nmse"]
        ),
        "unique_sequences": (
            dataset_evidence["unique_sequences"]
            >= thresholds["minimum_unique_sequences"]
        ),
        "prediction_positions": (
            prediction_positions >= thresholds["minimum_prediction_positions"]
        ),
    }
    frozen_gate_passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment": "native_bitnet_compiled_operator_controller_substitution",
        "status": (
            "frozen_compiled_controller_confirmation"
            if frozen_gate_passed
            else "compiled_controller_development_evaluation"
        ),
        "package": str(Path(package).resolve()),
        "controller": {
            "path": str(Path(controller).resolve()),
            **controller_runtime.metadata(),
        },
        "dataset": {
            **dataset_evidence,
            "role": (
                "frozen_confirmation"
                if sequence_count >= 8 and prediction_positions >= 256
                else "development_evaluation"
            ),
            "sequences": sequence_count,
            "tokens_per_sequence": tokens_per_sequence,
            "prediction_positions": prediction_positions,
        },
        "configuration": {
            "semantic_operator": "native_packed_bitnet_phase_stream",
            "episodic_operator": "native_streaming_attention",
            "native_packed_attention_projections": bool(native_projections),
            "local_window": local_window,
            "retrieval_candidates": retrieval_candidates,
            "retrieval_top_k": retrieval_top_k,
            "sink_tokens": sink_tokens,
            "controller_correction_enabled": False,
        },
        "timing": {
            "dense_attention_baseline_seconds": baseline_seconds,
            "compiled_candidate_seconds": candidate_seconds,
            "controller_replay_seconds": replay_seconds,
        },
        "native_mlp_calls": {
            "dense_attention_baseline": len(baseline_calls),
            "compiled_candidate": len(candidate_calls),
        },
        "trajectory": trajectory,
        "compiled_candidate_vs_dense_baseline": candidate_vs_baseline,
        "controller_replay_vs_compiled_candidate": replay_vs_candidate,
        "controller_replay_vs_dense_baseline": replay_vs_baseline,
        "thresholds": thresholds,
        "checks": checks,
        "frozen_gate_passed": frozen_gate_passed,
        "scope": {
            "source_mlp_tensors_loaded": False,
            "semantic_outputs_from_compiled_cpu_kernel": True,
            "episodic_outputs_from_native_bounded_kernel": True,
            "decoder_layer_residual_scaffold_used_for_operator_capture": True,
            "decoder_layer_residual_scaffold_used_for_controller_replay": False,
            "incremental_generation_through_controller_tested": False,
        },
        "decision": (
            "compiled_operator_controller_gate_pass_incremental_runtime_next"
            if frozen_gate_passed
            else "attribute_compiled_operator_failure_before_runtime_integration"
        ),
    }
    atomic_json(Path(out), report)
    return report


__all__ = [
    "evaluate_native_bitnet_controller_substitution",
    "replay_compiled_operator_trajectory",
]
