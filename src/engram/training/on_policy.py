"""On-policy recalibration for native-gate utility residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from engram.evaluation.mlp_intervention import (
    _evaluation_sequence_hashes,
    _relative_and_cosine_rows,
)
from engram.models.inspection import inspect_model, resolve_model_path
from engram.semantic.swiglu import silu
from engram.training.sparse_teacher import _ids, _load_jsonl
from engram.training.structured_experts import (
    LowRankUtilityResidual,
    _selected_channel_output,
    _stable_top_indices,
    _stats,
    _wrap_native_gate_channel_mlp_class,
    fit_low_rank_utility_residual,
    load_native_gate_utility_residual,
    native_gate_channel_traffic,
)
from engram.utils import atomic_json, sha256_file


def _import_causal_lm() -> tuple[Any, Any]:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except ImportError:
                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("the local Transformers causal-LM stack could not be imported") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _utility_residual_targets(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    *,
    input_coordinates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.argsort(-np.abs(states), axis=1, kind="stable")[
        :, :input_coordinates
    ]
    partial_states = np.zeros_like(states)
    partial_states[np.arange(len(states))[:, None], coordinates] = states[
        np.arange(len(states))[:, None], coordinates
    ]
    partial_gate = partial_states @ gate.T
    full_gate = states @ gate.T
    up_values = states @ up.T
    value_norms = np.linalg.norm(down, axis=0)[None, :]
    base_logits = np.log(np.abs(silu(partial_gate)) * value_norms + 1e-8)
    exact_logits = np.log(
        np.abs(silu(full_gate) * up_values) * value_norms + 1e-8
    )
    targets = np.clip(exact_logits - base_logits, -8.0, 8.0)
    targets -= np.mean(targets, axis=1, keepdims=True)
    return targets, base_logits, partial_gate, up_values


def _collect_sparse_states(
    model: Any,
    records: list[dict[str, Any]],
    tokenizer: Any,
    torch: Any,
    device: str,
    *,
    maximum_states: int,
) -> tuple[list[np.ndarray], int]:
    layers = len(model.model.layers)
    batches: list[list[np.ndarray]] = [[] for _ in range(layers)]
    counts = [0] * layers
    handles = []
    per_sequence = max(1, int(np.ceil(maximum_states / len(records))))

    def hook_for(layer: int):
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            remaining = maximum_states - counts[layer]
            if remaining <= 0:
                return
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            count = min(len(values), per_sequence, remaining)
            indices = torch.linspace(
                0, len(values) - 1, steps=count, device=values.device
            ).round().long()
            values = values[indices].cpu().numpy().astype(np.float64)
            batches[layer].append(values)
            counts[layer] += len(values)

        return capture

    for layer, decoder in enumerate(model.model.layers):
        handles.append(decoder.mlp.register_forward_pre_hook(hook_for(layer)))
    try:
        contributing_sequences = 0
        with torch.inference_mode():
            for record in records:
                input_ids = _ids(record, tokenizer, torch, device)
                attention_mask = torch.ones_like(input_ids)
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
                contributing_sequences += 1
                if min(counts) >= maximum_states:
                    break
    finally:
        for handle in handles:
            handle.remove()
    if min(counts) < maximum_states:
        raise ValueError(
            f"dataset produced only {min(counts)} states; {maximum_states} required"
        )
    return (
        [np.concatenate(values)[:maximum_states] for values in batches],
        contributing_sequences,
    )


def recalibrate_native_gate_residual(
    model: str | Path,
    calibration_dataset: str | Path,
    initial_residual: str | Path,
    out: str | Path,
    *,
    rank: int = 16,
    blends: Iterable[float] = (0.5, 0.65, 0.8, 1.0),
    input_fraction: float = 0.625,
    top_k: int = 512,
    regularization: float = 4000.0,
    fit_fraction: float = 0.75,
    fit_states: int = 512,
    validation_states: int = 128,
    device: str = "cpu",
) -> dict[str, Any]:
    """Refit residuals on states induced by the deployed hard sparse student."""

    try:
        import torch
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for recalibration") from exc
    AutoModelForCausalLM, AutoTokenizer = _import_causal_lm()
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not isinstance(fit_states, int) or fit_states <= rank:
        raise ValueError("fit_states must be an integer greater than rank")
    if not isinstance(validation_states, int) or validation_states <= 0:
        raise ValueError("validation_states must be a positive integer")
    if not np.isfinite(fit_fraction) or not 0 < fit_fraction < 1:
        raise ValueError("fit_fraction must lie in (0, 1)")
    blend_values = tuple(dict.fromkeys(float(value) for value in blends))
    if not blend_values or any(
        not np.isfinite(value) or value < 0 for value in blend_values
    ):
        raise ValueError("blends must be finite and nonnegative")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    dataset_path = Path(calibration_dataset)
    residual_path = Path(initial_residual)
    records = _load_jsonl(dataset_path, None)
    split = round(len(records) * fit_fraction)
    split = max(1, min(len(records) - 1, split))
    fit_records = records[:split]
    validation_records = records[split:]

    student = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to(device)
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:
        class TokenIdOnlyTokenizer:
            pad_token_id = student.config.pad_token_id
            eos_token_id = student.config.eos_token_id

        tokenizer = TokenIdOnlyTokenizer()
    fit_hashes = _evaluation_sequence_hashes(fit_records, tokenizer)
    validation_hashes = _evaluation_sequence_hashes(validation_records, tokenizer)
    if set(fit_hashes).intersection(validation_hashes):
        raise ValueError("on-policy fit and validation contain matching sequences")

    wrapper_type = _wrap_native_gate_channel_mlp_class(torch)
    wrappers = []
    initial_predictors = []
    initial_blend = None
    for layer, decoder in enumerate(student.model.layers):
        predictor, layer_blend = load_native_gate_utility_residual(
            residual_path,
            layer,
            expected_source_hash=inspection.source_hash,
        )
        if initial_blend is not None and layer_blend != initial_blend:
            raise ValueError("initial residual blends differ across layers")
        initial_blend = layer_blend
        wrapper = wrapper_type(
            decoder.mlp,
            top_k=top_k,
            input_fraction=input_fraction,
            utility_residual=predictor,
            residual_blend=layer_blend,
        ).to(device)
        wrapper.eval()
        wrapper.use_training_surrogate = False
        decoder.mlp = wrapper
        wrappers.append(wrapper)
        initial_predictors.append(predictor)
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    fit_inputs, fit_contributing_sequences = _collect_sparse_states(
        student,
        fit_records,
        tokenizer,
        torch,
        device,
        maximum_states=fit_states,
    )
    validation_inputs, validation_contributing_sequences = _collect_sparse_states(
        student,
        validation_records,
        tokenizer,
        torch,
        device,
        maximum_states=validation_states,
    )

    accumulators = {
        blend: {"relative": [], "cosine": []} for blend in blend_values
    }
    baseline = {"relative": [], "cosine": []}
    fitted: list[LowRankUtilityResidual] = []
    layer_reports = []
    input_coordinates = wrappers[0].input_coordinates
    for layer, wrapper in enumerate(wrappers):
        gate = wrapper.gate_weight.detach().cpu().numpy().astype(np.float64)
        up = wrapper.up_weight.detach().cpu().numpy().astype(np.float64)
        down = wrapper.down_weight.detach().cpu().numpy().astype(np.float64)
        targets, _, _, _ = _utility_residual_targets(
            fit_inputs[layer],
            gate,
            up,
            down,
            input_coordinates=input_coordinates,
        )
        predictor = fit_low_rank_utility_residual(
            fit_inputs[layer],
            targets,
            rank=rank,
            regularization=regularization,
        )
        predictor = LowRankUtilityResidual(
            predictor.input_factors.astype(np.float32),
            predictor.output_factors.astype(np.float32),
            predictor.bias.astype(np.float32),
        )
        fitted.append(predictor)
        states = validation_inputs[layer]
        _, base_logits, partial_gate, up_values = _utility_residual_targets(
            states,
            gate,
            up,
            down,
            input_coordinates=input_coordinates,
        )
        full_gate = states @ gate.T
        dense_target = (silu(full_gate) * up_values) @ down.T
        initial_logits = base_logits + initial_blend * initial_predictors[layer].predict(
            states
        )
        initial_selected = _stable_top_indices(initial_logits, top_k)
        initial_output = _selected_channel_output(
            partial_gate, up_values, down, initial_selected
        )
        initial_error, initial_cosine = _relative_and_cosine_rows(
            initial_output, dense_target
        )
        baseline["relative"].extend(initial_error.tolist())
        baseline["cosine"].extend(initial_cosine.tolist())
        predicted = predictor.predict(states)
        layer_entry = {"layer": layer, "configurations": []}
        for blend in blend_values:
            selected = _stable_top_indices(base_logits + blend * predicted, top_k)
            output = _selected_channel_output(partial_gate, up_values, down, selected)
            error, cosine = _relative_and_cosine_rows(output, dense_target)
            accumulators[blend]["relative"].extend(error.tolist())
            accumulators[blend]["cosine"].extend(cosine.tolist())
            layer_entry["configurations"].append(
                {"blend": blend, "local_relative_l2": float(np.mean(error))}
            )
        layer_reports.append(layer_entry)

    baseline_stats = _stats(baseline["relative"])
    configurations = []
    for blend, values in accumulators.items():
        configurations.append(
            {
                "blend": blend,
                "local_relative_l2": _stats(values["relative"]),
                "local_cosine": _stats(values["cosine"]),
            }
        )
    configurations.sort(key=lambda item: item["local_relative_l2"]["mean"])
    selected = configurations[0]
    improvement = 1.0 - selected["local_relative_l2"]["mean"] / baseline_stats["mean"]
    traffic = native_gate_channel_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        input_fraction=input_fraction,
        active_records=top_k,
    )
    predictor_bytes = fitted[0].parameter_bytes()
    traffic_fraction = (
        traffic.total_weight_bytes + predictor_bytes
    ) / traffic.dense_weight_bytes
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    tensor_path = target / "native_gate_on_policy_residual.safetensors"
    tensors = {}
    for layer, predictor in enumerate(fitted):
        prefix = f"layers.{layer}.utility_residual"
        tensors[f"{prefix}.input_factors"] = predictor.input_factors.astype(np.float32)
        tensors[f"{prefix}.output_factors"] = predictor.output_factors.astype(np.float32)
        tensors[f"{prefix}.bias"] = predictor.bias.astype(np.float32)
    save_file(
        tensors,
        tensor_path,
        metadata={
            "format": "engram_native_gate_utility_residual_v1",
            "source_model_hash": inspection.source_hash,
            "rank": str(rank),
            "blend": str(selected["blend"]),
            "parent_sha256": sha256_file(residual_path),
            "mlp_weights_sha256": inspection.source_hash,
            "calibration_dataset_sha256": sha256_file(dataset_path),
            "state_distribution": "hard_sparse_student",
            "input_fraction": str(input_fraction),
            "top_k": str(top_k),
        },
    )
    checks = {
        "sequence_disjoint_shadow": True,
        "local_improvement": improvement > 0,
        "material_local_improvement": improvement >= 0.10,
        "projected_traffic": traffic_fraction <= 0.45,
    }
    report = {
        "schema_version": 1,
        "experiment": "native_gate_on_policy_residual_recalibration",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "rank": rank,
            "blends": list(blend_values),
            "input_fraction": input_fraction,
            "top_k": top_k,
            "regularization": regularization,
            "fit_fraction": fit_fraction,
            "fit_states_per_layer": fit_states,
            "validation_states_per_layer": validation_states,
            "device": device,
        },
        "data_separation": {
            "fit_sequences": len(fit_records),
            "validation_sequences": len(validation_records),
            "fit_contributing_sequences": fit_contributing_sequences,
            "validation_contributing_sequences": validation_contributing_sequences,
            "state_sampling": "evenly_spaced_per_sequence",
            "overlapping_sequences": 0,
            "dataset_sha256": sha256_file(dataset_path),
        },
        "baseline": {
            "artifact_sha256": sha256_file(residual_path),
            "blend": initial_blend,
            "local_relative_l2": baseline_stats,
        },
        "artifact_binding": {
            "mlp_weights_sha256": inspection.source_hash,
            "parent_residual_sha256": sha256_file(residual_path),
            "input_fraction": input_fraction,
            "top_k": top_k,
        },
        "configurations": configurations,
        "selection": selected,
        "relative_improvement": improvement,
        "projected_traffic": {
            **traffic.to_dict(),
            "utility_residual_bytes": predictor_bytes,
            "with_utility_residual_fraction_of_dense": traffic_fraction,
        },
        "screen": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "run_candidate_policy_rollout_before_causal_gate"
                if all(checks.values())
                else "reject_on_policy_recalibration"
            ),
        },
        "artifact": {
            "path": str(tensor_path.resolve()),
            "sha256": sha256_file(tensor_path),
            "format": "engram_native_gate_utility_residual_v1",
        },
        "layers": layer_reports,
    }
    atomic_json(target / "native_gate_on_policy_residual.json", report)
    lines = [
        "# On-policy native-gate residual recalibration",
        "",
        f"Decision: **{report['screen']['decision']}**",
        "",
        f"Initial sparse-state local relative L2: {baseline_stats['mean']:.6f}.",
        f"Recalibrated sparse-state local relative L2: "
        f"{selected['local_relative_l2']['mean']:.6f}.",
        f"Relative improvement: {improvement:.2%}.",
        f"Selected blend: {selected['blend']:.2f}.",
        f"Projected traffic: {traffic_fraction:.6f}× dense.",
        "",
        "The validation trajectories are sequence-disjoint from the fitted trajectories; the",
        "untouched causal corpus remains a separate progression gate.",
        "",
    ]
    (target / "native_gate_on_policy_residual.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = ["recalibrate_native_gate_residual"]
