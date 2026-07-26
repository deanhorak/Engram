"""Oracle record-concentration analysis for packaged native BitNet MLPs.

This is deliberately a Gate-1 ceiling, not a practical router.  The evaluator
uses the exact dense record coefficients produced by the teacher, ranks records
by the norm of their individual down-projection contribution, and then measures
how well selected records reconstruct the full MLP output.  A practical
Milestone-2 implementation must later predict the same membership without
executing the dense gate/up path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.models.native_bitnet import (
    LoadedNativeBitNetArtifact,
    _activation_quant,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
)
from engram.evaluation.native_bitnet_parity import _logit_metrics, _tensor_metrics
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.utils import atomic_json, sha256_file


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize empty or non-finite oracle metrics")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _row_metrics(
    approximation: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    exact_norm = np.linalg.norm(exact, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    relative = np.linalg.norm(actual - exact, axis=1) / np.maximum(exact_norm, 1e-12)
    cosine = np.sum(actual * exact, axis=1) / np.maximum(
        actual_norm * exact_norm, 1e-12
    )
    cosine[(actual_norm <= 1e-12) & (exact_norm <= 1e-12)] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


def analyze_native_bitnet_layer_oracle(
    artifact: LoadedNativeBitNetArtifact,
    layer: int,
    hidden: np.ndarray,
    *,
    fractions: Sequence[float] = (0.05, 0.1, 0.15, 0.175, 0.25, 0.5, 1.0),
) -> dict[str, Any]:
    """Measure the exact additive-record oracle for one BitNet MLP layer."""

    states = np.asarray(hidden, dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != artifact.hidden_size:
        raise ValueError("hidden must be a [records, hidden_size] matrix")
    if not states.size or not np.all(np.isfinite(states)):
        raise ValueError("hidden must be non-empty and finite")
    requested = tuple(dict.fromkeys(float(value) for value in fractions))
    if not requested or any(
        not np.isfinite(value) or not 0 < value <= 1 for value in requested
    ):
        raise ValueError("fractions must lie in (0, 1]")

    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = np.asarray(decoded["gate_codes"], dtype=np.float32)
    up_codes = np.asarray(decoded["up_codes"], dtype=np.float32)
    down_codes = np.asarray(decoded["down_codes"], dtype=np.float32)
    quantized_state = _activation_quant(states)
    gate = (
        quantized_state @ gate_codes.T
        * np.asarray(decoded["gate_scale"], dtype=np.float32)
    )
    up = (
        quantized_state @ up_codes.T
        * np.asarray(decoded["up_scale"], dtype=np.float32)
    )
    raw_activation = np.maximum(gate, np.float32(0.0)) ** 2 * up
    variance = np.mean(
        raw_activation.astype(np.float32) ** 2,
        axis=1,
        keepdims=True,
    )
    normalized = raw_activation * np.reciprocal(
        np.sqrt(variance + np.float32(artifact.rms_norm_eps))
    )
    normalized *= np.asarray(decoded["ffn_sub_norm"], dtype=np.float32)
    coefficients = _activation_quant(normalized)
    down_scale = np.asarray(decoded["down_scale"], dtype=np.float32)
    reference = coefficients @ down_codes.T * down_scale

    # Each record contributes coefficient[i] * down[:, i].  Ranking by the
    # squared L2 norm of that vector is the strongest independent-record
    # magnitude oracle available before solving a combinatorial subset problem.
    value_norm_squared = np.sum(down_codes * down_codes, axis=0, dtype=np.float32)
    utility = coefficients * coefficients * value_norm_squared[None, :]
    order = np.argsort(-utility, axis=1, kind="stable")
    sorted_utility = np.take_along_axis(utility, order, axis=1)
    total_utility = np.sum(sorted_utility, axis=1)
    cumulative_utility = np.cumsum(sorted_utility, axis=1)
    width = artifact.intermediate_size

    fraction_reports: list[dict[str, Any]] = []
    for fraction in requested:
        count = min(width, max(1, int(np.ceil(fraction * width))))
        selected = order[:, :count]
        sparse_coefficients = np.zeros_like(coefficients)
        np.put_along_axis(
            sparse_coefficients,
            selected,
            np.take_along_axis(coefficients, selected, axis=1),
            axis=1,
        )
        approximation = sparse_coefficients @ down_codes.T * down_scale
        relative, cosine = _row_metrics(approximation, reference)
        captured = cumulative_utility[:, count - 1] / np.maximum(
            total_utility, np.float32(1e-20)
        )
        fraction_reports.append(
            {
                "requested_fraction": fraction,
                "record_count": count,
                "actual_fraction": count / width,
                "relative_l2": _summary(relative.tolist()),
                "cosine_similarity": _summary(cosine.tolist()),
                "independent_contribution_energy": _summary(captured.tolist()),
            }
        )

    required: dict[str, dict[str, float | int]] = {}
    for target in (0.9, 0.95, 0.99):
        counts = []
        for row, total in zip(cumulative_utility, total_utility, strict=True):
            if total <= 1e-20:
                counts.append(0)
            else:
                counts.append(int(np.searchsorted(row, target * total) + 1))
        fractions_required = np.asarray(counts, dtype=np.float64) / width
        required[str(target)] = {
            "mean_records": float(np.mean(counts)),
            "p95_records": float(np.percentile(counts, 95)),
            "mean_fraction": float(np.mean(fractions_required)),
            "p95_fraction": float(np.percentile(fractions_required, 95)),
        }

    zero_fraction = np.mean(coefficients == 0, axis=1)
    return {
        "layer": int(layer),
        "states": int(states.shape[0]),
        "hidden_size": artifact.hidden_size,
        "intermediate_size": width,
        "oracle": "exact_coefficient_times_down_column_l2",
        "decomposition": (
            "full native activation quantization and intermediate RMS "
            "normalization, followed by additive down-projection records"
        ),
        "coefficient_zero_fraction": _summary(zero_fraction.tolist()),
        "records_for_independent_contribution_energy": required,
        "fractions": fraction_reports,
    }


def _load_records(path: Path, *, offset: int, count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    usable = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record {line_number} must be an object")
            if usable >= offset:
                records.append(value)
                if len(records) == count:
                    break
            usable += 1
    if len(records) != count:
        raise ValueError(f"dataset contains only {len(records)} selected records")
    return records


def evaluate_native_bitnet_oracle(
    package: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    layers: Sequence[int] = (0, 14, 29),
    samples: int = 2,
    max_tokens: int = 8,
    record_offset: int = 0,
    fractions: Sequence[float] = (0.05, 0.1, 0.15, 0.175, 0.25, 0.5, 1.0),
    library: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Capture trained states and run the BitNet record-concentration oracle."""

    if samples <= 0 or max_tokens <= 0:
        raise ValueError("samples and max_tokens must be positive")
    if record_offset < 0:
        raise ValueError("record_offset must be non-negative")
    package_path = Path(package).resolve()
    dataset_path = Path(dataset).resolve()
    selected_layers = tuple(dict.fromkeys(int(value) for value in layers))
    if not selected_layers:
        raise ValueError("layers must not be empty")
    records = _load_records(dataset_path, offset=record_offset, count=samples)
    started = time.perf_counter()

    with NativeBitNetRuntime(
        package_path,
        library=library,
        threads=threads,
    ) as runtime:
        artifact = load_native_bitnet_artifact(runtime.artifact_path)
        if any(not 0 <= layer < len(artifact.layers) for layer in selected_layers):
            raise ValueError("requested layer is outside the package")
        captured: dict[int, list[np.ndarray]] = {layer: [] for layer in selected_layers}
        hooks = []

        def capture_input(layer: int):
            def hook(_module, args, kwargs):
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None:
                    raise RuntimeError("BitNet MLP hook received no hidden states")
                captured[layer].append(
                    hidden.detach().float().reshape(-1, artifact.hidden_size).cpu().numpy()
                )

            return hook

        for layer in selected_layers:
            hooks.append(
                runtime.model.model.layers[layer].mlp.register_forward_pre_hook(
                    capture_input(layer),
                    with_kwargs=True,
                )
            )
        try:
            import torch

            with torch.inference_mode():
                for record in records:
                    if "input_ids" in record:
                        token_ids = [int(value) for value in record["input_ids"]]
                    else:
                        token_ids = runtime.encode(str(record.get("text", "")))
                    token_ids = token_ids[:max_tokens]
                    if not token_ids:
                        raise ValueError("selected record tokenized to an empty sequence")
                    runtime.model(
                        input_ids=torch.tensor([token_ids], dtype=torch.long),
                        use_cache=False,
                    )
        finally:
            for hook in hooks:
                hook.remove()

        layer_reports = []
        for layer in selected_layers:
            if not captured[layer]:
                raise RuntimeError(f"no MLP inputs captured for layer {layer}")
            layer_reports.append(
                analyze_native_bitnet_layer_oracle(
                    artifact,
                    layer,
                    np.concatenate(captured[layer], axis=0),
                    fractions=fractions,
                )
            )

    target_fraction = min(
        layer_reports[0]["fractions"],
        key=lambda item: abs(item["actual_fraction"] - 0.25),
    )["actual_fraction"]
    target_rows = [
        min(
            report["fractions"],
            key=lambda item: abs(item["actual_fraction"] - target_fraction),
        )
        for report in layer_reports
    ]
    mean_relative = float(
        np.mean([row["relative_l2"]["mean"] for row in target_rows])
    )
    worst_layer_p95 = float(
        np.max([row["relative_l2"]["p95"] for row in target_rows])
    )
    mean_cosine = float(
        np.mean([row["cosine_similarity"]["mean"] for row in target_rows])
    )
    progression_passed = (
        target_fraction <= 0.25 + 1e-9
        and mean_relative <= 0.10
        and worst_layer_p95 <= 0.20
        and mean_cosine >= 0.99
    )
    result = {
        "experiment": "native_bitnet_oracle_record_concentration",
        "scope": "local_mlp_gate1_ceiling",
        "package": str(package_path),
        "artifact_sha256": artifact.payload_sha256,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_offset": record_offset,
            "samples": samples,
            "maximum_tokens_per_sample": max_tokens,
        },
        "layers": layer_reports,
        "progression_screen": {
            "target_fraction": target_fraction,
            "maximum_mean_relative_l2": 0.10,
            "maximum_worst_layer_p95_relative_l2": 0.20,
            "minimum_mean_cosine_similarity": 0.99,
            "observed_mean_relative_l2": mean_relative,
            "observed_worst_layer_p95_relative_l2": worst_layer_p95,
            "observed_mean_cosine_similarity": mean_cosine,
            "passed": progression_passed,
        },
        "decision": (
            "proceed_to_practical_router_and_causal_oracle_confirmation"
            if progression_passed
            else "oracle_concentration_insufficient_at_25_percent"
        ),
        "milestone_2_status": "blocked",
        "caveat": (
            "This oracle uses exact teacher coefficients and therefore cannot "
            "serve as a practical router or pass Milestone 2 by itself."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out, result)
    return result


def _oracle_mlp_class():
    import torch.nn as nn

    class NativeBitNetOracleMLP(nn.Module):
        def __init__(self, kernel, layer: int, top_k: int) -> None:
            super().__init__()
            self.kernel = kernel
            self.layer = int(layer)
            self.top_k = int(top_k)

        def forward(self, hidden_states):
            return self.kernel.forward_oracle(
                self.layer,
                hidden_states,
                top_k=self.top_k,
            )

    return NativeBitNetOracleMLP


def evaluate_native_bitnet_oracle_causal(
    package: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    fraction: float = 0.25,
    layer_fractions: Sequence[float] | None = None,
    sequence_count: int = 1,
    predictions_per_sequence: int = 8,
    record_offset: int = 0,
    library: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run an all-layer causal substitution using exact teacher memberships."""

    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    if sequence_count <= 0 or predictions_per_sequence <= 0:
        raise ValueError("causal evidence counts must be positive")
    if record_offset < 0:
        raise ValueError("record_offset must be non-negative")
    package_path = Path(package).resolve()
    dataset_path = Path(dataset).resolve()
    records = _load_records(
        dataset_path,
        offset=record_offset,
        count=sequence_count,
    )
    started = time.perf_counter()
    import torch
    import torch.nn.functional as functional

    with NativeBitNetRuntime(
        package_path,
        library=library,
        threads=threads,
    ) as runtime:
        width = int(runtime.manifest["model"]["intermediate_size"])
        layer_count = int(runtime.manifest["model"]["num_hidden_layers"])
        requested_layer_fractions = (
            tuple(float(value) for value in layer_fractions)
            if layer_fractions is not None
            else (float(fraction),) * layer_count
        )
        if len(requested_layer_fractions) != layer_count or any(
            not np.isfinite(value) or not 0 < value <= 1
            for value in requested_layer_fractions
        ):
            raise ValueError(
                "layer_fractions must contain one value in (0, 1] per layer"
            )
        layer_top_ks = tuple(
            min(width, max(1, int(np.ceil(value * width))))
            for value in requested_layer_fractions
        )
        required_tokens = predictions_per_sequence + 1
        encoded: list[list[int]] = []
        for index, record in enumerate(records):
            if "input_ids" in record:
                token_ids = [int(value) for value in record["input_ids"]]
            else:
                token_ids = runtime.encode(str(record.get("text", "")))
            if len(token_ids) < required_tokens:
                raise ValueError(
                    f"selected record {index} has {len(token_ids)} tokens; "
                    f"required {required_tokens}"
                )
            encoded.append(token_ids[:required_tokens])
        input_ids = torch.tensor(encoded, dtype=torch.long)

        with torch.inference_mode():
            baseline_started = time.perf_counter()
            reference = runtime.model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            baseline_seconds = time.perf_counter() - baseline_started

        OracleMLP = _oracle_mlp_class()
        for layer, module in enumerate(runtime.model.model.layers):
            module.mlp = OracleMLP(runtime.kernel, layer, layer_top_ks[layer])
        runtime.kernel.clear_metrics()
        with torch.inference_mode():
            oracle_started = time.perf_counter()
            candidate = runtime.model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
            )
            oracle_seconds = time.perf_counter() - oracle_started
        calls = list(runtime.kernel.calls)

    reference_logits = reference.logits[:, :-1, :]
    candidate_logits = candidate.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    logits = _logit_metrics(reference_logits, candidate_logits)
    hidden = _tensor_metrics(
        reference.hidden_states[-1][:, :-1, :],
        candidate.hidden_states[-1][:, :-1, :],
    )
    reference_nll = functional.cross_entropy(
        reference_logits.float().reshape(-1, reference_logits.shape[-1]),
        labels.reshape(-1),
    )
    candidate_nll = functional.cross_entropy(
        candidate_logits.float().reshape(-1, candidate_logits.shape[-1]),
        labels.reshape(-1),
    )
    nll_delta = float((candidate_nll - reference_nll).item())
    thresholds = {
        "maximum_teacher_student_kl": 0.05,
        "minimum_top1_agreement": 0.90,
        "maximum_nll_delta": 0.05,
        "maximum_final_hidden_relative_l2": 0.10,
    }
    quality_passed = (
        logits["mean_kl_divergence"] <= thresholds["maximum_teacher_student_kl"]
        and logits["top1_agreement"] >= thresholds["minimum_top1_agreement"]
        and nll_delta <= thresholds["maximum_nll_delta"]
        and hidden["relative_l2"] <= thresholds["maximum_final_hidden_relative_l2"]
    )
    if len(calls) != layer_count:
        raise RuntimeError("oracle kernel did not execute exactly once per layer")
    result = {
        "experiment": "native_bitnet_oracle_causal_substitution",
        "scope": "all_mlp_layers_exact_membership_ceiling",
        "package": str(package_path),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_offset": record_offset,
        },
        "configuration": {
            "requested_fraction": fraction if layer_fractions is None else None,
            "layer_requested_fractions": list(requested_layer_fractions),
            "layer_top_ks": list(layer_top_ks),
            "layer_actual_fractions": [value / width for value in layer_top_ks],
            "mean_actual_fraction": float(np.mean(layer_top_ks) / width),
            "maximum_actual_fraction": max(layer_top_ks) / width,
            "minimum_actual_fraction": min(layer_top_ks) / width,
            "intermediate_size": width,
            "layer_count": layer_count,
            "sequence_count": sequence_count,
            "predictions_per_sequence": predictions_per_sequence,
            "prediction_positions": sequence_count * predictions_per_sequence,
        },
        "quality": {
            "logits": logits,
            "reference_nll": float(reference_nll.item()),
            "candidate_nll": float(candidate_nll.item()),
            "nll_delta": nll_delta,
            "final_hidden": hidden,
        },
        "thresholds": thresholds,
        "quality_passed": quality_passed,
        "kernel": {
            "calls": len(calls),
            "baseline_seconds": baseline_seconds,
            "oracle_seconds": oracle_seconds,
            "gate_up_stream_bytes": sum(
                int(call["gate_up_stream_bytes"]) for call in calls
            ),
            "norm_stream_bytes": sum(
                int(call["norm_stream_bytes"]) for call in calls
            ),
            "selected_down_stream_bytes": sum(
                int(call["down_stream_bytes"]) for call in calls
            ),
            "note": (
                "exact membership still executes the full gate/up coefficient "
                "path; these bytes are not a practical-router traffic result"
            ),
        },
        "decision": (
            "train_practical_router_against_oracle_membership"
            if quality_passed
            else "reduce_error_before_practical_router_training"
        ),
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out, result)
    return result


def evaluate_native_bitnet_oracle_layer_sweep(
    package: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    fractions: Sequence[float] = (0.15, 0.20, 0.25, 0.30, 0.35),
    mean_budget: float = 0.25,
    sequence_count: int = 2,
    tokens_per_sequence: int = 16,
    record_offset: int = 0,
    library: str | Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Fit a layer-adaptive oracle budget from trained local MLP states."""

    requested = tuple(dict.fromkeys(float(value) for value in fractions))
    if not requested or any(
        not np.isfinite(value) or not 0 < value <= 1 for value in requested
    ):
        raise ValueError("fractions must lie in (0, 1]")
    if (
        not np.isfinite(mean_budget)
        or not min(requested) <= mean_budget <= max(requested)
    ):
        raise ValueError("mean_budget must lie within the swept fractions")
    if sequence_count <= 0 or tokens_per_sequence <= 0:
        raise ValueError("evidence counts must be positive")
    package_path = Path(package).resolve()
    dataset_path = Path(dataset).resolve()
    records = _load_records(
        dataset_path,
        offset=record_offset,
        count=sequence_count,
    )
    started = time.perf_counter()
    import torch

    with NativeBitNetRuntime(
        package_path,
        library=library,
        threads=threads,
    ) as runtime:
        width = int(runtime.manifest["model"]["intermediate_size"])
        layer_count = int(runtime.manifest["model"]["num_hidden_layers"])
        captured_inputs: dict[int, Any] = {}
        captured_outputs: dict[int, Any] = {}
        hooks = []

        def input_hook(layer: int):
            def capture(_module, args, kwargs):
                value = args[0] if args else kwargs.get("hidden_states")
                captured_inputs[layer] = value.detach().contiguous()

            return capture

        def output_hook(layer: int):
            def capture(_module, _args, value):
                captured_outputs[layer] = value.detach().contiguous()

            return capture

        for layer, module in enumerate(runtime.model.model.layers):
            hooks.append(
                module.mlp.register_forward_pre_hook(
                    input_hook(layer),
                    with_kwargs=True,
                )
            )
            hooks.append(module.mlp.register_forward_hook(output_hook(layer)))
        try:
            encoded = []
            for index, record in enumerate(records):
                token_ids = (
                    [int(value) for value in record["input_ids"]]
                    if "input_ids" in record
                    else runtime.encode(str(record.get("text", "")))
                )
                if len(token_ids) < tokens_per_sequence:
                    raise ValueError(
                        f"selected record {index} has fewer than "
                        f"{tokens_per_sequence} tokens"
                    )
                encoded.append(token_ids[:tokens_per_sequence])
            input_ids = torch.tensor(encoded, dtype=torch.long)
            with torch.inference_mode():
                runtime.model(input_ids=input_ids, use_cache=False)
        finally:
            for hook in hooks:
                hook.remove()
        if (
            set(captured_inputs) != set(range(layer_count))
            or set(captured_outputs) != set(range(layer_count))
        ):
            raise RuntimeError("failed to capture every trained MLP boundary")

        counts = tuple(
            min(width, max(1, int(np.ceil(value * width))))
            for value in requested
        )
        layer_reports = []
        runtime.kernel.clear_metrics()
        with torch.inference_mode():
            for layer in range(layer_count):
                arms = []
                reference = captured_outputs[layer]
                for requested_fraction, count in zip(
                    requested, counts, strict=True
                ):
                    approximation = runtime.kernel.forward_oracle(
                        layer,
                        captured_inputs[layer],
                        top_k=count,
                    )
                    metrics = _tensor_metrics(reference, approximation)
                    arms.append(
                        {
                            "requested_fraction": requested_fraction,
                            "top_k": count,
                            "actual_fraction": count / width,
                            **metrics,
                        }
                    )
                layer_reports.append({"layer": layer, "arms": arms})

    # Multiple-choice dynamic program. Later-layer errors receive a modestly
    # larger weight because less downstream computation remains to absorb them.
    budget_units = int(np.floor(mean_budget * layer_count * width))
    choices = list(counts)
    states: dict[int, tuple[float, tuple[int, ...]]] = {0: (0.0, ())}
    for layer, report in enumerate(layer_reports):
        weight = 1.0 + layer / max(1, layer_count - 1)
        updated: dict[int, tuple[float, tuple[int, ...]]] = {}
        for used, (score, selected) in states.items():
            for choice, arm in enumerate(report["arms"]):
                next_used = used + choices[choice]
                if next_used > budget_units:
                    continue
                next_score = score + weight * float(arm["relative_l2"])
                incumbent = updated.get(next_used)
                if incumbent is None or next_score < incumbent[0]:
                    updated[next_used] = (next_score, selected + (choice,))
        states = updated
    if not states:
        raise RuntimeError("no layer allocation satisfies the requested budget")
    used_units, (objective, selected_choices) = min(
        states.items(),
        key=lambda item: (item[1][0], -item[0]),
    )
    allocation = [
        layer_reports[layer]["arms"][choice]
        for layer, choice in enumerate(selected_choices)
    ]
    result = {
        "experiment": "native_bitnet_layer_adaptive_oracle_sweep",
        "package": str(package_path),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_offset": record_offset,
            "sequence_count": sequence_count,
            "tokens_per_sequence": tokens_per_sequence,
        },
        "fractions": list(requested),
        "mean_budget": mean_budget,
        "layers": layer_reports,
        "allocation": {
            "method": "weighted_local_relative_l2_multiple_choice_dp",
            "objective": objective,
            "requested_layer_fractions": [
                arm["requested_fraction"] for arm in allocation
            ],
            "layer_top_ks": [arm["top_k"] for arm in allocation],
            "mean_actual_fraction": float(
                np.mean([arm["actual_fraction"] for arm in allocation])
            ),
            "minimum_actual_fraction": min(
                arm["actual_fraction"] for arm in allocation
            ),
            "maximum_actual_fraction": max(
                arm["actual_fraction"] for arm in allocation
            ),
            "record_budget_used": used_units,
            "record_budget_available": budget_units,
        },
        "decision": "run_layer_adaptive_causal_development",
        "milestone_2_status": "blocked",
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(out, result)
    return result


__all__ = [
    "analyze_native_bitnet_layer_oracle",
    "evaluate_native_bitnet_oracle",
    "evaluate_native_bitnet_oracle_causal",
    "evaluate_native_bitnet_oracle_layer_sweep",
]
