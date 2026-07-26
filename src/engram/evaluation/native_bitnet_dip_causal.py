"""Causal model-shell evaluation for selected-record native BitNet DIP."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.models.native_bitnet import load_native_bitnet_artifact
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.semantic.native_bitnet_dip import (
    NativeBitNetDIPConfiguration,
    build_native_bitnet_dip_mlp,
)
from engram.semantic.dip import input_coordinate_count
from engram.utils import atomic_json, sha256_file


def _json_native(value: Any) -> Any:
    """Recursively convert NumPy scalars before an expensive report write."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_native(item) for item in value]
    if isinstance(value, tuple):
        return [_json_native(item) for item in value]
    return value


def _load_causal_records(
    path: Path,
    *,
    offset: int,
    count: int,
) -> list[dict[str, Any]]:
    if offset < 0 or count <= 0:
        raise ValueError("record offset/count must be non-negative/positive")
    selected: list[dict[str, Any]] = []
    usable = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record {line_number} must be an object"
                )
            if usable >= offset:
                selected.append(record)
                if len(selected) == count:
                    break
            usable += 1
    if len(selected) != count:
        raise ValueError(
            f"dataset contains only {len(selected)} selected records"
        )
    return selected


def _validate_configurations(
    configurations: Mapping[int, NativeBitNetDIPConfiguration],
    *,
    layer_count: int,
    require_all_layers: bool,
) -> dict[int, NativeBitNetDIPConfiguration]:
    if not configurations:
        raise ValueError("configurations must not be empty")
    result: dict[int, NativeBitNetDIPConfiguration] = {}
    for raw_layer, configuration in configurations.items():
        if (
            isinstance(raw_layer, bool)
            or not isinstance(raw_layer, int)
            or not 0 <= raw_layer < layer_count
        ):
            raise ValueError("DIP layer index is outside the artifact")
        if not isinstance(configuration, NativeBitNetDIPConfiguration):
            raise ValueError(
                "DIP configurations must be NativeBitNetDIPConfiguration values"
            )
        result[raw_layer] = configuration
    if require_all_layers and set(result) != set(range(layer_count)):
        raise ValueError(
            "full causal DIP evaluation requires one configuration per layer"
        )
    return dict(sorted(result.items()))


def _prediction_views(dense_result, sparse_result, input_ids):
    """Return aligned N prediction positions from N+1 input tokens."""

    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("causal scoring requires at least two input tokens")
    predictions = input_ids.shape[1] - 1
    dense_logits = dense_result.logits[:, :predictions].float()
    sparse_logits = sparse_result.logits[:, :predictions].float()
    dense_hidden = dense_result.hidden_states[-1][:, :predictions].float()
    sparse_hidden = sparse_result.hidden_states[-1][:, :predictions].float()
    labels = input_ids[:, 1:]
    if (
        dense_logits.shape[:2] != labels.shape
        or sparse_logits.shape != dense_logits.shape
        or dense_hidden.shape != sparse_hidden.shape
    ):
        raise RuntimeError("causal prediction tensors are not aligned")
    return dense_logits, sparse_logits, dense_hidden, sparse_hidden, labels


def _causal_evidence_passed(
    *,
    sequences: int,
    unique_sequences: int,
    predictions_per_sequence: int,
    prediction_positions: int,
    all_mlp_layers: bool,
) -> bool:
    return bool(
        all_mlp_layers
        and sequences >= 8
        and unique_sequences >= 8
        and predictions_per_sequence >= 32
        and prediction_positions >= 256
    )


def evaluate_native_bitnet_dip_causal(
    package: str | Path,
    dataset: str | Path,
    configurations: Mapping[int, NativeBitNetDIPConfiguration],
    *,
    out: str | Path,
    samples: int = 1,
    predictions_per_sequence: int = 8,
    record_offset: int = 0,
    library: str | Path | None = None,
    threads: int | None = None,
    require_all_layers: bool = True,
) -> dict[str, Any]:
    """Compare dense and selected-record MLPs inside the trained model shell.

    The caller supplies every per-layer ``q/C/K`` budget.  No schedule is baked
    into this evaluator, which allows robust schedules to be fitted on one
    corpus and confirmed unchanged on another.  Replacements are constructed
    once and toggled per record, so dense logits do not need to be retained for
    the whole corpus and decoded ternary weights are not rebuilt repeatedly.
    """

    if samples <= 0 or predictions_per_sequence <= 0:
        raise ValueError(
            "samples and predictions_per_sequence must be positive"
        )
    package_path = Path(package).resolve()
    dataset_path = Path(dataset).resolve()
    output_path = Path(out)
    checkpoint_path = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix or '.json'}"
    )
    records = _load_causal_records(
        dataset_path,
        offset=record_offset,
        count=samples,
    )
    started = time.perf_counter()
    sequence_reports: list[dict[str, Any]] = []
    total_tokens = 0
    total_kl = 0.0
    total_top1 = 0
    total_nll_tokens = 0
    total_dense_nll = 0.0
    total_sparse_nll = 0.0
    hidden_numerator_squared = 0.0
    hidden_denominator_squared = 0.0
    dense_seconds = 0.0
    sparse_seconds = 0.0
    selected_count_values: dict[int, list[int]] = {}
    token_selected_schedules: list[list[int]] = []
    sequence_token_ids: set[tuple[int, ...]] = set()

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("causal DIP evaluation requires torch") from exc

    with NativeBitNetRuntime(
        package_path,
        library=library,
        threads=threads,
    ) as runtime:
        artifact = load_native_bitnet_artifact(runtime.artifact_path)
        selected_configurations = _validate_configurations(
            configurations,
            layer_count=len(artifact.layers),
            require_all_layers=require_all_layers,
        )
        decoder = runtime.model.model.layers
        original_modules = {
            layer: decoder[layer].mlp for layer in selected_configurations
        }
        replacements = {
            layer: build_native_bitnet_dip_mlp(
                artifact,
                layer,
                input_fraction=configuration.input_fraction,
                candidate_count=configuration.candidate_count,
                top_k=configuration.top_k,
                rms_audit_count=configuration.rms_audit_count,
                energy_target=configuration.energy_target,
                minimum_top_k=configuration.minimum_top_k,
                maximum_top_k=configuration.maximum_top_k,
                rms_variance_scale=configuration.rms_variance_scale,
                rms_variance_bias=configuration.rms_variance_bias,
                output_scale=configuration.output_scale,
                rms_estimator=configuration.rms_estimator,
                rms_audit_strategy=configuration.rms_audit_strategy,
            )
            for layer, configuration in selected_configurations.items()
        }

        try:
            for sequence_index, record in enumerate(records):
                if "input_ids" in record:
                    token_ids = [int(value) for value in record["input_ids"]]
                else:
                    token_ids = runtime.encode(str(record.get("text", "")))
                required_tokens = predictions_per_sequence + 1
                if len(token_ids) < required_tokens:
                    raise ValueError(
                        f"selected record {sequence_index} has "
                        f"{len(token_ids)} tokens; required {required_tokens}"
                    )
                token_ids = token_ids[:required_tokens]
                sequence_token_ids.add(tuple(token_ids))
                input_ids = torch.tensor([token_ids], dtype=torch.long)

                for layer, original in original_modules.items():
                    decoder[layer].mlp = original
                dense_started = time.perf_counter()
                with torch.inference_mode():
                    dense_result = runtime.model(
                        input_ids=input_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                dense_seconds += time.perf_counter() - dense_started
                dense_logits = dense_result.logits.float()
                dense_hidden = dense_result.hidden_states[-1].float()

                for layer, replacement in replacements.items():
                    decoder[layer].mlp = replacement
                sparse_started = time.perf_counter()
                with torch.inference_mode():
                    sparse_result = runtime.model(
                        input_ids=input_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                sparse_seconds += time.perf_counter() - sparse_started
                (
                    dense_logits,
                    sparse_logits,
                    dense_hidden,
                    sparse_hidden,
                    targets,
                ) = _prediction_views(
                    dense_result,
                    sparse_result,
                    input_ids,
                )
                sequence_selection: dict[str, dict[str, float | int]] = {}
                sequence_layer_counts: dict[int, list[int]] = {}
                for layer, replacement in replacements.items():
                    last_result = replacement.last_result
                    if last_result is None:
                        raise RuntimeError(
                            f"DIP layer {layer} produced no selection metrics"
                        )
                    executed_counts = [
                        int(value)
                        for value in last_result.selected_counts.tolist()
                    ]
                    if len(executed_counts) != required_tokens:
                        raise RuntimeError(
                            f"DIP layer {layer} selected-count rows differ "
                            "from N+1 input positions"
                        )
                    counts = executed_counts[:-1]
                    selected_count_values.setdefault(layer, []).extend(counts)
                    sequence_layer_counts[layer] = counts
                    sequence_selection[str(layer)] = {
                        "minimum": min(counts),
                        "maximum": max(counts),
                        "mean": float(np.mean(counts)),
                    }
                for token_index in range(predictions_per_sequence):
                    token_selected_schedules.append(
                        [
                            sequence_layer_counts[layer][token_index]
                            for layer in selected_configurations
                        ]
                    )

                dense_log_probs = functional.log_softmax(
                    dense_logits,
                    dim=-1,
                )
                sparse_log_probs = functional.log_softmax(
                    sparse_logits,
                    dim=-1,
                )
                probabilities = dense_log_probs.exp()
                token_kl = torch.sum(
                    probabilities * (dense_log_probs - sparse_log_probs),
                    dim=-1,
                )
                token_top1 = dense_logits.argmax(dim=-1).eq(
                    sparse_logits.argmax(dim=-1)
                )
                hidden_delta = sparse_hidden - dense_hidden
                hidden_numerator = float(
                    torch.sum(hidden_delta * hidden_delta).item()
                )
                hidden_denominator = float(
                    torch.sum(dense_hidden * dense_hidden).item()
                )
                hidden_relative = np.sqrt(hidden_numerator) / max(
                    np.sqrt(hidden_denominator),
                    1e-12,
                )
                sequence_nll_delta: float | None = None
                dense_nll = -torch.gather(
                    dense_log_probs,
                    dim=-1,
                    index=targets.unsqueeze(-1),
                ).squeeze(-1)
                sparse_nll = -torch.gather(
                    sparse_log_probs,
                    dim=-1,
                    index=targets.unsqueeze(-1),
                ).squeeze(-1)
                nll_tokens = targets.numel()
                total_nll_tokens += nll_tokens
                total_dense_nll += float(dense_nll.sum().item())
                total_sparse_nll += float(sparse_nll.sum().item())
                sequence_nll_delta = float(
                    (sparse_nll.mean() - dense_nll.mean()).item()
                )

                tokens = predictions_per_sequence
                total_tokens += tokens
                total_kl += float(token_kl.sum().item())
                total_top1 += int(token_top1.sum().item())
                hidden_numerator_squared += hidden_numerator
                hidden_denominator_squared += hidden_denominator
                sequence_reports.append(
                    {
                        "sequence": sequence_index,
                        "input_tokens": required_tokens,
                        "prediction_positions": tokens,
                        "mean_kl_divergence": float(
                            token_kl.mean().item()
                        ),
                        "top1_agreement": float(
                            token_top1.float().mean().item()
                        ),
                        "nll_delta": sequence_nll_delta,
                        "final_hidden_relative_l2": float(hidden_relative),
                        "selected_records": sequence_selection,
                    }
                )
                checkpoint = _json_native(
                    {
                        "experiment": (
                            "native_bitnet_selected_record_dip_causal_checkpoint"
                        ),
                        "status": "in_progress",
                        "output": str(output_path.resolve()),
                        "completed_sequences": len(sequence_reports),
                        "requested_sequences": samples,
                        "completed_tokens": total_tokens,
                        "accumulators": {
                            "total_kl": total_kl,
                            "total_top1": total_top1,
                            "total_nll_tokens": total_nll_tokens,
                            "total_dense_nll": total_dense_nll,
                            "total_sparse_nll": total_sparse_nll,
                            "hidden_numerator_squared": (
                                hidden_numerator_squared
                            ),
                            "hidden_denominator_squared": (
                                hidden_denominator_squared
                            ),
                            "dense_seconds": dense_seconds,
                            "sparse_seconds": sparse_seconds,
                        },
                        "selected_count_values": selected_count_values,
                        "token_selected_schedules": token_selected_schedules,
                        "sequence_reports": sequence_reports,
                    }
                )
                # Fail before losing more completed work if a new metric is
                # non-finite or not JSON-compatible.
                json.dumps(checkpoint, allow_nan=False)
                atomic_json(checkpoint_path, checkpoint)
                del dense_result, sparse_result
        finally:
            for layer, original in original_modules.items():
                decoder[layer].mlp = original

    mean_kl = total_kl / total_tokens
    top1_agreement = total_top1 / total_tokens
    nll_delta = (
        (total_sparse_nll - total_dense_nll) / total_nll_tokens
        if total_nll_tokens
        else None
    )
    hidden_relative = np.sqrt(hidden_numerator_squared) / max(
        np.sqrt(hidden_denominator_squared),
        1e-12,
    )
    causal_passed = (
        mean_kl <= 0.05
        and top1_agreement >= 0.90
        and (nll_delta is None or nll_delta <= 0.05)
        and hidden_relative <= 0.10
    )
    input_counts = [
        input_coordinate_count(
            artifact.hidden_size,
            configuration.input_fraction,
        )
        for configuration in selected_configurations.values()
    ]
    candidate_counts = [
        configuration.candidate_count
        for configuration in selected_configurations.values()
    ]
    physical_reports = [
        native_bitnet_dip_physical_accounting(
            artifact.hidden_size,
            artifact.intermediate_size,
            input_counts=input_counts,
            candidate_counts=candidate_counts,
            top_ks=selected_schedule,
        )
        for selected_schedule in token_selected_schedules
    ]
    physical_fractions = [
        float(report["traffic"]["fraction_of_dense_q4"])
        for report in physical_reports
    ]
    worst_layer_fraction = max(
        float(layer["fraction_of_dense_q4"])
        for report in physical_reports
        for layer in report["traffic"]["layers"]
    )
    total_selected_records = sum(
        sum(values) for values in selected_count_values.values()
    )
    active_denominator = (
        total_tokens
        * len(selected_configurations)
        * artifact.intermediate_size
    )
    mean_active_fraction = total_selected_records / active_denominator
    active_budget_passed = mean_active_fraction <= 0.25
    scoring_protocol_valid = True
    evidence_passed = _causal_evidence_passed(
        sequences=samples,
        unique_sequences=len(sequence_token_ids),
        predictions_per_sequence=predictions_per_sequence,
        prediction_positions=total_tokens,
        all_mlp_layers=require_all_layers,
    )
    protocol_qualifying = scoring_protocol_valid and evidence_passed
    traffic_passed = (
        max(physical_fractions) <= 0.45
        and worst_layer_fraction <= 0.45
    )
    overall_gate_passed = (
        protocol_qualifying
        and causal_passed
        and traffic_passed
        and active_budget_passed
    )
    result = {
        "experiment": "native_bitnet_selected_record_dip_causal",
        "scope": (
            "all_mlp_layers"
            if require_all_layers
            else "configured_mlp_layers"
        ),
        "package": str(package_path),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "record_offset": record_offset,
            "samples": samples,
            "predictions_per_sequence": predictions_per_sequence,
            "required_input_tokens_per_sequence": (
                predictions_per_sequence + 1
            ),
            "prediction_positions": (
                samples * predictions_per_sequence
            ),
        },
        "configuration": {
            str(layer): {
                "input_fraction": configuration.input_fraction,
                "candidate_count": configuration.candidate_count,
                "top_k": configuration.top_k,
                "rms_audit_count": configuration.rms_audit_count,
                "energy_target": configuration.energy_target,
                "minimum_top_k": configuration.minimum_top_k,
                "maximum_top_k": configuration.maximum_top_k,
                "rms_variance_scale": configuration.rms_variance_scale,
                "rms_variance_bias": configuration.rms_variance_bias,
                "output_scale": configuration.output_scale,
                "rms_estimator": configuration.rms_estimator,
                "rms_audit_strategy": configuration.rms_audit_strategy,
            }
            for layer, configuration in selected_configurations.items()
        },
        "metrics": {
            "tokens": total_tokens,
            "mean_kl_divergence": mean_kl,
            "top1_agreement": top1_agreement,
            "nll_delta": nll_delta,
            "final_hidden_relative_l2": hidden_relative,
        },
        "selected_records": {
            str(layer): {
                "minimum": min(values),
                "maximum": max(values),
                "mean": float(np.mean(values)),
                "mean_fraction": float(
                    np.mean(values) / artifact.intermediate_size
                ),
            }
            for layer, values in selected_count_values.items()
        },
        "active_record_budget": {
            "total_selected_records": total_selected_records,
            "denominator_records": active_denominator,
            "mean_active_fraction": mean_active_fraction,
            "maximum_mean_active_fraction": 0.25,
            "passes_25_percent": active_budget_passed,
        },
        "physical_cold_traffic": {
            "tokens": len(physical_fractions),
            "mean_fraction_of_dense_q4": float(
                np.mean(physical_fractions)
            ),
            "maximum_fraction_of_dense_q4": max(physical_fractions),
            "minimum_fraction_of_dense_q4": min(physical_fractions),
            "worst_layer_fraction_of_dense_q4": worst_layer_fraction,
            "passes_45_percent": traffic_passed,
            "accounting": "native_bitnet_dip_dual_layout_v2",
        },
        "thresholds": {
            "maximum_mean_kl_divergence": 0.05,
            "minimum_top1_agreement": 0.90,
            "maximum_nll_delta": 0.05,
            "maximum_final_hidden_relative_l2": 0.10,
        },
        "causal_quality_passed": causal_passed,
        "scoring_protocol_valid": scoring_protocol_valid,
        "evidence_requirements": {
            "minimum_unique_sequences": 8,
            "minimum_predictions_per_sequence": 32,
            "minimum_prediction_positions": 256,
            "requires_all_mlp_layers": True,
        },
        "evidence_observed": {
            "unique_sequences": len(sequence_token_ids),
            "sequences": samples,
            "predictions_per_sequence": predictions_per_sequence,
            "prediction_positions": total_tokens,
            "all_mlp_layers": require_all_layers,
        },
        "evidence_passed": evidence_passed,
        "protocol_qualifying": protocol_qualifying,
        "overall_gate_passed": overall_gate_passed,
        "prediction_protocol": (
            "load N+1 tokens; score logits and final hidden on positions "
            "[:-1] against next-token labels [1:]"
        ),
        "sequence_reports": sequence_reports,
        "timing": {
            "dense_seconds": dense_seconds,
            "prototype_sparse_seconds": sparse_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "prototype_latency_is_a_gate": False,
        },
        "decision": (
            "proceed_to_native_selected_record_kernel"
            if overall_gate_passed
            else (
                "expand_unchanged_policy_to_qualifying_evidence"
                if causal_passed
                and traffic_passed
                and active_budget_passed
                and scoring_protocol_valid
                else "revise_q_c_k_schedule_or_estimator"
            )
        ),
        "milestone_2_status": "blocked",
    }
    result = _json_native(result)
    json.dumps(result, allow_nan=False)
    atomic_json(output_path, result)
    atomic_json(
        checkpoint_path,
        {
            "experiment": "native_bitnet_selected_record_dip_causal_checkpoint",
            "status": "complete",
            "output": str(output_path.resolve()),
            "completed_sequences": samples,
            "completed_tokens": total_tokens,
        },
    )
    return result


__all__ = ["evaluate_native_bitnet_dip_causal"]
