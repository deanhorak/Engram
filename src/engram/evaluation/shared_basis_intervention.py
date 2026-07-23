"""Causal evaluation with physical ESIB MLP substitution.

Unlike the legacy intervention harness, this evaluator does not run a dense
MLP and replace its result from a forward hook.  It first records a dense
teacher reference, destroys that model, loads a separate student, and replaces
the student's selected ``decoder.mlp`` modules with :class:`SharedBasisMLP`.
Every substituted forward therefore depends only on the authenticated ESIB
weights.
"""

from __future__ import annotations

import gc
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from engram.evaluation.gates import apply_mlp_intervention_gates
from engram.models.inspection import inspect_model, resolve_model_path
from engram.semantic.shared_basis import (
    SharedBasisArtifactSet,
    SharedBasisMLP,
    load_shared_basis_artifact_set,
    shared_basis_fixed_traffic,
    shared_basis_traffic,
)
from engram.utils import percentile, sha256_file, sha256_json


@dataclass(frozen=True)
class _TeacherExample:
    input_ids: Any
    logits: Any
    final_hidden: Any
    local_mlp_outputs: Mapping[int, Any]


def _load_jsonl(path: Path, max_records: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            if "input_ids" not in record and "text" not in record:
                raise ValueError(f"JSONL line {line_number} requires input_ids or text")
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError("dataset contains no JSONL records")
    return records


def _record_input_ids(record: Mapping[str, Any], tokenizer: Any | None) -> list[int]:
    if "input_ids" in record:
        values = record["input_ids"]
        if not isinstance(values, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) for value in values
        ):
            raise ValueError("input_ids must be a list of integers")
        return list(values)
    if tokenizer is None:
        raise RuntimeError("tokenizer is required for text records")
    values = tokenizer(str(record["text"]), add_special_tokens=True)["input_ids"]
    if not isinstance(values, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise RuntimeError("tokenizer returned invalid input_ids")
    return values


def _sequence_hash(input_ids: Sequence[int]) -> str:
    if len(input_ids) < 2:
        raise ValueError("sequence hashing requires at least two tokens")
    return sha256_json({"input_ids": list(input_ids)})


def _dataset_sequence_hashes(
    records: Sequence[Mapping[str, Any]], tokenizer: Any | None
) -> list[str]:
    return [
        _sequence_hash(values)
        for record in records
        if len(values := _record_input_ids(record, tokenizer)) >= 2
    ]


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size or not np.all(np.isfinite(array)):
        raise ValueError("cannot summarize an empty or non-finite metric")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": percentile(array, 95),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _relative_and_cosine_rows(
    approximation: Any,
    reference: Any,
) -> tuple[np.ndarray, np.ndarray]:
    approximate = approximation.detach().float().cpu().numpy()
    exact = reference.detach().float().cpu().numpy()
    if approximate.shape != exact.shape or approximate.ndim < 2:
        raise ValueError("student and teacher tensors have incompatible shapes")
    approximate = approximate.reshape(-1, approximate.shape[-1]).astype(np.float64)
    exact = exact.reshape(-1, exact.shape[-1]).astype(np.float64)
    exact_norm = np.linalg.norm(exact, axis=1)
    approximate_norm = np.linalg.norm(approximate, axis=1)
    relative = np.linalg.norm(approximate - exact, axis=1) / np.maximum(
        exact_norm, 1e-12
    )
    cosine = np.sum(approximate * exact, axis=1) / np.maximum(
        exact_norm * approximate_norm, 1e-12
    )
    both_zero = (exact_norm <= 1e-12) & (approximate_norm <= 1e-12)
    cosine[both_zero] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


def _quality_metrics(
    teacher_logits: Any,
    student_logits: Any,
    input_ids: Any,
    teacher_hidden: Any,
    student_hidden: Any,
    torch: Any,
) -> dict[str, np.ndarray]:
    functional = torch.nn.functional
    exact_logits = teacher_logits[:, :-1].detach().float().cpu()
    approximate_logits = student_logits[:, :-1].detach().float().cpu()
    targets = input_ids[:, 1:].detach().cpu()
    teacher_logp = functional.log_softmax(exact_logits, dim=-1)
    student_logp = functional.log_softmax(approximate_logits, dim=-1)
    teacher_probability = torch.exp(teacher_logp)
    kl = torch.clamp(
        torch.sum(teacher_probability * (teacher_logp - student_logp), dim=-1),
        min=0.0,
    )
    teacher_nll = -teacher_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    student_nll = -student_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    teacher_top = torch.topk(
        exact_logits, k=min(5, exact_logits.shape[-1]), dim=-1
    ).indices
    student_top = torch.topk(
        approximate_logits, k=min(5, approximate_logits.shape[-1]), dim=-1
    ).indices
    residual_relative, residual_cosine = _relative_and_cosine_rows(
        student_hidden, teacher_hidden
    )
    return {
        "teacher_student_kl": kl.numpy().reshape(-1),
        "teacher_nll": teacher_nll.numpy().reshape(-1),
        "student_nll": student_nll.numpy().reshape(-1),
        "nll_delta": (student_nll - teacher_nll).numpy().reshape(-1),
        "teacher_top1_agreement": (teacher_top[..., 0] == student_top[..., 0])
        .float()
        .numpy()
        .reshape(-1),
        "teacher_top5_contains_student_top1": (teacher_top == student_top[..., :1])
        .any(dim=-1)
        .float()
        .numpy()
        .reshape(-1),
        "teacher_student_top5_overlap": (
            (teacher_top.unsqueeze(-1) == student_top.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean(dim=-1)
            .numpy()
            .reshape(-1)
        ),
        "final_hidden_relative_l2": residual_relative,
        "final_hidden_cosine": residual_cosine,
    }


def _load_transformers() -> tuple[Any, Any, Any]:
    try:
        import torch
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except Exception:  # An installed binary can be ABI-incompatible.

                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for shared-basis causal evaluation"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _load_model(
    model_path: Path,
    *,
    device: str,
    torch: Any,
    auto_model: Any,
) -> Any:
    model = auto_model.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    model.eval()
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("model does not expose Llama-compatible decoder layers")
    return model


def _prepare_teacher_examples(
    model: Any,
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any | None,
    layer_indices: Sequence[int],
    *,
    device: str,
    torch: Any,
) -> tuple[list[_TeacherExample], dict[str, Any]]:
    captured: dict[int, Any] = {}
    handles = []

    def capture_for(layer: int):
        def capture(_: Any, __: tuple[Any, ...], output: Any) -> None:
            captured[layer] = output.detach().float().cpu()

        return capture

    for layer in layer_indices:
        handles.append(
            model.model.layers[layer].mlp.register_forward_hook(capture_for(layer))
        )
    examples: list[_TeacherExample] = []
    nll_values: list[float] = []
    skipped = 0
    input_token_positions = 0
    try:
        with torch.inference_mode():
            for record in records:
                values = _record_input_ids(record, tokenizer)
                if len(values) < 2:
                    skipped += 1
                    continue
                captured.clear()
                input_ids = torch.tensor([values], dtype=torch.long, device=device)
                output = model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                if set(captured) != set(layer_indices):
                    raise RuntimeError("teacher did not execute every selected MLP")
                logits = output.logits.detach().float().cpu()
                final_hidden = output.hidden_states[-1].detach().float().cpu()
                logp = torch.nn.functional.log_softmax(logits[:, :-1], dim=-1)
                targets = input_ids[:, 1:].cpu()
                nll_values.extend(
                    (-logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )
                examples.append(
                    _TeacherExample(
                        input_ids=input_ids.detach().cpu(),
                        logits=logits,
                        final_hidden=final_hidden,
                        local_mlp_outputs=dict(captured),
                    )
                )
                input_token_positions += len(values)
    finally:
        for handle in handles:
            handle.remove()
    if not examples:
        raise ValueError("evaluation produced no sequences with at least two tokens")
    mean_nll = float(np.mean(nll_values))
    return examples, {
        "sequences": len(examples),
        "skipped_short_sequences": skipped,
        "input_token_positions": input_token_positions,
        "next_token_positions": len(nll_values),
        "negative_log_likelihood": mean_nll,
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
    }


def _separation_metadata(
    calibration_path: Path,
    calibration_hashes: Sequence[str],
    evaluation_path: Path,
    evaluation_hashes: Sequence[str],
    *,
    artifact_set: SharedBasisArtifactSet,
) -> dict[str, Any]:
    overlap = set(calibration_hashes).intersection(evaluation_hashes)
    if overlap:
        raise ValueError(
            "shared-basis fitting and evaluation contain matching token sequences"
        )
    calibration_hash = sha256_file(calibration_path)
    declared_hash = artifact_set.provenance.get("calibration_dataset_hash")
    if declared_hash is None:
        declared_hash = artifact_set.provenance.get("trace_dataset_hash")
    if declared_hash is not None and declared_hash != calibration_hash:
        raise ValueError(
            "ESIB manifest calibration dataset hash does not match the supplied data"
        )
    ranks = {artifact.rank for artifact in artifact_set.artifacts.values()}
    return {
        "dataset_hash": calibration_hash,
        "dataset_files_differ": calibration_hash != sha256_file(evaluation_path),
        "trace_path": str(calibration_path.resolve()),
        "records_per_layer_limit": None,
        "regularization": None,
        "rank": next(iter(ranks)) if len(ranks) == 1 else None,
        "separation_method": "exact_token_sequence_hashes",
        "calibration_sequence_count": len(calibration_hashes),
        "calibration_unique_sequence_count": len(set(calibration_hashes)),
        "evaluation_sequence_count": len(evaluation_hashes),
        "evaluation_unique_sequence_count": len(set(evaluation_hashes)),
        "overlapping_sequence_count": 0,
        "record_level_disjoint": True,
        "held_out_from_evaluation": True,
        "artifact_set_sha256": artifact_set.artifact_set_sha256,
        "manifest_declared_dataset_hash": declared_hash,
    }


def _configuration_selection_metadata(
    path: Path,
    hashes: Sequence[str],
    evaluation_hashes: Sequence[str],
) -> dict[str, Any]:
    overlap = set(hashes).intersection(evaluation_hashes)
    if overlap:
        raise ValueError(
            "configuration-selection and evaluation contain matching token sequences"
        )
    return {
        "dataset_path": str(path.resolve()),
        "dataset_hash": sha256_file(path),
        "separation_method": "exact_token_sequence_hashes",
        "selection_sequence_count": len(hashes),
        "selection_unique_sequence_count": len(set(hashes)),
        "evaluation_sequence_count": len(evaluation_hashes),
        "evaluation_unique_sequence_count": len(set(evaluation_hashes)),
        "overlapping_sequence_count": 0,
        "held_out_from_configuration_selection": True,
    }


def _aggregate_traffic(
    artifact_set: SharedBasisArtifactSet,
    selected_by_layer: Mapping[int, Sequence[np.ndarray]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_layer = []
    dense_total = 0
    adversarial_total = 0
    observed_totals: np.ndarray | None = None
    for layer, artifact in artifact_set.artifacts.items():
        selected = np.concatenate(selected_by_layer[layer], axis=0)
        report = shared_basis_traffic(artifact, selected)
        per_layer.append({"layer": layer, **report})
        dense_total += int(report["dense_ideal_q4_bytes"])
        adversarial_total += int(report["adversarial_total_cold_bytes"])
        layer_observed = artifact.record_bytes[selected].sum(
            axis=1
        ) + shared_basis_fixed_traffic(artifact)
        if observed_totals is None:
            observed_totals = layer_observed.astype(np.int64, copy=True)
        else:
            if len(layer_observed) != len(observed_totals):
                raise RuntimeError("ESIB layers evaluated different token populations")
            observed_totals += layer_observed
    if observed_totals is None:
        raise RuntimeError("no ESIB traffic observations were recorded")
    return (
        {
            "kind": "authenticated_shared_basis_q3_q2_q4_hard_topk",
            "policy": (
                "worst-case physical cold reads: every authenticated fixed prefix "
                "plus each layer's largest possible set of aligned down-row records"
            ),
            "dense_ideal_q4_bytes": dense_total,
            "worst_case_total_cold_bytes": adversarial_total,
            "cold_fraction_of_dense_q4": adversarial_total / dense_total,
            "traffic_gate_maximum_fraction": 0.45,
            "traffic_gate_passed": adversarial_total / dense_total <= 0.45,
            "observed_total_cold_bytes": {
                "count": int(len(observed_totals)),
                "minimum": int(observed_totals.min()),
                "mean": float(observed_totals.mean()),
                "maximum": int(observed_totals.max()),
            },
            "observed_cold_fraction_of_dense_q4": {
                "mean": float(observed_totals.mean() / dense_total),
                "maximum": float(observed_totals.max() / dense_total),
            },
            "artifact_bytes_are_not_per_token_traffic": True,
        },
        per_layer,
    )


def evaluate_shared_basis_intervention(
    model: str | Path,
    dataset: str | Path,
    artifact_manifest: str | Path,
    *,
    calibration_dataset: str | Path,
    configuration_selection_dataset: str | Path | None = None,
    evaluation_role: str = "development",
    max_records: int | None = None,
    device: str = "cpu",
    execution_chunk_size: int = 64,
) -> dict[str, Any]:
    """Measure causal quality after physically replacing every source MLP.

    ``calibration_dataset`` must be the complete fitting/configuration corpus,
    not a token-sampled boundary trace.  Exact full token sequences are compared
    with the evaluation population before either model is run.
    """

    if evaluation_role not in {"development", "confirmation"}:
        raise ValueError("evaluation_role must be development or confirmation")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when provided")
    if evaluation_role == "confirmation" and configuration_selection_dataset is None:
        raise ValueError("confirmation requires a configuration_selection_dataset")
    torch, auto_model, auto_tokenizer = _load_transformers()
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    artifact_set = load_shared_basis_artifact_set(
        artifact_manifest,
        expected_source_model_hash=inspection.source_hash,
        expected_hidden_size=inspection.hidden_size,
        expected_intermediate_size=inspection.intermediate_size,
        expected_num_hidden_layers=inspection.num_hidden_layers,
        require_all_layers=True,
    )
    layer_indices = tuple(artifact_set.artifacts)
    top_ks = {artifact.top_k for artifact in artifact_set.artifacts.values()}
    if len(top_ks) != 1:
        raise ValueError(
            "the current intervention schema requires one shared top-K across layers"
        )
    top_k = next(iter(top_ks))

    dataset_path = Path(dataset)
    calibration_path = Path(calibration_dataset)
    if not dataset_path.is_file() or not calibration_path.is_file():
        raise ValueError("evaluation and calibration datasets must be files")
    evaluation_records = _load_jsonl(dataset_path, max_records)
    calibration_records = _load_jsonl(calibration_path)
    needs_tokenizer = any(
        "input_ids" not in record
        for record in (*evaluation_records, *calibration_records)
    )
    selection_records: list[dict[str, Any]] | None = None
    selection_path = (
        Path(configuration_selection_dataset)
        if configuration_selection_dataset is not None
        else None
    )
    if selection_path is not None:
        if not selection_path.is_file():
            raise ValueError("configuration-selection dataset must be a file")
        selection_records = _load_jsonl(selection_path)
        needs_tokenizer = needs_tokenizer or any(
            "input_ids" not in record for record in selection_records
        )
    tokenizer = (
        auto_tokenizer.from_pretrained(model_path, local_files_only=True)
        if needs_tokenizer
        else None
    )
    evaluation_hashes = _dataset_sequence_hashes(evaluation_records, tokenizer)
    calibration_hashes = _dataset_sequence_hashes(calibration_records, tokenizer)
    calibration = _separation_metadata(
        calibration_path,
        calibration_hashes,
        dataset_path,
        evaluation_hashes,
        artifact_set=artifact_set,
    )
    configuration_selection = None
    if selection_path is not None and selection_records is not None:
        configuration_selection = _configuration_selection_metadata(
            selection_path,
            _dataset_sequence_hashes(selection_records, tokenizer),
            evaluation_hashes,
        )

    teacher = _load_model(model_path, device=device, torch=torch, auto_model=auto_model)
    examples, baseline = _prepare_teacher_examples(
        teacher,
        evaluation_records,
        tokenizer,
        layer_indices,
        device=device,
        torch=torch,
    )
    if len(evaluation_hashes) != baseline["sequences"]:
        raise RuntimeError("evaluation provenance and execution counts differ")
    baseline["unique_sequences"] = len(set(evaluation_hashes))
    del teacher
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    student = _load_model(model_path, device=device, torch=torch, auto_model=auto_model)
    modules: dict[int, SharedBasisMLP] = {}
    for layer, artifact in artifact_set.artifacts.items():
        replacement = SharedBasisMLP(
            artifact,
            capture=True,
            execution_chunk_size=execution_chunk_size,
        ).to(device)
        student.model.layers[layer].mlp = replacement
        modules[layer] = replacement
    gc.collect()
    dense_mlp_names = [
        name
        for name, _ in student.named_parameters()
        if ".mlp.gate_proj." in name
        or ".mlp.up_proj." in name
        or ".mlp.down_proj." in name
    ]
    if dense_mlp_names:
        raise RuntimeError(
            "dense source MLP parameters remain after all-layer ESIB substitution"
        )

    quality: dict[str, list[float]] = defaultdict(list)
    local_relative: dict[int, list[float]] = defaultdict(list)
    local_cosine: dict[int, list[float]] = defaultdict(list)
    selected_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    with torch.inference_mode():
        for example in examples:
            input_ids = example.input_ids.to(device)
            output = student(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            for name, values in _quality_metrics(
                example.logits,
                output.logits,
                input_ids,
                example.final_hidden,
                output.hidden_states[-1],
                torch,
            ).items():
                quality[name].extend(
                    np.asarray(values, dtype=np.float64).reshape(-1).tolist()
                )
            for layer, module in modules.items():
                local_output, selected = module.pop_capture()
                relative, cosine = _relative_and_cosine_rows(
                    local_output,
                    example.local_mlp_outputs[layer],
                )
                local_relative[layer].extend(relative.tolist())
                local_cosine[layer].extend(cosine.tolist())
                selected_by_layer[layer].append(
                    selected.detach().cpu().numpy().astype(np.int64, copy=False)
                )

    quality_summary = {name: _stats(values) for name, values in sorted(quality.items())}
    quality_summary["student_perplexity"] = float(
        math.exp(min(quality_summary["student_nll"]["mean"], 50.0))
    )
    local_by_layer = [
        {
            "layer": layer,
            "tokens": len(local_relative[layer]),
            "mlp_output_relative_l2": _stats(local_relative[layer]),
            "mlp_output_cosine": _stats(local_cosine[layer]),
        }
        for layer in layer_indices
    ]
    all_relative = [value for layer in layer_indices for value in local_relative[layer]]
    all_cosine = [value for layer in layer_indices for value in local_cosine[layer]]
    projected, traffic_by_layer = _aggregate_traffic(artifact_set, selected_by_layer)
    arm = {
        "name": "shared_basis_authenticated_all_layers",
        "variant": "shared_basis",
        "scope": "all",
        "layer_indices": list(layer_indices),
        "top_k": top_k,
        "selection_scope": {
            "mode": "full_converted_width",
            "converted_width": inspection.intermediate_size,
            "scored_width": inspection.intermediate_size,
            "candidate_shortlist": False,
            "authentication": {
                "source": "strict_reloaded_artifact_header_and_manifest",
                "verified": True,
                "sha256": artifact_set.artifact_set_sha256,
            },
        },
        "candidate_recall_applicable": False,
        "local_mlp": {
            "tokens": len(all_relative),
            "mlp_output_relative_l2": _stats(all_relative),
            "mlp_output_cosine": _stats(all_cosine),
        },
        "local_mlp_by_layer": local_by_layer,
        "quality": quality_summary,
        "projected_accounting": projected,
        "traffic_by_layer": traffic_by_layer,
        "artifacts": [
            {
                "layer": layer,
                "sha256": artifact.artifact_sha256,
                "content_checksum": artifact.content_checksum,
                "rank": artifact.rank,
                "top_k": artifact.top_k,
                "serialized_bytes": len(artifact.payload),
            }
            for layer, artifact in artifact_set.artifacts.items()
        ],
        "execution": {
            "physical_mlp_substitution": True,
            "post_forward_output_replacement": False,
            "artifact_only_selected_mlp_forward": True,
            "dense_source_mlp_parameters_present_in_student": False,
            "decoded_float32_quality_execution": True,
            "native_compressed_kernel_timing_valid": False,
        },
        "timing_valid": False,
    }
    report = {
        "schema_version": 1,
        "experiment": "trained_teacher_mlp_intervention",
        "status": "physical_shared_basis_mlp_substitution_measurement",
        "source_model_hash": inspection.source_hash,
        "num_hidden_layers": inspection.num_hidden_layers,
        "intermediate_size": inspection.intermediate_size,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_hash": sha256_file(dataset_path),
        "evaluation_role": evaluation_role,
        "configuration_selection": configuration_selection,
        "selected_layers": list(layer_indices),
        "layer_mode": "all",
        "variants": ["shared_basis"],
        "top_ks": [top_k],
        "layer_top_ks": None,
        "candidate_counts": [],
        "input_fractions": [],
        "rank": calibration["rank"],
        "calibration": calibration,
        "artifact_manifest": {
            "path": str(artifact_set.manifest_path),
            "sha256": artifact_set.manifest_sha256,
            "artifact_set_sha256": artifact_set.artifact_set_sha256,
            "strict_reload_completed": True,
            "source_model_hash_verified": True,
            "all_layers_covered": True,
        },
        "baseline": baseline,
        "arms": [arm],
        "measurement_caveat": (
            "The student physically executes decoded artifact weights after its dense "
            "source MLP modules are removed. Causal quality is valid; Python decoded-"
            "float32 wall time is not a native compressed-kernel benchmark."
        ),
        "quality_targets_met": None,
    }
    return apply_mlp_intervention_gates(report)


__all__ = ["evaluate_shared_basis_intervention"]
