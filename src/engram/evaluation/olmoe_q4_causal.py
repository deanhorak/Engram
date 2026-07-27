"""All-layer causal screen for simulated groupwise-Q4 OLMoE experts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.olmoe_q4 import symmetric_groupwise_dequant
from engram.models.olmoe import audit_olmoe_source
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


def _quantize_olmoe_experts_in_place(
    model: object, *, bits: int = 4, group_size: int
) -> dict[str, int]:
    """Replace expert weights with decoded low-bit values, expert by expert."""

    import torch

    matrices = 0
    parameters = 0
    stored_bytes = 0
    with torch.no_grad():
        for layer in model.model.layers:  # type: ignore[attr-defined]
            experts = layer.mlp.experts
            for parameter in (experts.gate_up_proj, experts.down_proj):
                for expert in range(parameter.shape[0]):
                    source = parameter[expert].detach().float().cpu().numpy()
                    decoded, encoded_bytes = symmetric_groupwise_dequant(
                        source, bits=bits, group_size=group_size
                    )
                    parameter[expert].copy_(
                        torch.from_numpy(decoded).to(dtype=parameter.dtype)
                    )
                    matrices += 1
                    parameters += int(source.size)
                    stored_bytes += encoded_bytes
    return {
        "expert_matrices": matrices,
        "expert_parameters": parameters,
        "modelled_quantized_and_scale_bytes": stored_bytes,
        "scale_dtype": "bfloat16",
    }


def _read_records(path: Path, samples: int) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {exc}"
                ) from exc
            if len(records) == samples:
                break
    if not records:
        raise ValueError("dataset contains no JSONL records")
    return records


def evaluate_olmoe_q4_causal(
    model: str | Path,
    dataset: str | Path,
    out: str | Path,
    *,
    samples: int = 2,
    max_tokens: int = 16,
    bits: int = 4,
    group_size: int = 8,
    threads: int = 12,
) -> dict[str, Any]:
    """Compare the BF16 teacher with all expert matrices Q4-simulated in place."""

    if samples <= 0 or max_tokens < 2 or threads <= 0:
        raise ValueError("samples/threads must be positive and max_tokens at least 2")
    model_path = Path(model).expanduser().resolve()
    dataset_path = Path(dataset).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if audit.decision != "proceed_to_router_trace":
        raise ValueError("local OLMoE checkpoint failed exact source validation")
    records = _read_records(dataset_path, samples)
    try:
        import torch

        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for OLMoE evaluation") from exc
    torch.set_num_threads(threads)
    tokenizer = None
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    inputs = []
    for record in records:
        if "input_ids" in record:
            ids = torch.tensor([record["input_ids"]], dtype=torch.long)
        else:
            assert tokenizer is not None
            ids = tokenizer(str(record.get("text", "")), return_tensors="pt")[
                "input_ids"
            ]
        inputs.append(ids[:, :max_tokens])
    loaded = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).eval()

    def run() -> list[dict[str, torch.Tensor]]:
        outputs = []
        with torch.inference_mode():
            for ids in inputs:
                result = loaded(
                    input_ids=ids,
                    use_cache=False,
                    output_hidden_states=True,
                )
                outputs.append(
                    {
                        "logits": result.logits[0, :-1].float().cpu(),
                        "hidden": result.hidden_states[-1][0, :-1].float().cpu(),
                        "targets": ids[0, 1:].cpu(),
                    }
                )
        return outputs

    teacher = run()
    quantization = _quantize_olmoe_experts_in_place(
        loaded, bits=bits, group_size=group_size
    )
    student = run()
    kl_values = []
    top1_matches = []
    nll_deltas = []
    hidden_l2 = []
    for dense, quantized in zip(teacher, student, strict=True):
        teacher_log_probs = torch.log_softmax(dense["logits"], dim=-1)
        student_log_probs = torch.log_softmax(quantized["logits"], dim=-1)
        teacher_probs = torch.exp(teacher_log_probs)
        kl_values.append(
            torch.sum(
                teacher_probs * (teacher_log_probs - student_log_probs), dim=-1
            )
        )
        top1_matches.append(
            torch.argmax(dense["logits"], dim=-1)
            == torch.argmax(quantized["logits"], dim=-1)
        )
        targets = dense["targets"].unsqueeze(1)
        nll_deltas.append(
            -student_log_probs.gather(1, targets).squeeze(1)
            + teacher_log_probs.gather(1, targets).squeeze(1)
        )
        difference = torch.linalg.vector_norm(
            quantized["hidden"] - dense["hidden"], dim=-1
        )
        denominator = torch.clamp(
            torch.linalg.vector_norm(dense["hidden"], dim=-1), min=1e-12
        )
        hidden_l2.append(difference / denominator)
    kl = torch.cat(kl_values)
    top1 = torch.cat(top1_matches)
    nll = torch.cat(nll_deltas)
    hidden = torch.cat(hidden_l2)
    dims = audit.dimensions
    h = int(dims["hidden_size"] or 0)
    intermediate = int(dims["intermediate_size"] or 0)
    experts = int(dims["num_experts"] or 0)
    top_k = int(dims["num_experts_per_tok"] or 0)
    layers = int(dims["num_hidden_layers"] or 0)
    groups_gate_up = (h + group_size - 1) // group_size
    groups_down = (intermediate + group_size - 1) // group_size
    per_expert_bytes = (
        (2 * intermediate * h * bits + 7) // 8
        + 2 * intermediate * groups_gate_up * 2
        + (h * intermediate * bits + 7) // 8
        + h * groups_down * 2
    )
    selected_bytes = layers * top_k * per_expert_bytes
    router_bytes = layers * experts * h * 2
    baseline = layers * experts * 3 * h * intermediate // 2
    traffic_fraction = (selected_bytes + router_bytes) / baseline
    metrics = {
        "teacher_to_student_kl": float(torch.mean(kl)),
        "teacher_top1_agreement": float(torch.mean(top1.float())),
        "target_nll_delta": float(torch.mean(nll)),
        "final_hidden_relative_l2": float(torch.mean(hidden)),
        "maximum_position_kl": float(torch.max(kl)),
        "prediction_positions": int(kl.numel()),
        "sequences": len(inputs),
    }
    thresholds = {
        "maximum_kl": 0.05,
        "minimum_top1_agreement": 0.90,
        "maximum_nll_delta": 0.05,
        "maximum_hidden_relative_l2": 0.10,
        "maximum_traffic_fraction": 0.45,
    }
    quality_passed = (
        metrics["teacher_to_student_kl"] <= thresholds["maximum_kl"]
        and metrics["teacher_top1_agreement"]
        >= thresholds["minimum_top1_agreement"]
        and metrics["target_nll_delta"] <= thresholds["maximum_nll_delta"]
        and metrics["final_hidden_relative_l2"]
        <= thresholds["maximum_hidden_relative_l2"]
    )
    result = {
        "schema_version": 1,
        "experiment": "olmoe_all_layer_groupwise_quantized_causal",
        "model": str(model_path),
        "source_revision": audit.resolved_revision,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "input_identity": sha256_json(
            [ids[0].tolist() for ids in inputs]
        ),
        "group_size": group_size,
        "bits": bits,
        "threads": threads,
        "metrics": metrics,
        "traffic": {
            "selected_quantized_and_scale_bytes_per_token": selected_bytes,
            "router_bf16_bytes_per_token": router_bytes,
            "complete_modelled_bytes_per_token": selected_bytes + router_bytes,
            "all_expert_ideal_q4_bytes_per_token": baseline,
            "fraction_of_all_expert_ideal_q4": traffic_fraction,
            "measured_hardware_traffic": False,
        },
        "quantization": quantization,
        "thresholds": thresholds,
        "screen": {
            "quality_passed": quality_passed,
            "traffic_projection_passed": traffic_fraction <= 0.45,
            "evidence_floor_passed": len(inputs) >= 8 and int(kl.numel()) >= 256,
            "passed": (
                quality_passed
                and traffic_fraction <= 0.45
                and len(inputs) >= 8
                and int(kl.numel()) >= 256
            ),
            "scope": (
                "all-layer causal low-bit simulation; decoded weights execute in "
                "Transformers, so serialized CPU runtime traffic remains unproven"
            ),
        },
    }
    atomic_json(out, result)
    return result
