"""Sparse-teacher fine-tuning with frozen base weights and low-rank adapters."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation.gates import (
    MINIMUM_EVALUATION_SEQUENCES,
    MINIMUM_NEXT_TOKEN_POSITIONS,
    MINIMUM_ROUTED_CANDIDATE_RECALL,
    MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
    MLP_QUALITY_THRESHOLDS,
)
from engram.evaluation.mlp_intervention import (
    _evaluation_sequence_hashes,
    _quality_metrics,
    _relative_and_cosine_rows,
    _stats,
)
from engram.evaluation.router_sweep import _load_states, _membership, _sequence_hashes
from engram.models.inspection import inspect_model, resolve_model_path
from engram.semantic.multilabel_router import LowRankMultiLabelRouter
from engram.tracing.format import TraceReader
from engram.utils import atomic_json, sha256_file


def _load_jsonl(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    if not records:
        raise ValueError("training or validation dataset contains no records")
    return records


def _ids(record: dict[str, Any], tokenizer: Any, torch: Any, device: str) -> Any:
    if "input_ids" in record:
        values = record["input_ids"]
        if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
            raise ValueError("input_ids must be a list of integers")
        return torch.tensor([values], dtype=torch.long, device=device)
    return tokenizer(str(record["text"]), return_tensors="pt")["input_ids"].to(device)


def _wrap_sparse_mlp_class(torch: Any):
    class SparseStudentMLP(torch.nn.Module):
        def __init__(self, base: Any, router: LowRankMultiLabelRouter, *, top_k: int, candidates: int, adapter_rank: int):
            super().__init__()
            self.base = base
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            dtype = base.down_proj.weight.dtype
            self.router_input = torch.nn.Parameter(torch.tensor(router.input_factors, dtype=dtype))
            self.router_output = torch.nn.Parameter(torch.tensor(router.output_factors, dtype=dtype))
            self.router_bias = torch.nn.Parameter(torch.tensor(router.bias, dtype=dtype))
            intermediate = base.down_proj.weight.shape[1]
            hidden = base.down_proj.weight.shape[0]
            generator = torch.Generator(device="cpu").manual_seed(1701 + intermediate + hidden)
            self.adapter_a = torch.nn.Parameter(
                torch.randn(intermediate, adapter_rank, generator=generator, dtype=dtype) * 0.01
            )
            self.adapter_b = torch.nn.Parameter(torch.zeros(adapter_rank, hidden, dtype=dtype))
            self.adapter_scale = 1.0 / adapter_rank
            self.top_k = top_k
            self.candidates = candidates
            self.mode = "trained"
            self.last_output = None
            self.last_router_logits = None
            self.last_oracle = None
            self.last_recall = None

        def forward(self, hidden: Any) -> Any:
            shape = hidden.shape
            flat = hidden.reshape(-1, shape[-1])
            activations = self.base.act_fn(self.base.gate_proj(flat)) * self.base.up_proj(flat)
            dense = self.base.down_proj(activations)
            value_norms = torch.linalg.vector_norm(self.base.down_proj.weight.detach(), dim=0)
            exact_scores = torch.abs(activations) * value_norms.unsqueeze(0)
            oracle = torch.argsort(exact_scores, dim=1, descending=True, stable=True)[:, : self.top_k]
            self.last_oracle = oracle
            if self.mode == "identity":
                self.last_output = dense.reshape(*shape[:-1], -1)
                self.last_router_logits = None
                self.last_recall = None
                return self.last_output
            if self.mode == "oracle":
                active = oracle
                self.last_router_logits = None
                self.last_recall = None
            else:
                logits = (flat @ self.router_input) @ self.router_output + self.router_bias
                candidate_ids = torch.argsort(logits, dim=1, descending=True, stable=True)[:, : self.candidates]
                candidate_scores = exact_scores.gather(1, candidate_ids)
                local = torch.argsort(candidate_scores, dim=1, descending=True, stable=True)[:, : self.top_k]
                active = candidate_ids.gather(1, local)
                mask = torch.zeros_like(exact_scores, dtype=torch.bool)
                mask.scatter_(1, candidate_ids, True)
                self.last_recall = mask.gather(1, oracle).float().mean(dim=1)
                self.last_router_logits = logits
            selected = torch.zeros_like(activations)
            selected.scatter_(1, active, activations.gather(1, active))
            output = self.base.down_proj(selected)
            if self.mode == "trained":
                output = output + ((selected @ self.adapter_a) @ self.adapter_b) * self.adapter_scale
            self.last_output = output.reshape(*shape[:-1], -1)
            return self.last_output

    return SparseStudentMLP


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def train_sparse_student(
    model: str | Path,
    calibration_dataset: str | Path,
    validation_dataset: str | Path,
    calibration_traces: str | Path,
    out: str | Path,
    *,
    top_k: int = 768,
    candidate_count: int = 1280,
    router_rank: int = 16,
    router_regularization: float = 8000.0,
    adapter_rank: int = 8,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    local_weight: float = 1.0,
    hidden_weight: float = 0.25,
    logit_weight: float = 0.25,
    router_weight: float = 0.1,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as functional
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install engram-lm[conversion] for sparse-teacher training") from exc
    if epochs <= 0 or learning_rate <= 0.0 or adapter_rank <= 0:
        raise ValueError("epochs, learning_rate, and adapter_rank must be positive")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    if not 0 < top_k <= candidate_count <= inspection.intermediate_size:
        raise ValueError("require 0 < top_k <= candidate_count <= intermediate size")
    trace = TraceReader(calibration_traces)
    if trace.manifest["model_hash"] != inspection.source_hash or trace.manifest["split"] != "calibration":
        raise ValueError("calibration trace/model provenance mismatch")
    calibration_path = Path(calibration_dataset)
    validation_path = Path(validation_dataset)
    if trace.manifest["dataset_hash"] != sha256_file(calibration_path):
        raise ValueError("calibration trace/dataset hash mismatch")
    train_records = _load_jsonl(calibration_path, max_train_records)
    validation_records = _load_jsonl(validation_path, max_validation_records)
    teacher = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    student = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    calibration_sequence_hashes = _sequence_hashes(trace)
    validation_sequence_hashes = _evaluation_sequence_hashes(validation_records, tokenizer)
    if set(calibration_sequence_hashes).intersection(validation_sequence_hashes):
        raise ValueError("calibration and validation contain matching token sequences")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    SparseStudentMLP = _wrap_sparse_mlp_class(torch)
    wrappers = []
    for layer, decoder in enumerate(student.model.layers):
        teacher_mlp = teacher.model.layers[layer].mlp
        gate = teacher_mlp.gate_proj.weight.detach().cpu().numpy().astype(np.float64)
        up = teacher_mlp.up_proj.weight.detach().cpu().numpy().astype(np.float64)
        down = teacher_mlp.down_proj.weight.detach().cpu().numpy().astype(np.float64)
        states = _load_states(trace, layer, None)
        labels = _membership(states, gate, up, down, top_k).astype(np.float64)
        router = LowRankMultiLabelRouter.fit(
            states, labels, rank=router_rank, regularization=router_regularization
        )
        wrapper = SparseStudentMLP(
            decoder.mlp,
            router,
            top_k=top_k,
            candidates=candidate_count,
            adapter_rank=adapter_rank,
        ).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)

    teacher_targets: dict[int, Any] = {}
    handles = [
        layer.mlp.register_forward_hook(
            lambda module, args, output, index=index: teacher_targets.__setitem__(
                index, output.detach()
            )
        )
        for index, layer in enumerate(teacher.model.layers)
    ]
    optimizer = torch.optim.AdamW(
        [parameter for wrapper in wrappers for parameter in wrapper.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    history = []
    student.train()
    try:
        for epoch in range(epochs):
            for record_index, record in enumerate(train_records):
                input_ids = _ids(record, tokenizer, torch, device)
                if input_ids.shape[1] < 2:
                    continue
                teacher_targets.clear()
                with torch.no_grad():
                    teacher_output = teacher(
                        input_ids=input_ids,
                        use_cache=False,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                optimizer.zero_grad(set_to_none=True)
                student_output = student(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                local_loss = torch.stack(
                    [
                        functional.mse_loss(wrapper.last_output, teacher_targets[layer])
                        / torch.clamp(torch.mean(teacher_targets[layer] ** 2), min=1e-8)
                        for layer, wrapper in enumerate(wrappers)
                    ]
                ).mean()
                hidden_loss = torch.stack(
                    [
                        functional.mse_loss(student_hidden, teacher_hidden)
                        / torch.clamp(torch.mean(teacher_hidden ** 2), min=1e-8)
                        for student_hidden, teacher_hidden in zip(
                            student_output.hidden_states[1:], teacher_output.hidden_states[1:], strict=True
                        )
                    ]
                ).mean()
                teacher_logp = functional.log_softmax(teacher_output.logits.detach(), dim=-1)
                student_logp = functional.log_softmax(student_output.logits, dim=-1)
                logit_loss = functional.kl_div(student_logp, teacher_logp.exp(), reduction="batchmean") / input_ids.shape[1]
                router_loss = torch.stack(
                    [
                        functional.binary_cross_entropy_with_logits(
                            wrapper.last_router_logits,
                            torch.zeros_like(wrapper.last_router_logits).scatter(
                                1, wrapper.last_oracle, 1.0
                            ),
                        )
                        for wrapper in wrappers
                    ]
                ).mean()
                loss = (
                    local_weight * local_loss
                    + hidden_weight * hidden_loss
                    + logit_weight * logit_loss
                    + router_weight * router_loss
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
                optimizer.step()
                history.append(
                    {
                        "epoch": epoch,
                        "record": record_index,
                        "total": float(loss.detach()),
                        "local": float(local_loss.detach()),
                        "hidden": float(hidden_loss.detach()),
                        "logit": float(logit_loss.detach()),
                        "router": float(router_loss.detach()),
                    }
                )
    finally:
        for handle in handles:
            handle.remove()

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for layer, wrapper in enumerate(wrappers):
        for name in ("router_input", "router_output", "router_bias", "adapter_a", "adapter_b"):
            tensors[f"layer_{layer}.{name}"] = getattr(wrapper, name).detach().cpu().contiguous()
    tensor_path = target / "sparse_student.safetensors"
    save_file(tensors, tensor_path)

    student.eval()
    quality = defaultdict(list)
    local_error = []
    recalls = []
    teacher_nll = []
    input_positions = 0
    next_positions = 0
    validation_handles = [
        layer.mlp.register_forward_hook(
            lambda module, args, output, index=index: teacher_targets.__setitem__(
                index, output.detach()
            )
        )
        for index, layer in enumerate(teacher.model.layers)
    ]
    try:
        for record in validation_records:
            input_ids = _ids(record, tokenizer, torch, device)
            if input_ids.shape[1] < 2:
                continue
            input_positions += int(input_ids.shape[1])
            next_positions += int(input_ids.shape[1] - 1)
            teacher_targets.clear()
            with torch.inference_mode():
                teacher_output = teacher(
                    input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True
                )
                student_output = student(
                    input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True
                )
            metrics = _quality_metrics(
                teacher_output.logits,
                student_output.logits,
                input_ids,
                teacher_output.hidden_states[-1],
                student_output.hidden_states[-1],
                torch,
            )
            for name, values in metrics.items():
                quality[name].extend(np.asarray(values).reshape(-1).tolist())
            teacher_nll.extend(metrics["teacher_nll"].tolist())
            for layer, wrapper in enumerate(wrappers):
                relative, _ = _relative_and_cosine_rows(
                    wrapper.last_output.detach().cpu().numpy(),
                    teacher_targets[layer].detach().cpu().numpy(),
                )
                local_error.extend(relative.tolist())
                recalls.extend(wrapper.last_recall.detach().cpu().numpy().reshape(-1).tolist())
    finally:
        for handle in validation_handles:
            handle.remove()

    metrics_mean = {name: _mean(values) for name, values in quality.items()}
    checks = {
        "teacher_student_kl": metrics_mean["teacher_student_kl"] <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": metrics_mean["teacher_top1_agreement"] >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": metrics_mean["nll_delta"] <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": metrics_mean["final_hidden_relative_l2"] <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "candidate_recall": _mean(recalls) >= MINIMUM_ROUTED_CANDIDATE_RECALL,
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_sequence_hashes)) >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
    }
    report = {
        "schema_version": 1,
        "experiment": "sparse_teacher_finetuning",
        "status": "measured_local_model",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "top_k": top_k,
            "candidate_count": candidate_count,
            "router_rank": router_rank,
            "router_regularization": router_regularization,
            "adapter_rank": adapter_rank,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss_weights": {"local": local_weight, "hidden": hidden_weight, "logit": logit_weight, "router": router_weight},
        },
        "training": {"records": len(train_records), "steps": len(history), "history": history},
        "validation": {"records": len(validation_records), "input_token_positions": input_positions, "next_token_positions": next_positions},
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "calibration_sequences": len(calibration_sequence_hashes),
            "validation_sequences": len(validation_sequence_hashes),
            "overlapping_sequences": 0,
            "held_out": True,
        },
        "metrics": {
            **{name: _stats(values) for name, values in quality.items()},
            "local_mlp_relative_l2": _stats(local_error),
            "candidate_recall": _stats(recalls),
        },
        "gate": {"passed": all(checks.values()), "checks": checks, "decision": "eligible_for_intervention_artifact" if all(checks.values()) else "stop_before_serialization"},
        "artifact": {"path": str(tensor_path.resolve()), "sha256": sha256_file(tensor_path), "format": "safetensors_router_and_mlp_down_adapters_only"},
    }
    atomic_json(target / "sparse_teacher_training.json", report)
    metric = report["metrics"]
    lines = [
        "# Sparse-teacher fine-tuning pilot",
        "",
        f"Status: **{report['gate']['decision']}**",
        "",
        f"Training records/steps: {len(train_records)}/{len(history)}; validation records: "
        f"{len(validation_records)}.",
        "",
        "| Metric | Mean | Threshold | Pass |",
        "|---|---:|---:|---|",
        f"| Candidate recall | {metric['candidate_recall']['mean']:.6f} | ≥0.95 | "
        f"{'yes' if checks['candidate_recall'] else 'no'} |",
        f"| Teacher-student KL | {metric['teacher_student_kl']['mean']:.6f} | ≤0.05 | "
        f"{'yes' if checks['teacher_student_kl'] else 'no'} |",
        f"| Teacher top-1 agreement | {metric['teacher_top1_agreement']['mean']:.6f} | ≥0.90 | "
        f"{'yes' if checks['teacher_top1_agreement'] else 'no'} |",
        f"| NLL delta | {metric['nll_delta']['mean']:.6f} | ≤0.05 | "
        f"{'yes' if checks['nll_delta'] else 'no'} |",
        f"| Final hidden relative L2 | {metric['final_hidden_relative_l2']['mean']:.6f} | ≤0.10 | "
        f"{'yes' if checks['final_hidden_relative_l2'] else 'no'} |",
        "",
        "The artifact contains only router factors and sparse MLP down-adapter tensors. It is not",
        "eligible for package serialization unless every held-out gate check passes.",
        "",
    ]
    (target / "sparse_teacher_training.md").write_text("\n".join(lines), encoding="utf-8")
    return report


__all__ = ["train_sparse_student"]
