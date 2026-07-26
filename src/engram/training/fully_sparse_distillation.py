"""Whole-model distillation through exact hard Q-Sparse MLP execution.

CUDA is an optional training accelerator. Saved tensors are device-neutral,
and the deployment contract requires a separate CPU-only reload validation.
The artifact produced here is a training artifact, not yet the packed Q4
runtime representation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.gates import (
    MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION,
    MINIMUM_EVALUATION_SEQUENCES,
    MINIMUM_NEXT_TOKEN_POSITIONS,
    MINIMUM_UNIQUE_EVALUATION_SEQUENCES,
    MLP_QUALITY_THRESHOLDS,
)
from engram.evaluation.mlp_intervention import (
    _evaluation_sequence_hashes,
    _quality_metrics,
    _relative_and_cosine_rows,
)
from engram.models.inspection import inspect_model, resolve_model_path
from engram.training.fully_sparse import Q_SPARSE_REFERENCE, fully_sparse_mlp_traffic
from engram.training.on_policy import _import_causal_lm
from engram.training.sparse_teacher import (
    _batch_ids,
    _batches,
    _load_jsonl,
    _masked_mean,
    _normalized_masked_mse,
    _same_input_teacher_mlp_targets,
)
from engram.training.structured_experts import _stats
from engram.utils import atomic_json, sha256_file


def progressive_fully_sparse_counts(
    hidden_size: int,
    intermediate_size: int,
    *,
    target_input_count: int,
    target_intermediate_count: int,
    step: int,
    warmup_steps: int,
    anneal_steps: int,
) -> tuple[int, int]:
    """Return dense-to-target counts for a zero-based optimization step."""

    dimensions = (
        hidden_size,
        intermediate_size,
        target_input_count,
        target_intermediate_count,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in dimensions
    ):
        raise ValueError("dimensions and target counts must be integers")
    if not 0 < target_input_count <= hidden_size:
        raise ValueError("target_input_count must lie within hidden_size")
    if not 0 < target_intermediate_count <= intermediate_size:
        raise ValueError("target_intermediate_count must lie within intermediate_size")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (step, warmup_steps, anneal_steps)
    ):
        raise ValueError("step and schedule lengths must be nonnegative integers")
    if step < warmup_steps:
        progress = 0.0
    elif anneal_steps == 0:
        progress = 1.0
    else:
        linear = min(1.0, (step - warmup_steps + 1) / anneal_steps)
        progress = float(np.sin(0.5 * np.pi * linear) ** 2)
    input_count = round(hidden_size - progress * (hidden_size - target_input_count))
    intermediate_count = round(
        intermediate_size - progress * (intermediate_size - target_intermediate_count)
    )
    return (
        max(target_input_count, min(hidden_size, input_count)),
        max(
            target_intermediate_count,
            min(intermediate_size, intermediate_count),
        ),
    )


def fully_sparse_mlp_class(torch: Any):
    """Build a SwiGLU module with exact top-K activations and an STE backward."""

    class FullySparseMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            *,
            input_count: int,
            intermediate_count: int,
            residual_rank: int = 0,
        ):
            super().__init__()
            if (
                base.gate_proj.bias is not None
                or base.up_proj.bias is not None
                or base.down_proj.bias is not None
            ):
                raise ValueError("bias-enabled MLP projections are not supported")
            intermediate, hidden = base.gate_proj.weight.shape
            if not 0 < input_count <= hidden:
                raise ValueError("input_count must lie within hidden size")
            if not 0 < intermediate_count <= intermediate:
                raise ValueError("intermediate_count must lie within intermediate size")
            if (
                isinstance(residual_rank, bool)
                or not isinstance(residual_rank, int)
                or residual_rank < 0
                or residual_rank > hidden
            ):
                raise ValueError("residual_rank must lie within [0, hidden size]")
            self.gate_weight = torch.nn.Parameter(
                base.gate_proj.weight.detach().clone()
            )
            self.up_weight = torch.nn.Parameter(base.up_proj.weight.detach().clone())
            self.down_weight = torch.nn.Parameter(
                base.down_proj.weight.detach().clone()
            )
            if residual_rank:
                self.residual_input_weight = torch.nn.Parameter(
                    torch.empty(
                        residual_rank,
                        hidden,
                        dtype=self.gate_weight.dtype,
                        device=self.gate_weight.device,
                    )
                )
                torch.nn.init.normal_(
                    self.residual_input_weight,
                    mean=0.0,
                    std=hidden**-0.5,
                )
                self.residual_output_weight = torch.nn.Parameter(
                    torch.zeros(
                        hidden,
                        residual_rank,
                        dtype=self.gate_weight.dtype,
                        device=self.gate_weight.device,
                    )
                )
            else:
                self.register_parameter("residual_input_weight", None)
                self.register_parameter("residual_output_weight", None)
            self.act_fn = base.act_fn
            self.input_count = input_count
            self.intermediate_count = intermediate_count
            self.use_training_ste = True
            self.last_output = None
            self.last_input = None
            self.last_input_indices = None
            self.last_intermediate_indices = None
            self.last_surrogate_used = False

        def set_budget(self, *, input_count: int, intermediate_count: int) -> None:
            intermediate, hidden = self.gate_weight.shape
            if not 0 < input_count <= hidden:
                raise ValueError("input_count must lie within hidden size")
            if not 0 < intermediate_count <= intermediate:
                raise ValueError("intermediate_count must lie within intermediate size")
            self.input_count = input_count
            self.intermediate_count = intermediate_count

        @staticmethod
        def _hard_top_k(values: Any, count: int) -> tuple[Any, Any]:
            if count == values.shape[-1]:
                indices = torch.arange(
                    count, device=values.device, dtype=torch.long
                ).expand(*values.shape[:-1], -1)
                return values, indices
            indices = torch.topk(values.abs(), count, dim=-1, sorted=False).indices
            mask = torch.zeros_like(values).scatter(-1, indices, 1.0)
            return values * mask, indices

        def forward(self, hidden: Any) -> Any:
            self.last_input = hidden
            sparse_hidden, input_indices = self._hard_top_k(hidden, self.input_count)
            surrogate_used = bool(self.training and self.use_training_ste)
            if surrogate_used:
                # Forward remains exactly hard; backward is the identity.
                sparse_hidden = sparse_hidden + hidden - hidden.detach()
            gate = torch.nn.functional.linear(sparse_hidden, self.gate_weight)
            up = torch.nn.functional.linear(sparse_hidden, self.up_weight)
            activation = self.act_fn(gate) * up
            sparse_activation, intermediate_indices = self._hard_top_k(
                activation, self.intermediate_count
            )
            if surrogate_used:
                sparse_activation = sparse_activation + activation - activation.detach()
            output = torch.nn.functional.linear(sparse_activation, self.down_weight)
            if self.residual_input_weight is not None:
                output = output + torch.nn.functional.linear(
                    torch.nn.functional.linear(hidden, self.residual_input_weight),
                    self.residual_output_weight,
                )
            self.last_output = output
            self.last_input_indices = input_indices
            self.last_intermediate_indices = intermediate_indices
            self.last_surrogate_used = surrogate_used
            return output

    return FullySparseMLP


def validate_fully_sparse_artifact_cpu(
    artifact: str | Path,
    *,
    hidden_size: int,
    intermediate_size: int,
    input_count: int,
    intermediate_count: int,
) -> dict[str, Any]:
    """Reload a training artifact on CPU and execute its hard sparse math."""

    try:
        import torch
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for CPU artifact validation"
        ) from exc
    tensors = load_file(str(Path(artifact)), device="cpu")
    required = ("layer_0.gate", "layer_0.up", "layer_0.down")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"fully sparse artifact is missing tensors: {missing}")
    if any(tensor.device.type != "cpu" for tensor in tensors.values()):
        raise RuntimeError("fully sparse artifact did not reload entirely on CPU")
    gate, up, down = (tensors[name] for name in required)
    if tuple(gate.shape) != (intermediate_size, hidden_size):
        raise ValueError("artifact gate tensor has an incompatible shape")
    if tuple(up.shape) != tuple(gate.shape):
        raise ValueError("artifact up tensor has an incompatible shape")
    if tuple(down.shape) != (hidden_size, intermediate_size):
        raise ValueError("artifact down tensor has an incompatible shape")
    sample = torch.linspace(-1.0, 1.0, hidden_size).reshape(1, 1, -1)
    hidden_indices = torch.topk(sample.abs(), input_count, dim=-1, sorted=False).indices
    sparse_hidden = torch.zeros_like(sample).scatter(
        -1, hidden_indices, sample.gather(-1, hidden_indices)
    )
    activation = torch.nn.functional.silu(
        torch.nn.functional.linear(sparse_hidden, gate)
    ) * torch.nn.functional.linear(sparse_hidden, up)
    activation_indices = torch.topk(
        activation.abs(), intermediate_count, dim=-1, sorted=False
    ).indices
    sparse_activation = torch.zeros_like(activation).scatter(
        -1,
        activation_indices,
        activation.gather(-1, activation_indices),
    )
    output = torch.nn.functional.linear(sparse_activation, down)
    residual_input = tensors.get("layer_0.residual_input")
    residual_output = tensors.get("layer_0.residual_output")
    if (residual_input is None) != (residual_output is None):
        raise ValueError("artifact contains an incomplete low-rank residual")
    if residual_input is not None:
        output = output + torch.nn.functional.linear(
            torch.nn.functional.linear(sample, residual_input),
            residual_output,
        )
    finite = bool(torch.isfinite(output).all().item())
    if not finite:
        raise RuntimeError("CPU artifact validation produced non-finite output")
    return {
        "passed": True,
        "execution_device": str(output.device),
        "cuda_required": False,
        "hard_input_cardinality": input_count,
        "hard_intermediate_cardinality": intermediate_count,
        "output_shape": list(output.shape),
        "all_finite": finite,
    }


def train_fully_sparse_student(
    model: str | Path,
    training_dataset: str | Path,
    validation_dataset: str | Path,
    out: str | Path,
    *,
    input_fraction: float = 0.49,
    intermediate_fraction: float = 0.34,
    input_counts: Sequence[int] | None = None,
    intermediate_counts: Sequence[int] | None = None,
    steps: int = 8,
    warmup_steps: int = 1,
    anneal_steps: int = 6,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    backbone_learning_rate: float = 3e-6,
    local_weight: float = 0.25,
    hidden_weight: float = 0.5,
    logit_weight: float = 1.0,
    label_weight: float = 0.0,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
    device: str = "cuda",
    checkpoint_every: int = 0,
    resume: bool = False,
    coadapt_backbone: bool = False,
    coadapt_embeddings_and_head: bool = False,
    residual_rank: int = 0,
) -> dict[str, Any]:
    """Distill all student MLPs while executing the exact hard sparse path."""

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for fully sparse distillation"
        ) from exc
    AutoModelForCausalLM, AutoTokenizer = _import_causal_lm()
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (steps, warmup_steps, anneal_steps, checkpoint_every)
    ):
        raise ValueError("step counts must be nonnegative integers")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if steps and warmup_steps + anneal_steps > steps:
        raise ValueError("warmup_steps + anneal_steps cannot exceed steps")
    if not 0 < input_fraction <= 1 or not 0 < intermediate_fraction <= 1:
        raise ValueError("sparse fractions must lie in (0, 1]")
    scalars = (
        learning_rate,
        backbone_learning_rate,
        local_weight,
        hidden_weight,
        logit_weight,
        label_weight,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("learning rate and loss weights must be finite")
    if (
        learning_rate <= 0
        or backbone_learning_rate <= 0
        or any(value < 0 for value in scalars[2:])
    ):
        raise ValueError("learning rates must be positive and weights nonnegative")
    if coadapt_embeddings_and_head and not coadapt_backbone:
        raise ValueError("embedding/head co-adaptation requires backbone co-adaptation")
    if (
        isinstance(residual_rank, bool)
        or not isinstance(residual_rank, int)
        or residual_rank < 0
    ):
        raise ValueError("residual_rank must be a nonnegative integer")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    input_count = max(1, round(input_fraction * inspection.hidden_size))
    intermediate_count = max(
        1, round(intermediate_fraction * inspection.intermediate_size)
    )
    if (input_counts is None) != (intermediate_counts is None):
        raise ValueError(
            "input_counts and intermediate_counts must be provided together"
        )
    if input_counts is None:
        layer_input_counts = (input_count,) * inspection.num_hidden_layers
        layer_intermediate_counts = (intermediate_count,) * inspection.num_hidden_layers
    else:
        layer_input_counts = tuple(input_counts)
        layer_intermediate_counts = tuple(intermediate_counts or ())
        if (
            len(layer_input_counts) != inspection.num_hidden_layers
            or len(layer_intermediate_counts) != inspection.num_hidden_layers
        ):
            raise ValueError(
                "layer schedules must contain one count per transformer layer"
            )
    layer_traffic = [
        fully_sparse_mlp_traffic(
            inspection.hidden_size,
            inspection.intermediate_size,
            layer_input,
            layer_intermediate,
        )
        for layer_input, layer_intermediate in zip(
            layer_input_counts, layer_intermediate_counts, strict=True
        )
    ]
    residual_weights_per_layer = 2 * inspection.hidden_size * residual_rank
    projected_weights = (
        sum(int(layer["projected_weights_per_token_layer"]) for layer in layer_traffic)
        + inspection.num_hidden_layers * residual_weights_per_layer
    )
    dense_weights = sum(
        int(layer["dense_weights_per_token_layer"]) for layer in layer_traffic
    )
    traffic = {
        "bytes_per_weight": layer_traffic[0]["bytes_per_weight"],
        "input_counts": list(layer_input_counts),
        "intermediate_counts": list(layer_intermediate_counts),
        "layers": layer_traffic,
        "residual_rank": residual_rank,
        "residual_weights_per_token_layer": residual_weights_per_layer,
        "residual_weight_fraction_of_dense": (
            residual_weights_per_layer
            / int(layer_traffic[0]["dense_weights_per_token_layer"])
        ),
        "projected_weights_per_token": projected_weights,
        "dense_weights_per_token": dense_weights,
        "projected_bytes_per_token": (
            projected_weights * float(layer_traffic[0]["bytes_per_weight"])
        ),
        "dense_bytes_per_token": (
            dense_weights * float(layer_traffic[0]["bytes_per_weight"])
        ),
        "fraction_of_dense": projected_weights / dense_weights,
        "candidate_recall_applicable": False,
        "metadata_included": False,
    }
    training_path = Path(training_dataset)
    validation_path = Path(validation_dataset)
    train_records = _load_jsonl(training_path, max_train_records)
    validation_records = _load_jsonl(validation_path, max_validation_records)
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to(device)
    student = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to(device)
    if any(
        "input_ids" not in record for record in (*train_records, *validation_records)
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    else:

        class TokenIdOnlyTokenizer:
            pad_token_id = student.config.pad_token_id
            eos_token_id = student.config.eos_token_id

        tokenizer = TokenIdOnlyTokenizer()
    training_hashes = _evaluation_sequence_hashes(train_records, tokenizer)
    validation_hashes = _evaluation_sequence_hashes(validation_records, tokenizer)
    if set(training_hashes).intersection(validation_hashes):
        raise ValueError("training and validation contain matching token sequences")

    teacher.eval()
    student.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    wrapper_type = fully_sparse_mlp_class(torch)
    wrappers = []
    for layer, decoder in enumerate(student.model.layers):
        wrapper = wrapper_type(
            decoder.mlp,
            input_count=inspection.hidden_size,
            intermediate_count=inspection.intermediate_size,
            residual_rank=residual_rank,
        ).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)
    trainable_named = {
        f"layer_{layer}.{name}": parameter
        for layer, wrapper in enumerate(wrappers)
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }
    backbone_named: dict[str, Any] = {}
    if coadapt_backbone:
        for name, parameter in student.named_parameters():
            if (
                ".self_attn." in name
                or name.endswith("input_layernorm.weight")
                or name.endswith("post_attention_layernorm.weight")
                or name == "model.norm.weight"
                or (
                    coadapt_embeddings_and_head
                    and name in {"model.embed_tokens.weight", "lm_head.weight"}
                )
            ):
                parameter.requires_grad_(True)
                backbone_named[f"backbone.{name}"] = parameter
    trainable_named.update(backbone_named)
    trainable = list(trainable_named.values())
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for name, parameter in trainable_named.items()
                if not name.startswith("backbone.")
            ],
            "lr": learning_rate,
        }
    ]
    if backbone_named:
        parameter_groups.append(
            {
                "params": list(backbone_named.values()),
                "lr": backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)
    configuration = {
        "source_model_hash": inspection.source_hash,
        "training_dataset_hash": sha256_file(training_path),
        "validation_dataset_hash": sha256_file(validation_path),
        "input_count": input_count,
        "intermediate_count": intermediate_count,
        "layer_input_counts": list(layer_input_counts),
        "layer_intermediate_counts": list(layer_intermediate_counts),
        "warmup_steps": warmup_steps,
        "anneal_steps": anneal_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "coadapt_backbone": coadapt_backbone,
        "coadapt_embeddings_and_head": coadapt_embeddings_and_head,
        "residual_rank": residual_rank,
        "loss_weights": {
            "local": local_weight,
            "hidden": hidden_weight,
            "logit": logit_weight,
            "label": label_weight,
        },
    }
    checkpoint_path = target / "fully_sparse_training_checkpoint.pt"
    checkpoint_manifest_path = target / "fully_sparse_training_checkpoint.json"
    completed_steps = 0
    history: list[dict[str, int | float]] = []
    if resume:
        if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
            raise ValueError("resume requested but fully sparse checkpoint is missing")
        checkpoint_manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if checkpoint_manifest.get("configuration") != configuration:
            raise ValueError("fully sparse checkpoint configuration mismatch")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        saved = checkpoint["trainable_parameters"]
        if set(saved) != set(trainable_named):
            raise ValueError("fully sparse checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, parameter in trainable_named.items():
                parameter.copy_(saved[name].to(device=device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_steps = int(checkpoint_manifest["completed_steps"])
        history = list(checkpoint_manifest["history"])
        if completed_steps > steps:
            raise ValueError("requested total steps cannot precede the checkpoint")

    def save_checkpoint(completed: int) -> None:
        temporary = target / "fully_sparse_training_checkpoint.pt.tmp"
        torch.save(
            {
                "trainable_parameters": {
                    name: parameter.detach().cpu()
                    for name, parameter in trainable_named.items()
                },
                "optimizer": optimizer.state_dict(),
            },
            temporary,
        )
        temporary.replace(checkpoint_path)
        atomic_json(
            checkpoint_manifest_path,
            {
                "configuration": configuration,
                "completed_steps": completed,
                "history": history,
                "device_neutral": True,
            },
        )

    teacher_targets: dict[int, Any] = {}
    handles = [
        layer.mlp.register_forward_hook(
            lambda module, args, output, index=index: teacher_targets.__setitem__(
                index, output.detach()
            )
        )
        for index, layer in enumerate(teacher.model.layers)
    ]
    batches = list(_batches(train_records, batch_size))
    teacher_training_objectives = bool(local_weight or hidden_weight or logit_weight)
    try:
        for step in range(completed_steps, steps):
            current_budgets = [
                progressive_fully_sparse_counts(
                    inspection.hidden_size,
                    inspection.intermediate_size,
                    target_input_count=layer_input,
                    target_intermediate_count=layer_intermediate,
                    step=step,
                    warmup_steps=warmup_steps,
                    anneal_steps=anneal_steps,
                )
                for layer_input, layer_intermediate in zip(
                    layer_input_counts,
                    layer_intermediate_counts,
                    strict=True,
                )
            ]
            for wrapper, (current_input, current_intermediate) in zip(
                wrappers, current_budgets, strict=True
            ):
                wrapper.set_budget(
                    input_count=current_input,
                    intermediate_count=current_intermediate,
                )
                wrapper.train()
            batch_records = batches[step % len(batches)]
            input_ids, attention_mask, lengths = _batch_ids(
                batch_records, tokenizer, torch, device
            )
            if max(lengths) < 2:
                raise ValueError("training records must contain at least two tokens")
            valid_mask = attention_mask.bool()
            teacher_targets.clear()
            teacher_output = None
            if teacher_training_objectives:
                with torch.no_grad():
                    teacher_output = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=bool(hidden_weight),
                        return_dict=True,
                    )
            optimizer.zero_grad(set_to_none=True)
            student_output = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=bool(hidden_weight),
                return_dict=True,
            )
            zero = torch.zeros((), device=input_ids.device)
            if local_weight:
                same_input_targets = _same_input_teacher_mlp_targets(
                    [layer.mlp for layer in teacher.model.layers],
                    [wrapper.last_input for wrapper in wrappers],
                    torch,
                )
                local_loss = torch.stack(
                    [
                        _normalized_masked_mse(
                            wrapper.last_output,
                            same_input_targets[layer],
                            valid_mask,
                            torch,
                        )
                        for layer, wrapper in enumerate(wrappers)
                    ]
                ).mean()
            else:
                local_loss = zero
            if hidden_weight:
                assert teacher_output is not None
                hidden_loss = torch.stack(
                    [
                        _normalized_masked_mse(
                            student_hidden,
                            teacher_hidden,
                            valid_mask,
                            torch,
                        )
                        for student_hidden, teacher_hidden in zip(
                            student_output.hidden_states[1:],
                            teacher_output.hidden_states[1:],
                            strict=True,
                        )
                    ]
                ).mean()
            else:
                hidden_loss = zero
            if logit_weight:
                assert teacher_output is not None
                teacher_logp = torch.nn.functional.log_softmax(
                    teacher_output.logits.detach(), dim=-1
                )
                student_logp = torch.nn.functional.log_softmax(
                    student_output.logits, dim=-1
                )
                logit_rows = torch.nn.functional.kl_div(
                    student_logp, teacher_logp.exp(), reduction="none"
                ).sum(dim=-1)
                logit_loss = _masked_mean(logit_rows, valid_mask, torch)
            else:
                logit_loss = zero
            shift_logits = student_output.logits[:, :-1].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = valid_mask[:, 1:]
            label_rows = torch.nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction="none",
            ).reshape_as(shift_labels)
            label_loss = _masked_mean(label_rows, shift_mask, torch)
            loss = (
                local_weight * local_loss
                + hidden_weight * hidden_loss
                + logit_weight * logit_loss
                + label_weight * label_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "minimum_input_count": min(budget[0] for budget in current_budgets),
                    "maximum_input_count": max(budget[0] for budget in current_budgets),
                    "minimum_intermediate_count": min(
                        budget[1] for budget in current_budgets
                    ),
                    "maximum_intermediate_count": max(
                        budget[1] for budget in current_budgets
                    ),
                    "total": float(loss.detach()),
                    "local": float(local_loss.detach()),
                    "hidden": float(hidden_loss.detach()),
                    "logit": float(logit_loss.detach()),
                    "label": float(label_loss.detach()),
                }
            )
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_checkpoint(step + 1)
        if checkpoint_every and len(history) != completed_steps:
            save_checkpoint(len(history))

        for wrapper, layer_input, layer_intermediate in zip(
            wrappers,
            layer_input_counts,
            layer_intermediate_counts,
            strict=True,
        ):
            wrapper.set_budget(
                input_count=layer_input,
                intermediate_count=layer_intermediate,
            )
            wrapper.eval()
        quality: dict[str, list[float]] = {}
        local_error: list[float] = []
        input_positions = 0
        next_positions = 0
        for batch_records in _batches(validation_records, batch_size):
            input_ids, attention_mask, lengths = _batch_ids(
                batch_records, tokenizer, torch, device
            )
            teacher_targets.clear()
            with torch.inference_mode():
                teacher_output = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                student_output = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            if any(wrapper.last_surrogate_used for wrapper in wrappers):
                raise RuntimeError("validation executed a training-only STE")
            for row, length in enumerate(lengths):
                if length < 2:
                    continue
                input_positions += length
                next_positions += length - 1
                metrics = _quality_metrics(
                    teacher_output.logits[row : row + 1, :length],
                    student_output.logits[row : row + 1, :length],
                    input_ids[row : row + 1, :length],
                    teacher_output.hidden_states[-1][row : row + 1, :length],
                    student_output.hidden_states[-1][row : row + 1, :length],
                    torch,
                )
                for name, values in metrics.items():
                    quality.setdefault(name, []).extend(
                        np.asarray(values).reshape(-1).tolist()
                    )
                for layer, wrapper in enumerate(wrappers):
                    relative, _ = _relative_and_cosine_rows(
                        wrapper.last_output[row : row + 1, :length]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                        teacher_targets[layer][row : row + 1, :length]
                        .detach()
                        .float()
                        .cpu()
                        .numpy(),
                    )
                    local_error.extend(relative.tolist())
    finally:
        for handle in handles:
            handle.remove()

    artifact_tensors = {
        f"layer_{layer}.{name.removesuffix('_weight')}": parameter.detach()
        .cpu()
        .contiguous()
        for layer, wrapper in enumerate(wrappers)
        for name, parameter in wrapper.named_parameters()
    }
    artifact_tensors.update(
        {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in backbone_named.items()
        }
    )
    artifact_path = target / "fully_sparse_student.safetensors"
    save_file(artifact_tensors, artifact_path)
    cpu_reload = validate_fully_sparse_artifact_cpu(
        artifact_path,
        hidden_size=inspection.hidden_size,
        intermediate_size=inspection.intermediate_size,
        input_count=layer_input_counts[0],
        intermediate_count=layer_intermediate_counts[0],
    )
    metric_means = {name: float(np.mean(values)) for name, values in quality.items()}
    checks = {
        "teacher_student_kl": metric_means["teacher_student_kl"]
        <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": metric_means["teacher_top1_agreement"]
        >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": metric_means["nll_delta"]
        <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": metric_means["final_hidden_relative_l2"]
        <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_hashes)) >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
        "projected_traffic_before_metadata": (
            traffic["fraction_of_dense"] <= MAXIMUM_PROJECTED_MLP_TRAFFIC_FRACTION
        ),
        "cpu_artifact_reload": cpu_reload["passed"],
        "hard_validation_path": True,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "experiment": "whole_model_fully_sparse_distillation",
        "status": (
            "eligible_for_q4_cpu_kernel_development"
            if passed
            else "continue_training_or_reject_before_confirmation"
        ),
        "reference": Q_SPARSE_REFERENCE,
        "source_model_hash": inspection.source_hash,
        "configuration": {
            **configuration,
            "input_fraction_requested": input_fraction,
            "intermediate_fraction_requested": intermediate_fraction,
            "layer_adaptive_schedule": input_counts is not None,
            "training_device": device,
            "inference_device_contract": "cpu_only",
            "trainable_scope": (
                "all_student_mlp_projections_plus_attention_and_normalization"
                if coadapt_backbone
                else "all_student_mlp_projection_weights"
            ),
            "attention_and_normalization_frozen": not coadapt_backbone,
            "embeddings_and_head_frozen": not coadapt_embeddings_and_head,
            "low_rank_residual": {
                "rank": residual_rank,
                "input": "full_already_resident_hidden_state",
                "included_in_traffic": True,
            },
            "local_target": "frozen_teacher_mlp_on_detached_student_input",
            "teacher_forward_during_training": teacher_training_objectives,
            "straight_through_estimator": "training_only_identity_backward",
        },
        "training": {
            "records": len(train_records),
            "completed_before_resume": completed_steps,
            "requested_total_steps": steps,
            "history": history,
            "checkpoint": {
                "written": checkpoint_path.is_file(),
                "path": (
                    str(checkpoint_path.resolve())
                    if checkpoint_path.is_file()
                    else None
                ),
                "device_neutral": True,
            },
        },
        "validation": {
            "role": "development",
            "records": len(validation_records),
            "input_token_positions": input_positions,
            "next_token_positions": next_positions,
            "hard_sparse_path_only": True,
            "training_surrogate_used": False,
            "confirmation_opened": False,
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "training_sequences": len(training_hashes),
            "validation_sequences": len(validation_hashes),
            "overlapping_sequences": 0,
            "held_out": True,
            "training_dataset_hash": sha256_file(training_path),
            "validation_dataset_hash": sha256_file(validation_path),
        },
        "metrics": {
            **{name: _stats(values) for name, values in quality.items()},
            "local_mlp_relative_l2": _stats(local_error),
        },
        "traffic": traffic,
        "artifact": {
            "path": str(artifact_path.resolve()),
            "bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
            "device_neutral_tensors": True,
            "cpu_reload_validation": cpu_reload,
            "formal_deployment_artifact": False,
            "remaining_format_work": (
                "pack Q4 sparse rows with index metadata and execute them "
                "through the direct CPU kernel"
            ),
        },
        "gate": {
            "passed": passed,
            "checks": checks,
            "decision": (
                "implement_q4_cpu_kernel_then_run_confirmation"
                if passed
                else "continue_training_or_reject_before_confirmation"
            ),
        },
        "scope_caveat": (
            "CUDA may accelerate distillation only. The saved artifact reloads "
            "on CPU without CUDA, but it is not yet a packed Q4 CPU runtime "
            "artifact and its ideal traffic excludes sparse-index metadata."
        ),
    }
    atomic_json(target / "fully_sparse_distillation.json", report)
    lines = [
        "# Whole-model fully sparse distillation",
        "",
        f"Decision: **{report['gate']['decision']}**",
        "",
        f"Training device: `{device}`; inference contract: **CPU only**.",
        "",
        "| Metric | Mean | Threshold |",
        "|---|---:|---:|",
        f"| Teacher-student KL | {metric_means['teacher_student_kl']:.6f} | ≤0.05 |",
        f"| Teacher top-1 agreement | {metric_means['teacher_top1_agreement']:.6f} | ≥0.90 |",
        f"| NLL delta | {metric_means['nll_delta']:.6f} | ≤0.05 |",
        f"| Final hidden relative L2 | {metric_means['final_hidden_relative_l2']:.6f} | ≤0.10 |",
        f"| Local MLP relative L2 | {float(np.mean(local_error)):.6f} | diagnostic |",
        f"| Ideal Q4 traffic | {traffic['fraction_of_dense']:.6f}× | ≤0.45× |",
        "",
        "Validation used exact hard top-K forwards. The STE was disabled.",
        "The training artifact was independently reloaded and executed on CPU.",
        "Confirmation remains sealed until the development gate passes and a",
        "packed Q4 CPU kernel proves parity.",
        "",
    ]
    (target / "fully_sparse_distillation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report


__all__ = [
    "fully_sparse_mlp_class",
    "progressive_fully_sparse_counts",
    "train_fully_sparse_student",
    "validate_fully_sparse_artifact_cpu",
]
