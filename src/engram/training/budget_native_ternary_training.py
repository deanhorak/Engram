"""Continual quantization-aware distillation for grouped-ternary MLPs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.gates import (
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
from engram.training.budget_native_ternary import (
    confidence_weighted_kl,
    grouped_ternary_mlp_class,
    layer_quantization_strengths_for_step,
    masked_linear_cka_loss,
)
from engram.training.budget_native_ternary_codec import (
    budget_native_ternary_traffic,
    decode_budget_native_ternary_artifact,
    load_budget_native_ternary_artifact,
    save_budget_native_ternary_artifact,
)
from engram.training.on_policy import _import_causal_lm
from engram.training.sparse_teacher import (
    _batch_ids,
    _batches,
    _load_jsonl,
    _normalized_masked_mse,
)
from engram.training.structured_experts import _stats
from engram.utils import atomic_json, sha256_file


def _mean_metric(values: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(np.mean(rows)) for name, rows in values.items()}


def _next_token_cross_entropy(
    logits: Any,
    input_ids: Any,
    valid_mask: Any,
    torch: Any,
) -> Any:
    next_mask = valid_mask[:, 1:]
    if not bool(next_mask.any()):
        raise ValueError("training batch has no valid next-token position")
    rows = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).reshape_as(next_mask)
    return rows[next_mask].mean()


def _teacher_top1_cross_entropy(
    student_logits: Any,
    teacher_logits: Any,
    valid_mask: Any,
    torch: Any,
) -> Any:
    targets = teacher_logits.detach().argmax(dim=-1)
    return torch.nn.functional.cross_entropy(
        student_logits[valid_mask],
        targets[valid_mask],
    )


def train_budget_native_ternary_student(
    model: str | Path,
    training_dataset: str | Path,
    validation_dataset: str | Path,
    out: str | Path,
    *,
    group_size: int = 128,
    steps: int = 8,
    dense_warmup_steps: int = 0,
    anneal_steps: int = 4,
    transition_mode: str = "deepest_first",
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    backbone_learning_rate: float = 3e-5,
    local_weight: float = 0.1,
    hidden_weight: float = 0.5,
    final_hidden_weight: float = 1.0,
    final_cka_weight: float = 0.0,
    logit_weight: float = 1.0,
    teacher_top1_weight: float = 0.0,
    label_weight: float = 0.1,
    temperature: float = 1.0,
    confidence_weight: float = 0.5,
    coadapt_backbone: bool = False,
    coadapt_embeddings_and_head: bool = False,
    backbone_start_step: int | None = None,
    max_train_records: int | None = None,
    training_record_offset: int = 0,
    max_validation_records: int | None = None,
    device: str = "cpu",
    save_artifact: bool = True,
    checkpoint_every: int = 0,
    resume: bool = False,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train full-width grouped-ternary MLPs through their deployed forward.

    The source checkpoint supplies both the immutable dense teacher and float
    master initialization. Training retains optimizer state while a cosine
    schedule transitions the student from dense to hard grouped ternary. Every
    validation runs at strength 1.0. When ``save_artifact`` is true, validation
    executes weights decoded from the independently reloaded binary artifact.
    """

    try:
        import torch
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for grouped-ternary distillation"
        ) from exc
    AutoModelForCausalLM, AutoTokenizer = _import_causal_lm()
    for value, name, allow_zero in (
        (steps, "steps", True),
        (dense_warmup_steps, "dense_warmup_steps", True),
        (anneal_steps, "anneal_steps", False),
        (batch_size, "batch_size", False),
        (group_size, "group_size", False),
        (checkpoint_every, "checkpoint_every", True),
        (training_record_offset, "training_record_offset", True),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (0 if allow_zero else 1)
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    if steps and dense_warmup_steps + anneal_steps > steps:
        raise ValueError("warm-up and anneal phases exceed requested steps")
    if transition_mode not in {"global", "deepest_first"}:
        raise ValueError(
            "transition_mode must be 'global' or 'deepest_first'"
        )
    if backbone_start_step is None:
        backbone_start_step = dense_warmup_steps + anneal_steps
    if (
        isinstance(backbone_start_step, bool)
        or not isinstance(backbone_start_step, int)
        or backbone_start_step < 0
    ):
        raise ValueError("backbone_start_step must be a non-negative integer")
    scalars = (
        learning_rate,
        backbone_learning_rate,
        local_weight,
        hidden_weight,
        final_hidden_weight,
        final_cka_weight,
        logit_weight,
        teacher_top1_weight,
        label_weight,
        temperature,
        confidence_weight,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("learning rates and loss controls must be finite")
    if learning_rate <= 0 or backbone_learning_rate <= 0 or temperature <= 0:
        raise ValueError("learning rates and temperature must be positive")
    if any(value < 0 for value in scalars[2:9]) or confidence_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if coadapt_embeddings_and_head and not coadapt_backbone:
        raise ValueError(
            "embedding/head co-adaptation requires backbone co-adaptation"
        )
    if resume and not checkpoint_every:
        raise ValueError("resume requires checkpointing")
    if resume and initial_checkpoint is not None:
        raise ValueError(
            "resume and initial_checkpoint are mutually exclusive"
        )
    if not save_artifact and steps == 0:
        # A zero-step no-artifact call has no independently testable output
        # and is almost always an accidental misuse.
        raise ValueError("zero-step evaluation requires artifact serialization")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    traffic = budget_native_ternary_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        layer_count=inspection.num_hidden_layers,
        group_size=group_size,
    )
    if not traffic["passes_45_percent_traffic_gate"]:
        raise ValueError(
            "grouped-ternary layout exceeds the 45% traffic gate"
        )
    training_path = Path(training_dataset)
    validation_path = Path(validation_dataset)
    training_limit = (
        None
        if max_train_records is None
        else training_record_offset + max_train_records
    )
    train_records = _load_jsonl(training_path, training_limit)[
        training_record_offset:
    ]
    if not train_records:
        raise ValueError("training_record_offset leaves no training records")
    validation_records = _load_jsonl(
        validation_path, max_validation_records
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    student = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
    ).to(device)
    if any(
        "input_ids" not in record
        for record in (*train_records, *validation_records)
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
    else:

        class TokenIdOnlyTokenizer:
            pad_token_id = student.config.pad_token_id
            eos_token_id = student.config.eos_token_id

        tokenizer = TokenIdOnlyTokenizer()
    training_hashes = _evaluation_sequence_hashes(train_records, tokenizer)
    validation_hashes = _evaluation_sequence_hashes(
        validation_records, tokenizer
    )
    if set(training_hashes).intersection(validation_hashes):
        raise ValueError(
            "training and validation contain matching token sequences"
        )

    teacher.eval()
    student.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    wrapper_type = grouped_ternary_mlp_class(torch)
    wrappers = []
    for decoder in student.model.layers:
        wrapper = wrapper_type(
            decoder.mlp,
            group_size=group_size,
        ).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)
    mlp_named = {
        f"layers.{layer}.{name}": parameter
        for layer, wrapper in enumerate(wrappers)
        for name, parameter in wrapper.named_parameters()
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
                    and name
                    in {"model.embed_tokens.weight", "lm_head.weight"}
                )
            ):
                backbone_named[f"backbone.{name}"] = parameter
    trainable_named = {**mlp_named, **backbone_named}
    parameter_groups = [
        {"params": list(mlp_named.values()), "lr": learning_rate}
    ]
    if backbone_named:
        parameter_groups.append(
            {
                "params": list(backbone_named.values()),
                "lr": backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / "budget_native_ternary_checkpoint.pt"
    checkpoint_manifest_path = (
        target / "budget_native_ternary_checkpoint.json"
    )
    checkpoint_configuration = {
        "schema_version": 1,
        "source_model_hash": inspection.source_hash,
        "training_dataset_hash": sha256_file(training_path),
        "training_record_offset": training_record_offset,
        "max_train_records": max_train_records,
        "validation_dataset_hash": sha256_file(validation_path),
        "max_validation_records": max_validation_records,
        "group_size": group_size,
        "dense_warmup_steps": dense_warmup_steps,
        "anneal_steps": anneal_steps,
        "transition_mode": transition_mode,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "coadapt_backbone": coadapt_backbone,
        "coadapt_embeddings_and_head": coadapt_embeddings_and_head,
        "backbone_start_step": backbone_start_step,
        "local_weight": local_weight,
        "hidden_weight": hidden_weight,
        "final_hidden_weight": final_hidden_weight,
        "final_cka_weight": final_cka_weight,
        "logit_weight": logit_weight,
        "teacher_top1_weight": teacher_top1_weight,
        "label_weight": label_weight,
        "temperature": temperature,
        "confidence_weight": confidence_weight,
    }
    history: list[dict[str, Any]] = []
    completed_steps = 0
    initialization = {
        "source": "dense_source_model",
        "checkpoint": None,
        "optimizer_state_retained": False,
    }

    def load_checkpoint_parameters(
        checkpoint: dict[str, Any],
        *,
        label: str,
        allow_missing_backbone: bool = False,
    ) -> tuple[str, ...]:
        checkpoint_names = set(checkpoint["parameters"])
        current_names = set(trainable_named)
        unexpected = checkpoint_names.difference(current_names)
        missing = current_names.difference(checkpoint_names)
        missing_width = missing.intersection(mlp_named)
        if (
            unexpected
            or missing_width
            or (missing and not allow_missing_backbone)
        ):
            raise ValueError(
                f"grouped-ternary {label} parameter set mismatch"
            )
        with torch.no_grad():
            for name in checkpoint_names:
                parameter = trainable_named[name]
                source = checkpoint["parameters"][name]
                if source.shape != parameter.shape:
                    raise ValueError(
                        f"grouped-ternary {label} shape mismatch for {name}"
                    )
                parameter.copy_(source.to(device))
        return tuple(sorted(missing))

    if resume:
        if (
            not checkpoint_path.is_file()
            or not checkpoint_manifest_path.is_file()
        ):
            raise ValueError(
                "resume requested but grouped-ternary checkpoint is missing"
            )
        manifest = json.loads(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        if manifest["configuration"] != checkpoint_configuration:
            raise ValueError(
                "grouped-ternary checkpoint configuration mismatch"
            )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        load_checkpoint_parameters(checkpoint, label="checkpoint")
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        completed_steps = len(history)
        initialization = {
            "source": "resumed_checkpoint",
            "checkpoint": str(checkpoint_path.resolve()),
            "optimizer_state_retained": True,
        }
        if completed_steps > steps:
            raise ValueError(
                "checkpoint has more steps than the requested total"
            )
    elif initial_checkpoint is not None:
        initial_checkpoint_path = Path(initial_checkpoint)
        if not initial_checkpoint_path.is_file():
            raise ValueError(
                "grouped-ternary initial checkpoint is missing"
            )
        checkpoint = torch.load(
            initial_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        missing_initial = load_checkpoint_parameters(
            checkpoint,
            label="initial checkpoint",
            allow_missing_backbone=True,
        )
        optimizer_retained = not missing_initial
        if optimizer_retained:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group, learning_rate_value in zip(
                optimizer.param_groups,
                (learning_rate, backbone_learning_rate),
                strict=False,
            ):
                group["lr"] = learning_rate_value
        initialization = {
            "source": "initial_checkpoint",
            "checkpoint": str(initial_checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(initial_checkpoint_path),
            "optimizer_state_retained": optimizer_retained,
            "source_model_parameters_retained": list(missing_initial),
            "history_reset_for_new_protocol": True,
        }

    def save_checkpoint() -> None:
        checkpoint = {
            "parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in trainable_named.items()
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        atomic_json(
            checkpoint_manifest_path,
            {
                "configuration": checkpoint_configuration,
                "completed_steps": len(history),
                "device_neutral": True,
                "requested_total_steps": steps,
            },
        )

    teacher_targets: list[Any] = [None] * inspection.num_hidden_layers
    handles = []
    for layer, decoder in enumerate(teacher.model.layers):

        def capture(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            index=layer,
        ) -> None:
            teacher_targets[index] = output.detach()

        handles.append(decoder.mlp.register_forward_hook(capture))

    training_input_positions = 0
    training_next_positions = 0
    batches = _batches(train_records, batch_size)
    try:
        for step in range(completed_steps, steps):
            strengths = layer_quantization_strengths_for_step(
                step,
                total_steps=steps,
                dense_warmup_steps=dense_warmup_steps,
                anneal_steps=anneal_steps,
                layer_count=len(wrappers),
                transition_mode=transition_mode,
            )
            for wrapper, strength in zip(
                wrappers, strengths, strict=True
            ):
                wrapper.set_quantization_strength(strength)
                wrapper.train()
            backbone_active = (
                coadapt_backbone and step >= backbone_start_step
            )
            for parameter in backbone_named.values():
                parameter.requires_grad_(backbone_active)
            records = batches[step % len(batches)]
            input_ids, attention_mask, lengths = _batch_ids(
                records,
                tokenizer,
                torch,
                device,
            )
            valid_mask = attention_mask.bool()
            training_input_positions += sum(lengths)
            training_next_positions += sum(max(0, length - 1) for length in lengths)
            teacher_targets[:] = [None] * inspection.num_hidden_layers
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
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
            local_loss = torch.stack(
                [
                    _normalized_masked_mse(
                        wrapper.last_output,
                        teacher_targets[layer],
                        valid_mask,
                        torch,
                    )
                    for layer, wrapper in enumerate(wrappers)
                ]
            ).mean()
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
            final_hidden_loss = _normalized_masked_mse(
                student_output.hidden_states[-1],
                teacher_output.hidden_states[-1],
                valid_mask,
                torch,
            )
            final_cka_loss = masked_linear_cka_loss(
                student_output.hidden_states[-1],
                teacher_output.hidden_states[-1],
                valid_mask,
                torch,
            )
            logit_loss = confidence_weighted_kl(
                student_output.logits,
                teacher_output.logits,
                valid_mask,
                torch,
                temperature=temperature,
                confidence_weight=confidence_weight,
            )
            teacher_top1_loss = _teacher_top1_cross_entropy(
                student_output.logits,
                teacher_output.logits,
                valid_mask,
                torch,
            )
            label_loss = _next_token_cross_entropy(
                student_output.logits,
                input_ids,
                valid_mask,
                torch,
            )
            loss = (
                local_weight * local_loss
                + hidden_weight * hidden_loss
                + final_hidden_weight * final_hidden_loss
                + final_cka_weight * final_cka_loss
                + logit_weight * logit_loss
                + teacher_top1_weight * teacher_top1_loss
                + label_weight * label_loss
            )
            loss.backward()
            active_parameters = [
                *mlp_named.values(),
                *(
                    backbone_named.values()
                    if backbone_active
                    else ()
                ),
            ]
            torch.nn.utils.clip_grad_norm_(active_parameters, 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "quantization_strength": float(np.mean(strengths)),
                    "minimum_quantization_strength": min(strengths),
                    "maximum_quantization_strength": max(strengths),
                    "hard_ternary_layers": sum(
                        strength == 1.0 for strength in strengths
                    ),
                    "backbone_active": backbone_active,
                    "total": float(loss.detach()),
                    "local": float(local_loss.detach()),
                    "hidden": float(hidden_loss.detach()),
                    "final_hidden": float(final_hidden_loss.detach()),
                    "final_cka": float(final_cka_loss.detach()),
                    "logit": float(logit_loss.detach()),
                    "teacher_top1": float(teacher_top1_loss.detach()),
                    "label": float(label_loss.detach()),
                    "input_token_positions": sum(lengths),
                    "next_token_positions": sum(
                        max(0, length - 1) for length in lengths
                    ),
                }
            )
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_checkpoint()
        if checkpoint_every and len(history) != completed_steps:
            save_checkpoint()

        artifact = {
            "written": False,
            "reloaded_before_validation": False,
            "format": "engram_budget_native_grouped_ternary_v1",
            "mlp": None,
            "backbone": None,
        }
        if save_artifact:
            mlp_path = target / "budget_native_ternary_mlp.bin"
            save_budget_native_ternary_artifact(
                mlp_path,
                [wrapper.deployment_layer_weights() for wrapper in wrappers],
                group_size=group_size,
            )
            loaded = load_budget_native_ternary_artifact(mlp_path)
            decoded = decode_budget_native_ternary_artifact(loaded)
            if len(decoded) != len(wrappers):
                raise RuntimeError(
                    "grouped-ternary layer count changed on reload"
                )
            for wrapper, weights in zip(wrappers, decoded, strict=True):
                wrapper.install_deployment_weights(**weights)
            if mlp_path.stat().st_size != traffic["total_cold_bytes"]:
                raise RuntimeError(
                    "grouped-ternary file size differs from traffic accounting"
                )
            mlp_artifact = {
                "written": True,
                "reloaded_before_validation": True,
                "path": str(mlp_path.resolve()),
                "sha256": sha256_file(mlp_path),
                "bytes": mlp_path.stat().st_size,
                "traffic": traffic,
            }
            backbone_artifact: dict[str, Any] = {
                "written": False,
                "reloaded_before_validation": True,
                "required": bool(backbone_named),
                "source": "unchanged_source_model",
                "incremental_mlp_traffic_bytes": 0,
            }
            if backbone_named:
                backbone_path = (
                    target / "budget_native_ternary_backbone.safetensors"
                )
                save_file(
                    {
                        name: parameter.detach().cpu().contiguous()
                        for name, parameter in backbone_named.items()
                    },
                    backbone_path,
                    metadata={
                        "format": "engram_budget_native_ternary_backbone_v1",
                        "source_model_hash": inspection.source_hash,
                    },
                )
                reloaded_backbone = load_file(backbone_path, device=device)
                if set(reloaded_backbone) != set(backbone_named):
                    raise RuntimeError(
                        "grouped-ternary backbone tensor set changed on reload"
                    )
                with torch.no_grad():
                    for name, parameter in backbone_named.items():
                        value = reloaded_backbone[name]
                        if (
                            value.shape != parameter.shape
                            or value.dtype != parameter.dtype
                        ):
                            raise RuntimeError(
                                "grouped-ternary backbone artifact mismatch "
                                f"for {name}"
                            )
                        parameter.copy_(value)
                backbone_artifact.update(
                    written=True,
                    path=str(backbone_path.resolve()),
                    sha256=sha256_file(backbone_path),
                    source="serialized_safetensors_reload",
                )
            artifact.update(
                written=True,
                reloaded_before_validation=True,
                mlp=mlp_artifact,
                backbone=backbone_artifact,
            )
        else:
            for wrapper in wrappers:
                wrapper.set_quantization_strength(1.0)

        for wrapper in wrappers:
            wrapper.set_quantization_strength(1.0)
            wrapper.eval()
        student.eval()
        metrics: dict[str, list[float]] = {}
        local_error: list[float] = []
        input_positions = 0
        next_positions = 0
        for records in _batches(validation_records, batch_size):
            input_ids, attention_mask, lengths = _batch_ids(
                records,
                tokenizer,
                torch,
                device,
            )
            teacher_targets[:] = [None] * inspection.num_hidden_layers
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
            for row, length in enumerate(lengths):
                if length < 2:
                    continue
                input_positions += length
                next_positions += length - 1
                values = _quality_metrics(
                    teacher_output.logits[row : row + 1, :length],
                    student_output.logits[row : row + 1, :length],
                    input_ids[row : row + 1, :length],
                    teacher_output.hidden_states[-1][
                        row : row + 1, :length
                    ],
                    student_output.hidden_states[-1][
                        row : row + 1, :length
                    ],
                    torch,
                )
                for name, rows in values.items():
                    metrics.setdefault(name, []).extend(
                        np.asarray(rows).reshape(-1).tolist()
                    )
                for layer, wrapper in enumerate(wrappers):
                    relative, _ = _relative_and_cosine_rows(
                        wrapper.last_output[row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                        teacher_targets[layer][row : row + 1, :length]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                    local_error.extend(relative.tolist())
    finally:
        for handle in handles:
            handle.remove()

    means = _mean_metric(metrics)
    zero_fractions = {
        name: float(
            np.mean(
                [wrapper.zero_fractions()[name] for wrapper in wrappers]
            )
        )
        for name in ("gate", "up", "down")
    }
    checks = {
        "teacher_student_kl": means["teacher_student_kl"]
        <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": means["teacher_top1_agreement"]
        >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": means["nll_delta"]
        <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": means["final_hidden_relative_l2"]
        <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_hashes))
            >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
        "physical_traffic": traffic["fraction_of_dense_q4"] <= 0.45,
        "all_layers_grouped_ternary": len(wrappers)
        == inspection.num_hidden_layers,
        "hard_validation": all(
            wrapper.quantization_strength == 1.0 for wrapper in wrappers
        ),
        "serialized_reload": artifact["reloaded_before_validation"],
    }
    report = {
        "schema_version": 1,
        "experiment": "budget_native_grouped_ternary_qad",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "group_size": group_size,
            "steps": steps,
            "dense_warmup_steps": dense_warmup_steps,
            "anneal_steps": anneal_steps,
            "transition_mode": transition_mode,
            "schedule": (
                "dense then deepest-layer-first staggered half-cosine "
                "transition then hard ternary"
                if transition_mode == "deepest_first"
                else "dense then global half-cosine transition then hard ternary"
            ),
            "optimizer_state_retained_across_transition": True,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "backbone_learning_rate": backbone_learning_rate,
            "coadapt_backbone": coadapt_backbone,
            "coadapt_embeddings_and_head": coadapt_embeddings_and_head,
            "backbone_start_step": backbone_start_step,
            "backbone_trainable_parameters": int(
                sum(parameter.numel() for parameter in backbone_named.values())
            ),
            "incremental_mlp_traffic_from_backbone": 0,
            "device": device,
            "device_neutral_checkpoint": True,
            "loss_weights": {
                "local": local_weight,
                "hidden": hidden_weight,
                "final_hidden": final_hidden_weight,
                "final_cka": final_cka_weight,
                "logit": logit_weight,
                "teacher_top1": teacher_top1_weight,
                "label": label_weight,
            },
            "temperature": temperature,
            "confidence_weight": confidence_weight,
        },
        "training": {
            "records": len(train_records),
            "record_offset": training_record_offset,
            "completed_before_resume": completed_steps,
            "initialization": initialization,
            "input_token_positions_this_invocation": training_input_positions,
            "next_token_positions_this_invocation": training_next_positions,
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
            "records": len(validation_records),
            "input_token_positions": input_positions,
            "next_token_positions": next_positions,
            "quantization_strength": 1.0,
            "mlp_weight_source": (
                "serialized_reloaded_grouped_ternary_decode"
                if save_artifact
                else "in_memory_hard_grouped_ternary"
            ),
        },
        "data_separation": {
            "method": "exact_token_sequence_hashes",
            "overlapping_sequences": 0,
            "training_dataset_hash": sha256_file(training_path),
            "validation_dataset_hash": sha256_file(validation_path),
        },
        "metrics": {
            **{name: _stats(rows) for name, rows in metrics.items()},
            "local_mlp_relative_l2": _stats(local_error),
        },
        "zero_fraction": zero_fractions,
        "physical_traffic": traffic,
        "artifact": artifact,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_confirmation"
                if all(checks.values())
                else "continue_only_under_predeclared_training_rungs_or_stop"
            ),
        },
    }
    atomic_json(target / "budget_native_ternary_training.json", report)
    lines = [
        "# Budget-native grouped-ternary distillation",
        "",
        f"Decision: **{report['gate']['decision']}**",
        "",
        f"Complete MLP cold traffic: {traffic['fraction_of_dense_q4']:.6%} "
        "of dense ideal Q4.",
        "",
        "| Metric | Mean | Threshold |",
        "|---|---:|---:|",
        f"| Teacher-student KL | {means['teacher_student_kl']:.6f} | <=0.05 |",
        f"| Teacher top-1 agreement | {means['teacher_top1_agreement']:.6f} | >=0.90 |",
        f"| NLL delta | {means['nll_delta']:.6f} | <=+0.05 |",
        f"| Final hidden relative L2 | {means['final_hidden_relative_l2']:.6f} | <=0.10 |",
        f"| Local MLP relative L2 | {float(np.mean(local_error)):.6f} | diagnostic |",
        "",
        "Validation used hard grouped-ternary MLPs"
        + (
            " decoded from the independently reloaded artifact."
            if save_artifact
            else " from the in-memory training quantizer."
        ),
        "",
    ]
    (target / "budget_native_ternary_training.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    return report


__all__ = ["train_budget_native_ternary_student"]
