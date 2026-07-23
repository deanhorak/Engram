"""Structured fixed-width SwiGLU pruning and progressive distillation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
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
from engram.evaluation.router_sweep import _load_states
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.semantic.swiglu import silu
from engram.tracing.format import TraceReader
from engram.training.on_policy import _import_causal_lm
from engram.training.sparse_teacher import (
    _batch_ids,
    _batches,
    _load_jsonl,
    _normalized_masked_mse,
)
from engram.training.structured_experts import _load_trace_field, _stats
from engram.training.width_pruned_codec import (
    WidthPrunedQ4LayerWeights,
    decode_width_pruned_q4_artifact,
    load_width_pruned_q4_artifact,
    save_width_pruned_q4_artifact,
    width_pruned_q4_traffic,
)
from engram.utils import atomic_json, sha256_file


@dataclass(frozen=True)
class WidthPrunedTraffic:
    hidden_size: int
    source_intermediate_size: int
    target_intermediate_size: int
    weight_bytes: int
    dense_weight_bytes: int
    fraction_of_dense: float

    def to_dict(self) -> dict[str, int | float]:
        return dict(self.__dict__)


def width_pruned_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    target_intermediate_size: int,
    *,
    bytes_per_parameter: int = 4,
) -> WidthPrunedTraffic:
    values = (hidden_size, source_intermediate_size, target_intermediate_size)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("widths must be positive integers")
    if target_intermediate_size > source_intermediate_size:
        raise ValueError("target width cannot exceed source width")
    if not isinstance(bytes_per_parameter, int) or bytes_per_parameter <= 0:
        raise ValueError("bytes_per_parameter must be a positive integer")
    weight_bytes = 3 * hidden_size * target_intermediate_size * bytes_per_parameter
    dense_bytes = 3 * hidden_size * source_intermediate_size * bytes_per_parameter
    return WidthPrunedTraffic(
        hidden_size=hidden_size,
        source_intermediate_size=source_intermediate_size,
        target_intermediate_size=target_intermediate_size,
        weight_bytes=weight_bytes,
        dense_weight_bytes=dense_bytes,
        fraction_of_dense=weight_bytes / dense_bytes,
    )


def width_pruned_schedule_traffic(
    hidden_size: int,
    source_intermediate_size: int,
    layer_widths: Sequence[int],
    *,
    bytes_per_parameter: int = 4,
) -> dict[str, Any]:
    """Account compact MLP weights for an explicit per-layer width schedule."""

    widths = tuple(layer_widths)
    if not widths:
        raise ValueError("layer_widths must not be empty")
    layers = [
        width_pruned_traffic(
            hidden_size,
            source_intermediate_size,
            width,
            bytes_per_parameter=bytes_per_parameter,
        )
        for width in widths
    ]
    weight_bytes = sum(layer.weight_bytes for layer in layers)
    dense_weight_bytes = sum(layer.dense_weight_bytes for layer in layers)
    return {
        "hidden_size": hidden_size,
        "source_intermediate_size": source_intermediate_size,
        "target_intermediate_size": (widths[0] if len(set(widths)) == 1 else None),
        "target_intermediate_sizes": list(widths),
        "layer_widths": list(widths),
        "layer_count": len(widths),
        "bytes_per_parameter": bytes_per_parameter,
        "layers": [
            {
                "layer": layer,
                "target_intermediate_size": width,
                "weight_bytes": traffic.weight_bytes,
                "dense_weight_bytes": traffic.dense_weight_bytes,
                "fraction_of_dense": traffic.fraction_of_dense,
            }
            for layer, (width, traffic) in enumerate(zip(widths, layers, strict=True))
        ],
        "weight_bytes": weight_bytes,
        "dense_weight_bytes": dense_weight_bytes,
        "fraction_of_dense": weight_bytes / dense_weight_bytes,
    }


def _resolve_target_widths(
    target_width: int,
    target_widths: Sequence[int] | None,
    *,
    layer_count: int,
    source_intermediate_size: int,
) -> tuple[int, ...]:
    if target_widths is None:
        widths = (target_width,) * layer_count
    else:
        widths = tuple(target_widths)
        if len(widths) != layer_count:
            raise ValueError(
                "target_widths must contain exactly one width per transformer layer"
            )
    # Centralize validation with the public traffic-accounting implementation.
    width_pruned_schedule_traffic(
        1,
        source_intermediate_size,
        widths,
    )
    return widths


def select_width_channels(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    target_width: int,
    *,
    states: np.ndarray | None = None,
) -> np.ndarray:
    """Select a deterministic initialization subset for a compact MLP."""

    gate_matrix = np.asarray(gate, dtype=np.float64)
    up_matrix = np.asarray(up, dtype=np.float64)
    down_matrix = np.asarray(down, dtype=np.float64)
    if gate_matrix.ndim != 2 or up_matrix.shape != gate_matrix.shape:
        raise ValueError("gate and up must have matching rank-2 shapes")
    if down_matrix.shape != (gate_matrix.shape[1], gate_matrix.shape[0]):
        raise ValueError("down has an incompatible shape")
    if not isinstance(target_width, int) or not 0 < target_width <= len(gate_matrix):
        raise ValueError("target_width must lie within the source width")
    value_norms = np.linalg.norm(down_matrix, axis=0)
    if states is None:
        scores = (
            np.linalg.norm(gate_matrix, axis=1)
            * np.linalg.norm(up_matrix, axis=1)
            * value_norms
        )
    else:
        state_matrix = np.asarray(states, dtype=np.float64)
        if state_matrix.ndim != 2 or state_matrix.shape[1] != gate_matrix.shape[1]:
            raise ValueError("states have an incompatible shape")
        activations = silu(state_matrix @ gate_matrix.T) * (state_matrix @ up_matrix.T)
        scores = np.mean(np.abs(activations), axis=0) * value_norms
    return np.argsort(-scores, kind="stable")[:target_width]


def _fake_signed_q4_rows(weight: Any, torch: Any) -> Any:
    """Apply the deployment signed-Q4 row codec with an identity STE."""

    if weight.ndim != 2 or not weight.shape[0] or not weight.shape[1]:
        raise ValueError("Q4 weight must be a non-empty rank-2 tensor")
    with torch.no_grad():
        maximum = weight.detach().abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            maximum > 0,
            maximum / 7.0,
            torch.ones_like(maximum),
        )
        scale = scale.to(torch.float16).to(weight.dtype)
        quantized = torch.clamp(torch.round(weight.detach() / scale), -7, 7) * scale
    return weight + (quantized - weight).detach()


def _wrap_width_pruned_mlp_class(torch: Any):
    class WidthPrunedMLP(torch.nn.Module):
        def __init__(
            self,
            base: Any,
            selected: Any,
            *,
            fake_q4_training: bool = False,
        ):
            super().__init__()
            if not isinstance(fake_q4_training, bool):
                raise ValueError("fake_q4_training must be a boolean")
            if (
                base.gate_proj.bias is not None
                or base.up_proj.bias is not None
                or base.down_proj.bias is not None
            ):
                raise ValueError("bias-enabled MLP projections are not supported")
            indices = torch.as_tensor(
                selected, dtype=torch.long, device=base.gate_proj.weight.device
            )
            if indices.ndim != 1 or not len(indices):
                raise ValueError("selected must be a nonempty rank-1 index tensor")
            if torch.unique(indices).numel() != len(indices):
                raise ValueError("selected channels must be unique")
            intermediate, _ = base.gate_proj.weight.shape
            if int(indices.min()) < 0 or int(indices.max()) >= intermediate:
                raise ValueError("selected channel is outside the source width")
            self.register_buffer(
                "dense_gate_weight", base.gate_proj.weight.detach().clone()
            )
            self.register_buffer(
                "dense_up_weight", base.up_proj.weight.detach().clone()
            )
            self.register_buffer(
                "dense_down_weight", base.down_proj.weight.detach().clone()
            )
            self.gate_weight = torch.nn.Parameter(
                base.gate_proj.weight.detach()[indices].clone()
            )
            self.up_weight = torch.nn.Parameter(
                base.up_proj.weight.detach()[indices].clone()
            )
            self.down_weight = torch.nn.Parameter(
                base.down_proj.weight.detach()[:, indices].clone()
            )
            self.register_buffer("source_indices", indices.clone())
            self.act_fn = base.act_fn
            self.mode = "dense"
            self.fake_q4_training = fake_q4_training
            self.last_output = None

        @property
        def target_width(self) -> int:
            return int(self.gate_weight.shape[0])

        def _output(self, hidden: Any, gate: Any, up: Any, down: Any) -> Any:
            gate_values = torch.nn.functional.linear(hidden, gate)
            up_values = torch.nn.functional.linear(hidden, up)
            return torch.nn.functional.linear(
                self.act_fn(gate_values) * up_values, down
            )

        def forward(self, hidden: Any) -> Any:
            if self.mode == "dense":
                output = self._output(
                    hidden,
                    self.dense_gate_weight,
                    self.dense_up_weight,
                    self.dense_down_weight,
                )
            elif self.mode == "compact":
                gate = self.gate_weight
                up = self.up_weight
                down = self.down_weight
                if self.fake_q4_training:
                    gate = _fake_signed_q4_rows(gate, torch)
                    up = _fake_signed_q4_rows(up, torch)
                    # The artifact stores down.T rows and transposes them back
                    # after decode.  Fake quantization must use that orientation.
                    down = _fake_signed_q4_rows(down.T, torch).T
                output = self._output(
                    hidden,
                    gate,
                    up,
                    down,
                )
            else:
                raise ValueError(f"unsupported width-pruned mode {self.mode!r}")
            self.last_output = output
            return output

    return WidthPrunedMLP


def _mean_metric(values: dict[str, list[float]]) -> dict[str, float]:
    return {name: float(np.mean(rows)) for name, rows in values.items()}


def train_width_pruned_student(
    model: str | Path,
    training_dataset: str | Path,
    validation_dataset: str | Path,
    out: str | Path,
    *,
    calibration_traces: str | Path | None = None,
    target_width: int = 672,
    target_widths: Sequence[int] | None = None,
    steps: int = 8,
    replacement_steps: int = 0,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    local_weight: float = 1.0,
    hidden_weight: float = 0.25,
    logit_weight: float = 0.25,
    initialization_records: int = 512,
    local_warmup_steps: int = 0,
    local_batch_size: int = 32,
    local_learning_rate: float = 3e-4,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
    device: str = "cpu",
    save_artifact: bool = True,
    strict_q4_deployment: bool = False,
    fake_q4_training: bool = False,
    checkpoint_every: int = 0,
    resume: bool = False,
    initial_checkpoint: str | Path | None = None,
    coadapt_backbone: bool = False,
    coadapt_embeddings_and_head: bool = False,
    backbone_learning_rate: float = 3e-5,
) -> dict[str, Any]:
    """Progressively replace dense MLPs with compact contiguous-width students.

    When ``coadapt_backbone`` is enabled, the existing attention and normalization
    weights are optimized jointly with the compact MLPs.  The embeddings and
    output head can be included explicitly with ``coadapt_embeddings_and_head``.
    Those weights replace already-resident model weights at deployment and
    therefore add no incremental MLP cold-read traffic.  ``target_widths`` selects
    one width per transformer layer and takes precedence over the backward-
    compatible scalar ``target_width``.
    """

    try:
        import torch
        from safetensors.torch import save_file
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for width distillation"
        ) from exc
    AutoModelForCausalLM, AutoTokenizer = _import_causal_lm()
    if any(
        not isinstance(value, int) or value < 0
        for value in (steps, replacement_steps, local_warmup_steps, checkpoint_every)
    ):
        raise ValueError("step counts must be nonnegative integers")
    if any(
        not isinstance(value, int) or value <= 0
        for value in (batch_size, local_batch_size)
    ):
        raise ValueError("batch sizes must be positive integers")
    if not isinstance(initialization_records, int) or initialization_records <= 0:
        raise ValueError("initialization_records must be a positive integer")
    scalars = (
        learning_rate,
        backbone_learning_rate,
        local_learning_rate,
        local_weight,
        hidden_weight,
        logit_weight,
    )
    if any(not np.isfinite(value) for value in scalars):
        raise ValueError("learning rate and loss weights must be finite")
    if (
        learning_rate <= 0
        or backbone_learning_rate <= 0
        or local_learning_rate <= 0
        or any(value < 0 for value in scalars[3:])
    ):
        raise ValueError("learning rates must be positive and loss weights nonnegative")
    if local_warmup_steps and calibration_traces is None:
        raise ValueError("local warm-up requires calibration traces")
    if resume and initial_checkpoint is not None:
        raise ValueError("resume and initial_checkpoint are mutually exclusive")
    if strict_q4_deployment and not save_artifact:
        raise ValueError("strict Q4 deployment validation requires artifact saving")
    if not isinstance(fake_q4_training, bool):
        raise ValueError("fake_q4_training must be a boolean")
    if fake_q4_training and not strict_q4_deployment:
        raise ValueError(
            "fake Q4 training requires strict serialized Q4 deployment validation"
        )
    effective_replacement_steps = replacement_steps
    if steps:
        effective_replacement_steps = replacement_steps or max(1, steps // 2)
        if effective_replacement_steps > steps:
            raise ValueError("replacement_steps cannot exceed steps")
    elif replacement_steps:
        raise ValueError("replacement_steps requires nonzero training steps")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    layer_widths = _resolve_target_widths(
        target_width,
        target_widths,
        layer_count=inspection.num_hidden_layers,
        source_intermediate_size=inspection.intermediate_size,
    )
    reported_target_width = layer_widths[0] if len(set(layer_widths)) == 1 else None
    traffic = width_pruned_schedule_traffic(
        inspection.hidden_size,
        inspection.intermediate_size,
        layer_widths,
    )
    training_path = Path(training_dataset)
    validation_path = Path(validation_dataset)
    train_records = _load_jsonl(training_path, max_train_records)
    validation_records = _load_jsonl(validation_path, max_validation_records)
    trace_reader = TraceReader(calibration_traces) if calibration_traces else None
    if trace_reader is not None:
        if trace_reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("calibration trace/model hash mismatch")
        if trace_reader.manifest["split"] != "calibration":
            raise ValueError("width initialization requires calibration traces")

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
    wrapper_type = _wrap_width_pruned_mlp_class(torch)
    wrappers = []
    initialization = []
    for layer, decoder in enumerate(student.model.layers):
        gate, up, down = load_layer_mlp(model_path, layer)
        states = (
            _load_states(trace_reader, layer, initialization_records)
            if trace_reader is not None
            else None
        )
        selected = select_width_channels(
            gate,
            up,
            down,
            layer_widths[layer],
            states=states,
        )
        wrapper = wrapper_type(
            decoder.mlp,
            selected,
            fake_q4_training=fake_q4_training,
        ).to(device)
        decoder.mlp = wrapper
        wrappers.append(wrapper)
        initialization.append(
            {
                "layer": layer,
                "method": (
                    "mean_teacher_contribution"
                    if states is not None
                    else "weight_geometry"
                ),
                "records": len(states) if states is not None else 0,
                "target_width": layer_widths[layer],
            }
        )
    width_named = {
        f"layers.{layer}.{name}": parameter
        for layer, wrapper in enumerate(wrappers)
        for name, parameter in wrapper.named_parameters()
        if parameter.requires_grad
    }
    backbone_named: dict[str, Any] = {}
    if coadapt_embeddings_and_head and not coadapt_backbone:
        raise ValueError("embedding/head co-adaptation requires backbone co-adaptation")
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
    trainable_named = {**width_named, **backbone_named}
    checkpoint_lineage = None
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint)
        if not initial_path.is_file():
            raise ValueError("initial width-pruned checkpoint is missing")
        initial = torch.load(initial_path, map_location=device, weights_only=False)
        initial_parameters = initial.get("parameters", {})
        missing_width = set(width_named).difference(initial_parameters)
        unexpected = set(initial_parameters).difference(trainable_named)
        if missing_width or unexpected:
            raise ValueError("initial width-pruned checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, source in initial_parameters.items():
                parameter = trainable_named[name]
                if source.shape != parameter.shape:
                    raise ValueError(
                        f"initial width-pruned checkpoint shape mismatch for {name}"
                    )
                parameter.copy_(source.to(device))
        initial_manifest_path = initial_path.with_suffix(".json")
        if initial_manifest_path.is_file():
            initial_manifest = json.loads(
                initial_manifest_path.read_text(encoding="utf-8")
            )
            initial_configuration = initial_manifest.get("configuration", {})
            initial_widths = initial_configuration.get("target_widths")
            if initial_widths is None:
                initial_width = initial_configuration.get("target_width")
                initial_widths = [initial_width] * inspection.num_hidden_layers
            if (
                initial_configuration.get("source_model_hash") != inspection.source_hash
                or tuple(initial_widths) != layer_widths
            ):
                raise ValueError("initial width-pruned checkpoint model/width mismatch")
        checkpoint_lineage = {
            "mode": "parameters_only",
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
            "optimizer_restored": False,
            "history_restored": False,
            "backbone_parameters_restored": bool(
                set(backbone_named).intersection(initial_parameters)
            ),
        }
    trainable = list(trainable_named.values())
    parameter_groups = [{"params": list(width_named.values()), "lr": learning_rate}]
    if backbone_named:
        parameter_groups.append(
            {"params": list(backbone_named.values()), "lr": backbone_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.0)

    local_warmup = []
    if local_warmup_steps and not resume:
        for layer, wrapper in enumerate(wrappers):
            states = torch.from_numpy(
                _load_states(trace_reader, layer, initialization_records).astype(
                    np.float32
                )
            ).to(device)
            targets = torch.from_numpy(
                _load_trace_field(
                    trace_reader,
                    f"layer_{layer}_mlp_output",
                    initialization_records,
                ).astype(np.float32)
            ).to(device)
            wrapper.mode = "compact"
            wrapper.train()
            layer_optimizer = torch.optim.AdamW(
                wrapper.parameters(), lr=local_learning_rate, weight_decay=0.0
            )
            with torch.no_grad():
                before = wrapper(states)
                before_error, _ = _relative_and_cosine_rows(
                    before.cpu().numpy(), targets.cpu().numpy()
                )
            for local_step in range(local_warmup_steps):
                start = (local_step * local_batch_size) % len(states)
                indices = torch.arange(
                    start,
                    start + local_batch_size,
                    device=states.device,
                ) % len(states)
                output = wrapper(states[indices])
                target_rows = targets[indices]
                loss = torch.mean((output - target_rows) ** 2) / torch.clamp(
                    torch.mean(target_rows**2), min=1e-8
                )
                layer_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 1.0)
                layer_optimizer.step()
            wrapper.eval()
            with torch.no_grad():
                after = wrapper(states)
                after_error, _ = _relative_and_cosine_rows(
                    after.cpu().numpy(), targets.cpu().numpy()
                )
            wrapper.mode = "dense"
            local_warmup.append(
                {
                    "layer": layer,
                    "steps": local_warmup_steps,
                    "relative_l2_before": float(np.mean(before_error)),
                    "relative_l2_after": float(np.mean(after_error)),
                }
            )

    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    checkpoint_path = target / "width_pruned_checkpoint.pt"
    checkpoint_manifest_path = target / "width_pruned_checkpoint.json"
    checkpoint_configuration = {
        "schema_version": 1,
        "source_model_hash": inspection.source_hash,
        "training_dataset_hash": sha256_file(training_path),
        "validation_dataset_hash": sha256_file(validation_path),
        "target_width": reported_target_width,
        "target_widths": list(layer_widths),
        "replacement_steps": effective_replacement_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "coadapt_backbone": coadapt_backbone,
        "coadapt_embeddings_and_head": coadapt_embeddings_and_head,
        "backbone_learning_rate": backbone_learning_rate,
        "fake_q4_training": fake_q4_training,
        "local_weight": local_weight,
        "hidden_weight": hidden_weight,
        "logit_weight": logit_weight,
    }
    history = []
    completed_steps = 0
    if resume:
        if not checkpoint_path.is_file() or not checkpoint_manifest_path.is_file():
            raise ValueError("resume requested but width-pruned checkpoint is missing")
        manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
        manifest_configuration = dict(manifest["configuration"])
        # This flag was added after the first attention-only checkpoints.  A
        # missing value is exactly equivalent to the disabled mode and leaves
        # both the parameter set and optimizer state unchanged.
        manifest_configuration.setdefault("coadapt_embeddings_and_head", False)
        manifest_configuration.setdefault("fake_q4_training", False)
        if "target_widths" not in manifest_configuration:
            manifest_width = manifest_configuration.get("target_width")
            manifest_configuration["target_widths"] = [
                manifest_width
            ] * inspection.num_hidden_layers
        if manifest_configuration != checkpoint_configuration:
            raise ValueError("width-pruned checkpoint configuration mismatch")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if set(checkpoint["parameters"]) != set(trainable_named):
            raise ValueError("width-pruned checkpoint parameter set mismatch")
        with torch.no_grad():
            for name, parameter in trainable_named.items():
                parameter.copy_(checkpoint["parameters"][name].to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        completed_steps = len(history)
        checkpoint_lineage = manifest.get("lineage")
        if completed_steps > steps:
            raise ValueError("checkpoint has more steps than the requested total")

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
                "lineage": checkpoint_lineage,
            },
        )

    teacher_targets: list[Any] = [None] * inspection.num_hidden_layers
    handles = []
    for layer, decoder in enumerate(teacher.model.layers):

        def capture(
            _module: Any, _inputs: tuple[Any, ...], output: Any, *, index=layer
        ):
            teacher_targets[index] = output.detach()

        handles.append(decoder.mlp.register_forward_hook(capture))
    batches = _batches(train_records, batch_size)
    replacement_order = list(reversed(range(inspection.num_hidden_layers)))
    try:
        for step in range(completed_steps, steps):
            converted_count = min(
                inspection.num_hidden_layers,
                max(
                    1,
                    int(
                        np.ceil(
                            (step + 1)
                            * inspection.num_hidden_layers
                            / effective_replacement_steps
                        )
                    ),
                ),
            )
            converted = set(replacement_order[:converted_count])
            for layer, wrapper in enumerate(wrappers):
                wrapper.mode = "compact" if layer in converted else "dense"
                wrapper.train(layer in converted)
            records = batches[step % len(batches)]
            input_ids, attention_mask, _ = _batch_ids(records, tokenizer, torch, device)
            valid_mask = attention_mask.bool()
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
                        wrappers[layer].last_output,
                        teacher_targets[layer],
                        valid_mask,
                        torch,
                    )
                    for layer in sorted(converted)
                ]
            ).mean()
            hidden_loss = torch.stack(
                [
                    _normalized_masked_mse(
                        student_hidden, teacher_hidden, valid_mask, torch
                    )
                    for student_hidden, teacher_hidden in zip(
                        student_output.hidden_states[1:],
                        teacher_output.hidden_states[1:],
                        strict=True,
                    )
                ]
            ).mean()
            teacher_logp = torch.nn.functional.log_softmax(
                teacher_output.logits.detach(), dim=-1
            )
            student_logp = torch.nn.functional.log_softmax(
                student_output.logits, dim=-1
            )
            logit_rows = torch.nn.functional.kl_div(
                student_logp, teacher_logp.exp(), reduction="none"
            ).sum(dim=-1)
            logit_loss = logit_rows[valid_mask].mean()
            loss = (
                local_weight * local_loss
                + hidden_weight * hidden_loss
                + logit_weight * logit_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            history.append(
                {
                    "step": step + 1,
                    "converted_layers": converted_count,
                    "total": float(loss.detach()),
                    "local": float(local_loss.detach()),
                    "hidden": float(hidden_loss.detach()),
                    "logit": float(logit_loss.detach()),
                }
            )
            if checkpoint_every and (step + 1) % checkpoint_every == 0:
                save_checkpoint()

        if checkpoint_every and len(history) != completed_steps:
            save_checkpoint()

        artifact_format = (
            "engram_width_pruned_coadapted_v2"
            if coadapt_backbone
            else "engram_width_pruned_mlp_v1"
        )
        artifact: dict[str, Any] = {
            "written": False,
            "reloaded_before_validation": False,
            "format": artifact_format,
            "strict_q4_deployment": strict_q4_deployment,
            "fake_q4_training": fake_q4_training,
            "q4": None,
            "backbone": None,
        }
        strict_q4_traffic: dict[str, Any] | None = None
        if save_artifact:
            from safetensors.torch import load_file

            if strict_q4_deployment:
                q4_path = target / "width_pruned_mlp.q4.bin"
                q4_layers = [
                    WidthPrunedQ4LayerWeights(
                        wrapper.gate_weight,
                        wrapper.up_weight,
                        wrapper.down_weight,
                        wrapper.source_indices,
                    )
                    for wrapper in wrappers
                ]
                save_width_pruned_q4_artifact(
                    q4_path,
                    q4_layers,
                    source_intermediate_size=inspection.intermediate_size,
                )
                loaded_q4 = load_width_pruned_q4_artifact(q4_path)
                decoded_q4 = decode_width_pruned_q4_artifact(loaded_q4)
                if len(decoded_q4) != len(wrappers):
                    raise RuntimeError("width-pruned Q4 layer count changed on reload")
                with torch.no_grad():
                    for layer, (wrapper, decoded) in enumerate(
                        zip(wrappers, decoded_q4, strict=True)
                    ):
                        values = {
                            "gate_weight": decoded["gate"],
                            "up_weight": decoded["up"],
                            "down_weight": decoded["down"],
                            "source_indices": decoded["source_ids"],
                        }
                        for name, value in values.items():
                            destination = getattr(wrapper, name)
                            source = torch.from_numpy(value).to(
                                device=destination.device,
                                dtype=destination.dtype,
                            )
                            if source.shape != destination.shape:
                                raise RuntimeError(
                                    "width-pruned Q4 tensor shape changed on reload "
                                    f"at layer {layer} for {name}"
                                )
                            destination.copy_(source)
                strict_q4_traffic = width_pruned_q4_traffic(
                    inspection.hidden_size,
                    inspection.intermediate_size,
                    [wrapper.target_width for wrapper in wrappers],
                )
                if q4_path.stat().st_size != strict_q4_traffic["total_cold_bytes"]:
                    raise RuntimeError(
                        "width-pruned Q4 file size differs from strict traffic accounting"
                    )
                q4_artifact = {
                    "written": True,
                    "reloaded_before_validation": True,
                    "format": "engram_width_pruned_cache_aligned_q4_v1",
                    "path": str(q4_path.resolve()),
                    "sha256": sha256_file(q4_path),
                    "layers": len(decoded_q4),
                    "traffic": strict_q4_traffic,
                }
                backbone_artifact: dict[str, Any] = {
                    "written": False,
                    "reloaded_before_validation": True,
                    "required": bool(backbone_named),
                    "format": "engram_width_pruned_backbone_v1",
                    "source": "unchanged_source_model",
                }
                if backbone_named:
                    backbone_tensors = {
                        name: parameter.detach().cpu().contiguous()
                        for name, parameter in backbone_named.items()
                    }
                    backbone_path = target / "width_pruned_backbone.safetensors"
                    save_file(
                        backbone_tensors,
                        backbone_path,
                        metadata={
                            "format": backbone_artifact["format"],
                            "source_model_hash": inspection.source_hash,
                            "coadapt_backbone": "true",
                            "coadapt_embeddings_and_head": str(
                                coadapt_embeddings_and_head
                            ).lower(),
                        },
                    )
                    reloaded_backbone = load_file(backbone_path, device=device)
                    if set(reloaded_backbone) != set(backbone_named):
                        raise RuntimeError(
                            "width-pruned backbone tensor set changed on reload"
                        )
                    with torch.no_grad():
                        for name, parameter in backbone_named.items():
                            value = reloaded_backbone[name]
                            if (
                                value.shape != parameter.shape
                                or value.dtype != parameter.dtype
                            ):
                                raise RuntimeError(
                                    "width-pruned backbone artifact mismatch "
                                    f"for {name}"
                                )
                            parameter.copy_(value)
                    backbone_artifact.update(
                        written=True,
                        path=str(backbone_path.resolve()),
                        sha256=sha256_file(backbone_path),
                        tensors=len(backbone_tensors),
                        source="serialized_safetensors_reload",
                    )
                artifact.update(
                    written=True,
                    reloaded_before_validation=True,
                    format="engram_width_pruned_strict_q4_v1",
                    path=str(q4_path.resolve()),
                    sha256=sha256_file(q4_path),
                    tensors=len(wrappers) * 4 + len(backbone_named),
                    q4=q4_artifact,
                    backbone=backbone_artifact,
                )
            else:
                tensor_references = {
                    f"layers.{layer}.{name}": getattr(wrapper, name)
                    for layer, wrapper in enumerate(wrappers)
                    for name in (
                        "gate_weight",
                        "up_weight",
                        "down_weight",
                        "source_indices",
                    )
                }
                tensor_references.update(backbone_named)
                tensors = {
                    name: parameter.detach().cpu().contiguous()
                    for name, parameter in tensor_references.items()
                }
                tensor_path = target / "width_pruned_student.safetensors"
                save_file(
                    tensors,
                    tensor_path,
                    metadata={
                        "format": artifact["format"],
                        "source_model_hash": inspection.source_hash,
                        "target_intermediate_size": (
                            str(reported_target_width)
                            if reported_target_width is not None
                            else "variable"
                        ),
                        "target_intermediate_sizes": json.dumps(
                            layer_widths, separators=(",", ":")
                        ),
                        "coadapt_backbone": str(coadapt_backbone).lower(),
                        "coadapt_embeddings_and_head": str(
                            coadapt_embeddings_and_head
                        ).lower(),
                    },
                )
                reloaded = load_file(tensor_path, device=device)
                if set(reloaded) != set(tensor_references):
                    raise RuntimeError(
                        "width-pruned artifact tensor set changed on reload"
                    )
                with torch.no_grad():
                    for name, parameter in tensor_references.items():
                        value = reloaded[name]
                        if (
                            value.shape != parameter.shape
                            or value.dtype != parameter.dtype
                        ):
                            raise RuntimeError(
                                f"width-pruned artifact tensor mismatch for {name}"
                            )
                        parameter.copy_(value)
                artifact.update(
                    written=True,
                    reloaded_before_validation=True,
                    path=str(tensor_path.resolve()),
                    sha256=sha256_file(tensor_path),
                    tensors=len(tensors),
                )

        for wrapper in wrappers:
            # Strict causal validation must execute only the decoded bytes that
            # were reloaded above, never a second in-memory fake quantization.
            wrapper.fake_q4_training = False
            wrapper.mode = "compact"
            wrapper.eval()
        metrics: dict[str, list[float]] = {}
        local_error: list[float] = []
        input_positions = 0
        next_positions = 0
        for records in _batches(validation_records, batch_size):
            input_ids, attention_mask, lengths = _batch_ids(
                records, tokenizer, torch, device
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
                    teacher_output.hidden_states[-1][row : row + 1, :length],
                    student_output.hidden_states[-1][row : row + 1, :length],
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
    deployment_traffic_fraction = (
        float(strict_q4_traffic["fraction_of_dense_q4"])
        if strict_q4_traffic is not None
        else float(traffic["fraction_of_dense"])
    )
    checks = {
        "teacher_student_kl": means["teacher_student_kl"]
        <= MLP_QUALITY_THRESHOLDS["maximum_teacher_student_kl"],
        "teacher_top1_agreement": means["teacher_top1_agreement"]
        >= MLP_QUALITY_THRESHOLDS["minimum_teacher_top1_agreement"],
        "nll_delta": means["nll_delta"] <= MLP_QUALITY_THRESHOLDS["maximum_nll_delta"],
        "final_hidden_relative_l2": means["final_hidden_relative_l2"]
        <= MLP_QUALITY_THRESHOLDS["maximum_final_hidden_relative_l2"],
        "evidence_size": (
            len(validation_records) >= MINIMUM_EVALUATION_SEQUENCES
            and len(set(validation_hashes)) >= MINIMUM_UNIQUE_EVALUATION_SEQUENCES
            and next_positions >= MINIMUM_NEXT_TOKEN_POSITIONS
        ),
        "projected_traffic": deployment_traffic_fraction <= 0.45,
        "all_layers_compact": all(wrapper.mode == "compact" for wrapper in wrappers),
        "serialized_reload": artifact["reloaded_before_validation"],
        "q4_serialized_reload": (
            not strict_q4_deployment
            or bool(artifact["q4"] and artifact["q4"]["reloaded_before_validation"])
        ),
    }
    report = {
        "schema_version": 1,
        "experiment": "progressive_width_pruned_distillation",
        "source_model_hash": inspection.source_hash,
        "configuration": {
            "target_width": reported_target_width,
            "target_widths": list(layer_widths),
            "steps": steps,
            "replacement_steps": effective_replacement_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "coadapt_backbone": coadapt_backbone,
            "coadapt_embeddings_and_head": coadapt_embeddings_and_head,
            "backbone_learning_rate": backbone_learning_rate,
            "backbone_trainable_parameters": int(
                sum(parameter.numel() for parameter in backbone_named.values())
            ),
            "incremental_mlp_traffic_from_backbone": 0,
            "replacement_order": "deepest_first",
            "device": device,
            "device_neutral_artifact": True,
            "strict_q4_deployment": strict_q4_deployment,
            "fake_q4_training": fake_q4_training,
            "checkpoint_every": checkpoint_every,
            "resumed": resume,
            "initialized_from_checkpoint": checkpoint_lineage is not None,
            "loss_weights": {
                "local": local_weight,
                "hidden": hidden_weight,
                "logit": logit_weight,
            },
            "local_warmup_steps": local_warmup_steps,
            "local_batch_size": local_batch_size,
            "local_learning_rate": local_learning_rate,
        },
        "initialization": initialization,
        "local_warmup": local_warmup,
        "training": {
            "records": len(train_records),
            "completed_before_resume": completed_steps,
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
            "initial_checkpoint": checkpoint_lineage,
        },
        "validation": {
            "records": len(validation_records),
            "input_token_positions": input_positions,
            "next_token_positions": next_positions,
            "all_layers_compact": True,
            "mlp_weight_source": (
                "serialized_reloaded_q4_decode"
                if strict_q4_deployment
                else "float_compact_parameters"
            ),
            "fake_q4_disabled_after_serialized_reload": (
                fake_q4_training and strict_q4_deployment
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
        "projected_traffic": traffic,
        "strict_q4_traffic": strict_q4_traffic,
        "deployment_traffic_fraction": deployment_traffic_fraction,
        "gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "decision": (
                "eligible_for_confirmation"
                if all(checks.values())
                else "continue_width_distillation_or_stop"
            ),
        },
        "artifact": artifact,
    }
    atomic_json(target / "width_pruned_training.json", report)
    width_summary = (
        f"Target width: {reported_target_width}/{inspection.intermediate_size}"
        if reported_target_width is not None
        else (
            f"Target widths by layer: {list(layer_widths)}; mean "
            f"{float(np.mean(layer_widths)):.2f}/{inspection.intermediate_size}"
        )
    )
    lines = [
        "# Progressive fixed-width MLP distillation",
        "",
        f"Decision: **{report['gate']['decision']}**",
        "",
        f"{width_summary}; deployment MLP traffic: "
        f"{deployment_traffic_fraction:.6f}× dense.",
        "",
        "| Metric | Mean | Threshold |",
        "|---|---:|---:|",
        f"| Teacher-student KL | {means['teacher_student_kl']:.6f} | ≤0.05 |",
        f"| Teacher top-1 agreement | {means['teacher_top1_agreement']:.6f} | ≥0.90 |",
        f"| NLL delta | {means['nll_delta']:.6f} | ≤0.05 |",
        f"| Final hidden relative L2 | {means['final_hidden_relative_l2']:.6f} | ≤0.10 |",
        f"| Local MLP relative L2 | {float(np.mean(local_error)):.6f} | diagnostic |",
        "",
    ]
    if strict_q4_traffic is not None:
        lines.extend(
            [
                "Strict Q4 deployment validation used the atomically serialized, reloaded, "
                "and decoded compact MLP artifact.",
                "",
                f"Q4 cold bytes: {strict_q4_traffic['total_cold_bytes']}; ideal dense-Q4 "
                f"bytes: {strict_q4_traffic['dense_q4_source_mlp_bytes']}; fraction: "
                f"{strict_q4_traffic['fraction_of_dense_q4']:.6f}.",
                "",
            ]
        )
    (target / "width_pruned_training.md").write_text("\n".join(lines), encoding="utf-8")
    return report


__all__ = [
    "WidthPrunedTraffic",
    "select_width_channels",
    "train_width_pruned_student",
    "width_pruned_schedule_traffic",
    "width_pruned_traffic",
]
