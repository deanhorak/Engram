"""Activation-aware additive-quantization screen for one SwiGLU layer.

This module is deliberately a boundary experiment rather than a deployment
kernel.  It jointly encodes the three SwiGLU matrices as rows of
``[gate; up; down.T]`` and keeps the assignments produced by
``fit_product_additive`` fixed.  PyTorch then optimizes only the shared
codebooks and positive per-row scales against cached MLP-boundary outputs.

The final metrics are intentionally strict: optimized values are rounded to
their declared float16 storage types, codes are bit-packed again, the artifact
is serialized and reloaded, and *that decoded artifact* is used for reporting.
The default 2-stage, 128-entry, group-of-8 codec uses 14 bits per subvector.
For a 576-by-1536 SwiGLU layer its complete payload is 593,920 bytes, or
44.7531% of the three dense Q4 matrices.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.product_quantization import (
    ProductAdditiveEncoding,
    ProductAdditiveMetadata,
    _pack_fixed_width,
    decode_product_additive,
    fit_product_additive,
)
from engram.evaluation.router_sweep import _sequence_hashes
from engram.models.inspection import inspect_model, load_layer_mlp, resolve_model_path
from engram.tracing.format import TraceReader
from engram.training.structured_experts import _load_trace_field
from engram.utils import atomic_json, sha256_file


ACTIVATION_AWARE_AQ_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActivationAwareAQResult:
    """Strict result of a fixed-assignment activation-aware AQ screen."""

    encoding: ProductAdditiveEncoding
    report: dict[str, Any]
    artifact_path: Path | None


def _float_matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        matrix = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix)


def project_activation_aware_aq_storage(
    hidden_size: int,
    intermediate_size: int,
    *,
    group_size: int = 8,
    num_codebooks: int = 2,
    codebook_size: int = 128,
) -> dict[str, int | float | bool]:
    """Project exact array-payload bytes against three ideal dense Q4 matrices."""

    for value, name in (
        (hidden_size, "hidden_size"),
        (intermediate_size, "intermediate_size"),
        (group_size, "group_size"),
        (num_codebooks, "num_codebooks"),
        (codebook_size, "codebook_size"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_codebooks not in {1, 2}:
        raise ValueError("num_codebooks must be 1 or 2")
    if codebook_size < 2 or codebook_size > 65536:
        raise ValueError("codebook_size must be within [2, 65536]")

    rows = 3 * intermediate_size
    groups = math.ceil(hidden_size / group_size)
    code_bits = (codebook_size - 1).bit_length()
    information_bits = rows * groups * num_codebooks * code_bits
    packed_code_bytes = (information_bits + 7) // 8
    codebook_bytes = num_codebooks * codebook_size * group_size * 2
    record_scale_bytes = rows * 2
    total_payload_bytes = packed_code_bytes + codebook_bytes + record_scale_bytes
    runtime_header_bytes = 64
    total_bytes = total_payload_bytes + runtime_header_bytes
    dense_q4_bytes = (rows * hidden_size * 4 + 7) // 8
    traffic_fraction = total_bytes / dense_q4_bytes
    return {
        "records": rows,
        "groups_per_record": groups,
        "code_bits": code_bits,
        "information_bits": information_bits,
        "packed_code_bytes": packed_code_bytes,
        "fp16_codebook_bytes": codebook_bytes,
        "fp16_record_scale_bytes": record_scale_bytes,
        "total_payload_bytes": total_payload_bytes,
        "runtime_header_bytes": runtime_header_bytes,
        "total_cold_bytes": total_bytes,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": traffic_fraction,
        "passes_45_percent_traffic_gate": traffic_fraction <= 0.45,
    }


def save_activation_aware_aq_encoding(
    path: str | Path, encoding: ProductAdditiveEncoding
) -> Path:
    """Serialize a codec payload without pickle and atomically replace ``path``."""

    encoding.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = asdict(encoding.metadata)
    metadata["shape"] = list(encoding.metadata.shape)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                packed_codes=encoding.packed_codes,
                codebooks=encoding.codebooks,
                record_scales=(
                    np.empty(0, dtype=np.float16)
                    if encoding.record_scales is None
                    else encoding.record_scales
                ),
                metadata_json=np.asarray(
                    json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                ),
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_activation_aware_aq_encoding(path: str | Path) -> ProductAdditiveEncoding:
    """Load and fully validate an artifact written by the matching save helper."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            required = {
                "packed_codes",
                "codebooks",
                "record_scales",
                "metadata_json",
            }
            if set(payload.files) != required:
                raise ValueError("activation-aware AQ artifact has unexpected fields")
            raw_metadata = payload["metadata_json"]
            if raw_metadata.ndim != 0:
                raise ValueError("metadata_json must be a scalar string")
            metadata_values = json.loads(str(raw_metadata.item()))
            metadata_values["shape"] = tuple(metadata_values["shape"])
            metadata = ProductAdditiveMetadata(**metadata_values)
            raw_scales = np.ascontiguousarray(payload["record_scales"])
            scales = raw_scales if metadata.per_record_scale else None
            if not metadata.per_record_scale and raw_scales.size:
                raise ValueError("scale-free payload unexpectedly contains scales")
            encoding = ProductAdditiveEncoding(
                packed_codes=np.ascontiguousarray(payload["packed_codes"]),
                codebooks=np.ascontiguousarray(payload["codebooks"]),
                record_scales=scales,
                metadata=metadata,
            )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid activation-aware AQ artifact: {source}") from exc
    encoding.validate()
    return encoding


def _split_joint_weights(
    stacked: NDArray[np.float32], intermediate_size: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    gate = stacked[:intermediate_size]
    up = stacked[intermediate_size : 2 * intermediate_size]
    down = stacked[2 * intermediate_size :].T
    return gate, up, down


def run_activation_aware_aq_boundary_screen(
    gate_weight: ArrayLike,
    up_weight: ArrayLike,
    down_weight: ArrayLike,
    training_inputs: ArrayLike,
    training_outputs: ArrayLike,
    validation_inputs: ArrayLike,
    validation_outputs: ArrayLike,
    *,
    artifact_path: str | Path | None = None,
    report_path: str | Path | None = None,
    group_size: int = 8,
    num_codebooks: int = 2,
    codebook_size: int = 128,
    fit_iterations: int = 12,
    fit_sample_limit: int | None = 65_536,
    steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    cosine_loss_weight: float = 0.1,
    projection_loss_weight: float = 0.01,
    checkpoint_interval: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> ActivationAwareAQResult:
    """Optimize and strictly evaluate a fixed-code additive layer encoding.

    The target arrays are cached outputs captured at the exact MLP boundary.
    Validation arrays never participate in fitting or checkpoint selection.
    ``artifact_path`` is optional; when omitted, a temporary artifact still
    exercises the complete serialize/reload/decode path before it is removed.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for activation-aware AQ training"
        ) from exc

    gate = _float_matrix(gate_weight, "gate_weight")
    up = _float_matrix(up_weight, "up_weight")
    down = _float_matrix(down_weight, "down_weight")
    train_x = _float_matrix(training_inputs, "training_inputs")
    train_y = _float_matrix(training_outputs, "training_outputs")
    validation_x = _float_matrix(validation_inputs, "validation_inputs")
    validation_y = _float_matrix(validation_outputs, "validation_outputs")
    intermediate_size, hidden_size = gate.shape
    if up.shape != (intermediate_size, hidden_size):
        raise ValueError("up_weight must have the same shape as gate_weight")
    if down.shape != (hidden_size, intermediate_size):
        raise ValueError("down_weight must have shape [hidden_size, intermediate_size]")
    for name, inputs, outputs in (
        ("training", train_x, train_y),
        ("validation", validation_x, validation_y),
    ):
        if inputs.shape[1] != hidden_size:
            raise ValueError(f"{name}_inputs width must equal hidden_size")
        if outputs.shape != (inputs.shape[0], hidden_size):
            raise ValueError(
                f"{name}_outputs must have shape [records, hidden_size]"
            )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if (
        not np.isfinite(cosine_loss_weight)
        or cosine_loss_weight < 0
        or not np.isfinite(projection_loss_weight)
        or projection_loss_weight < 0
    ):
        raise ValueError("loss weights must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    interval = checkpoint_interval or max(1, steps // 8)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")

    stacked = np.ascontiguousarray(
        np.concatenate((gate, up, down.T), axis=0), dtype=np.float32
    )
    initial_encoding = fit_product_additive(
        stacked,
        group_size=group_size,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        iterations=fit_iterations,
        sample_limit=fit_sample_limit,
        seed=seed,
        per_record_scale=True,
    )
    initial_codes = initial_encoding.unpack_codes()
    if initial_encoding.record_scales is None:  # defensive; fit above requires scales
        raise RuntimeError("initial additive encoding unexpectedly has no row scales")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    codes_tensor = torch.from_numpy(initial_codes.astype(np.int64)).to(target_device)
    initial_codebooks = torch.from_numpy(
        initial_encoding.codebooks.astype(np.float32)
    ).to(target_device)
    minimum_scale = float(np.finfo(np.float16).tiny)
    maximum_scale = float(np.finfo(np.float16).max)
    initial_scales = torch.from_numpy(
        np.maximum(
            initial_encoding.record_scales.astype(np.float32), minimum_scale
        )
    ).to(target_device)

    class FixedAssignmentLayer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.codebooks = torch.nn.Parameter(initial_codebooks.clone())
            self.log_scales = torch.nn.Parameter(torch.log(initial_scales))
            self.register_buffer("codes", codes_tensor)

        @staticmethod
        def fake_float16(value: Any) -> Any:
            rounded = value.to(torch.float16).to(torch.float32)
            return value + (rounded - value).detach()

        def decoded_stacked(self) -> Any:
            stored_codebooks = self.fake_float16(self.codebooks)
            reconstructed = torch.zeros(
                (
                    stacked.shape[0],
                    initial_encoding.metadata.groups,
                    group_size,
                ),
                dtype=torch.float32,
                device=target_device,
            )
            for stage in range(num_codebooks):
                stage_codes = self.codes[:, stage, :].reshape(-1)
                selected = torch.index_select(
                    stored_codebooks[stage], 0, stage_codes
                ).reshape(stacked.shape[0], -1, group_size)
                reconstructed = reconstructed + selected
            positive_scales = torch.exp(self.log_scales)
            stored_scales = self.fake_float16(positive_scales)
            return (
                reconstructed.reshape(stacked.shape[0], -1)[:, :hidden_size]
                * stored_scales[:, None]
            )

        def weights(self) -> tuple[Any, Any, Any]:
            decoded = self.decoded_stacked()
            return (
                decoded[:intermediate_size],
                decoded[intermediate_size : 2 * intermediate_size],
                decoded[2 * intermediate_size :].T,
            )

    module = FixedAssignmentLayer().to(target_device)
    train_x_tensor = torch.from_numpy(train_x).to(target_device)
    train_y_tensor = torch.from_numpy(train_y).to(target_device)
    dense_gate = torch.from_numpy(gate).to(target_device)
    dense_up = torch.from_numpy(up).to(target_device)

    def boundary_output(inputs: Any, weights: tuple[Any, Any, Any]) -> Any:
        quant_gate, quant_up, quant_down = weights
        gate_projection = torch.nn.functional.linear(inputs, quant_gate)
        up_projection = torch.nn.functional.linear(inputs, quant_up)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate_projection) * up_projection,
            quant_down,
        )

    def objective(inputs: Any, targets: Any) -> Any:
        quant_gate, quant_up, quant_down = module.weights()
        gate_projection = torch.nn.functional.linear(inputs, quant_gate)
        up_projection = torch.nn.functional.linear(inputs, quant_up)
        output = torch.nn.functional.linear(
            torch.nn.functional.silu(gate_projection) * up_projection,
            quant_down,
        )
        epsilon = torch.finfo(torch.float32).eps
        output_nmse = torch.sum((output - targets) ** 2) / torch.clamp(
            torch.sum(targets**2), min=epsilon
        )
        cosine_distance = 1.0 - torch.nn.functional.cosine_similarity(
            output, targets, dim=-1, eps=1e-8
        ).mean()
        target_gate = torch.nn.functional.linear(inputs, dense_gate)
        target_up = torch.nn.functional.linear(inputs, dense_up)
        gate_nmse = torch.sum((gate_projection - target_gate) ** 2) / torch.clamp(
            torch.sum(target_gate**2), min=epsilon
        )
        up_nmse = torch.sum((up_projection - target_up) ** 2) / torch.clamp(
            torch.sum(target_up**2), min=epsilon
        )
        return (
            output_nmse
            + cosine_loss_weight * cosine_distance
            + projection_loss_weight * 0.5 * (gate_nmse + up_nmse)
        )

    optimizer = torch.optim.AdamW(
        module.parameters(), learning_rate, weight_decay=0.0
    )
    rng = np.random.default_rng(seed)
    minibatch_losses: list[float] = []
    checkpoints: list[dict[str, float | int]] = []
    with torch.no_grad():
        best_objective = float(objective(train_x_tensor, train_y_tensor).cpu())
    best_codebooks = module.codebooks.detach().clone()
    best_log_scales = module.log_scales.detach().clone()
    checkpoints.append({"step": 0, "full_training_objective": best_objective})

    module.train()
    for step in range(1, steps + 1):
        sample_size = min(batch_size, train_x.shape[0])
        indices = rng.choice(train_x.shape[0], size=sample_size, replace=False)
        batch_indices = torch.from_numpy(indices.astype(np.int64)).to(target_device)
        loss = objective(
            train_x_tensor.index_select(0, batch_indices),
            train_y_tensor.index_select(0, batch_indices),
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"activation-aware AQ loss became non-finite at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            module.codebooks.clamp_(-maximum_scale, maximum_scale)
            module.log_scales.clamp_(math.log(minimum_scale), math.log(maximum_scale))
        minibatch_losses.append(float(loss.detach().cpu()))
        if step % interval == 0 or step == steps:
            with torch.no_grad():
                full_objective = float(
                    objective(train_x_tensor, train_y_tensor).cpu()
                )
            checkpoints.append(
                {"step": step, "full_training_objective": full_objective}
            )
            if full_objective < best_objective:
                best_objective = full_objective
                best_codebooks = module.codebooks.detach().clone()
                best_log_scales = module.log_scales.detach().clone()

    with torch.no_grad():
        module.codebooks.copy_(best_codebooks)
        module.log_scales.copy_(best_log_scales)
        stored_codebooks = module.codebooks.to(torch.float16).cpu().numpy()
        stored_scales = torch.exp(module.log_scales).to(torch.float16).cpu().numpy()
    if not np.all(np.isfinite(stored_codebooks)) or not np.all(
        np.isfinite(stored_scales)
    ):
        raise RuntimeError("optimized AQ parameters do not fit in finite float16")
    if np.any(stored_scales <= 0):
        raise RuntimeError("optimized AQ row scales must remain strictly positive")

    optimized_encoding = ProductAdditiveEncoding(
        packed_codes=_pack_fixed_width(
            initial_codes, initial_encoding.metadata.code_bits
        ),
        codebooks=np.ascontiguousarray(stored_codebooks),
        record_scales=np.ascontiguousarray(stored_scales),
        metadata=initial_encoding.metadata,
    )
    optimized_encoding.validate()

    def strict_metrics(
        candidate_stacked: NDArray[np.float32],
        inputs: NDArray[np.float32],
        targets: NDArray[np.float32],
    ) -> dict[str, float]:
        candidate_gate, candidate_up, candidate_down = _split_joint_weights(
            candidate_stacked, intermediate_size
        )
        with torch.inference_mode():
            input_tensor = torch.from_numpy(inputs).to(target_device)
            target_tensor = torch.from_numpy(targets).to(target_device)
            candidate_weights = (
                torch.from_numpy(candidate_gate).to(target_device),
                torch.from_numpy(candidate_up).to(target_device),
                torch.from_numpy(candidate_down).to(target_device),
            )
            output = boundary_output(input_tensor, candidate_weights)
            difference = output - target_tensor
            target_energy = torch.clamp(
                torch.sum(target_tensor**2), min=torch.finfo(torch.float32).eps
            )
            output_nmse = torch.sum(difference**2) / target_energy
            relative = torch.linalg.vector_norm(difference, dim=-1) / torch.clamp(
                torch.linalg.vector_norm(target_tensor, dim=-1), min=1e-8
            )
            cosine = torch.nn.functional.cosine_similarity(
                output, target_tensor, dim=-1, eps=1e-8
            )
            candidate_gate_projection = torch.nn.functional.linear(
                input_tensor, candidate_weights[0]
            )
            candidate_up_projection = torch.nn.functional.linear(
                input_tensor, candidate_weights[1]
            )
            dense_gate_projection = torch.nn.functional.linear(input_tensor, dense_gate)
            dense_up_projection = torch.nn.functional.linear(input_tensor, dense_up)
            gate_nmse = torch.sum(
                (candidate_gate_projection - dense_gate_projection) ** 2
            ) / torch.clamp(
                torch.sum(dense_gate_projection**2),
                min=torch.finfo(torch.float32).eps,
            )
            up_nmse = torch.sum(
                (candidate_up_projection - dense_up_projection) ** 2
            ) / torch.clamp(
                torch.sum(dense_up_projection**2),
                min=torch.finfo(torch.float32).eps,
            )
            return {
                "output_nmse": float(output_nmse.cpu()),
                "mean_relative_l2": float(relative.mean().cpu()),
                "p95_relative_l2": float(
                    torch.quantile(relative, 0.95).cpu()
                ),
                "mean_cosine_similarity": float(cosine.mean().cpu()),
                "gate_projection_nmse": float(gate_nmse.cpu()),
                "up_projection_nmse": float(up_nmse.cpu()),
            }

    initial_decoded = decode_product_additive(initial_encoding)
    initial_validation_metrics = strict_metrics(
        initial_decoded, validation_x, validation_y
    )

    requested_artifact = Path(artifact_path) if artifact_path is not None else None
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None
    if requested_artifact is None:
        temporary_directory = tempfile.TemporaryDirectory(prefix="engram-aq-")
        serialization_path = Path(temporary_directory.name) / "encoding.npz"
    else:
        serialization_path = requested_artifact
    try:
        save_activation_aware_aq_encoding(serialization_path, optimized_encoding)
        serialized_bytes = serialization_path.stat().st_size
        serialized_sha256 = sha256_file(serialization_path)
        reloaded_encoding = load_activation_aware_aq_encoding(serialization_path)
        strict_decoded = decode_product_additive(reloaded_encoding)
        round_trip_exact = (
            np.array_equal(
                optimized_encoding.packed_codes, reloaded_encoding.packed_codes
            )
            and np.array_equal(
                optimized_encoding.codebooks, reloaded_encoding.codebooks
            )
            and np.array_equal(
                optimized_encoding.record_scales, reloaded_encoding.record_scales
            )
            and np.array_equal(
                decode_product_additive(optimized_encoding), strict_decoded
            )
        )
        if not round_trip_exact:
            raise RuntimeError("activation-aware AQ artifact failed its exact round trip")
        strict_training_metrics = strict_metrics(strict_decoded, train_x, train_y)
        strict_validation_metrics = strict_metrics(
            strict_decoded, validation_x, validation_y
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    traffic = project_activation_aware_aq_storage(
        hidden_size,
        intermediate_size,
        group_size=group_size,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )
    if reloaded_encoding.storage_bytes != traffic["total_payload_bytes"]:
        raise RuntimeError("projected and encoded payload byte accounting disagree")
    dense_weight_norm = float(np.linalg.norm(stacked))
    strict_weight_relative_l2 = float(
        np.linalg.norm(strict_decoded - stacked) / max(dense_weight_norm, 1e-12)
    )
    report: dict[str, Any] = {
        "schema_version": ACTIVATION_AWARE_AQ_SCHEMA_VERSION,
        "experiment": "activation_aware_additive_quantization_boundary_screen",
        "optimization_mode": "P_only_fixed_assignments",
        "configuration": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "group_size": group_size,
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "code_bits": initial_encoding.metadata.code_bits,
            "fit_iterations": fit_iterations,
            "fit_sample_limit": fit_sample_limit,
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "cosine_loss_weight": cosine_loss_weight,
            "projection_loss_weight": projection_loss_weight,
            "checkpoint_interval": interval,
            "seed": seed,
            "device": str(target_device),
        },
        "traffic": traffic,
        "storage_components": reloaded_encoding.storage_components(),
        "training": {
            "initial_full_objective": checkpoints[0]["full_training_objective"],
            "best_full_objective": best_objective,
            "minibatch_initial_loss": minibatch_losses[0],
            "minibatch_final_loss": minibatch_losses[-1],
            "checkpoints": checkpoints,
        },
        "validation": {
            "initial_fixed_code_decoded": initial_validation_metrics,
            "strict_reloaded_decoded": strict_validation_metrics,
            "strict_weight_relative_l2": strict_weight_relative_l2,
        },
        "strict_reloaded_training": strict_training_metrics,
        "artifact": {
            "path": str(requested_artifact) if requested_artifact else None,
            "archive_bytes_not_used_for_traffic": serialized_bytes,
            "sha256": serialized_sha256,
            "exact_array_round_trip": round_trip_exact,
            "validation_source": "serialized_reloaded_fp16_payload",
        },
        "interpretation": {
            "traffic_gate_passed": traffic["passes_45_percent_traffic_gate"],
            "quality_is_boundary_only": True,
            "validation_was_not_used_for_checkpoint_selection": True,
            "next_if_shared_codebooks_fail": (
                "screen multiple codebook sets at 2x6 or 3x4 bits while retaining "
                "the same strict sub-45% payload accounting"
            ),
        },
    }
    if report_path is not None:
        atomic_json(Path(report_path), report)
    return ActivationAwareAQResult(
        encoding=reloaded_encoding,
        report=report,
        artifact_path=requested_artifact,
    )


def train_activation_aware_aq_boundaries(
    model: str | Path,
    training_traces: str | Path,
    validation_traces: str | Path,
    out: str | Path,
    *,
    layers: list[int],
    fit_iterations: int = 12,
    fit_sample_limit: int | None = 65_536,
    steps: int = 128,
    batch_size: int = 64,
    learning_rate: float = 2e-3,
    cosine_loss_weight: float = 0.1,
    projection_loss_weight: float = 0.01,
    checkpoint_interval: int | None = None,
    maximum_mean_relative_l2: float = 0.10,
    max_train_records: int | None = 4096,
    max_validation_records: int | None = 2048,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run strict packed-artifact AQ screens from cached teacher boundaries."""

    if not layers:
        raise ValueError("at least one layer is required")
    if not np.isfinite(maximum_mean_relative_l2) or maximum_mean_relative_l2 <= 0:
        raise ValueError("maximum_mean_relative_l2 must be positive and finite")
    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    selected_layers = sorted(set(int(layer) for layer in layers))
    if selected_layers[0] < 0 or selected_layers[-1] >= inspection.num_hidden_layers:
        raise ValueError("layer index is outside the source model")

    training_reader = TraceReader(training_traces)
    validation_reader = TraceReader(validation_traces)
    for name, reader, split in (
        ("training", training_reader, "calibration"),
        ("validation", validation_reader, "validation"),
    ):
        if reader.manifest["model_hash"] != inspection.source_hash:
            raise ValueError(f"{name} trace/model hash mismatch")
        if reader.manifest["split"] != split:
            raise ValueError(f"expected {split!r} {name} traces")
    if training_reader.manifest["dataset_hash"] == validation_reader.manifest["dataset_hash"]:
        raise ValueError("training and validation boundary datasets must differ")
    if set(_sequence_hashes(training_reader)).intersection(
        _sequence_hashes(validation_reader)
    ):
        raise ValueError("training and validation boundary sequences overlap")

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    layer_reports: dict[str, Any] = {}
    for layer in selected_layers:
        gate, up, down = load_layer_mlp(model_path, layer)
        training_inputs = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_input", max_train_records
        ).astype(np.float32)
        training_outputs = _load_trace_field(
            training_reader, f"layer_{layer}_mlp_output", max_train_records
        ).astype(np.float32)
        validation_inputs = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_input", max_validation_records
        ).astype(np.float32)
        validation_outputs = _load_trace_field(
            validation_reader, f"layer_{layer}_mlp_output", max_validation_records
        ).astype(np.float32)
        result = run_activation_aware_aq_boundary_screen(
            gate,
            up,
            down,
            training_inputs,
            training_outputs,
            validation_inputs,
            validation_outputs,
            artifact_path=destination / f"layer_{layer}.aq.npz",
            report_path=destination / f"layer_{layer}.json",
            fit_iterations=fit_iterations,
            fit_sample_limit=fit_sample_limit,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            cosine_loss_weight=cosine_loss_weight,
            projection_loss_weight=projection_loss_weight,
            checkpoint_interval=checkpoint_interval,
            seed=seed + layer,
            device=device,
        )
        layer_reports[str(layer)] = result.report

    mean_errors = [
        float(report["validation"]["strict_reloaded_decoded"]["mean_relative_l2"])
        for report in layer_reports.values()
    ]
    traffic_passed = all(
        bool(report["traffic"]["passes_45_percent_traffic_gate"])
        for report in layer_reports.values()
    )
    report = {
        "schema_version": ACTIVATION_AWARE_AQ_SCHEMA_VERSION,
        "experiment": "activation_aware_additive_quantization_cached_boundaries",
        "model": str(model_path),
        "model_hash": inspection.source_hash,
        "training_dataset_hash": training_reader.manifest["dataset_hash"],
        "validation_dataset_hash": validation_reader.manifest["dataset_hash"],
        "layers": layer_reports,
        "screen": {
            "maximum_mean_relative_l2": maximum_mean_relative_l2,
            "mean_layer_relative_l2": float(np.mean(mean_errors)),
            "maximum_layer_relative_l2": float(np.max(mean_errors)),
            "traffic_passed": traffic_passed,
            "passed": traffic_passed
            and max(mean_errors) <= maximum_mean_relative_l2,
            "scope": "cached_mlp_boundaries_only_not_causal_gate",
        },
    }
    atomic_json(destination / "activation_aware_aq_boundaries.json", report)
    return report


__all__ = [
    "ACTIVATION_AWARE_AQ_SCHEMA_VERSION",
    "ActivationAwareAQResult",
    "load_activation_aware_aq_encoding",
    "project_activation_aware_aq_storage",
    "run_activation_aware_aq_boundary_screen",
    "save_activation_aware_aq_encoding",
    "train_activation_aware_aq_boundaries",
]
