"""Activation-aware P/V training for the projection-local AQ codec.

The deployment object is always a strict
``projection_2x7_scale6`` :class:`~engram.semantic.multiset_additive_quantization.MultiSetAdditiveEncoding`.
Training is allowed to keep float32 master parameters, but every reported
boundary metric is measured after float16 rounding, bit packing, serialization,
checksum validation, reload, and decode through the deployment codec.

``P`` steps optimize codebook vectors and positive scale centroids while codes
and scale indices remain fixed.  ``V`` steps update those discrete assignments
using cached teacher-boundary states.  Gate and up assignments use the exact
activation block Hessian and the current preactivation residual.  Down
assignments use the current nonlinear activation and exact MLP-output residual;
they never evaluate a nonlinear forward for each candidate.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from engram.semantic.multiset_additive_quantization import (
    PROJECTION_2X7_SCALE6,
    MultiSetAdditiveEncoding,
    decode_multiset_additive,
    fit_multiset_additive,
    load_multiset_additive,
    save_multiset_additive,
)


PROJECTION_AQ_BOUNDARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectionAQBoundaryResult:
    """Final strict encoding and diagnostics for one cached MLP boundary."""

    encoding: MultiSetAdditiveEncoding
    report: dict[str, Any]
    artifact_path: Path | None


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float32]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()  # type: ignore[union-attr]
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a non-empty rank-2 matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


def _positive_integer(value: object, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < (0 if allow_zero else 1):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _pack_fixed_width(values: NDArray[np.integer[Any]], bits: int) -> NDArray[np.uint8]:
    """Pack row-major unsigned values using the codec's public wire order."""

    flat = np.asarray(values).reshape(-1).astype(np.uint32, copy=False)
    if flat.size and int(flat.max()) >= 1 << bits:
        raise ValueError("a value is outside its packed bit width")
    result = np.zeros((flat.size * bits + 7) // 8, dtype=np.uint8)
    starts = np.arange(flat.size, dtype=np.int64) * bits
    for source_bit in range(bits):
        positions = starts + source_bit
        selected = ((flat >> source_bit) & 1).astype(np.uint8)
        np.bitwise_or.at(
            result,
            positions // 8,
            selected << (positions % 8).astype(np.uint8),
        )
    return np.ascontiguousarray(result)


def _unpack_fixed_width(
    packed: NDArray[np.uint8], count: int, bits: int
) -> NDArray[np.uint16]:
    source = np.asarray(packed, dtype=np.uint8).reshape(-1)
    result = np.zeros(count, dtype=np.uint16)
    starts = np.arange(count, dtype=np.int64) * bits
    for target_bit in range(bits):
        positions = starts + target_bit
        selected = (source[positions // 8] >> (positions % 8)) & 1
        result |= selected.astype(np.uint16) << target_bit
    return result


def _silu_numpy(values: NDArray[np.float32]) -> NDArray[np.float32]:
    # The stable branch avoids overflow in tests with deliberately broad inputs.
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = values[positive] / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = values[~positive] * exponent / (1.0 + exponent)
    return result


def _dense_boundary_output(
    inputs: NDArray[np.float32],
    gate: NDArray[np.float32],
    up: NDArray[np.float32],
    down: NDArray[np.float32],
) -> NDArray[np.float32]:
    return np.ascontiguousarray(
        (_silu_numpy(inputs @ gate.T) * (inputs @ up.T)) @ down.T,
        dtype=np.float32,
    )


def _split_stack(
    stacked: NDArray[np.float32], intermediate_size: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    return (
        np.ascontiguousarray(stacked[:intermediate_size]),
        np.ascontiguousarray(stacked[intermediate_size : 2 * intermediate_size]),
        np.ascontiguousarray(stacked[2 * intermediate_size :].T),
    )


def _strict_metrics(
    inputs: NDArray[np.float32],
    targets: NDArray[np.float32],
    stacked: NDArray[np.float32],
    dense_gate: NDArray[np.float32],
    dense_up: NDArray[np.float32],
    intermediate_size: int,
) -> dict[str, float]:
    gate, up, down = _split_stack(stacked, intermediate_size)
    output = _dense_boundary_output(inputs, gate, up, down)
    difference = output - targets
    target_norm = np.linalg.norm(targets, axis=1)
    relative = np.linalg.norm(difference, axis=1) / np.maximum(target_norm, 1e-8)
    output_norm = np.linalg.norm(output, axis=1)
    cosine = np.sum(output * targets, axis=1) / np.maximum(
        output_norm * target_norm, 1e-8
    )
    dense_gate_projection = inputs @ dense_gate.T
    dense_up_projection = inputs @ dense_up.T
    gate_difference = inputs @ gate.T - dense_gate_projection
    up_difference = inputs @ up.T - dense_up_projection
    epsilon = np.finfo(np.float32).eps
    return {
        "output_nmse": float(
            np.sum(difference.astype(np.float64) ** 2)
            / max(float(np.sum(targets.astype(np.float64) ** 2)), epsilon)
        ),
        "mean_relative_l2": float(np.mean(relative, dtype=np.float64)),
        "p95_relative_l2": float(np.quantile(relative, 0.95)),
        "mean_cosine_similarity": float(np.mean(cosine, dtype=np.float64)),
        "gate_projection_nmse": float(
            np.sum(gate_difference.astype(np.float64) ** 2)
            / max(
                float(np.sum(dense_gate_projection.astype(np.float64) ** 2)),
                epsilon,
            )
        ),
        "up_projection_nmse": float(
            np.sum(up_difference.astype(np.float64) ** 2)
            / max(
                float(np.sum(dense_up_projection.astype(np.float64) ** 2)),
                epsilon,
            )
        ),
    }


def _objective_from_metrics(
    metrics: dict[str, float],
    cosine_weight: float,
    gate_projection_weight: float,
    up_projection_weight: float,
) -> float:
    return float(
        metrics["output_nmse"]
        + cosine_weight * (1.0 - metrics["mean_cosine_similarity"])
        + gate_projection_weight * metrics["gate_projection_nmse"]
        + up_projection_weight * metrics["up_projection_nmse"]
    )


def _encoding_from_module(module: Any, initial: MultiSetAdditiveEncoding) -> MultiSetAdditiveEncoding:
    metadata = initial.metadata
    with module.torch.no_grad():
        codebooks = (
            module.codebooks.detach().to(module.torch.float16).cpu().numpy()
        )
        scale_codebooks = (
            module.torch.exp(module.log_scale_codebooks)
            .detach()
            .to(module.torch.float16)
            .cpu()
            .numpy()
        )
        codes = module.codes.detach().cpu().numpy().astype(np.uint16)
        scale_indices = (
            module.scale_indices.detach().cpu().numpy().astype(np.uint16)
        )
    encoding = MultiSetAdditiveEncoding(
        packed_codes=_pack_fixed_width(codes, metadata.code_bits),
        codebooks=np.ascontiguousarray(codebooks),
        metadata=metadata,
        packed_scale_indices=_pack_fixed_width(scale_indices, metadata.scale_bits),
        scale_codebooks=np.ascontiguousarray(scale_codebooks),
    )
    encoding.validate()
    return encoding


def _strict_round_trip(
    encoding: MultiSetAdditiveEncoding, path: Path
) -> tuple[MultiSetAdditiveEncoding, NDArray[np.float32], dict[str, Any]]:
    checksum = save_multiset_additive(path, encoding)
    serialized_bytes = path.stat().st_size
    if serialized_bytes != encoding.storage_bytes:
        raise RuntimeError("serialized AQ bytes disagree with codec accounting")
    reloaded = load_multiset_additive(path)
    decoded = decode_multiset_additive(reloaded)
    exact = (
        np.array_equal(encoding.packed_codes, reloaded.packed_codes)
        and np.array_equal(encoding.codebooks, reloaded.codebooks)
        and np.array_equal(
            encoding.packed_scale_indices, reloaded.packed_scale_indices
        )
        and np.array_equal(encoding.scale_codebooks, reloaded.scale_codebooks)
        and np.array_equal(decode_multiset_additive(encoding), decoded)
    )
    if not exact:
        raise RuntimeError("projection-local AQ failed its exact packed round trip")
    dense_q4_bytes = reloaded.metadata.dense_q4_bytes
    return reloaded, decoded, {
        "path": str(path),
        "checksum": checksum,
        "serialized_bytes": serialized_bytes,
        "dense_q4_bytes": dense_q4_bytes,
        "fraction_of_dense_q4": serialized_bytes / dense_q4_bytes,
        "exact_round_trip": True,
        "metrics_source": "checksum_validated_serialized_reload_decode",
    }


def train_projection_aq_boundaries(
    gate_weight: ArrayLike,
    up_weight: ArrayLike,
    down_weight: ArrayLike,
    train_inputs: ArrayLike,
    train_outputs: ArrayLike,
    validation_inputs: ArrayLike,
    validation_outputs: ArrayLike,
    *,
    artifact_dir: str | Path | None = None,
    p_steps_per_cycle: int = 64,
    v_cycles: int = 2,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    checkpoint_interval: int = 16,
    fit_iterations: int = 12,
    fit_sample_limit: int | None = 65_536,
    v_max_records: int | None = 1024,
    v_change_fraction: float = 0.01,
    selection_records: int | None = 512,
    cosine_weight: float = 0.1,
    gate_projection_weight: float = 0.025,
    up_projection_weight: float = 0.025,
    maximum_mean_relative_l2: float = 0.08,
    maximum_p95_relative_l2: float = 0.18,
    minimum_mean_cosine: float = 0.99,
    seed: int = 0,
    device: str = "cpu",
    enforce_traffic_gate: bool = True,
) -> ProjectionAQBoundaryResult:
    """Fit one projection-local AQ artifact against cached teacher boundaries.

    Checkpoints and discrete-update acceptance decisions use a fixed subset of the
    *training* records.  Validation records are decoded and measured only after
    a cycle is complete, so they cannot select a checkpoint.
    """

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "projection AQ training requires torch; install engram-lm[conversion]"
        ) from exc

    gate = _matrix(gate_weight, "gate_weight")
    up = _matrix(up_weight, "up_weight")
    down = _matrix(down_weight, "down_weight")
    training_x = _matrix(train_inputs, "train_inputs")
    training_y = _matrix(train_outputs, "train_outputs")
    validation_x = _matrix(validation_inputs, "validation_inputs")
    validation_y = _matrix(validation_outputs, "validation_outputs")
    if gate.shape != up.shape:
        raise ValueError("gate_weight and up_weight must have identical shapes")
    intermediate_size, hidden_size = gate.shape
    if down.shape != (hidden_size, intermediate_size):
        raise ValueError(
            "down_weight must have shape [hidden_size, intermediate_size]"
        )
    for name, inputs, outputs in (
        ("training", training_x, training_y),
        ("validation", validation_x, validation_y),
    ):
        if inputs.shape[1] != hidden_size:
            raise ValueError(f"{name} inputs disagree with hidden_size")
        if outputs.shape != (inputs.shape[0], hidden_size):
            raise ValueError(f"{name} outputs must match input rows and hidden_size")

    p_steps = _positive_integer(p_steps_per_cycle, "p_steps_per_cycle")
    cycles = _positive_integer(v_cycles, "v_cycles")
    minibatch_size = _positive_integer(batch_size, "batch_size")
    interval = _positive_integer(checkpoint_interval, "checkpoint_interval")
    fit_rounds = _positive_integer(fit_iterations, "fit_iterations")
    if fit_sample_limit is not None:
        fit_sample_limit = _positive_integer(fit_sample_limit, "fit_sample_limit")
    if v_max_records is not None:
        v_max_records = _positive_integer(v_max_records, "v_max_records")
    if selection_records is not None:
        selection_records = _positive_integer(selection_records, "selection_records")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for value, name in (
        (learning_rate, "learning_rate"),
        (maximum_mean_relative_l2, "maximum_mean_relative_l2"),
        (maximum_p95_relative_l2, "maximum_p95_relative_l2"),
        (minimum_mean_cosine, "minimum_mean_cosine"),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if minimum_mean_cosine > 1.0:
        raise ValueError("minimum_mean_cosine must not exceed 1")
    if (
        not np.isfinite(v_change_fraction)
        or v_change_fraction <= 0
        or v_change_fraction > 1
    ):
        raise ValueError("v_change_fraction must be finite and within (0, 1]")
    for value, name in (
        (cosine_weight, "cosine_weight"),
        (gate_projection_weight, "gate_projection_weight"),
        (up_projection_weight, "up_projection_weight"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not isinstance(enforce_traffic_gate, bool):
        raise ValueError("enforce_traffic_gate must be a boolean")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    stacked = np.ascontiguousarray(np.concatenate((gate, up, down.T), axis=0))
    initial_encoding = fit_multiset_additive(
        stacked,
        profile=PROJECTION_2X7_SCALE6,
        iterations=fit_rounds,
        sample_limit=fit_sample_limit,
        seed=int(seed),
    )
    metadata = initial_encoding.metadata
    expected_codebook_shape = (3, 2, 128, 8)
    if initial_encoding.codebooks.shape != expected_codebook_shape:
        raise RuntimeError(
            f"primary codec returned {initial_encoding.codebooks.shape}, "
            f"expected {expected_codebook_shape}"
        )
    if metadata.fraction_of_dense_q4 >= 0.45 and enforce_traffic_gate:
        raise ValueError(
            "the complete serialized codec is not below 45% of dense Q4 for "
            f"this shape ({metadata.fraction_of_dense_q4:.6f})"
        )

    initial_codes = initial_encoding.unpack_codes().astype(np.int64)
    assert initial_encoding.packed_scale_indices is not None
    assert initial_encoding.scale_codebooks is not None
    initial_scale_indices = _unpack_fixed_width(
        initial_encoding.packed_scale_indices,
        metadata.shape[0],
        metadata.scale_bits,
    ).astype(np.int64)
    initial_scales = initial_encoding.scale_codebooks.astype(np.float32)
    minimum_scale = float(np.nextafter(np.float16(0), np.float16(1)))
    initial_scales = np.maximum(initial_scales, minimum_scale)

    class ProjectionAQModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.torch = torch
            self.codebooks = torch.nn.Parameter(
                torch.from_numpy(initial_encoding.codebooks.astype(np.float32))
            )
            self.log_scale_codebooks = torch.nn.Parameter(
                torch.log(torch.from_numpy(initial_scales))
            )
            self.register_buffer("codes", torch.from_numpy(initial_codes))
            self.register_buffer(
                "scale_indices", torch.from_numpy(initial_scale_indices)
            )

        @staticmethod
        def fake_float16(value: Any) -> Any:
            rounded = value.to(torch.float16).to(torch.float32)
            return value + (rounded - value).detach()

        def stored_parameters(self) -> tuple[Any, Any]:
            return (
                self.fake_float16(self.codebooks),
                self.fake_float16(torch.exp(self.log_scale_codebooks)),
            )

        def normalized_projection(self, projection: int, codebooks: Any | None = None) -> Any:
            if codebooks is None:
                codebooks = self.stored_parameters()[0]
            start = projection * intermediate_size
            projection_codes = self.codes[start : start + intermediate_size]
            reconstructed = torch.zeros(
                (intermediate_size, metadata.groups, metadata.group_size),
                dtype=torch.float32,
                device=target_device,
            )
            for stage in range(metadata.num_stages):
                reconstructed = reconstructed + codebooks[projection, stage][
                    projection_codes[:, stage, :]
                ]
            return reconstructed.reshape(intermediate_size, metadata.padded_width)[
                :, :hidden_size
            ]

        def weights(self) -> tuple[Any, Any, Any]:
            codebooks, scale_codebooks = self.stored_parameters()
            projections = []
            for projection in range(3):
                start = projection * intermediate_size
                indices = self.scale_indices[start : start + intermediate_size]
                scales = scale_codebooks[projection][indices]
                projections.append(
                    self.normalized_projection(projection, codebooks) * scales[:, None]
                )
            return projections[0], projections[1], projections[2].T

    module = ProjectionAQModule().to(target_device)
    train_x_tensor = torch.from_numpy(training_x).to(target_device)
    train_y_tensor = torch.from_numpy(training_y).to(target_device)
    dense_gate_tensor = torch.from_numpy(gate).to(target_device)
    dense_up_tensor = torch.from_numpy(up).to(target_device)

    def torch_objective(inputs: Any, targets: Any) -> tuple[Any, dict[str, Any]]:
        quant_gate, quant_up, quant_down = module.weights()
        gate_projection = functional.linear(inputs, quant_gate)
        up_projection = functional.linear(inputs, quant_up)
        output = functional.linear(
            functional.silu(gate_projection) * up_projection, quant_down
        )
        epsilon = torch.finfo(torch.float32).eps
        output_nmse = torch.sum((output - targets) ** 2) / torch.clamp(
            torch.sum(targets**2), min=epsilon
        )
        cosine_distance = 1.0 - functional.cosine_similarity(
            output, targets, dim=1, eps=1e-8
        ).mean()
        dense_gate_projection = functional.linear(inputs, dense_gate_tensor)
        dense_up_projection = functional.linear(inputs, dense_up_tensor)
        gate_nmse = torch.sum((gate_projection - dense_gate_projection) ** 2) / torch.clamp(
            torch.sum(dense_gate_projection**2), min=epsilon
        )
        up_nmse = torch.sum((up_projection - dense_up_projection) ** 2) / torch.clamp(
            torch.sum(dense_up_projection**2), min=epsilon
        )
        loss = (
            output_nmse
            + cosine_weight * cosine_distance
            + gate_projection_weight * gate_nmse
            + up_projection_weight * up_nmse
        )
        return loss, {
            "output_nmse": output_nmse,
            "cosine_distance": cosine_distance,
            "gate_projection_nmse": gate_nmse,
            "up_projection_nmse": up_nmse,
        }

    selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53454C]))
    selection_count = min(
        training_x.shape[0],
        training_x.shape[0] if selection_records is None else selection_records,
    )
    selection_ids = np.sort(
        selection_rng.choice(training_x.shape[0], selection_count, replace=False)
    ).astype(np.int64)
    selection_tensor = torch.from_numpy(selection_ids).to(target_device)
    selection_x = train_x_tensor.index_select(0, selection_tensor)
    selection_y = train_y_tensor.index_select(0, selection_tensor)

    def selection_objective() -> float:
        module.eval()
        with torch.no_grad():
            value = float(torch_objective(selection_x, selection_y)[0].cpu())
        return value

    def snapshot() -> tuple[Any, Any, Any, Any]:
        return (
            module.codebooks.detach().clone(),
            module.log_scale_codebooks.detach().clone(),
            module.codes.detach().clone(),
            module.scale_indices.detach().clone(),
        )

    def restore(state: tuple[Any, Any, Any, Any]) -> None:
        with torch.no_grad():
            module.codebooks.copy_(state[0])
            module.log_scale_codebooks.copy_(state[1])
            module.codes.copy_(state[2])
            module.scale_indices.copy_(state[3])

    def p_optimize(cycle: int) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            (module.codebooks, module.log_scale_codebooks), lr=learning_rate
        )
        phase_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x50535445, cycle])
        )
        order = phase_rng.permutation(training_x.shape[0])
        cursor = 0
        initial_value = selection_objective()
        best_value = initial_value
        best_parameters = (
            module.codebooks.detach().clone(),
            module.log_scale_codebooks.detach().clone(),
        )
        checkpoints: list[dict[str, float | int]] = [
            {"step": 0, "selection_training_objective": initial_value}
        ]
        minibatch_losses: list[float] = []
        module.train()
        for step in range(1, p_steps + 1):
            if cursor + minibatch_size > order.size:
                order = phase_rng.permutation(training_x.shape[0])
                cursor = 0
            ids = order[cursor : cursor + min(minibatch_size, order.size)]
            cursor += ids.size
            ids_tensor = torch.from_numpy(ids.astype(np.int64)).to(target_device)
            loss, _ = torch_objective(
                train_x_tensor.index_select(0, ids_tensor),
                train_y_tensor.index_select(0, ids_tensor),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"P loss became non-finite in cycle {cycle}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (module.codebooks, module.log_scale_codebooks), 1.0
            )
            optimizer.step()
            with torch.no_grad():
                module.codebooks.clamp_(-65_504.0, 65_504.0)
                module.log_scale_codebooks.clamp_(
                    math.log(minimum_scale), math.log(65_504.0)
                )
            minibatch_losses.append(float(loss.detach().cpu()))
            if step % interval == 0 or step == p_steps:
                current = selection_objective()
                checkpoints.append(
                    {"step": step, "selection_training_objective": current}
                )
                if current < best_value:
                    best_value = current
                    best_parameters = (
                        module.codebooks.detach().clone(),
                        module.log_scale_codebooks.detach().clone(),
                    )
                module.train()
        with torch.no_grad():
            module.codebooks.copy_(best_parameters[0])
            module.log_scale_codebooks.copy_(best_parameters[1])
        return {
            "initial_selection_training_objective": initial_value,
            "best_selection_training_objective": best_value,
            "initial_minibatch_loss": minibatch_losses[0],
            "final_minibatch_loss": minibatch_losses[-1],
            "checkpoints": checkpoints,
            "fixed_codes_and_scale_indices": True,
        }

    def v_reassign(cycle: int) -> dict[str, Any]:
        v_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x56535445, cycle])
        )
        v_count = min(
            training_x.shape[0],
            training_x.shape[0] if v_max_records is None else v_max_records,
        )
        ids = np.sort(
            v_rng.choice(training_x.shape[0], v_count, replace=False)
        ).astype(np.int64)
        ids_tensor = torch.from_numpy(ids).to(target_device)
        x = train_x_tensor.index_select(0, ids_tensor)
        target_output = train_y_tensor.index_select(0, ids_tensor)
        if metadata.padded_width == hidden_size:
            padded_x = x
        else:
            padded_x = functional.pad(
                x, (0, metadata.padded_width - hidden_size)
            )
        codes_changed = [0, 0, 0]
        codes_proposed = [0, 0, 0]
        scale_indices_changed = [0, 0, 0]
        scale_indices_proposed = [0, 0, 0]

        def trust_limited_assignments(
            current_ids: Any, scores: Any
        ) -> tuple[Any, int, int]:
            """Apply only the strongest improving changes in one coordinate block."""

            best_scores, best_ids = torch.min(scores, dim=1)
            proposed_mask = (best_ids != current_ids) & (best_scores < -1e-10)
            candidates = torch.nonzero(proposed_mask, as_tuple=False).flatten()
            proposed = int(candidates.numel())
            budget = max(
                1, int(math.ceil(current_ids.numel() * v_change_fraction))
            )
            if candidates.numel() > budget:
                strongest = torch.topk(
                    best_scores.index_select(0, candidates),
                    k=budget,
                    largest=False,
                ).indices
                candidates = candidates.index_select(0, strongest)
            selected = current_ids.clone()
            if candidates.numel():
                selected[candidates] = best_ids[candidates]
            return selected, proposed, int(candidates.numel())
        with torch.no_grad():
            codebooks = module.codebooks.to(torch.float16).to(torch.float32)
            scale_codebooks = torch.exp(module.log_scale_codebooks).to(
                torch.float16
            ).to(torch.float32)

            # Exact block-coordinate projection loss for gate and up.  Each
            # preactivation row is independent, and E is updated after every
            # input-position block, making its Hessian score exact.
            for projection, dense_weight in (
                (0, dense_gate_tensor),
                (1, dense_up_tensor),
            ):
                row_start = projection * intermediate_size
                projection_codes = module.codes[
                    row_start : row_start + intermediate_size
                ]
                projection_scale_indices = module.scale_indices[
                    row_start : row_start + intermediate_size
                ]
                scales = scale_codebooks[projection][projection_scale_indices]
                quant_weight = module.normalized_projection(
                    projection, codebooks
                ) * scales[:, None]
                error = x @ quant_weight.T - x @ dense_weight.T
                for stage in range(metadata.num_stages):
                    stage_codebook = codebooks[projection, stage]
                    for group in range(metadata.groups):
                        start = group * metadata.group_size
                        stop = start + metadata.group_size
                        x_group = padded_x[:, start:stop]
                        gradient = error.T @ x_group
                        hessian = x_group.T @ x_group
                        current_ids = projection_codes[:, stage, group]
                        current_vectors = stage_codebook[current_ids]
                        differences = (
                            stage_codebook[None, :, :] - current_vectors[:, None, :]
                        )
                        scaled_differences = differences * scales[:, None, None]
                        linear = 2.0 * torch.sum(
                            scaled_differences * gradient[:, None, :], dim=2
                        )
                        quadratic = torch.einsum(
                            "rkd,de,rke->rk",
                            scaled_differences,
                            hessian,
                            scaled_differences,
                        )
                        selected, proposed, applied = trust_limited_assignments(
                            current_ids, linear + quadratic
                        )
                        codes_proposed[projection] += proposed
                        codes_changed[projection] += applied
                        chosen_delta = scaled_differences[
                            torch.arange(intermediate_size, device=target_device),
                            selected,
                        ]
                        projection_codes[:, stage, group].copy_(selected)
                        error.add_(x_group @ chosen_delta.T)

                normalized = module.normalized_projection(projection, codebooks)
                base_projection = x @ normalized.T
                current_scales = scale_codebooks[projection][
                    projection_scale_indices
                ]
                scale_deltas = (
                    scale_codebooks[projection][None, :]
                    - current_scales[:, None]
                )
                linear_coeff = torch.sum(base_projection * error, dim=0)
                quadratic_coeff = torch.sum(base_projection**2, dim=0)
                scale_scores = (
                    2.0 * scale_deltas * linear_coeff[:, None]
                    + scale_deltas**2 * quadratic_coeff[:, None]
                )
                selected_scales, proposed, applied = trust_limited_assignments(
                    projection_scale_indices, scale_scores
                )
                scale_indices_proposed[projection] += proposed
                scale_indices_changed[projection] += applied
                projection_scale_indices.copy_(selected_scales)

            # The down score uses the exact current nonlinear activation and
            # output residual.  Rows are updated in parallel with the diagonal
            # of A^T A; the output residual is recomputed after each AQ stage.
            quant_gate, quant_up, quant_down = module.weights()
            activation = functional.silu(x @ quant_gate.T) * (x @ quant_up.T)
            output_error = activation @ quant_down.T - target_output
            projection = 2
            row_start = 2 * intermediate_size
            down_codes = module.codes[row_start : row_start + intermediate_size]
            down_scale_indices = module.scale_indices[
                row_start : row_start + intermediate_size
            ]
            down_scales = scale_codebooks[projection][down_scale_indices]
            activation_energy = torch.sum(activation**2, dim=0)
            for stage in range(metadata.num_stages):
                stage_codebook = codebooks[projection, stage]
                for group in range(metadata.groups):
                    start = group * metadata.group_size
                    stop = min(start + metadata.group_size, hidden_size)
                    width = stop - start
                    if width <= 0:
                        continue
                    current_ids = down_codes[:, stage, group]
                    current_vectors = stage_codebook[current_ids, :width]
                    differences = (
                        stage_codebook[None, :, :width]
                        - current_vectors[:, None, :]
                    )
                    scaled_differences = differences * down_scales[:, None, None]
                    gradient = activation.T @ output_error[:, start:stop]
                    linear = 2.0 * torch.sum(
                        scaled_differences * gradient[:, None, :], dim=2
                    )
                    quadratic = activation_energy[:, None] * torch.sum(
                        scaled_differences**2, dim=2
                    )
                    selected, proposed, applied = trust_limited_assignments(
                        current_ids, linear + quadratic
                    )
                    codes_proposed[projection] += proposed
                    codes_changed[projection] += applied
                    chosen_delta = scaled_differences[
                        torch.arange(intermediate_size, device=target_device),
                        selected,
                    ]
                    down_codes[:, stage, group].copy_(selected)
                    output_error[:, start:stop].add_(activation @ chosen_delta)

            normalized_down = module.normalized_projection(projection, codebooks)
            current_scales = scale_codebooks[projection][down_scale_indices]
            scale_deltas = (
                scale_codebooks[projection][None, :] - current_scales[:, None]
            )
            down_gradient = activation.T @ output_error
            linear_coeff = torch.sum(down_gradient * normalized_down, dim=1)
            quadratic_coeff = activation_energy * torch.sum(
                normalized_down**2, dim=1
            )
            scale_scores = (
                2.0 * scale_deltas * linear_coeff[:, None]
                + scale_deltas**2 * quadratic_coeff[:, None]
            )
            selected_scales, proposed, applied = trust_limited_assignments(
                down_scale_indices, scale_scores
            )
            scale_indices_proposed[projection] += proposed
            scale_indices_changed[projection] += applied
            down_scale_indices.copy_(selected_scales)

        return {
            "training_records": v_count,
            "maximum_change_fraction_per_coordinate_block": v_change_fraction,
            "code_changes_proposed_by_projection": codes_proposed,
            "code_changes_by_projection": codes_changed,
            "scale_index_changes_proposed_by_projection": scale_indices_proposed,
            "scale_index_changes_by_projection": scale_indices_changed,
            "total_code_changes_proposed": int(sum(codes_proposed)),
            "total_code_changes": int(sum(codes_changed)),
            "total_scale_index_changes_proposed": int(
                sum(scale_indices_proposed)
            ),
            "total_scale_index_changes": int(sum(scale_indices_changed)),
            "gate_up_score": (
                "exact_block_hessian_with_current_preactivation_residual"
            ),
            "down_score": (
                "exact_current_output_residual_with_activation_hessian_diagonal"
            ),
            "nonlinear_candidate_forwards": 0,
        }

    requested_directory = Path(artifact_dir) if artifact_dir is not None else None
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if requested_directory is None:
        temporary = tempfile.TemporaryDirectory(prefix="engram-projection-aq-")
        serialization_directory = Path(temporary.name)
    else:
        serialization_directory = requested_directory
        serialization_directory.mkdir(parents=True, exist_ok=True)

    cycle_reports: list[dict[str, Any]] = []
    try:
        baseline_path = serialization_directory / "initial.projection-aq"
        baseline_encoding, baseline_decoded, baseline_artifact = _strict_round_trip(
            initial_encoding, baseline_path
        )
        if baseline_artifact["fraction_of_dense_q4"] >= 0.45 and enforce_traffic_gate:
            raise RuntimeError("serialized initial artifact violates the 45% traffic gate")
        initial_training_metrics = _strict_metrics(
            training_x, training_y, baseline_decoded, gate, up, intermediate_size
        )
        initial_validation_metrics = _strict_metrics(
            validation_x,
            validation_y,
            baseline_decoded,
            gate,
            up,
            intermediate_size,
        )
        final_encoding = baseline_encoding
        final_decoded = baseline_decoded
        final_artifact = baseline_artifact

        for cycle in range(1, cycles + 1):
            before_training_objective = selection_objective()
            p_report = p_optimize(cycle)
            after_p_training_objective = selection_objective()
            # P is selected only on the fixed training subset.  Retain that
            # state if the subsequent approximate discrete sweep (especially
            # the diagonal-Hessian down update) does not improve the same
            # deployable objective.
            after_p = snapshot()
            v_report = v_reassign(cycle)
            after_v_training_objective = selection_objective()
            accepted = after_v_training_objective <= after_p_training_objective
            if not accepted:
                restore(after_p)
            selected_training_objective = selection_objective()
            cycle_path = serialization_directory / f"cycle_{cycle:02d}.projection-aq"
            final_encoding, final_decoded, final_artifact = _strict_round_trip(
                _encoding_from_module(module, initial_encoding), cycle_path
            )
            if final_artifact["fraction_of_dense_q4"] >= 0.45 and enforce_traffic_gate:
                raise RuntimeError(
                    f"serialized cycle {cycle} artifact violates the 45% traffic gate"
                )
            training_metrics = _strict_metrics(
                training_x,
                training_y,
                final_decoded,
                gate,
                up,
                intermediate_size,
            )
            validation_metrics = _strict_metrics(
                validation_x,
                validation_y,
                final_decoded,
                gate,
                up,
                intermediate_size,
            )
            cycle_reports.append(
                {
                    "cycle": cycle,
                    "v": v_report,
                    "p": p_report,
                    "selection": {
                        "before_training_objective": before_training_objective,
                        "after_v_training_objective": after_v_training_objective,
                        "after_p_training_objective": after_p_training_objective,
                        "selected_training_objective": selected_training_objective,
                        "cycle_update_accepted": accepted,
                        "source": "fixed_training_subset_never_validation",
                    },
                    "strict_reloaded_training": training_metrics,
                    "strict_reloaded_validation": validation_metrics,
                    "artifact": final_artifact,
                }
            )

        final_training_metrics = cycle_reports[-1]["strict_reloaded_training"]
        final_validation_metrics = cycle_reports[-1]["strict_reloaded_validation"]
        initial_mean = initial_validation_metrics["mean_relative_l2"]
        final_mean = final_validation_metrics["mean_relative_l2"]
        relative_improvement = (
            (initial_mean - final_mean) / initial_mean if initial_mean > 0 else 0.0
        )
        traffic_passed = final_artifact["fraction_of_dense_q4"] < 0.45
        quality_checks = {
            "mean_relative_l2": final_mean <= maximum_mean_relative_l2,
            "p95_relative_l2": (
                final_validation_metrics["p95_relative_l2"]
                <= maximum_p95_relative_l2
            ),
            "mean_cosine_similarity": (
                final_validation_metrics["mean_cosine_similarity"]
                >= minimum_mean_cosine
            ),
        }
        report: dict[str, Any] = {
            "schema_version": PROJECTION_AQ_BOUNDARY_SCHEMA_VERSION,
            "experiment": "projection_local_activation_aware_aq_boundaries",
            "codec_profile": PROJECTION_2X7_SCALE6,
            "configuration": {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "p_steps_per_cycle": p_steps,
                "v_cycles": cycles,
                "batch_size": minibatch_size,
                "learning_rate": learning_rate,
                "checkpoint_interval": interval,
                "fit_iterations": fit_rounds,
                "fit_sample_limit": fit_sample_limit,
                "v_max_records": v_max_records,
                "v_change_fraction": v_change_fraction,
                "selection_records": selection_count,
                "cosine_weight": cosine_weight,
                "gate_projection_weight": gate_projection_weight,
                "up_projection_weight": up_projection_weight,
                "seed": int(seed),
                "device": str(target_device),
            },
            "objective": (
                "output_nmse + 0.1*cosine_distance + "
                "0.025*gate_projection_nmse + 0.025*up_projection_nmse"
                if (
                    cosine_weight == 0.1
                    and gate_projection_weight == 0.025
                    and up_projection_weight == 0.025
                )
                else "configured_weighted_boundary_objective"
            ),
            "initial": {
                "strict_reloaded_training": initial_training_metrics,
                "strict_reloaded_validation": initial_validation_metrics,
                "training_objective": _objective_from_metrics(
                    initial_training_metrics,
                    cosine_weight,
                    gate_projection_weight,
                    up_projection_weight,
                ),
                "artifact": baseline_artifact,
            },
            "cycles": cycle_reports,
            "final": {
                "strict_reloaded_training": final_training_metrics,
                "strict_reloaded_validation": final_validation_metrics,
                "validation_mean_relative_l2_absolute_improvement": (
                    initial_mean - final_mean
                ),
                "validation_mean_relative_l2_fractional_improvement": (
                    relative_improvement
                ),
                "artifact": final_artifact,
            },
            "traffic": {
                "storage_components": final_encoding.storage_components(),
                "serialized_artifact_bytes": final_artifact["serialized_bytes"],
                "dense_q4_bytes": final_artifact["dense_q4_bytes"],
                "fraction_of_dense_q4": final_artifact[
                    "fraction_of_dense_q4"
                ],
                "strictly_below_45_percent": traffic_passed,
                "all_codebooks_and_scale_metadata_included": True,
            },
            "screen": {
                "criteria": {
                    "maximum_mean_relative_l2": maximum_mean_relative_l2,
                    "maximum_p95_relative_l2": maximum_p95_relative_l2,
                    "minimum_mean_cosine_similarity": minimum_mean_cosine,
                    "maximum_fraction_of_dense_q4": 0.45,
                },
                "checks": {**quality_checks, "cold_traffic": traffic_passed},
                "passed": traffic_passed and all(quality_checks.values()),
                "scope": "held_out_cached_mlp_boundary_not_causal_gate",
            },
            "validity": {
                "validation_was_not_used_for_checkpoint_selection": True,
                "all_reported_metrics_use_serialized_reloaded_decode": True,
                "oracle_is_not_deployable": True,
            },
        }
        final_path = (
            requested_directory / f"cycle_{cycles:02d}.projection-aq"
            if requested_directory is not None
            else None
        )
        return ProjectionAQBoundaryResult(
            encoding=final_encoding,
            report=report,
            artifact_path=final_path,
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = [
    "PROJECTION_AQ_BOUNDARY_SCHEMA_VERSION",
    "ProjectionAQBoundaryResult",
    "train_projection_aq_boundaries",
]
