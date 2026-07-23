"""P/V boundary training for position-local 3x4-bit additive quantization.

This is the predeclared fallback to the projection-wide 2x7-bit codec.  Each
projection/input-position bucket owns a small three-stage, 16-entry codebook;
set selection is static and therefore adds no learned router or posting-list
traffic.  The canonical 36-bucket SmolLM2 layout contains 108 sets.

All reported metrics come from the checksum-validated serialized artifact after
reload and decode.  Validation data is observational only: P checkpoints and V
acceptance use a fixed subset of training states.
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
    POSITION_3X4_FP16_SCALE,
    MultiSetAdditiveEncoding,
    decode_multiset_additive,
    fit_multiset_additive,
    load_multiset_additive,
    save_multiset_additive,
)
from engram.training.projection_aq_boundaries import (
    _matrix,
    _objective_from_metrics,
    _pack_fixed_width,
    _positive_integer,
    _strict_metrics,
)


POSITION_AQ_BOUNDARY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PositionAQBoundaryResult:
    """Final strict position-local encoding and boundary-screen report."""

    encoding: MultiSetAdditiveEncoding
    report: dict[str, Any]
    artifact_path: Path | None


def _encoding_from_module(
    module: Any, initial: MultiSetAdditiveEncoding
) -> MultiSetAdditiveEncoding:
    metadata = initial.metadata
    with module.torch.no_grad():
        codebooks = (
            module.codebooks.detach().to(module.torch.float16).cpu().numpy()
        )
        row_scales = (
            module.torch.exp(module.log_row_scales)
            .detach()
            .to(module.torch.float16)
            .cpu()
            .numpy()
        )
        codes = module.codes.detach().cpu().numpy().astype(np.uint16)
    encoding = MultiSetAdditiveEncoding(
        packed_codes=_pack_fixed_width(codes, metadata.code_bits),
        codebooks=np.ascontiguousarray(codebooks),
        metadata=metadata,
        row_scales=np.ascontiguousarray(row_scales),
        set_mapping=np.ascontiguousarray(initial.set_mapping),
    )
    encoding.validate()
    return encoding


def _strict_round_trip(
    encoding: MultiSetAdditiveEncoding, path: Path
) -> tuple[MultiSetAdditiveEncoding, NDArray[np.float32], dict[str, Any]]:
    checksum = save_multiset_additive(path, encoding)
    serialized_bytes = path.stat().st_size
    if serialized_bytes != encoding.storage_bytes:
        raise RuntimeError("serialized position-AQ bytes disagree with accounting")
    reloaded = load_multiset_additive(path)
    decoded = decode_multiset_additive(reloaded)
    exact = (
        np.array_equal(encoding.packed_codes, reloaded.packed_codes)
        and np.array_equal(encoding.codebooks, reloaded.codebooks)
        and np.array_equal(encoding.row_scales, reloaded.row_scales)
        and np.array_equal(encoding.set_mapping, reloaded.set_mapping)
        and np.array_equal(decode_multiset_additive(encoding), decoded)
    )
    if not exact:
        raise RuntimeError("position-local AQ failed its exact packed round trip")
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


def train_position_aq_boundaries(
    gate_weight: ArrayLike,
    up_weight: ArrayLike,
    down_weight: ArrayLike,
    train_inputs: ArrayLike,
    train_outputs: ArrayLike,
    validation_inputs: ArrayLike,
    validation_outputs: ArrayLike,
    *,
    artifact_dir: str | Path | None = None,
    position_buckets: int = 36,
    p_steps_per_cycle: int = 64,
    v_cycles: int = 2,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    checkpoint_interval: int = 16,
    fit_iterations: int = 12,
    fit_sample_limit: int | None = 65_536,
    v_max_records: int | None = 1024,
    v_trust_fraction: float = 0.01,
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
) -> PositionAQBoundaryResult:
    """Fit a static position-local additive artifact to one MLP boundary."""

    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "position AQ training requires torch; install engram-lm[conversion]"
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

    buckets = _positive_integer(position_buckets, "position_buckets")
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
    for value, name in (
        (cosine_weight, "cosine_weight"),
        (gate_projection_weight, "gate_projection_weight"),
        (up_projection_weight, "up_projection_weight"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if not np.isfinite(v_trust_fraction) or not 0 < v_trust_fraction <= 1:
        raise ValueError("v_trust_fraction must be within (0, 1]")
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
        profile=POSITION_3X4_FP16_SCALE,
        position_buckets=buckets,
        iterations=fit_rounds,
        sample_limit=fit_sample_limit,
        seed=int(seed),
    )
    metadata = initial_encoding.metadata
    expected_shape = (3 * buckets, 3, 16, 8)
    if initial_encoding.codebooks.shape != expected_shape:
        raise RuntimeError(
            f"position codec returned {initial_encoding.codebooks.shape}, "
            f"expected {expected_shape}"
        )
    if metadata.fraction_of_dense_q4 >= 0.45 and enforce_traffic_gate:
        raise ValueError(
            "the complete serialized codec is not below 45% of dense Q4 for "
            f"this shape ({metadata.fraction_of_dense_q4:.6f})"
        )
    assert initial_encoding.row_scales is not None
    assert initial_encoding.set_mapping is not None
    minimum_scale = float(np.nextafter(np.float16(0), np.float16(1)))
    initial_scales = np.maximum(
        initial_encoding.row_scales.astype(np.float32), minimum_scale
    )
    initial_codes = initial_encoding.unpack_codes().astype(np.int64)
    mapping = initial_encoding.set_mapping.astype(np.int64)

    class PositionAQModule(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.torch = torch
            self.codebooks = torch.nn.Parameter(
                torch.from_numpy(initial_encoding.codebooks.astype(np.float32))
            )
            self.log_row_scales = torch.nn.Parameter(
                torch.log(torch.from_numpy(initial_scales))
            )
            self.register_buffer("codes", torch.from_numpy(initial_codes))
            self.register_buffer("set_mapping", torch.from_numpy(mapping))

        @staticmethod
        def fake_float16(value: Any) -> Any:
            rounded = value.to(torch.float16).to(torch.float32)
            return value + (rounded - value).detach()

        def stored_parameters(self) -> tuple[Any, Any]:
            return (
                self.fake_float16(self.codebooks),
                self.fake_float16(torch.exp(self.log_row_scales)),
            )

        def normalized_projection(
            self, projection: int, codebooks: Any | None = None
        ) -> Any:
            if codebooks is None:
                codebooks = self.stored_parameters()[0]
            start = projection * intermediate_size
            projection_codes = self.codes[start : start + intermediate_size]
            set_ids = self.set_mapping[projection]
            reconstructed = torch.zeros(
                (intermediate_size, metadata.groups, metadata.group_size),
                dtype=torch.float32,
                device=target_device,
            )
            for stage in range(metadata.num_stages):
                reconstructed = reconstructed + codebooks[
                    set_ids[None, :], stage, projection_codes[:, stage, :]
                ]
            return reconstructed.reshape(intermediate_size, metadata.padded_width)[
                :, :hidden_size
            ]

        def weights(self) -> tuple[Any, Any, Any]:
            codebooks, row_scales = self.stored_parameters()
            projections = []
            for projection in range(3):
                start = projection * intermediate_size
                scales = row_scales[start : start + intermediate_size]
                projections.append(
                    self.normalized_projection(projection, codebooks) * scales[:, None]
                )
            return projections[0], projections[1], projections[2].T

    module = PositionAQModule().to(target_device)
    train_x_tensor = torch.from_numpy(training_x).to(target_device)
    train_y_tensor = torch.from_numpy(training_y).to(target_device)
    dense_gate_tensor = torch.from_numpy(gate).to(target_device)
    dense_up_tensor = torch.from_numpy(up).to(target_device)

    def torch_objective(inputs: Any, targets: Any) -> Any:
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
        return (
            output_nmse
            + cosine_weight * cosine_distance
            + gate_projection_weight * gate_nmse
            + up_projection_weight * up_nmse
        )

    selection_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x50534C]))
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
            return float(torch_objective(selection_x, selection_y).cpu())

    def snapshot() -> tuple[Any, Any, Any]:
        return (
            module.codebooks.detach().clone(),
            module.log_row_scales.detach().clone(),
            module.codes.detach().clone(),
        )

    def restore(state: tuple[Any, Any, Any]) -> None:
        with torch.no_grad():
            module.codebooks.copy_(state[0])
            module.log_row_scales.copy_(state[1])
            module.codes.copy_(state[2])

    def p_optimize(cycle: int) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            (module.codebooks, module.log_row_scales), lr=learning_rate
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x505354, cycle])
        )
        order = rng.permutation(training_x.shape[0])
        cursor = 0
        initial_value = selection_objective()
        best_value = initial_value
        best_parameters = (
            module.codebooks.detach().clone(),
            module.log_row_scales.detach().clone(),
        )
        checkpoints: list[dict[str, float | int]] = [
            {"step": 0, "selection_training_objective": initial_value}
        ]
        losses: list[float] = []
        module.train()
        for step in range(1, p_steps + 1):
            if cursor + minibatch_size > order.size:
                order = rng.permutation(training_x.shape[0])
                cursor = 0
            ids = order[cursor : cursor + min(minibatch_size, order.size)]
            cursor += ids.size
            ids_tensor = torch.from_numpy(ids.astype(np.int64)).to(target_device)
            loss = torch_objective(
                train_x_tensor.index_select(0, ids_tensor),
                train_y_tensor.index_select(0, ids_tensor),
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"position-AQ P loss became non-finite in cycle {cycle}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (module.codebooks, module.log_row_scales), 1.0
            )
            optimizer.step()
            with torch.no_grad():
                module.codebooks.clamp_(-65_504.0, 65_504.0)
                module.log_row_scales.clamp_(
                    math.log(minimum_scale), math.log(65_504.0)
                )
            losses.append(float(loss.detach().cpu()))
            if step % interval == 0 or step == p_steps:
                current = selection_objective()
                checkpoints.append(
                    {"step": step, "selection_training_objective": current}
                )
                if current < best_value:
                    best_value = current
                    best_parameters = (
                        module.codebooks.detach().clone(),
                        module.log_row_scales.detach().clone(),
                    )
                module.train()
        with torch.no_grad():
            module.codebooks.copy_(best_parameters[0])
            module.log_row_scales.copy_(best_parameters[1])
        return {
            "initial_selection_training_objective": initial_value,
            "best_selection_training_objective": best_value,
            "initial_minibatch_loss": losses[0],
            "final_minibatch_loss": losses[-1],
            "checkpoints": checkpoints,
            "fixed_codes": True,
        }

    def v_reassign(cycle: int) -> dict[str, Any]:
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, 0x565354, cycle])
        )
        count = min(
            training_x.shape[0],
            training_x.shape[0] if v_max_records is None else v_max_records,
        )
        ids = np.sort(rng.choice(training_x.shape[0], count, replace=False)).astype(
            np.int64
        )
        ids_tensor = torch.from_numpy(ids).to(target_device)
        x = train_x_tensor.index_select(0, ids_tensor)
        target_output = train_y_tensor.index_select(0, ids_tensor)
        padded_x = (
            x
            if metadata.padded_width == hidden_size
            else functional.pad(x, (0, metadata.padded_width - hidden_size))
        )
        original_codes = module.codes.detach().cpu().numpy().astype(np.int64)
        proposed_codes = original_codes.copy()
        proposal_margins = np.zeros(original_codes.shape, dtype=np.float32)
        with torch.no_grad():
            codebooks = module.codebooks.to(torch.float16).to(torch.float32)
            row_scales = torch.exp(module.log_row_scales).to(torch.float16).to(
                torch.float32
            )
            for projection, dense_weight in (
                (0, dense_gate_tensor),
                (1, dense_up_tensor),
            ):
                row_start = projection * intermediate_size
                projection_codes = module.codes[
                    row_start : row_start + intermediate_size
                ]
                scales = row_scales[row_start : row_start + intermediate_size]
                quant_weight = module.normalized_projection(
                    projection, codebooks
                ) * scales[:, None]
                error = x @ quant_weight.T - x @ dense_weight.T
                target_energy = torch.clamp(
                    torch.sum((x @ dense_weight.T) ** 2),
                    min=torch.finfo(torch.float32).eps,
                )
                projection_weight = (
                    gate_projection_weight if projection == 0 else up_projection_weight
                )
                for stage in range(metadata.num_stages):
                    for group in range(metadata.groups):
                        start = group * metadata.group_size
                        stop = start + metadata.group_size
                        x_group = padded_x[:, start:stop]
                        gradient = error.T @ x_group
                        hessian = x_group.T @ x_group
                        set_id = module.set_mapping[projection, group]
                        stage_codebook = codebooks[set_id, stage]
                        current_ids = projection_codes[:, stage, group]
                        current_vectors = stage_codebook[current_ids]
                        differences = (
                            stage_codebook[None, :, :] - current_vectors[:, None, :]
                        ) * scales[:, None, None]
                        scores = 2.0 * torch.sum(
                            differences * gradient[:, None, :], dim=2
                        ) + torch.einsum(
                            "rkd,de,rke->rk", differences, hessian, differences
                        )
                        selected = torch.argmin(scores, dim=1)
                        best_scores = scores.gather(1, selected[:, None])[:, 0]
                        proposed_codes[
                            row_start : row_start + intermediate_size, stage, group
                        ] = selected.cpu().numpy()
                        proposal_margins[
                            row_start : row_start + intermediate_size, stage, group
                        ] = (
                            torch.clamp(-best_scores, min=0)
                            * projection_weight
                            / target_energy
                        ).cpu().numpy()

            quant_gate, quant_up, quant_down = module.weights()
            activation = functional.silu(x @ quant_gate.T) * (x @ quant_up.T)
            output_error = activation @ quant_down.T - target_output
            target_output_energy = torch.clamp(
                torch.sum(target_output**2), min=torch.finfo(torch.float32).eps
            )
            projection = 2
            row_start = 2 * intermediate_size
            down_codes = module.codes[row_start : row_start + intermediate_size]
            down_scales = row_scales[row_start : row_start + intermediate_size]
            activation_energy = torch.sum(activation**2, dim=0)
            for stage in range(metadata.num_stages):
                for group in range(metadata.groups):
                    start = group * metadata.group_size
                    stop = min(start + metadata.group_size, hidden_size)
                    width = stop - start
                    if width <= 0:
                        continue
                    set_id = module.set_mapping[projection, group]
                    stage_codebook = codebooks[set_id, stage, :, :width]
                    current_ids = down_codes[:, stage, group]
                    current_vectors = stage_codebook[current_ids]
                    differences = (
                        stage_codebook[None, :, :] - current_vectors[:, None, :]
                    ) * down_scales[:, None, None]
                    gradient = activation.T @ output_error[:, start:stop]
                    scores = 2.0 * torch.sum(
                        differences * gradient[:, None, :], dim=2
                    ) + activation_energy[:, None] * torch.sum(
                        differences**2, dim=2
                    )
                    selected = torch.argmin(scores, dim=1)
                    best_scores = scores.gather(1, selected[:, None])[:, 0]
                    proposed_codes[
                        row_start : row_start + intermediate_size, stage, group
                    ] = selected.cpu().numpy()
                    proposal_margins[
                        row_start : row_start + intermediate_size, stage, group
                    ] = (
                        torch.clamp(-best_scores, min=0) / target_output_energy
                    ).cpu().numpy()

            changed = (proposed_codes != original_codes) & (proposal_margins > 0)
            proposed_flat = np.flatnonzero(changed.reshape(-1))
            trust_limit = max(
                1, int(math.floor(original_codes.size * v_trust_fraction))
            )
            applied_count = min(trust_limit, proposed_flat.size)
            if applied_count:
                flat_margins = proposal_margins.reshape(-1)
                if proposed_flat.size > applied_count:
                    local = np.argpartition(
                        -flat_margins[proposed_flat], applied_count - 1
                    )[:applied_count]
                    applied_flat = proposed_flat[local]
                else:
                    applied_flat = proposed_flat
                # Stable margin/index ordering makes ties reproducible and
                # simplifies artifact comparisons in deterministic tests.
                order = np.lexsort(
                    (applied_flat, -flat_margins[applied_flat])
                )
                applied_flat = applied_flat[order]
                updated = original_codes.reshape(-1).copy()
                proposed_values = proposed_codes.reshape(-1)
                updated[applied_flat] = proposed_values[applied_flat]
                module.codes.copy_(
                    torch.from_numpy(updated.reshape(original_codes.shape)).to(
                        target_device
                    )
                )
                applied_margins = flat_margins[applied_flat]
            else:
                applied_margins = np.empty(0, dtype=np.float32)

        proposed_by_projection = []
        applied_by_projection = []
        final_codes = module.codes.detach().cpu().numpy()
        for projection in range(3):
            start = projection * intermediate_size
            stop = start + intermediate_size
            proposed_by_projection.append(int(changed[start:stop].sum()))
            applied_by_projection.append(
                int((final_codes[start:stop] != original_codes[start:stop]).sum())
            )

        return {
            "training_records": count,
            "proposed_code_changes_by_projection": proposed_by_projection,
            "applied_code_changes_by_projection": applied_by_projection,
            "total_proposed_code_changes": int(proposed_flat.size),
            "total_applied_code_changes": int(applied_count),
            "total_code_assignments": int(original_codes.size),
            "trust_fraction": v_trust_fraction,
            "trust_limit_assignments": trust_limit,
            "strongest_applied_normalized_margin": (
                float(applied_margins[0]) if applied_margins.size else 0.0
            ),
            "weakest_applied_normalized_margin": (
                float(applied_margins[-1]) if applied_margins.size else 0.0
            ),
            "candidate_count_per_assignment": 16,
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
        temporary = tempfile.TemporaryDirectory(prefix="engram-position-aq-")
        serialization_directory = Path(temporary.name)
    else:
        serialization_directory = requested_directory
        serialization_directory.mkdir(parents=True, exist_ok=True)

    cycle_reports: list[dict[str, Any]] = []
    try:
        baseline_encoding, baseline_decoded, baseline_artifact = _strict_round_trip(
            initial_encoding, serialization_directory / "initial.position-aq"
        )
        if baseline_artifact["fraction_of_dense_q4"] >= 0.45 and enforce_traffic_gate:
            raise RuntimeError("serialized initial artifact violates the 45% traffic gate")
        initial_training = _strict_metrics(
            training_x, training_y, baseline_decoded, gate, up, intermediate_size
        )
        initial_validation = _strict_metrics(
            validation_x,
            validation_y,
            baseline_decoded,
            gate,
            up,
            intermediate_size,
        )
        final_encoding = baseline_encoding
        final_artifact = baseline_artifact
        for cycle in range(1, cycles + 1):
            before = selection_objective()
            p_report = p_optimize(cycle)
            after_p = selection_objective()
            p_state = snapshot()
            v_report = v_reassign(cycle)
            after_v = selection_objective()
            accepted = after_v <= after_p
            v_report["tentatively_applied_code_changes_by_projection"] = v_report[
                "applied_code_changes_by_projection"
            ]
            v_report["total_tentatively_applied_code_changes"] = v_report[
                "total_applied_code_changes"
            ]
            if not accepted:
                restore(p_state)
                v_report["applied_code_changes_by_projection"] = [0, 0, 0]
                v_report["total_applied_code_changes"] = 0
            v_report["retained_after_training_acceptance"] = accepted
            selected = selection_objective()
            final_encoding, decoded, final_artifact = _strict_round_trip(
                _encoding_from_module(module, initial_encoding),
                serialization_directory / f"cycle_{cycle:02d}.position-aq",
            )
            if final_artifact["fraction_of_dense_q4"] >= 0.45 and enforce_traffic_gate:
                raise RuntimeError(
                    f"serialized cycle {cycle} artifact violates the 45% traffic gate"
                )
            strict_training = _strict_metrics(
                training_x, training_y, decoded, gate, up, intermediate_size
            )
            strict_validation = _strict_metrics(
                validation_x, validation_y, decoded, gate, up, intermediate_size
            )
            cycle_reports.append(
                {
                    "cycle": cycle,
                    "p": p_report,
                    "v": v_report,
                    "selection": {
                        "before_training_objective": before,
                        "after_p_training_objective": after_p,
                        "after_v_training_objective": after_v,
                        "selected_training_objective": selected,
                        "v_update_accepted": accepted,
                        "source": "fixed_training_subset_never_validation",
                    },
                    "strict_reloaded_training": strict_training,
                    "strict_reloaded_validation": strict_validation,
                    "artifact": final_artifact,
                }
            )

        final_training = cycle_reports[-1]["strict_reloaded_training"]
        final_validation = cycle_reports[-1]["strict_reloaded_validation"]
        initial_mean = initial_validation["mean_relative_l2"]
        final_mean = final_validation["mean_relative_l2"]
        improvement = (
            (initial_mean - final_mean) / initial_mean if initial_mean > 0 else 0.0
        )
        traffic_passed = final_artifact["fraction_of_dense_q4"] < 0.45
        quality = {
            "mean_relative_l2": final_mean <= maximum_mean_relative_l2,
            "p95_relative_l2": (
                final_validation["p95_relative_l2"] <= maximum_p95_relative_l2
            ),
            "mean_cosine_similarity": (
                final_validation["mean_cosine_similarity"] >= minimum_mean_cosine
            ),
        }
        report: dict[str, Any] = {
            "schema_version": POSITION_AQ_BOUNDARY_SCHEMA_VERSION,
            "experiment": "position_local_activation_aware_aq_boundaries",
            "codec_profile": POSITION_3X4_FP16_SCALE,
            "configuration": {
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "position_buckets": buckets,
                "num_codebook_sets": metadata.num_codebook_sets,
                "p_steps_per_cycle": p_steps,
                "v_cycles": cycles,
                "batch_size": minibatch_size,
                "learning_rate": learning_rate,
                "checkpoint_interval": interval,
                "fit_iterations": fit_rounds,
                "fit_sample_limit": fit_sample_limit,
                "v_max_records": v_max_records,
                "v_trust_fraction": v_trust_fraction,
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
                "strict_reloaded_training": initial_training,
                "strict_reloaded_validation": initial_validation,
                "training_objective": _objective_from_metrics(
                    initial_training,
                    cosine_weight,
                    gate_projection_weight,
                    up_projection_weight,
                ),
                "artifact": baseline_artifact,
            },
            "cycles": cycle_reports,
            "final": {
                "strict_reloaded_training": final_training,
                "strict_reloaded_validation": final_validation,
                "validation_mean_relative_l2_absolute_improvement": (
                    initial_mean - final_mean
                ),
                "validation_mean_relative_l2_fractional_improvement": improvement,
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
                "all_codebooks_scales_and_set_mapping_included": True,
            },
            "screen": {
                "criteria": {
                    "maximum_mean_relative_l2": maximum_mean_relative_l2,
                    "maximum_p95_relative_l2": maximum_p95_relative_l2,
                    "minimum_mean_cosine_similarity": minimum_mean_cosine,
                    "maximum_fraction_of_dense_q4": 0.45,
                },
                "checks": {**quality, "cold_traffic": traffic_passed},
                "passed": traffic_passed and all(quality.values()),
                "scope": "held_out_cached_mlp_boundary_not_causal_gate",
            },
            "validity": {
                "validation_was_not_used_for_checkpoint_selection": True,
                "all_reported_metrics_use_serialized_reloaded_decode": True,
                "set_selection_is_static": True,
                "oracle_is_not_deployable": True,
            },
        }
        final_path = (
            requested_directory / f"cycle_{cycles:02d}.position-aq"
            if requested_directory is not None
            else None
        )
        return PositionAQBoundaryResult(
            encoding=final_encoding, report=report, artifact_path=final_path
        )
    finally:
        if temporary is not None:
            temporary.cleanup()


__all__ = [
    "POSITION_AQ_BOUNDARY_SCHEMA_VERSION",
    "PositionAQBoundaryResult",
    "train_position_aq_boundaries",
]
