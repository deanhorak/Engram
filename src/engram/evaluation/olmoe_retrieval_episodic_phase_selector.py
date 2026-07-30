"""Phase-conditioned causal selector for the full-visible C28 basis.

The native bounded-attention kernel already produces a normalized mass over
28 visible values per query head.  This selector retains the low-rank
mass-feature correction and adds a schedule-relative table:

``delta = gauge_clamp((relu(f @ U + E) @ V + B) + active * T[phase])``

``phase`` is the causal offset inside an active eight-token episodic read
span.  It is deliberately independent of absolute position and token
identity.  Outside an active span the table contribution is exactly zero.
No new key/value state, sidecar, or value-read pass is introduced.

This module contains the deterministic numerical and training reference.
Protocol authentication and experiment orchestration live in the companion
runner.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass_selector


_PRODUCTION_LAYERS = 16
_PRODUCTION_HEADS = 16
_PRODUCTION_COMPONENTS = 28
_PRODUCTION_HEAD_DIMENSION = 128
_PRODUCTION_RANK = 16
_PRODUCTION_PHASES = 8
_PRODUCTION_POSITIONS = 128
_PRODUCTION_FIXED_TRAFFIC_BYTES = 714_866_688
_PRODUCTION_DENSE_BYTES = 2_164_260_864
_PRODUCTION_EXACT_51_HEAD_CEILING_BYTES = 973_384_704
_PRODUCTION_BASE_STATE_BYTES = 10_534_912


float32_to_bf16_bits = mass_selector.float32_to_bf16_bits
bf16_bits_to_float32 = mass_selector.bf16_bits_to_float32


@dataclass(frozen=True)
class TrainingConfig(mass_selector.TrainingConfig):
    """Predecessor optimizer contract plus explicit phase-table decay."""

    t_weight_decay: float = 1.0e-4

    def validate(self) -> None:
        super().validate()
        if (
            not np.isfinite(self.t_weight_decay)
            or self.t_weight_decay < 0.0
        ):
            raise ValueError("phase-selector training configuration is invalid")


@dataclass(frozen=True)
class PhaseSelectorShape:
    """Static shape of one phase-conditioned selector artifact."""

    layers: int = _PRODUCTION_LAYERS
    heads: int = _PRODUCTION_HEADS
    components: int = _PRODUCTION_COMPONENTS
    head_dimension: int = _PRODUCTION_HEAD_DIMENSION
    rank: int = _PRODUCTION_RANK
    phases: int = _PRODUCTION_PHASES

    def validate(self) -> None:
        if (
            min(
                self.layers,
                self.heads,
                self.components,
                self.head_dimension,
                self.rank,
                self.phases,
            )
            <= 0
        ):
            raise ValueError("phase-selector dimensions must be positive")

    @property
    def hidden_size(self) -> int:
        self.validate()
        return self.heads * self.head_dimension

    @property
    def mass_shape(self) -> mass_selector.SelectorShape:
        self.validate()
        return mass_selector.SelectorShape(
            layers=self.layers,
            heads=self.heads,
            components=self.components,
            head_dimension=self.head_dimension,
            rank=self.rank,
        )

    @property
    def mass_parameter_count(self) -> int:
        return self.mass_shape.parameter_count

    @property
    def phase_parameter_count(self) -> int:
        self.validate()
        return self.phases * self.layers * self.heads * self.components

    @property
    def parameter_count(self) -> int:
        return self.mass_parameter_count + self.phase_parameter_count


@dataclass(frozen=True)
class PhaseSelectorParameters:
    """FP32 audit parameters.

    U/V/E/B are the predecessor mass branch. T is indexed by the causal
    offset within an active episodic read span.
    """

    U: np.ndarray
    V: np.ndarray
    E: np.ndarray
    B: np.ndarray
    T: np.ndarray

    def validate(self, shape: PhaseSelectorShape) -> None:
        shape.validate()
        expected = {
            "U": (shape.layers, shape.components, shape.rank),
            "V": (shape.layers, shape.rank, shape.components),
            "E": (shape.layers, shape.heads, shape.rank),
            "B": (shape.layers, shape.heads, shape.components),
            "T": (
                shape.phases,
                shape.layers,
                shape.heads,
                shape.components,
            ),
        }
        for name, value in self.as_dict().items():
            array = np.asarray(value)
            if (
                array.shape != expected[name]
                or array.dtype != np.float32
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"phase-selector {name} parameters are invalid")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "U": self.U,
            "V": self.V,
            "E": self.E,
            "B": self.B,
            "T": self.T,
        }


@dataclass(frozen=True)
class TrainingResult:
    """One deterministic final-step fit."""

    parameters: PhaseSelectorParameters
    initial_loss: float
    final_loss: float
    learning_rate_sha256: str
    schedule_sha256: str
    steps: int
    device: str
    train_mass_branch: bool


def initialize_parameters(
    shape: PhaseSelectorShape,
    *,
    seed: int,
    standard_deviation: float = 0.02,
) -> PhaseSelectorParameters:
    """Initialize at exact native attention with a live mass gradient path."""

    initialized = mass_selector.initialize_parameters(
        shape.mass_shape,
        seed=seed,
        standard_deviation=standard_deviation,
    )
    return parameters_from_mass(initialized, shape)


def parameters_from_mass(
    parameters: mass_selector.SelectorParameters,
    shape: PhaseSelectorShape,
) -> PhaseSelectorParameters:
    """Lift an authenticated mass branch and initialize its phase table to zero."""

    shape.validate()
    parameters.validate(shape.mass_shape)
    result = PhaseSelectorParameters(
        U=np.ascontiguousarray(parameters.U.copy(), dtype=np.float32),
        V=np.ascontiguousarray(parameters.V.copy(), dtype=np.float32),
        E=np.ascontiguousarray(parameters.E.copy(), dtype=np.float32),
        B=np.ascontiguousarray(parameters.B.copy(), dtype=np.float32),
        T=np.zeros(
            (
                shape.phases,
                shape.layers,
                shape.heads,
                shape.components,
            ),
            dtype=np.float32,
        ),
    )
    result.validate(shape)
    return result


def causal_read_phase(
    positions: np.ndarray,
    span_starts: np.ndarray,
    active: np.ndarray,
    *,
    phase_count: int = _PRODUCTION_PHASES,
) -> np.ndarray:
    """Derive a schedule-relative phase, invariant to joint position shifts.

    All three inputs must have the same shape.  Inactive rows return phase
    zero solely as a safe table index; the explicit ``active`` mask controls
    whether the table contributes.
    """

    position_array = np.asarray(positions)
    start_array = np.asarray(span_starts)
    active_array = np.asarray(active)
    if (
        position_array.shape != start_array.shape
        or position_array.shape != active_array.shape
        or position_array.dtype.kind not in "iu"
        or start_array.dtype.kind not in "iu"
        or active_array.dtype != np.bool_
        or isinstance(phase_count, bool)
        or not isinstance(phase_count, int)
        or phase_count <= 0
    ):
        raise ValueError("phase-selector read schedule is invalid")
    offset = position_array.astype(np.int64) - start_array.astype(np.int64)
    if np.any(active_array & ((offset < 0) | (offset >= phase_count))):
        raise ValueError("phase-selector active phase is out of range")
    phase = np.where(active_array, offset, np.int64(0))
    return np.ascontiguousarray(phase, dtype=np.int64)


def _validate_phase_inputs(
    native_mass: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    shape: PhaseSelectorShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mass = np.ascontiguousarray(native_mass, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    mass_selector.centered_log_mass_features(mass, mask, shape.mass_shape)
    phase_input = np.asarray(phase)
    active_input = np.asarray(active)
    prefix = mass.shape[:-3]
    if (
        phase_input.shape != prefix
        or active_input.shape != prefix
        or phase_input.dtype.kind not in "iu"
        or active_input.dtype != np.bool_
    ):
        raise ValueError("phase-selector phase or active mask shape is invalid")
    phase_array = np.ascontiguousarray(phase_input, dtype=np.int64)
    active_array = np.ascontiguousarray(active_input, dtype=bool)
    if np.any((phase_array < 0) | (phase_array >= shape.phases)):
        raise ValueError("phase-selector phase is out of range")
    return mass, mask, phase_array, active_array


def _selector_delta(
    features: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    parameters: PhaseSelectorParameters,
    *,
    delta_clamp: float,
) -> np.ndarray:
    """Apply the mass map and one phase-table row with FP32 accumulation."""

    hidden = np.einsum(
        "...lhc,lcr->...lhr",
        features,
        parameters.U,
        optimize=True,
    )
    hidden += parameters.E
    np.maximum(hidden, 0.0, out=hidden)
    raw_delta = np.einsum(
        "...lhr,lrc->...lhc",
        hidden,
        parameters.V,
        optimize=True,
    )
    raw_delta += parameters.B
    phase_delta = parameters.T[phase]
    raw_delta += np.where(
        active[..., None, None, None],
        phase_delta,
        np.float32(0.0),
    )
    return mass_selector._gauge_and_clamp(
        raw_delta,
        valid,
        clamp=delta_clamp,
    )


def selector_forward(
    native_mass: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    parameters: PhaseSelectorParameters,
    shape: PhaseSelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the cached-native-mass FP32 reference."""

    parameters.validate(shape)
    mass, mask, phase_array, active_array = _validate_phase_inputs(
        native_mass,
        valid,
        phase,
        active,
        shape,
    )
    features = mass_selector.centered_log_mass_features(
        mass,
        mask,
        shape.mass_shape,
        clip=feature_clip,
    )
    delta = _selector_delta(
        features,
        mask,
        phase_array,
        active_array,
        parameters,
        delta_clamp=delta_clamp,
    )
    coefficients = mass_selector.coefficients_from_mass(mass, mask, delta)
    return coefficients, delta


def selector_forward_from_scores(
    native_scores: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    parameters: PhaseSelectorParameters,
    shape: PhaseSelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the candidate fused native-score path."""

    parameters.validate(shape)
    scores = np.ascontiguousarray(native_scores, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    native_mass = mass_selector._masked_softmax(scores, mask)
    _, mask, phase_array, active_array = _validate_phase_inputs(
        native_mass,
        mask,
        phase,
        active,
        shape,
    )
    features = mass_selector.centered_score_features(
        scores,
        mask,
        shape.mass_shape,
        clip=feature_clip,
    )
    delta = _selector_delta(
        features,
        mask,
        phase_array,
        active_array,
        parameters,
        delta_clamp=delta_clamp,
    )
    coefficients = mass_selector.coefficients_from_scores(scores, mask, delta)
    return coefficients, delta


def quantize_parameters_bf16(
    parameters: PhaseSelectorParameters,
    shape: PhaseSelectorShape,
) -> tuple[PhaseSelectorParameters, dict[str, np.ndarray]]:
    """Serialize every deployed parameter with explicit BF16 RNE."""

    parameters.validate(shape)
    bits = {
        name: float32_to_bf16_bits(value)
        for name, value in parameters.as_dict().items()
    }
    decoded = PhaseSelectorParameters(
        **{
            name: bf16_bits_to_float32(bits[name])
            for name in ("U", "V", "E", "B", "T")
        }
    )
    decoded.validate(shape)
    return decoded, bits


def production_resource_contract(
    shape: PhaseSelectorShape = PhaseSelectorShape(),
) -> dict[str, Any]:
    """Return conservative all-token accounting under the exact 51-head cap."""

    if shape != PhaseSelectorShape():
        raise ValueError("production resource accounting requires production shape")
    parameter_count = shape.parameter_count
    parameter_bytes = parameter_count * 2
    weight_traffic = parameter_bytes * _PRODUCTION_POSITIONS
    total = _PRODUCTION_FIXED_TRAFFIC_BYTES + weight_traffic
    combined_state = _PRODUCTION_BASE_STATE_BYTES + parameter_bytes
    macs_per_token = (
        shape.layers
        * shape.heads
        * (shape.components * shape.rank + shape.rank * shape.components)
    )
    phase_additions_per_active_token = (
        shape.layers * shape.heads * shape.components
    )
    if (
        shape.mass_parameter_count != 25_600
        or shape.phase_parameter_count != 57_344
        or parameter_count != 82_944
        or parameter_bytes != 165_888
        or weight_traffic != 21_233_664
        or total != 736_100_352
        or combined_state != 10_700_800
        or macs_per_token != 229_376
        or phase_additions_per_active_token != 7_168
        or total >= _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES
    ):
        raise AssertionError("phase-selector production accounting changed")
    return {
        "parameter_count": parameter_count,
        "mass_selector_parameter_count": shape.mass_parameter_count,
        "phase_table_parameter_count": shape.phase_parameter_count,
        "serialized_parameter_dtype": "BF16",
        "serialized_parameter_bytes": parameter_bytes,
        "fixed_attention_state_bytes": _PRODUCTION_BASE_STATE_BYTES,
        "combined_attention_and_selector_state_bytes": combined_state,
        "mass_selector_multiply_accumulates_per_token": macs_per_token,
        "mass_selector_multiply_accumulates_per_128_token_sequence": (
            macs_per_token * _PRODUCTION_POSITIONS
        ),
        "phase_table_additions_per_active_token": phase_additions_per_active_token,
        "phase_table_operation": "one indexed BF16 row add per active token",
        "conservative_selector_weight_traffic_bytes_per_128_token_sequence": (
            weight_traffic
        ),
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _PRODUCTION_FIXED_TRAFFIC_BYTES
        ),
        "total_logical_traffic_bytes_per_128_token_sequence": total,
        "dense_full_context_logical_read_bytes": _PRODUCTION_DENSE_BYTES,
        "fraction_of_dense_full_context_logical_reads": (
            total / _PRODUCTION_DENSE_BYTES
        ),
        "exact_51_head_equivalent_ceiling_bytes": (
            _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES
        ),
        "remaining_headroom_below_exact_51_head_ceiling_bytes": (
            _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES - total
        ),
        "new_KV_state_bytes": 0,
        "new_KV_read_traffic_bytes": 0,
        "persistent_value_sidecar_bytes": 0,
        "single_full_value_accumulation_pass": True,
        "selector_weight_traffic_assumes_reload_for_every_token": True,
    }


def direct_post_wo_error_energy(
    coefficients: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    shape: PhaseSelectorShape,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact post-Wo recovery with the shared mass-selector routine."""

    return mass_selector.direct_post_wo_error_energy(
        coefficients,
        visible_values,
        base_heads,
        target_residual,
        output_projection,
        shape.mass_shape,
    )


def _torch_forward(
    torch: Any,
    native_mass: Any,
    valid: Any,
    phase: Any,
    active: Any,
    U: Any,
    V: Any,
    E: Any,
    B: Any,
    T: Any,
    *,
    feature_clip: float,
    delta_clamp: float,
) -> Any:
    """Torch forward with layout [layer, batch, head, component]."""

    log_mass = torch.where(
        valid,
        torch.log(native_mass),
        torch.zeros_like(native_mass),
    )
    count = valid.sum(dim=-1, keepdim=True)
    feature_mean = torch.where(valid, log_mass, 0.0).sum(
        dim=-1,
        keepdim=True,
    ) / count
    features = torch.where(
        valid,
        torch.clamp(log_mass - feature_mean, -feature_clip, feature_clip),
        0.0,
    )
    hidden = torch.relu(
        torch.einsum("lbhc,lcr->lbhr", features, U) + E[:, None]
    )
    raw_delta = torch.einsum("lbhr,lrc->lbhc", hidden, V) + B[:, None]
    # A direct advanced-index gather has a repeated-index scatter in its
    # backward pass on CUDA.  Use a tiny eight-way one-hot contraction during
    # training so deterministic-algorithm mode never depends on atomic scatter
    # order.  The deployment forward above remains one indexed table-row add.
    phase_features = torch.nn.functional.one_hot(
        phase,
        num_classes=T.shape[0],
    ).to(dtype=raw_delta.dtype)
    phase_delta = torch.einsum("lbp,plhc->lbhc", phase_features, T)
    raw_delta = raw_delta + torch.where(
        active[..., None, None],
        phase_delta,
        torch.zeros_like(phase_delta),
    )
    delta_mean = torch.where(valid, raw_delta, 0.0).sum(
        dim=-1,
        keepdim=True,
    ) / count
    delta = torch.where(
        valid,
        torch.clamp(raw_delta - delta_mean, -delta_clamp, delta_clamp),
        0.0,
    )
    logits = torch.where(
        valid,
        log_mass + delta,
        torch.full_like(log_mass, -torch.inf),
    )
    return torch.softmax(logits, dim=-1)


def fit_direct_post_wo(
    native_mass: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    *,
    training_records: Sequence[int],
    shape: PhaseSelectorShape,
    config: TrainingConfig,
    device: str = "cpu",
    initial_parameters: PhaseSelectorParameters | None = None,
    train_mass_branch: bool = True,
) -> TrainingResult:
    """Fit one fixed final-step model without consulting heldout records.

    Array layout is ``[records, reads, layers, ...]``.  Supplying an
    authenticated ``initial_parameters`` artifact and setting
    ``train_mass_branch=False`` leaves U/V/E/B byte-exact while fitting only
    T.  No choice between joint and frozen-mass training is made implicitly.
    """

    config.validate()
    shape.validate()
    mass, mask, phase_array, active_array = _validate_phase_inputs(
        native_mass,
        valid,
        phase,
        active,
        shape,
    )
    values = np.ascontiguousarray(visible_values, dtype=np.float32)
    base = np.ascontiguousarray(base_heads, dtype=np.float32)
    target = np.ascontiguousarray(target_residual, dtype=np.float32)
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if mass.ndim != 5:
        raise ValueError("phase-selector training masses must be rank five")
    records, reads = mass.shape[:2]
    expected_mass = (records, reads, shape.layers, shape.heads, shape.components)
    if (
        mass.shape != expected_mass
        or mask.shape != expected_mass
        or phase_array.shape != (records, reads)
        or active_array.shape != (records, reads)
        or values.shape != expected_mass + (shape.head_dimension,)
        or base.shape
        != (
            records,
            reads,
            shape.layers,
            shape.heads,
            shape.head_dimension,
        )
        or target.shape != (records, reads, shape.layers, shape.hidden_size)
        or weights.shape != (shape.layers, shape.hidden_size, shape.hidden_size)
        or not all(
            np.isfinite(array).all()
            for array in (values, base, target, weights)
        )
    ):
        raise ValueError("phase-selector training tensor shapes are invalid")
    if np.any(~mask):
        values = values.copy()
        values[~mask] = 0.0
    train = tuple(int(value) for value in training_records)
    if (
        not train
        or len(set(train)) != len(train)
        or min(train) < 0
        or max(train) >= records
    ):
        raise ValueError("phase-selector training records are invalid")
    rows_per_epoch = len(train) * reads
    if (
        rows_per_epoch % config.rows_per_layer_per_step
        or config.steps
        != config.epochs * rows_per_epoch // config.rows_per_layer_per_step
    ):
        raise ValueError("phase-selector fixed epoch schedule does not match data")
    if not isinstance(train_mass_branch, bool):
        raise ValueError("phase-selector mass-training flag is invalid")

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("phase-selector training requires torch") from error
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("phase-selector CUDA device is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError(
                "phase-selector deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.init_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.init_seed)
    torch_device = torch.device(device)

    initialized = (
        initialize_parameters(
            shape,
            seed=config.init_seed,
            standard_deviation=config.initial_u_standard_deviation,
        )
        if initial_parameters is None
        else initial_parameters
    )
    initialized.validate(shape)
    parameters = {
        name: torch.nn.Parameter(
            torch.from_numpy(value.copy()).to(torch_device),
            requires_grad=train_mass_branch or name == "T",
        )
        for name, value in initialized.as_dict().items()
    }
    if train_mass_branch:
        optimizer_groups = [
            {
                "params": [parameters["U"], parameters["V"]],
                "weight_decay": config.uv_weight_decay,
            },
            {
                "params": [parameters["E"], parameters["B"]],
                "weight_decay": 0.0,
            },
            {
                "params": [parameters["T"]],
                "weight_decay": config.t_weight_decay,
            },
        ]
    else:
        optimizer_groups = [
            {
                "params": [parameters["T"]],
                "weight_decay": config.t_weight_decay,
            }
        ]
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=config.peak_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
    )
    schedule = mass_selector.build_training_schedule(
        train,
        read_positions=reads,
        layers=shape.layers,
        epochs=config.epochs,
        rows_per_layer_per_step=config.rows_per_layer_per_step,
        seed=config.shuffle_seed,
    )
    rates = mass_selector.learning_rate_schedule(config)
    flat_mass = mass.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.components,
    )
    flat_mask = mask.reshape(flat_mass.shape)
    flat_phase = phase_array.reshape(records * reads)
    flat_active = active_array.reshape(records * reads)
    flat_values = values.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.components,
        shape.head_dimension,
    )
    flat_base = base.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.head_dimension,
    )
    flat_target = target.reshape(
        records * reads,
        shape.layers,
        shape.hidden_size,
    )
    train_target = target[np.asarray(train, dtype=np.int64)]
    target_denominator = float(
        np.mean(np.sum(train_target.astype(np.float64) ** 2, axis=-1))
    )
    if not np.isfinite(target_denominator) or target_denominator <= 0.0:
        raise ValueError("phase-selector training target energy is invalid")

    torch_mass = torch.from_numpy(flat_mass).to(torch_device)
    torch_mask = torch.from_numpy(flat_mask).to(torch_device)
    torch_phase = torch.from_numpy(flat_phase).to(torch_device)
    torch_active = torch.from_numpy(flat_active).to(torch_device)
    torch_values = torch.from_numpy(flat_values).to(torch_device)
    torch_base = torch.from_numpy(flat_base).to(torch_device)
    torch_target = torch.from_numpy(flat_target).to(torch_device)
    torch_weights = torch.from_numpy(weights).to(torch_device)
    torch_schedule = torch.from_numpy(schedule).to(torch_device)
    layer_indices = torch.arange(
        shape.layers,
        dtype=torch.long,
        device=torch_device,
    )[:, None]

    trainable_parameters = tuple(
        value for value in parameters.values() if value.requires_grad
    )
    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = float(rates[step])
        optimizer.zero_grad(set_to_none=True)
        row_ids = torch_schedule[step]
        batch_mass = torch_mass[row_ids, layer_indices]
        batch_mask = torch_mask[row_ids, layer_indices]
        batch_phase = torch_phase[row_ids]
        batch_active = torch_active[row_ids]
        batch_values = torch_values[row_ids, layer_indices]
        batch_base = torch_base[row_ids, layer_indices]
        batch_target = torch_target[row_ids, layer_indices]
        alpha = _torch_forward(
            torch,
            batch_mass,
            batch_mask,
            batch_phase,
            batch_active,
            parameters["U"],
            parameters["V"],
            parameters["E"],
            parameters["B"],
            parameters["T"],
            feature_clip=config.feature_clip,
            delta_clamp=config.delta_clamp,
        )
        candidate = torch.einsum("lbhc,lbhcd->lbhd", alpha, batch_values)
        pre_wo = (candidate - batch_base).reshape(
            shape.layers,
            config.rows_per_layer_per_step,
            shape.hidden_size,
        )
        correction = torch.bmm(pre_wo, torch_weights.transpose(1, 2))
        error = batch_target - correction
        total_error = torch.sum(error * error)
        total_rows = shape.layers * config.rows_per_layer_per_step
        loss = total_error / (total_rows * target_denominator)
        if not torch.isfinite(loss):
            raise ValueError("phase-selector training loss became non-finite")
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    fitted = PhaseSelectorParameters(
        **{
            name: np.ascontiguousarray(
                value.detach().cpu().numpy(),
                dtype=np.float32,
            )
            for name, value in parameters.items()
        }
    )
    fitted.validate(shape)
    if not train_mass_branch:
        for name in ("U", "V", "E", "B"):
            if not np.array_equal(
                fitted.as_dict()[name],
                initialized.as_dict()[name],
            ):
                raise AssertionError("frozen phase-selector mass branch changed")
    return TrainingResult(
        parameters=fitted,
        initial_loss=initial_loss,
        final_loss=final_loss,
        learning_rate_sha256=hashlib.sha256(rates.tobytes(order="C")).hexdigest(),
        schedule_sha256=mass_selector.schedule_sha256(schedule),
        steps=config.steps,
        device=str(torch_device),
        train_mass_branch=train_mass_branch,
    )


def fit_phase_table_direct_post_wo(
    native_mass: np.ndarray,
    valid: np.ndarray,
    phase: np.ndarray,
    active: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    *,
    base_parameters: (
        mass_selector.SelectorParameters | PhaseSelectorParameters
    ),
    training_records: Sequence[int],
    shape: PhaseSelectorShape,
    config: TrainingConfig,
    device: str = "cpu",
) -> TrainingResult:
    """Fit only T on top of a byte-frozen predecessor mass branch.

    A mass-selector artifact is lifted with an exact-zero phase table.  A
    phase-selector artifact is accepted only when its table is exactly zero,
    preventing accidental continuation from a selected phase checkpoint.
    """

    if isinstance(base_parameters, mass_selector.SelectorParameters):
        initialized = parameters_from_mass(base_parameters, shape)
    elif isinstance(base_parameters, PhaseSelectorParameters):
        base_parameters.validate(shape)
        if np.any(base_parameters.T != 0.0):
            raise ValueError("phase-table base parameters must have zero T")
        initialized = base_parameters
    else:
        raise TypeError("phase-table base parameters have an invalid type")
    return fit_direct_post_wo(
        native_mass,
        valid,
        phase,
        active,
        visible_values,
        base_heads,
        target_residual,
        output_projection,
        training_records=training_records,
        shape=shape,
        config=config,
        device=device,
        initial_parameters=initialized,
        train_mass_branch=False,
    )


def parameters_sha256(
    parameters: PhaseSelectorParameters,
    shape: PhaseSelectorShape,
) -> str:
    parameters.validate(shape)
    digest = hashlib.sha256()
    for name in ("U", "V", "E", "B", "T"):
        digest.update(name.encode("ascii"))
        digest.update(parameters.as_dict()[name].tobytes(order="C"))
    return digest.hexdigest()
