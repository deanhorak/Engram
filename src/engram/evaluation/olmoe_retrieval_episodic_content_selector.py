"""Content-aware causal selector for the full-visible C28 attention basis.

The mass-only selector can see which entries native attention preferred, but
not what those entries contain.  This selector keeps that inexpensive branch
and adds a low-rank bilinear interaction between each already-computed
post-query-normalization, pre-RoPE native query and a BF16 sidecar stored with
each already-cached value.

Production execution remains one-pass:

* project the current post-query-normalization, pre-RoPE query to four FP32
  features;
* project each value once when it enters a cache and store four BF16 values;
* score the 28 entries from native scores, the mass branch, and sidecars;
* accumulate the selected values once with the corrected softmax weights.

No additional key/value vector is stored or read.  The sidecar and selector
weight traffic are counted conservatively against the historical exact
51-head logical-read ceiling.
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import engram.evaluation.olmoe_retrieval_episodic_mass_selector as mass_selector


_PRODUCTION_LAYERS = 16
_PRODUCTION_HEADS = 16
_PRODUCTION_COMPONENTS = 28
_PRODUCTION_HEAD_DIMENSION = 128
_PRODUCTION_MASS_RANK = 16
_PRODUCTION_CONTENT_RANK = 4
_PRODUCTION_POSITIONS = 128
_PRODUCTION_FIXED_TRAFFIC_BYTES = 714_866_688
_PRODUCTION_DENSE_BYTES = 2_164_260_864
_PRODUCTION_EXACT_51_HEAD_CEILING_BYTES = 973_384_704
_PRODUCTION_BASE_STATE_BYTES = 10_534_912


@dataclass(frozen=True)
class ContentSelectorShape:
    """Static shape of the deployable selector."""

    layers: int = _PRODUCTION_LAYERS
    heads: int = _PRODUCTION_HEADS
    components: int = _PRODUCTION_COMPONENTS
    head_dimension: int = _PRODUCTION_HEAD_DIMENSION
    mass_rank: int = _PRODUCTION_MASS_RANK
    content_rank: int = _PRODUCTION_CONTENT_RANK

    def validate(self) -> None:
        if (
            min(
                self.layers,
                self.heads,
                self.components,
                self.head_dimension,
                self.mass_rank,
                self.content_rank,
            )
            <= 0
        ):
            raise ValueError("content-selector dimensions must be positive")

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
            rank=self.mass_rank,
        )

    @property
    def parameter_count(self) -> int:
        self.validate()
        return (
            self.mass_shape.parameter_count
            + 2
            * self.layers
            * self.heads
            * self.head_dimension
            * self.content_rank
        )


@dataclass(frozen=True)
class ContentSelectorParameters:
    """FP32 audit parameters.

    U/V/E/B are the complementary mass branch. Q and P are independent
    per-layer/per-head query and value projections.
    """

    U: np.ndarray
    V: np.ndarray
    E: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    P: np.ndarray

    def validate(self, shape: ContentSelectorShape) -> None:
        shape.validate()
        expected = {
            "U": (shape.layers, shape.components, shape.mass_rank),
            "V": (shape.layers, shape.mass_rank, shape.components),
            "E": (shape.layers, shape.heads, shape.mass_rank),
            "B": (shape.layers, shape.heads, shape.components),
            "Q": (
                shape.layers,
                shape.heads,
                shape.head_dimension,
                shape.content_rank,
            ),
            "P": (
                shape.layers,
                shape.heads,
                shape.head_dimension,
                shape.content_rank,
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
                raise ValueError(f"content-selector {name} parameters are invalid")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "U": self.U,
            "V": self.V,
            "E": self.E,
            "B": self.B,
            "Q": self.Q,
            "P": self.P,
        }


@dataclass(frozen=True)
class TrainingResult:
    """One deterministic final-step fit."""

    parameters: ContentSelectorParameters
    initial_loss: float
    final_loss: float
    learning_rate_sha256: str
    schedule_sha256: str
    steps: int
    device: str


def initialize_parameters(
    shape: ContentSelectorShape,
    *,
    seed: int,
    standard_deviation: float = 0.02,
) -> ContentSelectorParameters:
    """Initialize at exact native attention with live gradient paths."""

    shape.validate()
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not np.isfinite(standard_deviation)
        or standard_deviation <= 0.0
    ):
        raise ValueError("content-selector initialization is invalid")
    generator = np.random.Generator(np.random.PCG64(seed))
    U = generator.normal(
        0.0,
        standard_deviation,
        size=(shape.layers, shape.components, shape.mass_rank),
    ).astype(np.float32)
    Q = generator.normal(
        0.0,
        standard_deviation,
        size=(
            shape.layers,
            shape.heads,
            shape.head_dimension,
            shape.content_rank,
        ),
    ).astype(np.float32)
    return ContentSelectorParameters(
        U=np.ascontiguousarray(U),
        V=np.zeros(
            (shape.layers, shape.mass_rank, shape.components),
            dtype=np.float32,
        ),
        E=np.zeros(
            (shape.layers, shape.heads, shape.mass_rank),
            dtype=np.float32,
        ),
        B=np.zeros(
            (shape.layers, shape.heads, shape.components),
            dtype=np.float32,
        ),
        Q=np.ascontiguousarray(Q),
        P=np.zeros(
            (
                shape.layers,
                shape.heads,
                shape.head_dimension,
                shape.content_rank,
            ),
            dtype=np.float32,
        ),
    )


def _validate_content_inputs(
    native_mass: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
    visible_values: np.ndarray,
    shape: ContentSelectorShape,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mass = np.ascontiguousarray(native_mass, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    query_array = np.ascontiguousarray(query, dtype=np.float32)
    values = np.ascontiguousarray(visible_values, dtype=np.float32)
    mass_selector.centered_log_mass_features(
        mass,
        mask,
        shape.mass_shape,
    )
    prefix = mass.shape[:-3]
    if (
        query_array.shape
        != prefix
        + (
            shape.layers,
            shape.heads,
            shape.head_dimension,
        )
        or values.shape
        != prefix
        + (
            shape.layers,
            shape.heads,
            shape.components,
            shape.head_dimension,
        )
        or not np.isfinite(query_array).all()
        or not np.isfinite(values).all()
    ):
        raise ValueError("content-selector query or value tensors are invalid")
    # Native trace buffers may retain finite payload bytes in a padded older
    # slot even though its valid-kind bit is zero.  Production never reads
    # such a value.  Canonicalize it here so the cached reference models the
    # masked lifecycle rather than depending on stale trace scratch.
    if np.any(~mask):
        values = values.copy()
        values[~mask] = 0.0
    return mass, mask, query_array, values


def _selector_delta(
    features: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
    visible_values: np.ndarray,
    parameters: ContentSelectorParameters,
    *,
    delta_clamp: float,
    quantize_sidecars: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mass_hidden = np.einsum(
        "...lhc,lcr->...lhr",
        features,
        parameters.U,
        optimize=True,
    )
    mass_hidden += parameters.E
    np.maximum(mass_hidden, 0.0, out=mass_hidden)
    raw_delta = np.einsum(
        "...lhr,lrc->...lhc",
        mass_hidden,
        parameters.V,
        optimize=True,
    )
    raw_delta += parameters.B

    query_features = np.einsum(
        "...lhd,lhdr->...lhr",
        query,
        parameters.Q,
        optimize=True,
    )
    value_sidecars = np.einsum(
        "...lhcd,lhdr->...lhcr",
        visible_values,
        parameters.P,
        optimize=True,
    )
    if quantize_sidecars:
        value_sidecars = mass_selector.bf16_bits_to_float32(
            mass_selector.float32_to_bf16_bits(value_sidecars)
        )
    raw_delta += (
        np.einsum(
            "...lhr,...lhcr->...lhc",
            query_features,
            value_sidecars,
            optimize=True,
        )
        / np.float32(math.sqrt(parameters.Q.shape[-1]))
    )
    delta = mass_selector._gauge_and_clamp(
        raw_delta,
        valid,
        clamp=delta_clamp,
    )
    return delta, np.ascontiguousarray(value_sidecars, dtype=np.float32)


def selector_forward(
    native_mass: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
    visible_values: np.ndarray,
    parameters: ContentSelectorParameters,
    shape: ContentSelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
    quantize_sidecars: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the cached-mass reference.

    ``quantize_sidecars`` models persistent BF16 sidecar storage.  Parameter
    BF16 rounding is explicit and remains the caller's responsibility.
    """

    parameters.validate(shape)
    mass, mask, query_array, values = _validate_content_inputs(
        native_mass,
        valid,
        query,
        visible_values,
        shape,
    )
    features = mass_selector.centered_log_mass_features(
        mass,
        mask,
        shape.mass_shape,
        clip=feature_clip,
    )
    delta, sidecars = _selector_delta(
        features,
        mask,
        query_array,
        values,
        parameters,
        delta_clamp=delta_clamp,
        quantize_sidecars=quantize_sidecars,
    )
    coefficients = mass_selector.coefficients_from_mass(mass, mask, delta)
    return coefficients, delta, sidecars


def selector_forward_from_scores(
    native_scores: np.ndarray,
    valid: np.ndarray,
    query: np.ndarray,
    visible_values: np.ndarray,
    parameters: ContentSelectorParameters,
    shape: ContentSelectorShape,
    *,
    feature_clip: float = 16.0,
    delta_clamp: float = 16.0,
    quantize_sidecars: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the candidate fused native-score path."""

    parameters.validate(shape)
    scores = np.ascontiguousarray(native_scores, dtype=np.float32)
    mask = np.ascontiguousarray(valid, dtype=bool)
    query_array = np.ascontiguousarray(query, dtype=np.float32)
    values = np.ascontiguousarray(visible_values, dtype=np.float32)
    # Validate the content tensors using a normalized mass derived from scores.
    native_mass = mass_selector._masked_softmax(scores, mask)
    _validate_content_inputs(
        native_mass,
        mask,
        query_array,
        values,
        shape,
    )
    features = mass_selector.centered_score_features(
        scores,
        mask,
        shape.mass_shape,
        clip=feature_clip,
    )
    delta, sidecars = _selector_delta(
        features,
        mask,
        query_array,
        values,
        parameters,
        delta_clamp=delta_clamp,
        quantize_sidecars=quantize_sidecars,
    )
    coefficients = mass_selector.coefficients_from_scores(scores, mask, delta)
    return coefficients, delta, sidecars


def quantize_parameters_bf16(
    parameters: ContentSelectorParameters,
    shape: ContentSelectorShape,
) -> tuple[ContentSelectorParameters, dict[str, np.ndarray]]:
    parameters.validate(shape)
    bits = {
        name: mass_selector.float32_to_bf16_bits(value)
        for name, value in parameters.as_dict().items()
    }
    decoded = ContentSelectorParameters(
        **{
            name: mass_selector.bf16_bits_to_float32(bits[name])
            for name in ("U", "V", "E", "B", "Q", "P")
        }
    )
    decoded.validate(shape)
    return decoded, bits


def production_resource_contract(
    shape: ContentSelectorShape = ContentSelectorShape(),
) -> dict[str, Any]:
    """Return conservative accounting under the exact 51-head ceiling."""

    if shape != ContentSelectorShape():
        raise ValueError("production resource accounting requires production shape")
    parameter_count = shape.parameter_count
    parameter_bytes = parameter_count * 2
    weight_traffic = parameter_bytes * _PRODUCTION_POSITIONS
    sidecar_bytes_per_entry = shape.content_rank * 2
    sidecar_read_traffic = (
        _PRODUCTION_POSITIONS
        * shape.layers
        * shape.heads
        * shape.components
        * sidecar_bytes_per_entry
    )
    # Pessimistically count three writes of every incoming sketch: recent,
    # transfer to older, and episodic storage.
    sidecar_write_traffic = (
        _PRODUCTION_POSITIONS
        * 3
        * shape.layers
        * shape.heads
        * sidecar_bytes_per_entry
    )
    sidecar_state = (
        shape.layers
        * shape.heads
        * sidecar_bytes_per_entry
        * (16 + 8 + 32)
    )
    total = (
        _PRODUCTION_FIXED_TRAFFIC_BYTES
        + weight_traffic
        + sidecar_read_traffic
        + sidecar_write_traffic
    )
    macs_per_token = (
        shape.layers
        * shape.heads
        * (
            shape.components * shape.mass_rank
            + shape.mass_rank * shape.components
            + 2 * shape.head_dimension * shape.content_rank
            + shape.components * shape.content_rank
        )
    )
    combined_state = _PRODUCTION_BASE_STATE_BYTES + parameter_bytes + sidecar_state
    if (
        parameter_count != 287_744
        or parameter_bytes != 575_488
        or weight_traffic != 73_662_464
        or sidecar_read_traffic != 7_340_032
        or sidecar_write_traffic != 786_432
        or sidecar_state != 114_688
        or total != 796_655_616
        or macs_per_token != 520_192
        or combined_state != 11_225_088
        or total >= _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES
    ):
        raise AssertionError("content-selector production accounting changed")
    return {
        "parameter_count": parameter_count,
        "serialized_parameter_dtype": "BF16",
        "serialized_parameter_bytes": parameter_bytes,
        "fixed_attention_state_bytes": _PRODUCTION_BASE_STATE_BYTES,
        "value_sidecar_state_bytes": sidecar_state,
        "combined_attention_selector_and_sidecar_state_bytes": combined_state,
        "selector_multiply_accumulates_per_token": macs_per_token,
        "selector_multiply_accumulates_per_128_token_sequence": (
            macs_per_token * _PRODUCTION_POSITIONS
        ),
        "conservative_selector_weight_traffic_bytes_per_128_token_sequence": (
            weight_traffic
        ),
        "value_sidecar_read_traffic_bytes_per_128_token_sequence": (
            sidecar_read_traffic
        ),
        "pessimistic_value_sidecar_write_traffic_bytes_per_128_token_sequence": (
            sidecar_write_traffic
        ),
        "fixed_combined_attention_and_episodic_traffic_bytes": (
            _PRODUCTION_FIXED_TRAFFIC_BYTES
        ),
        "total_logical_traffic_bytes_per_128_token_sequence": total,
        "dense_full_context_logical_read_bytes": _PRODUCTION_DENSE_BYTES,
        "fraction_of_dense_full_context_logical_reads": total
        / _PRODUCTION_DENSE_BYTES,
        "exact_51_head_equivalent_ceiling_bytes": (
            _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES
        ),
        "remaining_headroom_below_exact_51_head_ceiling_bytes": (
            _PRODUCTION_EXACT_51_HEAD_CEILING_BYTES - total
        ),
        "new_full_key_or_value_state_bytes": 0,
        "new_full_key_or_value_read_traffic_bytes": 0,
        "single_full_value_accumulation_pass": True,
        "unfused_second_full_value_read_pass_authorized": False,
        "selector_weight_traffic_assumes_reload_for_every_token": True,
    }


def _torch_forward(
    torch: Any,
    native_mass: Any,
    valid: Any,
    query: Any,
    visible_values: Any,
    U: Any,
    V: Any,
    E: Any,
    B: Any,
    Q: Any,
    P: Any,
    *,
    feature_clip: float,
    delta_clamp: float,
    content_rank: int,
) -> Any:
    """Torch forward with layout [layer, batch, head, ...]."""

    log_mass = torch.where(valid, torch.log(native_mass), torch.zeros_like(native_mass))
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
    mass_hidden = torch.relu(
        torch.einsum("lbhc,lcr->lbhr", features, U) + E[:, None]
    )
    raw_delta = torch.einsum("lbhr,lrc->lbhc", mass_hidden, V) + B[:, None]
    query_features = torch.einsum("lbhd,lhdr->lbhr", query, Q)
    value_sidecars = torch.einsum("lbhcd,lhdr->lbhcr", visible_values, P)
    raw_delta = raw_delta + torch.einsum(
        "lbhr,lbhcr->lbhc",
        query_features,
        value_sidecars,
    ) / math.sqrt(content_rank)
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
    query: np.ndarray,
    visible_values: np.ndarray,
    base_heads: np.ndarray,
    target_residual: np.ndarray,
    output_projection: np.ndarray,
    *,
    training_records: Sequence[int],
    shape: ContentSelectorShape,
    config: mass_selector.TrainingConfig,
    device: str = "cpu",
) -> TrainingResult:
    """Fit one fixed final-step model without consulting heldout records."""

    config.validate()
    shape.validate()
    native, mask, query_array, values = _validate_content_inputs(
        native_mass,
        valid,
        query,
        visible_values,
        shape,
    )
    base = np.ascontiguousarray(base_heads, dtype=np.float32)
    target = np.ascontiguousarray(target_residual, dtype=np.float32)
    weights = np.ascontiguousarray(output_projection, dtype=np.float32)
    if native.ndim != 5:
        raise ValueError("content-selector training masses must be rank five")
    records, reads = native.shape[:2]
    if (
        base.shape
        != (
            records,
            reads,
            shape.layers,
            shape.heads,
            shape.head_dimension,
        )
        or target.shape != (records, reads, shape.layers, shape.hidden_size)
        or weights.shape != (shape.layers, shape.hidden_size, shape.hidden_size)
    ):
        raise ValueError("content-selector training tensor shapes are invalid")
    train = tuple(int(value) for value in training_records)
    if (
        not train
        or len(set(train)) != len(train)
        or min(train) < 0
        or max(train) >= records
    ):
        raise ValueError("content-selector training records are invalid")
    rows_per_epoch = len(train) * reads
    if (
        rows_per_epoch % config.rows_per_layer_per_step
        or config.steps
        != config.epochs * rows_per_epoch // config.rows_per_layer_per_step
    ):
        raise ValueError("content-selector fixed epoch schedule does not match data")

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project dependency
        raise RuntimeError("content-selector training requires torch") from error
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError("content-selector CUDA device is unavailable")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError(
                "content-selector deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG"
            )
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.init_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.init_seed)
    torch_device = torch.device(device)

    initialized = initialize_parameters(
        shape,
        seed=config.init_seed,
        standard_deviation=config.initial_u_standard_deviation,
    )
    parameters = {
        name: torch.nn.Parameter(torch.from_numpy(value.copy()).to(torch_device))
        for name, value in initialized.as_dict().items()
    }
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    parameters["U"],
                    parameters["V"],
                    parameters["Q"],
                    parameters["P"],
                ],
                "weight_decay": config.uv_weight_decay,
            },
            {
                "params": [parameters["E"], parameters["B"]],
                "weight_decay": 0.0,
            },
        ],
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
    flat_native = native.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.components,
    )
    flat_mask = mask.reshape(flat_native.shape)
    flat_query = query_array.reshape(
        records * reads,
        shape.layers,
        shape.heads,
        shape.head_dimension,
    )
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
        raise ValueError("content-selector training target energy is invalid")

    torch_native = torch.from_numpy(flat_native).to(torch_device)
    torch_mask = torch.from_numpy(flat_mask).to(torch_device)
    torch_query = torch.from_numpy(flat_query).to(torch_device)
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

    initial_loss = float("nan")
    final_loss = float("nan")
    for step in range(config.steps):
        for group in optimizer.param_groups:
            group["lr"] = float(rates[step])
        optimizer.zero_grad(set_to_none=True)
        row_ids = torch_schedule[step]
        batch_native = torch_native[row_ids, layer_indices]
        batch_mask = torch_mask[row_ids, layer_indices]
        batch_query = torch_query[row_ids, layer_indices]
        batch_values = torch_values[row_ids, layer_indices]
        batch_base = torch_base[row_ids, layer_indices]
        batch_target = torch_target[row_ids, layer_indices]
        alpha = _torch_forward(
            torch,
            batch_native,
            batch_mask,
            batch_query,
            batch_values,
            parameters["U"],
            parameters["V"],
            parameters["E"],
            parameters["B"],
            parameters["Q"],
            parameters["P"],
            feature_clip=config.feature_clip,
            delta_clamp=config.delta_clamp,
            content_rank=shape.content_rank,
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
            raise ValueError("content-selector training loss became non-finite")
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            tuple(parameters.values()),
            config.gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    fitted = ContentSelectorParameters(
        **{
            name: np.ascontiguousarray(
                value.detach().cpu().numpy(),
                dtype=np.float32,
            )
            for name, value in parameters.items()
        }
    )
    fitted.validate(shape)
    return TrainingResult(
        parameters=fitted,
        initial_loss=initial_loss,
        final_loss=final_loss,
        learning_rate_sha256=hashlib.sha256(rates.tobytes(order="C")).hexdigest(),
        schedule_sha256=mass_selector.schedule_sha256(schedule),
        steps=config.steps,
        device=str(torch_device),
    )


def parameters_sha256(
    parameters: ContentSelectorParameters,
    shape: ContentSelectorShape,
) -> str:
    parameters.validate(shape)
    digest = hashlib.sha256()
    for name in ("U", "V", "E", "B", "Q", "P"):
        digest.update(name.encode("ascii"))
        digest.update(parameters.as_dict()[name].tobytes(order="C"))
    return digest.hexdigest()
