"""Teacher-forced interventions at trained Hugging Face MLP boundaries.

This module answers a question that proxy routing metrics cannot: if a sparse
MLP approximation replaces the exact MLP inside the original transformer, how
much does the model's residual stream and next-token distribution change?

The Hugging Face MLP still executes before a forward hook replaces its output.
Consequently, wall-clock timing from this evaluator is not an inference-speed
measurement.  Byte and MAC figures are explicitly projected accounting only.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from engram.models.inspection import inspect_model, resolve_model_path
from engram.evaluation.gates import (
    MAGNITUDE_REFERENCE_CAVEAT,
    apply_mlp_intervention_gates,
)
from engram.semantic.multilabel_router import (
    LowRankMultiLabelRouter,
    OverlappingCoverageRouter,
)
from engram.semantic.dip import input_coordinate_count, projected_dip_traffic
from engram.semantic.swiglu import neuron_activations
from engram.tracing.format import TraceReader
from engram.utils import percentile, sha256_file, sha256_json

SUPPORTED_VARIANTS = (
    "identity",
    "oracle",
    "rank16",
    "overlap",
    "dip",
    "dip_paq",
)
SUPPORTED_LAYER_MODES = ("all", "individual", "both")
SUPPORTED_EVALUATION_ROLES = ("development", "confirmation")


def _load_jsonl(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record at line {line_number} must be an object"
                )
            if "input_ids" not in record and "text" not in record:
                raise ValueError(
                    f"JSONL record at line {line_number} requires 'input_ids' or 'text'"
                )
            records.append(record)
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError("dataset contains no JSONL records")
    return records


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        raise ValueError("cannot summarize an empty metric")
    if not np.all(np.isfinite(array)):
        raise ValueError("metric contains a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": percentile(array, 95),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _token_sequence_hash(input_ids: Sequence[int]) -> str:
    values = list(input_ids)
    if not values or any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in values
    ):
        raise ValueError("token sequences must contain at least one integer")
    return sha256_json({"input_ids": [int(value) for value in values]})


def _evaluation_sequence_hashes(
    records: Sequence[dict[str, Any]], tokenizer: Any | None
) -> list[str]:
    hashes: list[str] = []
    for record_index, record in enumerate(records):
        if "input_ids" in record:
            values = record["input_ids"]
            if not isinstance(values, list):
                raise ValueError(f"record {record_index} input_ids must be a list")
        else:
            if tokenizer is None:
                raise RuntimeError("tokenizer was not loaded for text input")
            values = tokenizer(str(record["text"]), add_special_tokens=True)[
                "input_ids"
            ]
        if len(values) < 2:
            continue
        hashes.append(_token_sequence_hash(values))
    return hashes


def _trace_sequence_hashes(reader: TraceReader) -> list[str]:
    """Reconstruct exact calibration token sequences from trace provenance."""

    sequences: dict[int, list[int]] = defaultdict(list)
    try:
        shards = reader.iter_shards(["sample_id", "token_id"])
        for shard in shards:
            sample_ids = np.asarray(shard["sample_id"])
            token_ids = np.asarray(shard["token_id"])
            if (
                sample_ids.ndim != 1
                or token_ids.ndim != 1
                or sample_ids.shape != token_ids.shape
            ):
                raise ValueError(
                    "trace sample_id/token_id fields must be matching vectors"
                )
            for sample_id, token_id in zip(sample_ids, token_ids, strict=True):
                if not np.issubdtype(type(sample_id), np.integer) or not np.issubdtype(
                    type(token_id), np.integer
                ):
                    raise ValueError("trace sample_id/token_id fields must be integral")
                sequences[int(sample_id)].append(int(token_id))
    except KeyError as exc:
        raise ValueError(
            "calibration traces require sample_id and token_id provenance fields"
        ) from exc
    if not sequences:
        raise ValueError("calibration traces contain no token-sequence provenance")
    return [_token_sequence_hash(sequences[index]) for index in sorted(sequences)]


def _relative_and_cosine_rows(
    approximation: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    approximate = np.asarray(approximation, dtype=np.float64)
    exact = np.asarray(reference, dtype=np.float64)
    if approximate.shape != exact.shape or approximate.ndim < 2:
        raise ValueError(
            "approximation and reference must have the same [..., width] shape"
        )
    approximate = approximate.reshape(-1, approximate.shape[-1])
    exact = exact.reshape(-1, exact.shape[-1])
    reference_norm = np.linalg.norm(exact, axis=1)
    approximation_norm = np.linalg.norm(approximate, axis=1)
    relative = np.linalg.norm(approximate - exact, axis=1) / np.maximum(
        reference_norm, 1e-12
    )
    denominator = reference_norm * approximation_norm
    cosine = np.sum(approximate * exact, axis=1) / np.maximum(denominator, 1e-12)
    both_zero = (reference_norm <= 1e-12) & (approximation_norm <= 1e-12)
    cosine[both_zero] = 1.0
    return relative, np.clip(cosine, -1.0, 1.0)


@dataclass
class _LocalAccumulator:
    relative_l2: list[float] = field(default_factory=list)
    cosine: list[float] = field(default_factory=list)
    candidate_recall: list[float] = field(default_factory=list)
    oracle_score_mass_recall: list[float] = field(default_factory=list)
    candidate_count: list[float] = field(default_factory=list)
    posting_groups: list[float] = field(default_factory=list)
    posting_entries: list[float] = field(default_factory=list)

    def add_outputs(self, approximation: Any, reference: Any) -> None:
        approximate = approximation.detach().float().cpu().numpy()
        exact = reference.detach().float().cpu().numpy()
        relative, cosine = _relative_and_cosine_rows(approximate, exact)
        self.relative_l2.extend(relative.tolist())
        self.cosine.extend(cosine.tolist())

    def add_recall(
        self,
        hits: Any,
        *,
        top_k: int,
        candidate_count: int,
        score_mass_recall: Any | None = None,
    ) -> None:
        values = hits.detach().float().cpu().numpy().reshape(-1) / float(top_k)
        self.candidate_recall.extend(values.tolist())
        self.candidate_count.extend([float(candidate_count)] * values.size)
        if score_mass_recall is not None:
            mass_values = score_mass_recall.detach().float().cpu().numpy().reshape(-1)
            if mass_values.size != values.size:
                raise ValueError(
                    "score-mass recall and candidate hits must have matching rows"
                )
            self.oracle_score_mass_recall.extend(mass_values.tolist())

    def add_posting_work(self, groups: Sequence[int], entries: Sequence[int]) -> None:
        self.posting_groups.extend(float(value) for value in groups)
        self.posting_entries.extend(float(value) for value in entries)

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tokens": len(self.relative_l2),
            "mlp_output_relative_l2": _stats(self.relative_l2),
            "mlp_output_cosine": _stats(self.cosine),
        }
        if self.candidate_recall:
            result["candidate_recall"] = _stats(self.candidate_recall)
            result["candidate_count"] = _stats(self.candidate_count)
        if self.oracle_score_mass_recall:
            result["oracle_score_mass_recall"] = _stats(self.oracle_score_mass_recall)
        if self.posting_groups:
            result["posting_groups_selected"] = _stats(self.posting_groups)
            result["posting_entries_scanned"] = _stats(self.posting_entries)
        return result


@dataclass(frozen=True)
class _PreparedExample:
    record_index: int
    input_type: str
    input_ids: Any
    logits: Any
    final_hidden: Any


@dataclass(frozen=True)
class _Arm:
    variant: str
    layer_indices: tuple[int, ...]
    scope: str
    top_k: int | None = None
    candidate_count: int | None = None
    rank: int | None = None
    posting_groups: int | None = None
    posting_size: int | None = None
    input_fraction: float | None = None
    layer_top_ks: tuple[int, ...] | None = None
    quantization: str | None = None

    @property
    def name(self) -> str:
        if self.variant == "identity":
            prefix = "identity"
        elif self.variant == "oracle":
            prefix = (
                f"oracle_layer_adaptive_mean_{self.top_k}"
                if self.layer_top_ks is not None
                else f"oracle_top_{self.top_k}"
            )
        elif self.variant == "rank16":
            prefix = (
                f"rank{self.rank}_candidates_{self.candidate_count}_top_{self.top_k}"
            )
        elif self.variant == "overlap":
            prefix = (
                f"overlap_rank{self.rank}_{self.posting_groups}x{self.posting_size}_"
                f"candidates_{self.candidate_count}_top_{self.top_k}"
            )
        elif self.variant == "dip":
            fraction = format(float(self.input_fraction), ".12g").replace(".", "p")
            prefix = (
                f"dip_input_{fraction}_candidates_{self.candidate_count}_"
                f"top_{self.top_k}"
            )
        else:
            assert self.variant == "dip_paq"
            fraction = format(float(self.input_fraction), ".12g").replace(".", "p")
            prefix = (
                f"dip_paq_{self.quantization}_input_{fraction}_"
                f"candidates_{self.candidate_count}_top_{self.top_k}"
            )
        if self.scope == "individual":
            return f"{prefix}_layer_{self.layer_indices[0]}"
        if len(self.layer_indices) == 1:
            return f"{prefix}_selected_layer_{self.layer_indices[0]}"
        return f"{prefix}_all_selected_layers"


@dataclass(frozen=True)
class _PAQLayer:
    """Decoded tensors plus physical storage metadata for one PAQ MLP layer.

    The tensors intentionally remain decoded float32 in this quality harness.
    The associated encodings are the source of byte accounting; evaluator wall
    time is not a compressed-kernel benchmark.
    """

    gate: Any
    up: Any
    down_records: Any
    group_size: int
    groups_per_record: int
    num_codebooks: int
    code_bits: int
    encoded_bits_per_weight: float
    codebook_bytes: int
    scale_bytes_per_gate_record: int
    scale_bytes_per_up_record: int
    scale_bytes_per_down_record: int
    codec: str


def _load_layer_states(reader: TraceReader, layer: int, limit: int) -> np.ndarray:
    field_name = f"layer_{layer}_mlp_input"
    batches: list[np.ndarray] = []
    count = 0
    for shard in reader.iter_shards([field_name]):
        batch = np.asarray(shard[field_name], dtype=np.float64)
        remaining = limit - count
        if remaining <= 0:
            break
        batches.append(batch[:remaining])
        count += min(len(batch), remaining)
    if not batches:
        raise ValueError(f"calibration traces contain no states for layer {layer}")
    return np.concatenate(batches, axis=0)


def _oracle_membership(
    states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    top_k: int,
) -> np.ndarray:
    activations = neuron_activations(states, gate, up)
    scores = np.abs(activations) * np.linalg.norm(down, axis=0)[None, :]
    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    membership = np.zeros(scores.shape, dtype=np.float64)
    membership[np.arange(scores.shape[0])[:, None], order] = 1.0
    return membership


def _fit_rank_routers(
    model: Any,
    inspection: Any,
    calibration: TraceReader,
    layer_indices: Sequence[int],
    top_ks: Sequence[int],
    *,
    rank: int,
    regularization: float,
    calibration_records: int,
) -> dict[tuple[int, int], LowRankMultiLabelRouter]:
    routers: dict[tuple[int, int], LowRankMultiLabelRouter] = {}
    for layer_index in layer_indices:
        module = model.model.layers[layer_index].mlp
        gate = module.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        up = module.up_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        down = module.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        states = _load_layer_states(calibration, layer_index, calibration_records)
        if states.shape[1] != inspection.hidden_size:
            raise ValueError(f"calibration state width mismatch at layer {layer_index}")
        for top_k in top_ks:
            membership = _oracle_membership(states, gate, up, down, top_k)
            routers[(layer_index, top_k)] = LowRankMultiLabelRouter.fit(
                states,
                membership,
                rank=rank,
                regularization=regularization,
            )
    return routers


def _fit_overlap_routers(
    model: Any,
    inspection: Any,
    calibration: TraceReader,
    layer_indices: Sequence[int],
    top_ks: Sequence[int],
    candidate_counts: Sequence[int],
    *,
    rank: int,
    regularization: float,
    calibration_records: int,
    groups: int,
    posting_size: int,
    iterations: int,
    max_replication: int,
) -> dict[tuple[int, int, int], OverlappingCoverageRouter]:
    routers: dict[tuple[int, int, int], OverlappingCoverageRouter] = {}
    for layer_index in layer_indices:
        module = model.model.layers[layer_index].mlp
        gate = module.gate_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        up = module.up_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        down = module.down_proj.weight.detach().float().cpu().numpy().astype(np.float64)
        states = _load_layer_states(calibration, layer_index, calibration_records)
        if states.shape[1] != inspection.hidden_size:
            raise ValueError(f"calibration state width mismatch at layer {layer_index}")
        for top_k in top_ks:
            membership = _oracle_membership(states, gate, up, down, top_k)
            for candidate_count in candidate_counts:
                routers[(layer_index, top_k, candidate_count)] = (
                    OverlappingCoverageRouter.fit(
                        states,
                        membership,
                        gate,
                        up,
                        down.T,
                        rank=rank,
                        groups=groups,
                        posting_size=posting_size,
                        candidate_count=candidate_count,
                        regularization=regularization,
                        iterations=iterations,
                        max_replication=max_replication,
                    )
                )
    return routers


def _fit_paq_layers(
    model: Any,
    layer_indices: Sequence[int],
    *,
    group_size: int,
    num_codebooks: int,
    codebook_size: int,
    iterations: int,
    sample_limit: int | None,
    seed: int,
    torch: Any,
) -> dict[int, _PAQLayer]:
    """Fit record-local product/additive codecs without modifying the teacher."""

    from engram.semantic.product_quantization import (
        decode_product_additive,
        fit_product_additive,
    )

    result: dict[int, _PAQLayer] = {}
    code_bits = max(1, (codebook_size - 1).bit_length())
    bits_per_weight = num_codebooks * code_bits / group_size
    for layer_index in layer_indices:
        module = model.model.layers[layer_index].mlp
        matrices = {
            "gate": module.gate_proj.weight.detach().float().cpu().numpy(),
            "up": module.up_proj.weight.detach().float().cpu().numpy(),
            "down": module.down_proj.weight.detach().float().cpu().numpy().T,
        }
        encodings = {
            name: fit_product_additive(
                matrix,
                group_size=group_size,
                num_codebooks=num_codebooks,
                codebook_size=codebook_size,
                iterations=iterations,
                sample_limit=sample_limit,
                seed=seed + layer_index * 11 + offset,
                per_record_scale=True,
            )
            for offset, (name, matrix) in enumerate(matrices.items())
        }
        decoded = {
            name: torch.from_numpy(decode_product_additive(encoding))
            for name, encoding in encodings.items()
        }
        records = matrices["gate"].shape[0]

        def scale_bytes_per_record(name: str) -> int:
            scales = encodings[name].record_scales
            return 0 if scales is None else int(scales.nbytes // records)

        result[layer_index] = _PAQLayer(
            gate=decoded["gate"],
            up=decoded["up"],
            down_records=decoded["down"],
            group_size=group_size,
            groups_per_record=encodings["gate"].metadata.groups,
            num_codebooks=num_codebooks,
            code_bits=code_bits,
            encoded_bits_per_weight=bits_per_weight,
            codebook_bytes=sum(
                int(encoding.codebooks.nbytes) for encoding in encodings.values()
            ),
            scale_bytes_per_gate_record=scale_bytes_per_record("gate"),
            scale_bytes_per_up_record=scale_bytes_per_record("up"),
            scale_bytes_per_down_record=scale_bytes_per_record("down"),
            codec=(
                f"product_additive_{num_codebooks}x{code_bits}_g{group_size}_"
                "fp16_scale"
            ),
        )
    return result


def _candidate_ids(
    router: LowRankMultiLabelRouter, hidden: Any, candidate_count: int, torch: Any
) -> Any:
    shape = hidden.shape
    flat = (
        hidden.detach().float().cpu().numpy().reshape(-1, shape[-1]).astype(np.float64)
    )
    scores = (flat @ router.input_factors) @ router.output_factors + router.bias
    candidates = np.argsort(-scores, axis=1, kind="stable")[:, :candidate_count].copy()
    return torch.from_numpy(candidates).to(device=hidden.device, dtype=torch.long)


def _overlap_candidate_ids(
    router: OverlappingCoverageRouter, hidden: Any, candidate_count: int, torch: Any
) -> tuple[Any, list[int], list[int]]:
    shape = hidden.shape
    flat = (
        hidden.detach().float().cpu().numpy().reshape(-1, shape[-1]).astype(np.float64)
    )
    candidates = np.empty((flat.shape[0], candidate_count), dtype=np.int64)
    group_counts: list[int] = []
    posting_entries: list[int] = []
    for row_index, state in enumerate(flat):
        indices, groups, entries = router.candidates(
            state, candidate_count=candidate_count
        )
        candidates[row_index] = indices
        group_counts.append(int(groups.size))
        posting_entries.append(int(entries))
    return (
        torch.from_numpy(candidates).to(device=hidden.device, dtype=torch.long),
        group_counts,
        posting_entries,
    )


def _input_coordinate_count(hidden_size: int, input_fraction: float) -> int:
    """Resolve a fractional DIP budget to a deterministic non-empty coordinate count."""

    return input_coordinate_count(hidden_size, input_fraction)


def _dip_candidate_ids(
    module: Any,
    hidden: Any,
    *,
    input_fraction: float,
    candidate_count: int,
    torch: Any,
) -> Any:
    """Select candidates from a predictor-free partial SwiGLU evaluation.

    This quality evaluator uses dense tensor operations to express the partial
    projection. Projected accounting, rather than evaluator wall time, captures
    the weight reads of a kernel that gathers only the selected input columns.
    """

    original_shape = hidden.shape
    flat_hidden = hidden.reshape(-1, original_shape[-1])
    q = _input_coordinate_count(original_shape[-1], input_fraction)
    coordinate_ids = torch.argsort(
        torch.abs(flat_hidden), dim=1, descending=True, stable=True
    )[:, :q]
    partial_hidden = torch.zeros_like(flat_hidden)
    partial_hidden.scatter_(1, coordinate_ids, flat_hidden.gather(1, coordinate_ids))
    approximate_activations = module.act_fn(
        module.gate_proj(partial_hidden)
    ) * module.up_proj(partial_hidden)
    value_norms = torch.linalg.vector_norm(
        module.down_proj.weight.detach().to(dtype=approximate_activations.dtype), dim=0
    )
    proxy_scores = torch.abs(approximate_activations) * value_norms.unsqueeze(0)
    return torch.argsort(proxy_scores, dim=1, descending=True, stable=True)[
        :, :candidate_count
    ]


def _sparse_replacement(
    module: Any,
    hidden: Any,
    *,
    top_k: int,
    candidate_ids: Any | None,
    torch: Any,
) -> tuple[Any, Any | None, Any | None]:
    """Return an exact-weight sparse reconstruction and candidate coverage."""

    original_shape = hidden.shape
    flat_hidden = hidden.reshape(-1, original_shape[-1])
    activations = module.act_fn(module.gate_proj(flat_hidden)) * module.up_proj(
        flat_hidden
    )
    value_norms = torch.linalg.vector_norm(
        module.down_proj.weight.detach().to(dtype=activations.dtype), dim=0
    )
    scores = torch.abs(activations) * value_norms.unsqueeze(0)
    oracle_ids = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :top_k]
    hits = None
    score_mass_recall = None
    if candidate_ids is None:
        active_ids = oracle_ids
    else:
        candidate_mask = torch.zeros_like(scores, dtype=torch.bool)
        candidate_mask.scatter_(1, candidate_ids, True)
        hits = candidate_mask.gather(1, oracle_ids).sum(dim=1)
        oracle_scores = scores.gather(1, oracle_ids)
        captured_oracle_scores = oracle_scores * candidate_mask.gather(1, oracle_ids)
        oracle_score_mass = oracle_scores.sum(dim=1)
        score_mass_recall = torch.where(
            oracle_score_mass > 0,
            captured_oracle_scores.sum(dim=1) / oracle_score_mass,
            torch.ones_like(oracle_score_mass),
        )
        # Restore record-index order before the stable score sort so exact-score
        # ties follow the same rule as the full-width magnitude oracle.
        candidate_ids_by_index = torch.sort(candidate_ids, dim=1).values
        candidate_scores = scores.gather(1, candidate_ids_by_index)
        local_order = torch.argsort(
            candidate_scores, dim=1, descending=True, stable=True
        )[:, :top_k]
        active_ids = candidate_ids_by_index.gather(1, local_order)
    active_values = activations.gather(1, active_ids)
    masked = torch.zeros_like(activations)
    masked.scatter_(1, active_ids, active_values)
    replacement = module.down_proj(masked).reshape(*original_shape[:-1], -1)
    return replacement, hits, score_mass_recall


def _paq_dip_replacement(
    module: Any,
    quantized: _PAQLayer,
    hidden: Any,
    *,
    input_fraction: float,
    candidate_count: int,
    top_k: int,
    torch: Any,
) -> tuple[Any, Any, Any]:
    """Execute DIP selection and reconstruction entirely with decoded PAQ weights.

    Dense source weights are consulted only for the diagnostic oracle-membership
    recall.  Candidate generation, exact candidate completion, reranking, and
    the returned MLP output all use the quantized representation.
    """

    original_shape = hidden.shape
    flat_hidden = hidden.reshape(-1, original_shape[-1]).float()
    gate = quantized.gate.to(device=flat_hidden.device, dtype=flat_hidden.dtype)
    up = quantized.up.to(device=flat_hidden.device, dtype=flat_hidden.dtype)
    down_records = quantized.down_records.to(
        device=flat_hidden.device, dtype=flat_hidden.dtype
    )

    q = _input_coordinate_count(original_shape[-1], input_fraction)
    coordinate_ids = torch.argsort(
        torch.abs(flat_hidden), dim=1, descending=True, stable=True
    )[:, :q]
    partial_hidden = torch.zeros_like(flat_hidden)
    partial_hidden.scatter_(
        1, coordinate_ids, flat_hidden.gather(1, coordinate_ids)
    )
    partial_activations = module.act_fn(partial_hidden @ gate.T) * (
        partial_hidden @ up.T
    )
    quantized_value_norms = torch.linalg.vector_norm(down_records, dim=1)
    proxy_scores = torch.abs(partial_activations) * quantized_value_norms.unsqueeze(0)
    candidate_ids = torch.argsort(
        proxy_scores, dim=1, descending=True, stable=True
    )[:, :candidate_count]

    quantized_activations = module.act_fn(flat_hidden @ gate.T) * (flat_hidden @ up.T)
    candidate_ids_by_index = torch.sort(candidate_ids, dim=1).values
    quantized_scores = (
        torch.abs(quantized_activations) * quantized_value_norms.unsqueeze(0)
    )
    candidate_scores = quantized_scores.gather(1, candidate_ids_by_index)
    local_order = torch.argsort(
        candidate_scores, dim=1, descending=True, stable=True
    )[:, :top_k]
    active_ids = candidate_ids_by_index.gather(1, local_order)
    active_values = quantized_activations.gather(1, active_ids)
    selected_down = down_records[active_ids]
    replacement = torch.sum(active_values.unsqueeze(-1) * selected_down, dim=1)

    # Full-precision work below is diagnostics only and is deliberately excluded
    # from the proposed inference traffic.
    exact_activations = module.act_fn(module.gate_proj(flat_hidden)) * module.up_proj(
        flat_hidden
    )
    exact_value_norms = torch.linalg.vector_norm(
        module.down_proj.weight.detach().to(dtype=flat_hidden.dtype), dim=0
    )
    exact_scores = torch.abs(exact_activations) * exact_value_norms.unsqueeze(0)
    oracle_ids = torch.argsort(
        exact_scores, dim=1, descending=True, stable=True
    )[:, :top_k]
    candidate_mask = torch.zeros_like(exact_scores, dtype=torch.bool)
    candidate_mask.scatter_(1, candidate_ids, True)
    hits = candidate_mask.gather(1, oracle_ids).sum(dim=1)
    oracle_scores = exact_scores.gather(1, oracle_ids)
    captured = oracle_scores * candidate_mask.gather(1, oracle_ids)
    oracle_mass = oracle_scores.sum(dim=1)
    score_mass_recall = torch.where(
        oracle_mass > 0,
        captured.sum(dim=1) / oracle_mass,
        torch.ones_like(oracle_mass),
    )
    return replacement.reshape(*original_shape[:-1], -1), hits, score_mass_recall


def _quality_metrics(
    teacher_logits: Any,
    student_logits: Any,
    input_ids: Any,
    teacher_hidden: Any,
    student_hidden: Any,
    torch: Any,
) -> dict[str, np.ndarray]:
    import torch.nn.functional as functional

    exact_logits = teacher_logits[:, :-1].detach().float().cpu()
    approximate_logits = student_logits[:, :-1].detach().float().cpu()
    targets = input_ids[:, 1:].detach().cpu()
    teacher_logp = functional.log_softmax(exact_logits, dim=-1)
    student_logp = functional.log_softmax(approximate_logits, dim=-1)
    teacher_probability = torch.exp(teacher_logp)
    kl = torch.clamp(
        torch.sum(teacher_probability * (teacher_logp - student_logp), dim=-1),
        min=0.0,
    )
    teacher_nll = -teacher_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    student_nll = -student_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    teacher_top = torch.topk(
        exact_logits, k=min(5, exact_logits.shape[-1]), dim=-1
    ).indices
    student_top = torch.topk(
        approximate_logits, k=min(5, approximate_logits.shape[-1]), dim=-1
    ).indices
    top1 = (teacher_top[..., 0] == student_top[..., 0]).float()
    top5_contains = (teacher_top == student_top[..., :1]).any(dim=-1).float()
    top5_overlap = (
        (teacher_top.unsqueeze(-1) == student_top.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    residual_relative, residual_cosine = _relative_and_cosine_rows(
        student_hidden.detach().float().cpu().numpy(),
        teacher_hidden.detach().float().cpu().numpy(),
    )
    return {
        "teacher_student_kl": kl.detach().cpu().numpy().reshape(-1),
        "teacher_nll": teacher_nll.detach().cpu().numpy().reshape(-1),
        "student_nll": student_nll.detach().cpu().numpy().reshape(-1),
        "nll_delta": (student_nll - teacher_nll).detach().cpu().numpy().reshape(-1),
        "teacher_top1_agreement": top1.detach().cpu().numpy().reshape(-1),
        "teacher_top5_contains_student_top1": top5_contains.detach()
        .cpu()
        .numpy()
        .reshape(-1),
        "teacher_student_top5_overlap": top5_overlap.detach().cpu().numpy().reshape(-1),
        "final_hidden_relative_l2": residual_relative,
        "final_hidden_cosine": residual_cosine,
    }


def _prepare_examples(
    model: Any,
    tokenizer: Any | None,
    records: Sequence[dict[str, Any]],
    *,
    device: str,
    torch: Any,
) -> tuple[list[_PreparedExample], dict[str, Any]]:
    examples: list[_PreparedExample] = []
    teacher_nll: list[float] = []
    skipped = 0
    input_token_positions = 0
    import torch.nn.functional as functional

    with torch.inference_mode():
        for record_index, record in enumerate(records):
            if "input_ids" in record:
                values = record["input_ids"]
                if not isinstance(values, list) or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in values
                ):
                    raise ValueError(
                        f"record {record_index} input_ids must be a list of integers"
                    )
                input_ids = torch.tensor([values], dtype=torch.long, device=device)
            else:
                if tokenizer is None:
                    raise RuntimeError("tokenizer was not loaded for text input")
                input_ids = tokenizer(str(record["text"]), return_tensors="pt")[
                    "input_ids"
                ].to(device)
            if input_ids.shape[1] < 2:
                skipped += 1
                continue
            input_token_positions += int(input_ids.shape[1])
            output = model(
                input_ids=input_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = output.logits.detach().float().cpu()
            final_hidden = output.hidden_states[-1].detach().float().cpu()
            logp = functional.log_softmax(logits[:, :-1], dim=-1)
            targets = input_ids[:, 1:].cpu()
            teacher_nll.extend(
                (-logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
                .numpy()
                .reshape(-1)
                .tolist()
            )
            examples.append(
                _PreparedExample(
                    record_index=record_index,
                    input_type=str(record.get("input_type", "unspecified")),
                    input_ids=input_ids.detach().cpu(),
                    logits=logits,
                    final_hidden=final_hidden,
                )
            )
    if not examples:
        raise ValueError(
            "evaluation dataset produced no sequences with at least two tokens"
        )
    mean_nll = float(np.mean(teacher_nll))
    return examples, {
        "sequences": len(examples),
        "skipped_short_sequences": skipped,
        "input_token_positions": input_token_positions,
        "next_token_positions": len(teacher_nll),
        "negative_log_likelihood": mean_nll,
        "perplexity": float(math.exp(min(mean_nll, 50.0))),
    }


def _make_arms(
    variants: Sequence[str],
    layer_mode: str,
    layer_indices: tuple[int, ...],
    top_ks: tuple[int, ...],
    candidate_counts: tuple[int, ...],
    input_fractions: tuple[float, ...],
    *,
    rank: int,
    posting_groups: int,
    posting_size: int,
    paq_label: str,
) -> list[_Arm]:
    arms: list[_Arm] = []
    if "identity" in variants:
        arms.append(_Arm("identity", layer_indices, "all"))
    scopes: list[tuple[str, tuple[int, ...]]] = []
    if layer_mode in {"all", "both"}:
        scopes.append(("all", layer_indices))
    if layer_mode in {"individual", "both"}:
        scopes.extend(("individual", (layer,)) for layer in layer_indices)
    for top_k in top_ks:
        if "oracle" in variants:
            arms.extend(
                _Arm("oracle", scope_layers, scope, top_k)
                for scope, scope_layers in scopes
            )
        if (
            "rank16" in variants
            or "overlap" in variants
            or "dip" in variants
            or "dip_paq" in variants
        ):
            for candidate_count in candidate_counts:
                for variant in ("rank16", "overlap"):
                    if variant in variants:
                        arms.extend(
                            _Arm(
                                variant,
                                scope_layers,
                                scope,
                                top_k,
                                candidate_count,
                                rank,
                                posting_groups if variant == "overlap" else None,
                                posting_size if variant == "overlap" else None,
                            )
                            for scope, scope_layers in scopes
                        )
                if "dip" in variants:
                    for input_fraction in input_fractions:
                        arms.extend(
                            _Arm(
                                "dip",
                                scope_layers,
                                scope,
                                top_k,
                                candidate_count,
                                input_fraction=input_fraction,
                            )
                            for scope, scope_layers in scopes
                        )
                if "dip_paq" in variants:
                    for input_fraction in input_fractions:
                        arms.extend(
                            _Arm(
                                "dip_paq",
                                scope_layers,
                                scope,
                                top_k,
                                candidate_count,
                                input_fraction=input_fraction,
                                quantization=paq_label,
                            )
                            for scope, scope_layers in scopes
                        )
    return arms


def _evaluate_arm(
    model: Any,
    examples: Sequence[_PreparedExample],
    arm: _Arm,
    routers: dict[tuple[int, int], LowRankMultiLabelRouter],
    overlap_routers: dict[tuple[int, int, int], OverlappingCoverageRouter],
    paq_layers: dict[int, _PAQLayer],
    *,
    hidden_size: int,
    intermediate_size: int,
    paq_cacheline_amplification: float,
    torch: Any,
    device: str,
) -> dict[str, Any]:
    local_by_layer = {index: _LocalAccumulator() for index in arm.layer_indices}
    layer_top_k = (
        dict(zip(arm.layer_indices, arm.layer_top_ks, strict=True))
        if arm.layer_top_ks is not None
        else {}
    )
    quality: dict[str, list[float]] = defaultdict(list)
    handles = []

    def hook_for(layer_index: int):
        accumulator = local_by_layer[layer_index]

        def hook(module: Any, args: tuple[Any, ...], output: Any) -> Any:
            if not args:
                raise RuntimeError("MLP hook did not receive hidden-state input")
            hidden = args[0]
            if arm.variant == "identity":
                replacement = output
                hits = None
            else:
                assert arm.top_k is not None
                candidates = None
                if arm.variant == "rank16":
                    assert arm.candidate_count is not None
                    candidates = _candidate_ids(
                        routers[(layer_index, arm.top_k)],
                        hidden,
                        arm.candidate_count,
                        torch,
                    )
                elif arm.variant == "overlap":
                    assert arm.candidate_count is not None
                    candidates, group_counts, posting_entries = _overlap_candidate_ids(
                        overlap_routers[(layer_index, arm.top_k, arm.candidate_count)],
                        hidden,
                        arm.candidate_count,
                        torch,
                    )
                    accumulator.add_posting_work(group_counts, posting_entries)
                elif arm.variant == "dip":
                    assert arm.candidate_count is not None
                    assert arm.input_fraction is not None
                    candidates = _dip_candidate_ids(
                        module,
                        hidden,
                        input_fraction=arm.input_fraction,
                        candidate_count=arm.candidate_count,
                        torch=torch,
                    )
                elif arm.variant == "dip_paq":
                    assert arm.candidate_count is not None
                    assert arm.input_fraction is not None
                    replacement, hits, score_mass_recall = _paq_dip_replacement(
                        module,
                        paq_layers[layer_index],
                        hidden,
                        input_fraction=arm.input_fraction,
                        candidate_count=arm.candidate_count,
                        top_k=layer_top_k.get(layer_index, arm.top_k),
                        torch=torch,
                    )
                    accumulator.add_recall(
                        hits,
                        top_k=layer_top_k.get(layer_index, arm.top_k),
                        candidate_count=arm.candidate_count,
                        score_mass_recall=score_mass_recall,
                    )
                    accumulator.add_outputs(replacement, output)
                    return replacement.to(dtype=output.dtype, device=output.device)
                active_top_k = layer_top_k.get(layer_index, arm.top_k)
                replacement, hits, score_mass_recall = _sparse_replacement(
                    module,
                    hidden,
                    top_k=active_top_k,
                    candidate_ids=candidates,
                    torch=torch,
                )
                if hits is not None:
                    accumulator.add_recall(
                        hits,
                        top_k=active_top_k,
                        candidate_count=arm.candidate_count,
                        score_mass_recall=score_mass_recall,
                    )
            accumulator.add_outputs(replacement, output)
            return replacement.to(dtype=output.dtype, device=output.device)

        return hook

    for layer_index in arm.layer_indices:
        handles.append(
            model.model.layers[layer_index].mlp.register_forward_hook(
                hook_for(layer_index)
            )
        )
    try:
        with torch.inference_mode():
            for example in examples:
                input_ids = example.input_ids.to(device)
                output = model(
                    input_ids=input_ids,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                metrics = _quality_metrics(
                    example.logits,
                    output.logits,
                    input_ids,
                    example.final_hidden,
                    output.hidden_states[-1],
                    torch,
                )
                for name, values in metrics.items():
                    quality[name].extend(np.asarray(values, dtype=np.float64).tolist())
    finally:
        for handle in handles:
            handle.remove()

    metric_summary = {name: _stats(values) for name, values in sorted(quality.items())}
    mean_nll = metric_summary["student_nll"]["mean"]
    metric_summary["student_perplexity"] = float(math.exp(min(mean_nll, 50.0)))
    local_layers = [
        {"layer": layer, **local_by_layer[layer].summary()}
        for layer in arm.layer_indices
    ]
    combined = _LocalAccumulator()
    for layer in arm.layer_indices:
        accumulator = local_by_layer[layer]
        combined.relative_l2.extend(accumulator.relative_l2)
        combined.cosine.extend(accumulator.cosine)
        combined.candidate_recall.extend(accumulator.candidate_recall)
        combined.oracle_score_mass_recall.extend(accumulator.oracle_score_mass_recall)
        combined.candidate_count.extend(accumulator.candidate_count)
        combined.posting_groups.extend(accumulator.posting_groups)
        combined.posting_entries.extend(accumulator.posting_entries)

    if arm.variant == "identity":
        projected = {
            "kind": "dense_identity",
            "dense_mlp_weight_bytes_per_token": len(arm.layer_indices)
            * 3
            * hidden_size
            * intermediate_size
            * 4,
        }
    elif arm.variant == "oracle":
        assert arm.top_k is not None
        active_records = (
            sum(arm.layer_top_ks)
            if arm.layer_top_ks is not None
            else len(arm.layer_indices) * arm.top_k
        )
        projected = {
            "kind": "full_information_magnitude_reference_only",
            "active_record_fraction": active_records
            / (len(arm.layer_indices) * intermediate_size),
            "selected_value_bytes_per_token": active_records * hidden_size * 4,
            "layer_top_ks": list(arm.layer_top_ks)
            if arm.layer_top_ks is not None
            else None,
            "candidate_selection_cost": "unavailable oracle; deliberately not estimated",
        }
    elif arm.variant == "rank16":
        assert arm.top_k is not None and arm.candidate_count is not None
        router = routers[(arm.layer_indices[0], arm.top_k)]
        projected = {
            "kind": f"rank{arm.rank}_router_plus_exact_candidate_rerank",
            "active_record_fraction": arm.top_k / intermediate_size,
            "candidate_record_fraction": arm.candidate_count / intermediate_size,
            "router_parameter_bytes_per_invocation": sum(
                routers[(layer, arm.top_k)].parameter_bytes()
                for layer in arm.layer_indices
            ),
            "router_macs_per_token": len(arm.layer_indices)
            * router.rank
            * (hidden_size + intermediate_size),
            "candidate_key_bytes_per_token": len(arm.layer_indices)
            * arm.candidate_count
            * 2
            * hidden_size
            * 4,
            "selected_value_bytes_per_token": len(arm.layer_indices)
            * arm.top_k
            * hidden_size
            * 4,
        }
    elif arm.variant == "overlap":
        assert arm.top_k is not None and arm.candidate_count is not None
        first_router = overlap_routers[
            (arm.layer_indices[0], arm.top_k, arm.candidate_count)
        ]
        projected = {
            "kind": f"rank{arm.rank}_overlapping_postings_plus_exact_candidate_rerank",
            "active_record_fraction": arm.top_k / intermediate_size,
            "candidate_record_fraction": arm.candidate_count / intermediate_size,
            "router_parameter_bytes_per_invocation": sum(
                overlap_routers[
                    (layer, arm.top_k, arm.candidate_count)
                ].router_parameter_bytes()
                for layer in arm.layer_indices
            ),
            "posting_storage_bytes": sum(
                overlap_routers[(layer, arm.top_k, arm.candidate_count)].posting_bytes
                for layer in arm.layer_indices
            ),
            "router_macs_per_token": len(arm.layer_indices)
            * first_router.rank
            * (hidden_size + first_router.groups),
            "candidate_key_bytes_per_token": len(arm.layer_indices)
            * arm.candidate_count
            * 2
            * hidden_size
            * 4,
            "selected_value_bytes_per_token": len(arm.layer_indices)
            * arm.top_k
            * hidden_size
            * 4,
        }
    elif arm.variant == "dip":
        assert arm.top_k is not None and arm.candidate_count is not None
        assert arm.input_fraction is not None
        input_coordinates = _input_coordinate_count(hidden_size, arm.input_fraction)
        layers = len(arm.layer_indices)
        dtype_bytes = 4
        traffic = projected_dip_traffic(
            hidden_size,
            intermediate_size,
            input_fraction=arm.input_fraction,
            candidate_count=arm.candidate_count,
            top_k=arm.top_k,
            bytes_per_element=dtype_bytes,
        )
        scalar_reads = layers * traffic.total_elements
        dense_scalar_reads = layers * traffic.dense_elements
        partial_key_bytes = layers * traffic.partial_projection_elements * dtype_bytes
        completion_key_bytes = (
            layers * traffic.candidate_completion_elements * dtype_bytes
        )
        selected_value_bytes = layers * traffic.selected_down_elements * dtype_bytes
        total_bytes = layers * traffic.total_bytes
        dense_bytes = layers * traffic.dense_bytes
        projected = {
            "kind": "dynamic_input_pruning_plus_exact_candidate_completion",
            "active_record_fraction": arm.top_k / intermediate_size,
            "candidate_record_fraction": arm.candidate_count / intermediate_size,
            "input_fraction": arm.input_fraction,
            "input_coordinate_count": input_coordinates,
            "partial_key_bytes_per_token": partial_key_bytes,
            "candidate_completion_key_bytes_per_token": completion_key_bytes,
            "selected_value_bytes_per_token": selected_value_bytes,
            "projected_weight_scalar_reads_per_token": scalar_reads,
            "dense_mlp_weight_scalar_reads_per_token": dense_scalar_reads,
            "total_mlp_weight_bytes_per_token": total_bytes,
            "dense_mlp_weight_bytes_per_token": dense_bytes,
            "projected_weight_traffic_fraction": total_bytes / dense_bytes,
            "projected_weight_traffic_reduction": dense_bytes / total_bytes,
            "accounting_formula_scalar_reads_per_layer": "2*I*q + 2*C*(H-q) + K*H",
        }
    else:
        assert arm.variant == "dip_paq"
        assert arm.top_k is not None and arm.candidate_count is not None
        assert arm.input_fraction is not None
        layers = len(arm.layer_indices)
        input_coordinates = _input_coordinate_count(hidden_size, arm.input_fraction)
        traffic = projected_dip_traffic(
            hidden_size,
            intermediate_size,
            input_fraction=arm.input_fraction,
            candidate_count=arm.candidate_count,
            top_k=arm.top_k,
            bytes_per_element=1,
        )
        packed_code_bytes = 0
        physical_code_bytes = 0
        scale_bytes = 0
        codebook_bytes = 0
        norm_bytes = 0
        index_bytes = 0
        codecs: set[str] = set()
        for layer_index in arm.layer_indices:
            layer = paq_layers[layer_index]
            codecs.add(layer.codec)
            # A PAQ code identifies an entire subvector.  Arbitrary top-|x|
            # coordinates normally touch every group, so the strict traffic
            # bound reads all gate/up code pairs once and the selected down
            # records.  Candidate completion reuses the already fetched key
            # codes; charging scalar q elements here would be falsely low.
            layer_code_count = (
                (2 * intermediate_size + arm.top_k)
                * layer.groups_per_record
                * layer.num_codebooks
            )
            layer_code_bytes = math.ceil(
                layer_code_count * layer.code_bits / 8.0
            )
            packed_code_bytes += layer_code_bytes
            physical_code_bytes += math.ceil(
                layer_code_bytes * paq_cacheline_amplification
            )
            scale_bytes += intermediate_size * (
                layer.scale_bytes_per_gate_record
                + layer.scale_bytes_per_up_record
            ) + arm.top_k * layer.scale_bytes_per_down_record
            codebook_bytes += layer.codebook_bytes
            # One cached FP16 norm per down record supports proxy scoring.
            norm_bytes += intermediate_size * 2
            # uint16 coordinate, candidate, and selected-record IDs.
            index_bytes += (
                input_coordinates + arm.candidate_count + arm.top_k
            ) * 2
        dense_q4_bytes = layers * (3 * hidden_size * intermediate_size // 2)
        warm_bytes = physical_code_bytes + scale_bytes + norm_bytes + index_bytes
        cold_bytes = warm_bytes + codebook_bytes
        projected = {
            "kind": "dynamic_input_pruning_plus_product_additive_quantized_records",
            "codec": sorted(codecs),
            "active_record_fraction": arm.top_k / intermediate_size,
            "candidate_record_fraction": arm.candidate_count / intermediate_size,
            "input_fraction": arm.input_fraction,
            "input_coordinate_count": input_coordinates,
            "encoded_bits_per_weight": paq_layers[
                arm.layer_indices[0]
            ].encoded_bits_per_weight,
            "group_size": paq_layers[arm.layer_indices[0]].group_size,
            "groups_per_record": paq_layers[
                arm.layer_indices[0]
            ].groups_per_record,
            "code_access_policy": (
                "strict arbitrary-coordinate bound: all gate/up subvector codes "
                "plus selected down-record codes"
            ),
            "logical_packed_code_bytes_per_token": packed_code_bytes,
            "cacheline_amplification": paq_cacheline_amplification,
            "physical_packed_code_bytes_per_token": physical_code_bytes,
            "record_scale_bytes_per_token": scale_bytes,
            "down_norm_bytes_per_token": norm_bytes,
            "index_bytes_per_token": index_bytes,
            "codebook_cold_bytes_per_token": codebook_bytes,
            "warm_total_bytes_per_token": warm_bytes,
            "cold_total_bytes_per_token": cold_bytes,
            "dense_q4_mlp_bytes_per_token": dense_q4_bytes,
            "warm_fraction_of_dense_q4": warm_bytes / dense_q4_bytes,
            "cold_fraction_of_dense_q4": cold_bytes / dense_q4_bytes,
            "projected_weight_traffic_fraction": cold_bytes / dense_q4_bytes,
            "traffic_gate_maximum_fraction": 0.45,
            "traffic_gate_passed": cold_bytes / dense_q4_bytes <= 0.45,
            "algorithmic_scalar_work_formula_per_layer": (
                "2*I*q + 2*C*(H-q) + K*H"
            ),
            "packed_code_read_formula_per_layer": (
                "(2*I + K)*ceil(H/group_size)*num_codebooks*code_bits"
            ),
            "accounting_policy": (
                "cold gate includes bit-packed codes with measured DIP cache-line "
                "amplification, FP16 record scales, FP16 down norms, uint16 indices, "
                "and all layer-local FP16 codebooks; baseline is dense Q4 MLP weights"
            ),
        }
    return {
        "name": arm.name,
        "variant": arm.variant,
        "scope": arm.scope,
        "layer_indices": list(arm.layer_indices),
        "top_k": arm.top_k,
        "layer_top_ks": list(arm.layer_top_ks)
        if arm.layer_top_ks is not None
        else None,
        "candidate_count": arm.candidate_count,
        "input_fraction": arm.input_fraction,
        "quantization": arm.quantization,
        "quality": metric_summary,
        "local_mlp": combined.summary(),
        "local_mlp_by_layer": local_layers,
        "projected_accounting": projected,
        "router_training": (
            [
                {
                    "layer": layer,
                    **overlap_routers[
                        (layer, arm.top_k, arm.candidate_count)
                    ].training_metadata,
                }
                for layer in arm.layer_indices
            ]
            if arm.variant == "overlap"
            else None
        ),
        "timing_valid": False,
    }


def evaluate_mlp_interventions(
    model: str | Path,
    dataset: str | Path,
    *,
    calibration_traces: str | Path | None = None,
    variants: Sequence[str] = ("identity", "oracle"),
    top_ks: Sequence[int] = (256,),
    candidate_counts: Sequence[int] = (512,),
    input_fractions: Sequence[float] = (0.75,),
    rank: int = 16,
    regularization: float = 1000.0,
    calibration_records: int = 128,
    posting_groups: int = 96,
    posting_size: int = 32,
    overlap_iterations: int = 8,
    max_replication: int = 4,
    layers: Sequence[int] | None = None,
    layer_mode: str = "both",
    max_records: int | None = None,
    device: str = "cpu",
    allow_calibration_overlap: bool = False,
    evaluation_role: str = "development",
    configuration_selection_traces: str | Path | None = None,
    layer_top_ks: Sequence[int] | None = None,
    paq_group_size: int = 8,
    paq_codebooks: int = 2,
    paq_codebook_size: int = 128,
    paq_iterations: int = 8,
    paq_sample_limit: int | None = 8192,
    paq_seed: int = 73,
    paq_cacheline_amplification: float = 12.0 / 11.0,
) -> dict[str, Any]:
    """Measure teacher-forced quality after replacing selected transformer MLPs.

    ``oracle`` retains the contribution-magnitude top-K records using every
    activation. It is a full-information reference, not a provably optimal
    K-subset. ``rank16`` learns that membership from calibration traces, selects
    a candidate set, and exactly reranks only within that set. ``overlap`` predicts
    a learned combination of overlapping coverage-trained postings. ``dip`` uses
    the largest-magnitude hidden coordinates for a partial predictor-free SwiGLU
    evaluation, then exactly completes and reranks only its candidate records.
    """

    try:
        import torch
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except ImportError:

                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] to evaluate Hugging Face interventions"
        ) from exc

    dataset_path = Path(dataset)
    if not dataset_path.is_file():
        raise ValueError(f"evaluation dataset is not a file: {dataset_path}")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive")
    if calibration_records <= 0:
        raise ValueError("calibration_records must be positive")
    normalized_variants = tuple(dict.fromkeys(str(value) for value in variants))
    unknown_variants = set(normalized_variants) - set(SUPPORTED_VARIANTS)
    if unknown_variants:
        raise ValueError(
            f"unsupported intervention variants: {sorted(unknown_variants)}"
        )
    if not normalized_variants:
        raise ValueError("at least one intervention variant is required")
    if layer_mode not in SUPPORTED_LAYER_MODES:
        raise ValueError(f"layer_mode must be one of {SUPPORTED_LAYER_MODES}")
    if evaluation_role not in SUPPORTED_EVALUATION_ROLES:
        raise ValueError(f"evaluation_role must be one of {SUPPORTED_EVALUATION_ROLES}")

    model_path = resolve_model_path(model)
    inspection = inspect_model(model_path)
    requires_top_k = any(value != "identity" for value in normalized_variants)
    top_k_values = (
        tuple(dict.fromkeys(int(value) for value in top_ks)) if requires_top_k else ()
    )
    candidate_values = tuple(dict.fromkeys(int(value) for value in candidate_counts))
    input_fraction_values = tuple(
        dict.fromkeys(float(value) for value in input_fractions)
    )
    if requires_top_k and (
        not top_k_values
        or any(
            value <= 0 or value > inspection.intermediate_size for value in top_k_values
        )
    ):
        raise ValueError("top_ks must lie within the intermediate size")
    learned_variants = {"rank16", "overlap"}.intersection(normalized_variants)
    candidate_variants = {"rank16", "overlap", "dip", "dip_paq"}.intersection(
        normalized_variants
    )
    if candidate_variants:
        if not candidate_values or any(
            value <= 0 or value > inspection.intermediate_size
            for value in candidate_values
        ):
            raise ValueError("candidate_counts must lie within the intermediate size")
        if any(
            candidate < top_k
            for candidate in candidate_values
            for top_k in top_k_values
        ):
            raise ValueError("every candidate_count must be at least every top_k")
    if {"dip", "dip_paq"}.intersection(normalized_variants) and (
        not input_fraction_values
        or any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in input_fraction_values
        )
    ):
        raise ValueError("input_fractions must be finite and lie in (0, 1]")
    if (
        configuration_selection_traces is not None
        and not {"dip", "dip_paq"}.intersection(normalized_variants)
        and layer_top_ks is None
    ):
        raise ValueError(
            "configuration_selection_traces requires DIP or layer_top_ks"
        )
    if "dip_paq" in normalized_variants:
        for name, value in (
            ("paq_group_size", paq_group_size),
            ("paq_codebooks", paq_codebooks),
            ("paq_codebook_size", paq_codebook_size),
            ("paq_iterations", paq_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if paq_codebooks > 2:
            raise ValueError("paq_codebooks must be one or two")
        if paq_codebook_size > 256:
            raise ValueError("paq_codebook_size must not exceed 256")
        if paq_sample_limit is not None and paq_sample_limit <= 0:
            raise ValueError("paq_sample_limit must be positive when provided")
        if (
            not math.isfinite(paq_cacheline_amplification)
            or paq_cacheline_amplification < 1.0
        ):
            raise ValueError("paq_cacheline_amplification must be finite and >= 1")
    if evaluation_role == "confirmation" and configuration_selection_traces is None:
        raise ValueError(
            "confirmation evaluation requires configuration_selection_traces"
        )
    if learned_variants:
        if calibration_traces is None:
            raise ValueError("routed interventions require calibration_traces")
        if rank <= 0 or rank > min(
            inspection.hidden_size, inspection.intermediate_size
        ):
            raise ValueError("rank must lie within the router dimensions")
        if not math.isfinite(regularization) or regularization <= 0:
            raise ValueError("regularization must be finite and positive")
    if "overlap" in normalized_variants:
        if posting_groups <= 0 or posting_size <= 0:
            raise ValueError("posting_groups and posting_size must be positive")
        slots = posting_groups * posting_size
        if slots < inspection.intermediate_size:
            raise ValueError(
                "overlapping postings require at least one slot per record"
            )
        if slots > inspection.intermediate_size * max_replication:
            raise ValueError("max_replication is too small for the posting slots")
        if overlap_iterations <= 0:
            raise ValueError("overlap_iterations must be positive")

    if layers is None:
        layer_indices = tuple(range(inspection.num_hidden_layers))
    else:
        layer_indices = tuple(dict.fromkeys(int(value) for value in layers))
        if not layer_indices or any(
            value < 0 or value >= inspection.num_hidden_layers
            for value in layer_indices
        ):
            raise ValueError("layers must contain valid layer indices")
    adaptive_top_ks = (
        tuple(int(value) for value in layer_top_ks)
        if layer_top_ks is not None
        else None
    )
    if adaptive_top_ks is not None:
        if len(adaptive_top_ks) != len(layer_indices) or any(
            value <= 0 or value > inspection.intermediate_size
            for value in adaptive_top_ks
        ):
            raise ValueError(
                "layer_top_ks must provide one valid active count per selected layer"
            )
        if set(normalized_variants) - {"identity"}:
            raise ValueError(
                "layer_top_ks must be evaluated with the identity variant only"
            )

    records = _load_jsonl(dataset_path, max_records)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)
    loaded_model.eval()
    if not hasattr(loaded_model, "model") or not hasattr(loaded_model.model, "layers"):
        raise RuntimeError(
            "loaded model does not expose Llama-compatible decoder layers"
        )
    if len(loaded_model.model.layers) != inspection.num_hidden_layers:
        raise RuntimeError("loaded layer count differs from inspected configuration")
    tokenizer = None
    if any("input_ids" not in record for record in records):
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    evaluation_sequence_hashes = _evaluation_sequence_hashes(records, tokenizer)

    configuration_selection_metadata = None
    if configuration_selection_traces is not None:
        selection = TraceReader(configuration_selection_traces)
        if selection.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("configuration-selection trace/model hash mismatch")
        selection_sequence_hashes = _trace_sequence_hashes(selection)
        selection_overlap = set(selection_sequence_hashes).intersection(
            evaluation_sequence_hashes
        )
        if selection_overlap:
            raise ValueError(
                "configuration-selection and evaluation contain matching token sequences"
            )
        configuration_selection_metadata = {
            "trace_path": str(Path(configuration_selection_traces).resolve()),
            "dataset_hash": selection.manifest["dataset_hash"],
            "separation_method": "exact_token_sequence_hashes",
            "selection_sequence_count": len(selection_sequence_hashes),
            "selection_unique_sequence_count": len(set(selection_sequence_hashes)),
            "evaluation_sequence_count": len(evaluation_sequence_hashes),
            "evaluation_unique_sequence_count": len(set(evaluation_sequence_hashes)),
            "overlapping_sequence_count": 0,
            "held_out_from_configuration_selection": True,
        }

    routers: dict[tuple[int, int], LowRankMultiLabelRouter] = {}
    overlap_routers: dict[tuple[int, int, int], OverlappingCoverageRouter] = {}
    calibration_metadata = None
    if learned_variants:
        assert calibration_traces is not None
        calibration = TraceReader(calibration_traces)
        if calibration.manifest["model_hash"] != inspection.source_hash:
            raise ValueError("calibration trace/model hash mismatch")
        if calibration.manifest["split"] != "calibration":
            raise ValueError("expected 'calibration' trace split")
        evaluation_hash = sha256_file(dataset_path)
        calibration_sequence_hashes = _trace_sequence_hashes(calibration)
        overlapping_sequence_hashes = set(calibration_sequence_hashes).intersection(
            evaluation_sequence_hashes
        )
        record_level_disjoint = not overlapping_sequence_hashes
        if not record_level_disjoint and not allow_calibration_overlap:
            raise ValueError(
                "calibration and evaluation contain matching token sequences; "
                "held-out evaluation is required"
            )
        if "rank16" in normalized_variants:
            routers = _fit_rank_routers(
                loaded_model,
                inspection,
                calibration,
                layer_indices,
                top_k_values,
                rank=rank,
                regularization=regularization,
                calibration_records=calibration_records,
            )
        if "overlap" in normalized_variants:
            overlap_routers = _fit_overlap_routers(
                loaded_model,
                inspection,
                calibration,
                layer_indices,
                top_k_values,
                candidate_values,
                rank=rank,
                regularization=regularization,
                calibration_records=calibration_records,
                groups=posting_groups,
                posting_size=posting_size,
                iterations=overlap_iterations,
                max_replication=max_replication,
            )
        calibration_metadata = {
            "trace_path": str(Path(calibration_traces).resolve()),
            "dataset_hash": calibration.manifest["dataset_hash"],
            "dataset_files_differ": calibration.manifest["dataset_hash"]
            != evaluation_hash,
            "separation_method": "exact_token_sequence_hashes",
            "calibration_sequence_count": len(calibration_sequence_hashes),
            "calibration_unique_sequence_count": len(set(calibration_sequence_hashes)),
            "evaluation_sequence_count": len(evaluation_sequence_hashes),
            "evaluation_unique_sequence_count": len(set(evaluation_sequence_hashes)),
            "overlapping_sequence_count": len(overlapping_sequence_hashes),
            "record_level_disjoint": record_level_disjoint,
            "records_per_layer_limit": calibration_records,
            "regularization": regularization,
            "rank": rank,
            "posting_groups": (
                posting_groups if "overlap" in normalized_variants else None
            ),
            "posting_size": posting_size if "overlap" in normalized_variants else None,
            "overlap_iterations": (
                overlap_iterations if "overlap" in normalized_variants else None
            ),
            "max_replication": (
                max_replication if "overlap" in normalized_variants else None
            ),
            "held_out_from_evaluation": record_level_disjoint,
        }

    examples, baseline = _prepare_examples(
        loaded_model,
        tokenizer,
        records,
        device=device,
        torch=torch,
    )
    if len(evaluation_sequence_hashes) != baseline["sequences"]:
        raise RuntimeError(
            "evaluation sequence provenance count differs from executed examples"
        )
    baseline["unique_sequences"] = len(set(evaluation_sequence_hashes))
    paq_layers: dict[int, _PAQLayer] = {}
    paq_label = (
        f"{paq_codebooks}x{max(1, (paq_codebook_size - 1).bit_length())}"
        f"_g{paq_group_size}"
    )
    if "dip_paq" in normalized_variants:
        paq_layers = _fit_paq_layers(
            loaded_model,
            layer_indices,
            group_size=paq_group_size,
            num_codebooks=paq_codebooks,
            codebook_size=paq_codebook_size,
            iterations=paq_iterations,
            sample_limit=paq_sample_limit,
            seed=paq_seed,
            torch=torch,
        )
    arms = _make_arms(
        normalized_variants,
        layer_mode,
        layer_indices,
        top_k_values,
        candidate_values,
        input_fraction_values,
        rank=rank,
        posting_groups=posting_groups,
        posting_size=posting_size,
        paq_label=paq_label,
    )
    if adaptive_top_ks is not None:
        adaptive_mean = sum(adaptive_top_ks) / len(adaptive_top_ks)
        arms.append(
            _Arm(
                "oracle",
                layer_indices,
                "all",
                int(round(adaptive_mean)),
                layer_top_ks=adaptive_top_ks,
            )
        )
    results = [
        _evaluate_arm(
            loaded_model,
            examples,
            arm,
            routers,
            overlap_routers,
            paq_layers,
            hidden_size=inspection.hidden_size,
            intermediate_size=inspection.intermediate_size,
            paq_cacheline_amplification=paq_cacheline_amplification,
            torch=torch,
            device=device,
        )
        for arm in arms
    ]
    report = {
        "schema_version": 1,
        "experiment": "trained_teacher_mlp_intervention",
        "status": "local_teacher_intervention_measurement",
        "source_model_hash": inspection.source_hash,
        "num_hidden_layers": inspection.num_hidden_layers,
        "intermediate_size": inspection.intermediate_size,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_hash": sha256_file(dataset_path),
        "evaluation_role": evaluation_role,
        "configuration_selection": configuration_selection_metadata,
        "selected_layers": list(layer_indices),
        "layer_mode": layer_mode,
        "variants": list(normalized_variants),
        "top_ks": list(top_k_values),
        "layer_top_ks": list(adaptive_top_ks)
        if adaptive_top_ks is not None
        else None,
        "candidate_counts": list(candidate_values) if candidate_variants else [],
        "input_fractions": (
            list(input_fraction_values)
            if {"dip", "dip_paq"}.intersection(normalized_variants)
            else []
        ),
        "product_additive_quantization": (
            {
                "group_size": paq_group_size,
                "num_codebooks": paq_codebooks,
                "codebook_size": paq_codebook_size,
                "iterations": paq_iterations,
                "sample_limit": paq_sample_limit,
                "seed": paq_seed,
                "cacheline_amplification": paq_cacheline_amplification,
                "quality_execution": "decoded_float32_from_bit_packed_encoding",
                "timing_valid": False,
            }
            if "dip_paq" in normalized_variants
            else None
        ),
        "rank": rank if learned_variants else None,
        "calibration": calibration_metadata,
        "baseline": baseline,
        "arms": results,
        "measurement_caveat": (
            "Forward hooks replace MLP outputs after the dense Hugging Face MLP executes; "
            "quality metrics are valid, but evaluator wall time is not an inference benchmark."
        ),
        "oracle_caveat": MAGNITUDE_REFERENCE_CAVEAT,
        "quality_targets_met": None,
    }
    return apply_mlp_intervention_gates(report)


__all__ = [
    "SUPPORTED_EVALUATION_ROLES",
    "SUPPORTED_LAYER_MODES",
    "SUPPORTED_VARIANTS",
    "evaluate_mlp_interventions",
]
