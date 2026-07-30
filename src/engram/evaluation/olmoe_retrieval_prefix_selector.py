"""Cached, train-only prefix selector for the OLMoE retrieval experiment.

This command consumes the already-authenticated retrieval protocol and
completed training checkpoint.  It performs no backward pass and does not
load a dense teacher.  Instead, it exhaustively fits two balanced prototype
head masks from the cached per-record M1 gradients, proves that the resulting
partition is recoverable from the causal input prefix, and evaluates both
exact-51 masks on every training record through the packaged native Q7
runtime.

Development outcomes are not inputs to this screen.  The sealed confirmation
file remains unopened.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import engram.evaluation.olmoe_causal_head_gate as causal_gate
import engram.evaluation.olmoe_retrieval_head_selector as retrieval
from engram.utils import atomic_json, sha256_file, sha256_json


_SCHEMA_VERSION = 1
_EXPERIMENT = "olmoe_q7_retrieval_prefix_selector_train_screen"
_RECORDS = 8
_PROTOTYPE_COUNT = 2
_CLUSTER_SIZE = 4
_PREFIX_LAST_INPUT_INDEX = 96
_PREFIX_INPUT_COUNT = _PREFIX_LAST_INPUT_INDEX + 1
_D_LATER = "D_later_half"
_D_EARLIER = "D_earlier_half"
_PROTOTYPE_NAMES = (_D_LATER, _D_EARLIER)
_EXPECTED_D_LATER_INDICES = (0, 1, 4, 6)
_EXPECTED_D_EARLIER_INDICES = (2, 3, 5, 7)
_D_LATER_MASK_SHA256 = (
    "5abc4ff25d8054acb960650ec098de569fd8662ed914e2fff51e62abbdc856b5"
)
_D_EARLIER_MASK_SHA256 = (
    "5d3d37a5985801bafbc27fa114aaeec4c48ea1ccf4d4af40d487e984e0ef2846"
)


def _progress(message: str) -> None:
    print(f"[retrieval-prefix-selector] {message}", file=sys.stderr, flush=True)


def _fact_anchor_ids(tokenizer_path: str | Path) -> dict[str, tuple[int, ...]]:
    """Return the four-token source headers used to identify fact order."""

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval prefix screening"
        ) from exc
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path).resolve()))
    anchors = {
        label: tuple(
            int(value)
            for value in tokenizer.encode(
                f" Key {label} has code",
                add_special_tokens=False,
            ).ids
        )
        for label in retrieval._LABELS
    }
    return _validate_fact_anchor_ids(anchors)


def _validate_fact_anchor_ids(
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(anchors, Mapping) or set(anchors) != set(retrieval._LABELS):
        raise ValueError("retrieval prefix tokenizer anchors are invalid")
    result: dict[str, tuple[int, ...]] = {}
    for label in retrieval._LABELS:
        values = anchors[label]
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, Sequence)
            or len(values) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in values
            )
        ):
            raise ValueError("retrieval prefix tokenizer anchors are invalid")
        result[label] = tuple(int(value) for value in values)
    if len(set(result.values())) != len(retrieval._LABELS):
        raise ValueError("retrieval prefix tokenizer anchors are not unique")
    return result


def _fact_order_from_causal_prefix(
    input_ids: Sequence[int],
    anchors: Mapping[str, Sequence[int]],
) -> tuple[str, ...]:
    """Read fact order without inspecting any token after input row 96."""

    headers = _validate_fact_anchor_ids(anchors)
    if (
        isinstance(input_ids, (str, bytes))
        or not isinstance(input_ids, Sequence)
        or len(input_ids) < _PREFIX_INPUT_COUNT
    ):
        raise ValueError("retrieval causal prefix input IDs are invalid")
    prefix = tuple(input_ids[:_PREFIX_INPUT_COUNT])
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in prefix
    ):
        raise ValueError("retrieval causal prefix input IDs are invalid")
    order: list[str] = []
    for start in retrieval._FACT_ANCHORS:
        observed = tuple(int(value) for value in prefix[start : start + 4])
        matches = [label for label, header in headers.items() if observed == header]
        if len(matches) != 1:
            raise ValueError("retrieval fact header is not identifiable from prefix")
        order.append(matches[0])
    if len(set(order)) != len(retrieval._LABELS):
        raise ValueError("retrieval causal prefix does not contain each fact once")
    return tuple(order)


def _prefix_partition(
    records: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Allocate records by whether source fact D appears in slots 2/3."""

    if (
        not isinstance(records, Sequence)
        or len(records) != _RECORDS
        or [record.get("record_index") for record in records] != list(range(_RECORDS))
    ):
        raise ValueError("retrieval prefix training records are invalid")
    later: list[int] = []
    earlier: list[int] = []
    orders: list[dict[str, Any]] = []
    for record in records:
        index = int(record["record_index"])
        order = _fact_order_from_causal_prefix(record.get("input_ids"), anchors)
        d_slot = order.index("D")
        allocation = _D_LATER if d_slot >= 2 else _D_EARLIER
        (later if allocation == _D_LATER else earlier).append(index)
        orders.append(
            {
                "record_index": index,
                "prefix_fact_order": list(order),
                "D_source_slot": d_slot,
                "allocation": allocation,
            }
        )
    if len(later) != _CLUSTER_SIZE or len(earlier) != _CLUSTER_SIZE:
        raise ValueError("retrieval prefix rule did not produce a balanced partition")
    return {
        "rule": "fact D source slot >= 2 using only input_ids[0:97]",
        "last_input_index_observed": _PREFIX_LAST_INPUT_INDEX,
        "future_answer_tokens_observed": False,
        "clusters": {
            _D_LATER: later,
            _D_EARLIER: earlier,
        },
        "records": orders,
    }


def _boolean_exact_51(value: Any, label: str) -> np.ndarray:
    mask = retrieval._boolean_mask(value, label)
    if int(mask.sum()) != retrieval._RESCUED_HEADS:
        raise ValueError(f"{label} does not contain exactly 51 heads")
    return mask


def _training_state(
    training: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Extract only the cached M1 gradients and recorded M2 train outcomes."""

    if not isinstance(training, Mapping):
        raise ValueError("retrieval prefix checkpoint training payload is invalid")
    entries = training.get("masks")
    if not isinstance(entries, Mapping):
        raise ValueError("retrieval prefix checkpoint masks are invalid")
    m1 = entries.get("M1")
    m2 = entries.get("M2")
    if not isinstance(m1, Mapping) or not isinstance(m2, Mapping):
        raise ValueError("retrieval prefix M1/M2 checkpoint entries are invalid")
    base_mask = _boolean_exact_51(m1.get("mask"), "retrieval prefix M1")
    global_mask = _boolean_exact_51(m2.get("mask"), "retrieval prefix M2")
    m1_rows = m1.get("records")
    m2_rows = m2.get("records")
    expected_indices = list(range(_RECORDS))
    expected_ids = [record.get("record_id") for record in records]
    if (
        not isinstance(m1_rows, list)
        or not isinstance(m2_rows, list)
        or len(m1_rows) != _RECORDS
        or len(m2_rows) != _RECORDS
        or [row.get("record_index") for row in m1_rows] != expected_indices
        or [row.get("record_index") for row in m2_rows] != expected_indices
        or [row.get("record_id") for row in m1_rows] != expected_ids
        or [row.get("record_id") for row in m2_rows] != expected_ids
    ):
        raise ValueError("retrieval prefix cached record ordering changed")
    gradients = tuple(
        retrieval._finite_matrix(
            row.get("gradient"),
            f"retrieval prefix M1 record {index} gradient",
        )
        for index, row in enumerate(m1_rows)
    )
    baseline: list[float] = []
    for index, row in enumerate(m2_rows):
        loss = row.get("loss")
        value = loss.get("answer_cross_entropy") if isinstance(loss, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"retrieval prefix M2 record {index} loss is invalid")
        baseline.append(float(value))
    return {
        "base_mask": base_mask,
        "global_mask": global_mask,
        "gradients": gradients,
        "global_M2_answer_cross_entropy": baseline,
    }


def _proximal_objective(
    gradient: np.ndarray,
    base_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> float:
    values = np.asarray(gradient, dtype=np.float64)
    base = np.asarray(base_mask, dtype=np.bool_)
    candidate = np.asarray(candidate_mask, dtype=np.bool_)
    if (
        values.shape != (retrieval._LAYERS, retrieval._HEADS)
        or base.shape != values.shape
        or candidate.shape != values.shape
        or int(base.sum()) != retrieval._RESCUED_HEADS
        or int(candidate.sum()) != retrieval._RESCUED_HEADS
        or not np.isfinite(values).all()
    ):
        raise ValueError("retrieval prefix proximal objective inputs are invalid")
    rms = float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise ValueError("retrieval prefix record gradient RMS is invalid")
    delta = candidate.astype(np.float64) - base.astype(np.float64)
    return float(
        np.sum(values * delta, dtype=np.float64)
        + 0.5
        * (rms + causal_gate._PROJECTED_GRADIENT_EPSILON)
        * np.sum(np.square(delta), dtype=np.float64)
    )


def _project_average_gradient(
    base_mask: np.ndarray,
    gradients: Sequence[np.ndarray],
) -> dict[str, Any]:
    average = np.mean(np.stack(gradients), axis=0, dtype=np.float64)
    scores, mask, rms = causal_gate._projected_gate_step(base_mask, average)
    return {
        "average_gradient": average,
        "average_gradient_sha256": sha256_json(average.tolist()),
        "scores": scores,
        "scores_sha256": sha256_json(scores.tolist()),
        "mask": mask,
        "mask_sha256": sha256_json(mask.tolist()),
        "gradient_rms": rms,
    }


def _balanced_partitions() -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Enumerate each unlabeled 4/4 split exactly once."""

    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for partners in itertools.combinations(range(1, _RECORDS), _CLUSTER_SIZE - 1):
        first = (0, *partners)
        second = tuple(index for index in range(_RECORDS) if index not in first)
        result.append((first, second))
    expected = 35
    if len(result) != expected or len(set(result)) != expected:
        raise AssertionError("retrieval balanced-partition enumeration changed")
    return result


def _derive_balanced_partition(
    base_mask: np.ndarray,
    global_mask: np.ndarray,
    gradients: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Fit the minimax, then mean-regret, balanced two-prototype split."""

    base = np.asarray(base_mask, dtype=np.bool_)
    global_m2 = np.asarray(global_mask, dtype=np.bool_)
    values = tuple(np.asarray(gradient, dtype=np.float64) for gradient in gradients)
    if (
        base.shape != (retrieval._LAYERS, retrieval._HEADS)
        or global_m2.shape != base.shape
        or int(base.sum()) != retrieval._RESCUED_HEADS
        or int(global_m2.sum()) != retrieval._RESCUED_HEADS
        or len(values) != _RECORDS
        or any(
            value.shape != base.shape or not np.isfinite(value).all()
            for value in values
        )
    ):
        raise ValueError("retrieval cached partition inputs are invalid")
    individual_masks: list[np.ndarray] = []
    individual_objectives: list[float] = []
    global_regrets: list[float] = []
    for gradient in values:
        _scores, mask, _rms = causal_gate._projected_gate_step(base, gradient)
        individual_masks.append(mask)
        individual = _proximal_objective(gradient, base, mask)
        individual_objectives.append(individual)
        global_regrets.append(
            _proximal_objective(gradient, base, global_m2) - individual
        )
    candidates: list[dict[str, Any]] = []
    for clusters in _balanced_partitions():
        prototypes = tuple(
            _project_average_gradient(
                base,
                [values[index] for index in cluster],
            )
            for cluster in clusters
        )
        regrets: list[float] = []
        for index, gradient in enumerate(values):
            cluster_index = 0 if index in clusters[0] else 1
            regret = (
                _proximal_objective(
                    gradient,
                    base,
                    prototypes[cluster_index]["mask"],
                )
                - individual_objectives[index]
            )
            if regret < -1.0e-10:
                raise ValueError("retrieval prefix proximal regret became negative")
            regrets.append(max(0.0, regret))
        candidates.append(
            {
                "clusters": clusters,
                "prototypes": prototypes,
                "regrets": tuple(regrets),
                "key": (
                    max(regrets),
                    float(np.mean(regrets, dtype=np.float64)),
                    clusters[0],
                ),
            }
        )
    best = min(candidates, key=lambda candidate: candidate["key"])
    return {
        "candidate_partition_count": len(candidates),
        "selection_order": [
            "minimum maximum assigned proximal regret",
            "minimum mean assigned proximal regret",
            "lexicographically smallest canonical cluster containing record 0",
        ],
        "clusters": best["clusters"],
        "prototypes": best["prototypes"],
        "assigned_regrets": best["regrets"],
        "global_M2_regrets": tuple(global_regrets),
        "individual_mask_sha256": tuple(
            sha256_json(mask.tolist()) for mask in individual_masks
        ),
        "objective": {
            "maximum_assigned_proximal_regret": float(best["key"][0]),
            "mean_assigned_proximal_regret": float(best["key"][1]),
        },
        "global_M2_objective": {
            "maximum_proximal_regret": float(max(global_regrets)),
            "mean_proximal_regret": float(np.mean(global_regrets, dtype=np.float64)),
        },
    }


def _bind_and_validate_prefix_prototypes(
    partition: Mapping[str, Any],
    prefix: Mapping[str, Any],
    global_mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    clusters = partition.get("clusters")
    prefix_clusters = prefix.get("clusters")
    if (
        clusters != (_EXPECTED_D_LATER_INDICES, _EXPECTED_D_EARLIER_INDICES)
        or not isinstance(prefix_clusters, Mapping)
        or tuple(prefix_clusters.get(_D_LATER, ())) != _EXPECTED_D_LATER_INDICES
        or tuple(prefix_clusters.get(_D_EARLIER, ())) != _EXPECTED_D_EARLIER_INDICES
    ):
        raise ValueError(
            "retrieval cached partition does not equal the causal prefix rule"
        )
    raw_prototypes = partition.get("prototypes")
    if not isinstance(raw_prototypes, tuple) or len(raw_prototypes) != 2:
        raise ValueError("retrieval prefix prototypes are invalid")
    expected_hashes = {
        _D_LATER: _D_LATER_MASK_SHA256,
        _D_EARLIER: _D_EARLIER_MASK_SHA256,
    }
    result: dict[str, dict[str, Any]] = {}
    for name, cluster, prototype in zip(
        _PROTOTYPE_NAMES,
        clusters,
        raw_prototypes,
        strict=True,
    ):
        if not isinstance(prototype, Mapping):
            raise ValueError("retrieval prefix prototype is invalid")
        mask = np.asarray(prototype.get("mask"), dtype=np.bool_)
        scores = np.asarray(prototype.get("scores"), dtype=np.float64)
        digest = sha256_json(mask.tolist())
        if (
            mask.shape != (retrieval._LAYERS, retrieval._HEADS)
            or scores.shape != mask.shape
            or int(mask.sum()) != retrieval._RESCUED_HEADS
            or not np.isfinite(scores).all()
            or prototype.get("mask_sha256") != digest
            or digest != expected_hashes[name]
        ):
            raise ValueError(f"retrieval prefix {name} frozen mask changed")
        selected = causal_gate._selected_head_rows(mask, scores)
        if len(selected) != retrieval._RESCUED_HEADS:
            raise ValueError("retrieval prefix selected-head ranking is invalid")
        additions = np.argwhere(mask & ~global_mask)
        removals = np.argwhere(global_mask & ~mask)
        result[name] = {
            "cluster_record_indices": list(cluster),
            "mask": mask,
            "scores": scores,
            "report": {
                "mask": mask.tolist(),
                "mask_sha256": digest,
                "selected_head_count": int(mask.sum()),
                "selected_heads": selected,
                "average_gradient_sha256": prototype["average_gradient_sha256"],
                "projected_scores_sha256": prototype["scores_sha256"],
                "average_gradient_rms": float(prototype["gradient_rms"]),
                "global_M2_overlap": int(np.logical_and(mask, global_mask).sum()),
                "added_vs_global_M2": [
                    {"layer": int(layer), "head": int(head)}
                    for layer, head in additions
                ],
                "removed_vs_global_M2": [
                    {"layer": int(layer), "head": int(head)} for layer, head in removals
                ],
            },
        }
    if np.array_equal(result[_D_LATER]["mask"], result[_D_EARLIER]["mask"]):
        raise ValueError("retrieval prefix prototype masks are not distinct")
    return result


def _evaluate_prototype_transfer(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    prototypes: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Run two masks over all eight train records: 16 native sequences."""

    results: dict[str, list[dict[str, Any]]] = {}
    for name in _PROTOTYPE_NAMES:
        prototype = prototypes[name]
        mask = np.asarray(prototype["mask"], dtype=np.bool_)
        selected_heads = retrieval._mask_pairs(mask)
        rows: list[dict[str, Any]] = []
        with retrieval._open_native_runtime(context, selected_heads) as runtime:
            for record in records:
                index = int(record["record_index"])
                label = f"{name} train record {index + 1}/{_RECORDS}"
                _progress(f"{label}: starting native Q7 transfer evaluation")
                _logits, _hidden, evidence = retrieval._execute_native_record(
                    runtime,
                    record=record,
                    context=context,
                    selected_heads=selected_heads,
                    progress_label=label,
                )
                rows.append(dict(evidence))
                _progress(f"{label}: complete")
        results[name] = rows
    return results


def _transfer_gate(
    *,
    records: Sequence[Mapping[str, Any]],
    baseline: Sequence[float],
    prefix: Mapping[str, Any],
    evaluations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the full 8x2 matrix and apply the strict train-only gate."""

    if len(records) != _RECORDS or len(baseline) != _RECORDS:
        raise ValueError("retrieval prefix transfer population is invalid")
    prefix_rows = prefix.get("records")
    if not isinstance(prefix_rows, list) or len(prefix_rows) != _RECORDS:
        raise ValueError("retrieval prefix allocation evidence is invalid")
    allocations = {int(row["record_index"]): row["allocation"] for row in prefix_rows}
    normalized: dict[str, list[dict[str, Any]]] = {}
    for name in _PROTOTYPE_NAMES:
        rows = evaluations.get(name)
        if (
            not isinstance(rows, Sequence)
            or len(rows) != _RECORDS
            or [row.get("record_index") for row in rows] != list(range(_RECORDS))
            or [row.get("record_id") for row in rows]
            != [record.get("record_id") for record in records]
        ):
            raise ValueError(f"retrieval prefix {name} transfer rows are invalid")
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            value = row.get("answer_cross_entropy")
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("retrieval prefix native transfer loss is invalid")
            normalized_rows.append(dict(row))
        normalized[name] = normalized_rows
    matrix: list[dict[str, Any]] = []
    assigned_losses: list[float] = []
    deltas: list[float] = []
    for index, record in enumerate(records):
        allocation = allocations[index]
        if allocation not in _PROTOTYPE_NAMES:
            raise ValueError("retrieval prefix allocation name is invalid")
        losses = {
            name: float(normalized[name][index]["answer_cross_entropy"])
            for name in _PROTOTYPE_NAMES
        }
        assigned = losses[allocation]
        reference = float(baseline[index])
        delta = assigned - reference
        assigned_losses.append(assigned)
        deltas.append(delta)
        matrix.append(
            {
                "record_index": index,
                "record_id": record["record_id"],
                "prefix_allocation": allocation,
                "prototype_answer_cross_entropy": losses,
                "assigned_answer_cross_entropy": assigned,
                "global_M2_answer_cross_entropy": reference,
                "assigned_minus_global_M2": delta,
                "no_record_regression": delta <= 0.0,
            }
        )
    baseline_worst = float(max(baseline))
    baseline_mean = float(np.mean(baseline, dtype=np.float64))
    assigned_worst = float(max(assigned_losses))
    assigned_mean = float(np.mean(assigned_losses, dtype=np.float64))
    cluster_summaries: dict[str, dict[str, float]] = {}
    cluster_checks: dict[str, bool] = {}
    for name in _PROTOTYPE_NAMES:
        indices = [index for index in range(_RECORDS) if allocations[index] == name]
        if len(indices) != _CLUSTER_SIZE:
            raise ValueError("retrieval prefix assigned cluster size changed")
        reference_values = [float(baseline[index]) for index in indices]
        candidate_values = [assigned_losses[index] for index in indices]
        reference_worst = float(max(reference_values))
        reference_mean = float(np.mean(reference_values, dtype=np.float64))
        candidate_worst = float(max(candidate_values))
        candidate_mean = float(np.mean(candidate_values, dtype=np.float64))
        cluster_summaries[name] = {
            "global_M2_maximum_answer_cross_entropy": reference_worst,
            "global_M2_mean_answer_cross_entropy": reference_mean,
            "assigned_maximum_answer_cross_entropy": candidate_worst,
            "assigned_mean_answer_cross_entropy": candidate_mean,
            "assigned_maximum_minus_global_M2": (candidate_worst - reference_worst),
            "assigned_mean_minus_global_M2": candidate_mean - reference_mean,
        }
        cluster_checks[f"{name}_worst_strictly_improved"] = (
            candidate_worst < reference_worst
        )
        cluster_checks[f"{name}_mean_strictly_improved"] = (
            candidate_mean < reference_mean
        )
    checks = {
        "complete_8_by_2_transfer_matrix": (
            sum(len(rows) for rows in normalized.values())
            == _RECORDS * _PROTOTYPE_COUNT
        ),
        "assigned_worst_strictly_improved": assigned_worst < baseline_worst,
        "assigned_mean_strictly_improved": assigned_mean < baseline_mean,
        "no_record_regression": all(delta <= 0.0 for delta in deltas),
        **cluster_checks,
    }
    return {
        "native_sequence_forwards": _RECORDS * _PROTOTYPE_COUNT,
        "native_token_steps": (
            _RECORDS * _PROTOTYPE_COUNT * retrieval._PREDICTION_POSITIONS
        ),
        "prototype_evidence": normalized,
        "matrix": matrix,
        "summaries": {
            "global_M2": {
                "maximum_answer_cross_entropy": baseline_worst,
                "mean_answer_cross_entropy": baseline_mean,
            },
            "assigned_prefix_prototypes": {
                "maximum_answer_cross_entropy": assigned_worst,
                "mean_answer_cross_entropy": assigned_mean,
                "maximum_minus_global_M2": assigned_worst - baseline_worst,
                "mean_minus_global_M2": assigned_mean - baseline_mean,
            },
            "assigned_clusters": cluster_summaries,
        },
        "gate_checks": checks,
        "passed": all(checks.values()),
    }


def screen_retrieval_prefix_selector(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    training_checkpoint: str | Path,
    training_checkpoint_sha256: str,
    out: str | Path,
) -> dict[str, Any]:
    """Authenticate cached training, fit two prototypes, and screen on train."""

    output = retrieval._new_output(out, "retrieval prefix selector result")
    started = time.perf_counter()
    _progress("authenticating frozen protocol and cached training checkpoint")
    context = retrieval._authenticate_fit_screen(protocol, protocol_sha256)
    (
        training,
        selection,
        _selected_heads,
        checkpoint_descriptor,
    ) = retrieval._load_training_checkpoint(
        training_checkpoint,
        training_checkpoint_sha256,
        context=context,
    )
    if (
        selection.get("screen_eligible") is not True
        or selection.get("selected_mask_name") != "M2"
    ):
        raise ValueError("retrieval prefix screen requires eligible global M2")
    records = context["train_records"]
    state = _training_state(training, records)
    partition = _derive_balanced_partition(
        state["base_mask"],
        state["global_mask"],
        state["gradients"],
    )
    prefix = _prefix_partition(
        records,
        _fact_anchor_ids(context["tokenizer_path"]),
    )
    prototypes = _bind_and_validate_prefix_prototypes(
        partition,
        prefix,
        state["global_mask"],
    )
    _progress(
        "cached partition and frozen exact-51 masks validated; "
        "starting 16 native train-sequence forwards"
    )
    evaluations = _evaluate_prototype_transfer(
        context=context,
        records=records,
        prototypes=prototypes,
    )
    transfer = _transfer_gate(
        records=records,
        baseline=state["global_M2_answer_cross_entropy"],
        prefix=prefix,
        evaluations=evaluations,
    )
    post_authentication = dict(retrieval._fit_post_authentication(context))
    checkpoint_path = Path(checkpoint_descriptor["path"]).resolve()
    post_authentication["training_checkpoint"] = (
        sha256_file(checkpoint_path) == checkpoint_descriptor["sha256"]
    )
    if not post_authentication or not all(post_authentication.values()):
        raise ValueError("retrieval prefix post-run authentication failed")
    status = (
        "train_prefix_gate_passed" if transfer["passed"] else "train_prefix_gate_failed"
    )
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _EXPERIMENT,
        "status": status,
        "protocol": {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "training_checkpoint": checkpoint_descriptor,
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "method": {
            "cached_M1_gradients_only": True,
            "backward_passes": 0,
            "dense_teacher_forwards": 0,
            "development_outcomes_used": False,
            "development_candidate_forwards": 0,
            "balanced_partition": {
                "candidate_partition_count": partition["candidate_partition_count"],
                "selection_order": partition["selection_order"],
                "clusters": {
                    _D_LATER: list(partition["clusters"][0]),
                    _D_EARLIER: list(partition["clusters"][1]),
                },
                "objective": partition["objective"],
                "global_M2_objective": partition["global_M2_objective"],
                "assigned_proximal_regrets": list(partition["assigned_regrets"]),
                "global_M2_proximal_regrets": list(partition["global_M2_regrets"]),
                "individual_mask_sha256": list(partition["individual_mask_sha256"]),
            },
            "causal_prefix_rule": prefix,
            "prototypes": {
                name: prototypes[name]["report"] for name in _PROTOTYPE_NAMES
            },
        },
        "train_transfer": transfer,
        "decision": {
            "passed": transfer["passed"],
            "confirmation_authorized": False,
            "next_step": (
                "freeze a separate prefix-conditioned development screen"
                if transfer["passed"]
                else "reject this two-prototype prefix selector"
            ),
        },
        "post_run_authentication": post_authentication,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cached train-only OLMoE retrieval prefix selector",
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--training-checkpoint-sha256", required=True)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    value = screen_retrieval_prefix_selector(
        protocol=args.protocol,
        protocol_sha256=args.protocol_sha256,
        training_checkpoint=args.training_checkpoint,
        training_checkpoint_sha256=args.training_checkpoint_sha256,
        out=args.out,
    )
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
