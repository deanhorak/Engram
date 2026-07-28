from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import engram.evaluation.olmoe_causal_head_gate as causal_gate
import engram.evaluation.olmoe_native_layer_rescue as layer_rescue
from engram.runtime.native_attention import NativeStreamingAttention
from engram.utils import atomic_json, sha256_file


_MODEL = {
    "layers": 16,
    "hidden_size": 2_048,
    "intermediate_size": 1_024,
    "experts": 64,
    "vocab_size": 50_304,
    "query_heads": 16,
    "key_value_heads": 16,
    "head_dimension": 128,
    "top_k": 8,
    "q7_group_size": 64,
}


def _independent_streaming_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    local_window: int = 16,
    older_candidates: int = 8,
    older_top_k: int = 4,
    sink_tokens: int = 2,
    scale: float | None = None,
) -> np.ndarray:
    """Small, deliberately direct reference for the native cache policy."""

    batches, heads, tokens, dimension = query.shape
    assert key.shape == value.shape == query.shape
    selected_scale = (
        1.0 / math.sqrt(float(dimension)) if scale is None else float(scale)
    )
    output = np.zeros_like(query)
    for batch in range(batches):
        for head in range(heads):
            recent: list[dict[str, object]] = []
            older: list[dict[str, object] | None] = [
                None for _ in range(older_candidates)
            ]
            for position in range(tokens):
                if len(recent) == local_window:
                    evicted = recent.pop(0)
                    evicted_position = int(evicted["position"])
                    destination: int | None
                    if evicted_position < sink_tokens:
                        destination = evicted_position
                    else:
                        destination = next(
                            (
                                index
                                for index in range(sink_tokens, older_candidates)
                                if older[index] is None
                            ),
                            None,
                        )
                        if destination is None:
                            destination = min(
                                range(sink_tokens, older_candidates),
                                key=lambda index: (
                                    float(older[index]["score"]),  # type: ignore[index]
                                    int(older[index]["position"]),  # type: ignore[index]
                                ),
                            )
                            incumbent = older[destination]
                            assert incumbent is not None
                            if float(evicted["mass"]) < float(incumbent["score"]):
                                destination = None
                    if destination is not None:
                        older[destination] = {
                            "key": evicted["key"],
                            "value": evicted["value"],
                            "score": float(evicted["mass"]),
                            "position": evicted_position,
                        }

                recent.append(
                    {
                        "key": key[batch, head, position].copy(),
                        "value": value[batch, head, position].copy(),
                        "mass": 0.0,
                        "position": position,
                    }
                )
                active = [
                    index for index, entry in enumerate(older) if entry is not None
                ]
                older_logits = {
                    index: float(
                        np.dot(
                            query[batch, head, position],
                            older[index]["key"],  # type: ignore[index]
                        )
                        * selected_scale
                    )
                    for index in active
                }
                selected = sorted(
                    active,
                    key=lambda index: (
                        -older_logits[index],
                        int(older[index]["position"]),  # type: ignore[index]
                    ),
                )[:older_top_k]
                local_logits = [
                    float(
                        np.dot(
                            query[batch, head, position],
                            entry["key"],
                        )
                        * selected_scale
                    )
                    for entry in recent
                ]
                visible_logits = np.asarray(
                    local_logits + [older_logits[index] for index in selected],
                    dtype=np.float64,
                )
                visible_weights = np.exp(visible_logits - np.max(visible_logits))
                visible_weights /= np.sum(visible_weights)
                row = np.zeros(dimension, dtype=np.float64)
                for index, entry in enumerate(recent):
                    weight = float(visible_weights[index])
                    entry["mass"] = float(entry["mass"]) + weight
                    row += weight * np.asarray(entry["value"])
                for selected_index, older_index in enumerate(selected):
                    weight = float(visible_weights[len(recent) + selected_index])
                    entry = older[older_index]
                    assert entry is not None
                    entry["score"] = float(entry["score"]) + weight
                    row += weight * np.asarray(entry["value"])
                output[batch, head, position] = row
    return output


def _independent_full_causal_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    scale: float | None = None,
) -> np.ndarray:
    batches, heads, tokens, dimension = query.shape
    selected_scale = (
        1.0 / math.sqrt(float(dimension)) if scale is None else float(scale)
    )
    output = np.zeros_like(query)
    for batch in range(batches):
        for head in range(heads):
            for position in range(tokens):
                logits = (
                    key[batch, head, : position + 1]
                    @ query[batch, head, position]
                    * selected_scale
                )
                weights = np.exp(logits - np.max(logits))
                weights /= np.sum(weights)
                output[batch, head, position] = (
                    weights @ value[batch, head, : position + 1]
                )
    return output


def _differentiable_full_causal_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(float(query.shape[-1]))
    scores = torch.einsum("bhtd,bhsd->bhts", query, key) * scale
    causal_mask = torch.ones(
        query.shape[-2],
        key.shape[-2],
        dtype=torch.bool,
        device=query.device,
    ).tril()
    scores = scores.masked_fill(~causal_mask, -torch.inf)
    return torch.einsum(
        "bhts,bhsd->bhtd",
        torch.softmax(scores, dim=-1),
        value,
    )


def _toy_qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = np.random.default_rng(20260728)
    arrays = tuple(
        torch.from_numpy(generator.normal(size=(1, 2, 32, 4)).astype(np.float32))
        for _ in range(3)
    )
    return arrays  # type: ignore[return-value]


def _adversarial_cancellation_qkv(
    *,
    heads: int = 1,
    tokens: int = 128,
    dimension: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert tokens == dimension == 128
    query = np.ones(
        (1, heads, tokens, dimension),
        dtype=np.float32,
    )
    key = np.empty_like(query)
    generator = np.random.default_rng(2000)
    epsilon = np.float32(1.1920929e-7)
    for head in range(heads):
        for position in range(tokens):
            row = np.concatenate(
                (
                    np.ones(64, dtype=np.float32),
                    -np.ones(64, dtype=np.float32),
                )
            )
            generator.shuffle(row)
            perturbed = generator.choice(
                dimension,
                size=8,
                replace=False,
            )
            row[perturbed] += (
                generator.integers(-4, 5, size=8).astype(np.float32) * epsilon
            )
            key[0, head, position] = row
    value_generator = np.random.default_rng(20260728)
    value = value_generator.normal(size=query.shape).astype(np.float32)
    return query, key, value


def _fake_protocol_context() -> dict[str, object]:
    record_ids = [f"record-{index}" for index in range(8)]
    split = layer_rescue._record_split(record_ids)
    rows_by_index = {int(row["sequence_index"]): row for row in split["ranked_records"]}
    frozen_order = (0, 1, 3, 4, 7, 2, 5, 6)
    split["ranked_records"] = [rows_by_index[index] for index in frozen_order]
    split["selection"] = [rows_by_index[index] for index in (0, 1)]
    split["internal_holdout"] = [rows_by_index[index] for index in (3, 4, 7, 2, 5, 6)]
    split["split_identity"] = "frozen-test-split"
    layer_source_hash = "d" * 64
    return {
        "sustained_protocol": {
            "source_revision": "revision",
            "source_config_sha256": "config",
            "source_index_sha256": "index",
            "source_shard_sha256": {"model.safetensors": "shard"},
        },
        "identities": {
            "package_manifest_sha256": "package",
            "native_library_sha256": "reference-library",
            "dataset_sha256": "dataset",
            "corpus_manifest_sha256": "corpus",
            "teacher_reference_sha256": "teacher-reference",
            "teacher_arrays_sha256": "teacher-arrays",
        },
        "hashes": {
            "sustained_protocol_sha256": "sustained-protocol",
            "sustained_result_sha256": "sustained-result",
            "control_protocol_sha256": "control-protocol",
            "control_result_sha256": "control-result",
        },
        "sweep_hashes": {
            "sweep_protocol_sha256": "sweep-protocol",
            "sweep_result_sha256": "sweep-result",
        },
        "control_source_hash": "control-source",
        "sweep_source_hash": "sweep-source",
        "layer_rescue_protocol_sha256": "layer-protocol",
        "layer_rescue_result_sha256": "layer-result",
        "layer_rescue_protocol": {
            "rescue_source_sha256": layer_source_hash,
            "rescue_source_inventory_sha256": {
                "src/engram/evaluation/olmoe_native_layer_rescue.py": (
                    layer_source_hash
                )
            },
        },
        "layer_rescue_historical_source_inventory": {
            "src/engram/evaluation/olmoe_native_layer_rescue.py": (layer_source_hash)
        },
        "candidate_library_sha256": "layered-library",
        "record_ids": record_ids,
        "split": split,
        "input_ids": [[index] + [0] * 128 for index in range(8)],
        "model": deepcopy(_MODEL),
    }


def _synthetic_loss(
    total: float,
    objective: dict[str, object],
) -> dict[str, object]:
    components = {name: 0.0 for name in causal_gate._LOSS_COMPONENT_NAMES}
    components["kl"] = (
        total
        * float(objective["teacher_to_student_kl_normalizer"])
        / float(objective["teacher_to_student_kl_weight"])
    )
    band_names = [str(row["name"]) for row in objective["bands"]]
    bands = {
        f"{band_name}_{component}": components[component]
        for band_name in band_names
        for component in causal_gate._LOSS_COMPONENT_NAMES
    }
    return {**components, "total": total, "bands": bands}


def _synthetic_native_layers(
    protocol: dict[str, object],
) -> list[dict[str, object]]:
    library_hash = str(protocol["training_attention_library_sha256"])
    expected_counts = causal_gate._expected_visible_counts(
        128,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
    ).tolist()

    def metrics(policy):
        fixed, heavy_range = causal_gate._native_metric_expectations(
            protocol["model"],
            policy,
        )
        return {
            **fixed,
            "heavy_hitter_updates": heavy_range[0],
        }

    sparse_metrics = metrics(protocol["base_attention_policy"])
    full_metrics = metrics(protocol["rescue_attention_policy"])
    layers = []
    for layer in range(16):
        sparse_schedule_seconds = 0.01
        sparse_actual_seconds = 0.02
        sparse_surrogate_seconds = 0.03
        full_actual_seconds = 0.02
        full_surrogate_seconds = 0.03
        layers.append(
            {
                "layer": layer,
                "sparse": {
                    "mode": ("native_exact_sparse_forward_gathered_surrogate_backward"),
                    "schedule": {
                        "expected_visible_counts": expected_counts,
                        "observed_visible_count_minimum": min(expected_counts),
                        "observed_visible_count_maximum": max(expected_counts),
                        "maximum_row_sum_error": 1.0e-7,
                        "minimum_positive_weight": 1.0e-5,
                        "indices_sha256": f"{layer + 1:064x}",
                        "native_metrics": deepcopy(sparse_metrics),
                        "elapsed_seconds": sparse_schedule_seconds,
                        "attention_library_sha256": library_hash,
                    },
                    "actual_value": {
                        "native_metrics": deepcopy(sparse_metrics),
                        "elapsed_seconds": sparse_actual_seconds,
                        "attention_library_sha256": library_hash,
                    },
                    "surrogate_elapsed_seconds": sparse_surrogate_seconds,
                    "total_elapsed_seconds": (
                        sparse_schedule_seconds
                        + sparse_actual_seconds
                        + sparse_surrogate_seconds
                    ),
                    "exact_forward_sha256": f"{layer + 17:064x}",
                },
                "full": {
                    "mode": (
                        "native_exact_W128_forward_full_causal_surrogate_backward"
                    ),
                    "actual_value": {
                        "native_metrics": deepcopy(full_metrics),
                        "elapsed_seconds": full_actual_seconds,
                        "attention_library_sha256": library_hash,
                    },
                    "surrogate_elapsed_seconds": full_surrogate_seconds,
                    "total_elapsed_seconds": (
                        full_actual_seconds + full_surrogate_seconds
                    ),
                    "exact_forward_sha256": f"{layer + 33:064x}",
                },
            }
        )
    return layers


def _synthetic_training_chain() -> tuple[dict[str, object], dict[str, object]]:
    context = _fake_protocol_context()
    prerequisite_hashes = {
        f"{name}_sha256": f"{index + 4:064x}"
        for index, name in enumerate(causal_gate._BOUNDARY_ARTIFACT_NAMES)
    }
    protocol = causal_gate._build_gate_protocol(
        context,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256="1" * 64,
        source_inventory={},
        framework_contract={"synthetic_framework": "bound"},
        attention_library_path=Path("/test/libengram_attention.so"),
        attention_library_sha256="2" * 64,
        device="cpu",
        threads=12,
    )
    masks = {"M0": np.zeros((16, 16), dtype=np.bool_)}
    gradients = [
        (
            np.linspace(-2.0, 1.0, 256, dtype=np.float64).reshape(16, 16),
            np.linspace(-1.0, 2.0, 256, dtype=np.float64).reshape(16, 16),
        ),
        (
            np.linspace(1.5, -2.5, 256, dtype=np.float64).reshape(16, 16),
            np.linspace(0.5, -1.5, 256, dtype=np.float64).reshape(16, 16),
        ),
    ]
    scores_by_mask: dict[str, np.ndarray] = {}
    steps = []
    for step_number, record_gradients in enumerate(gradients, start=1):
        input_name = f"M{step_number - 1}"
        output_name = f"M{step_number}"
        mean_gradient = np.stack(record_gradients).mean(
            axis=0,
            dtype=np.float64,
        )
        scores, output_mask, rms = causal_gate._projected_gate_step(
            masks[input_name],
            mean_gradient,
        )
        masks[output_name] = output_mask
        scores_by_mask[output_name] = scores
        steps.append(
            {
                "step": step_number,
                "input_mask_name": input_name,
                "output_mask_name": output_name,
                "input_mask": masks[input_name].tolist(),
                "input_mask_sha256": causal_gate.sha256_json(
                    masks[input_name].tolist()
                ),
                "output_mask": output_mask.tolist(),
                "output_mask_sha256": causal_gate.sha256_json(output_mask.tolist()),
                "record_gradients": [
                    {
                        "sequence_index": sequence_index,
                        "gradient": gradient.tolist(),
                    }
                    for sequence_index, gradient in enumerate(record_gradients)
                ],
                "mean_gradient": mean_gradient.tolist(),
                "mean_gradient_root_mean_square": rms,
                "projected_score": scores.tolist(),
                "output_selected_flat_indices": np.flatnonzero(output_mask.reshape(-1))
                .astype(int)
                .tolist(),
                "head_churn_from_input": int(
                    np.count_nonzero(masks[input_name] != output_mask)
                ),
            }
        )

    loss_by_mask = {
        "M0": (10.0, 11.0),
        "M1": (8.0, 9.0),
        "M2": (6.0, 7.0),
    }
    selection_ids = {
        int(row["sequence_index"]): str(row["record_id"])
        for row in protocol["training_data_access"]["selection_records"]
    }
    evaluations = {}
    for name in ("M0", "M1", "M2"):
        backward = name != "M2"
        gradient_step = None if name == "M2" else int(name[1])
        records = []
        for sequence_index in (0, 1):
            layers = _synthetic_native_layers(protocol)
            timing = causal_gate._diagnostic_timing_summary(layers)
            elapsed = (
                sum(float(value) for key, value in timing.items() if key != "layers")
                + 0.25
            )
            records.append(
                {
                    "sequence_index": sequence_index,
                    "record_id": selection_ids[sequence_index],
                    "mask_sha256": causal_gate.sha256_json(masks[name].tolist()),
                    "selected_head_count": int(masks[name].sum()),
                    "loss": _synthetic_loss(
                        loss_by_mask[name][sequence_index],
                        protocol["objective"],
                    ),
                    "backward": backward,
                    "gradient": (
                        None
                        if gradient_step is None
                        else gradients[gradient_step][sequence_index].tolist()
                    ),
                    "native_oracle_layers": layers,
                    "native_oracle_timing": timing,
                    "elapsed_seconds": elapsed,
                }
            )
        evaluations[name] = {
            "mask_name": name,
            "mask": masks[name].tolist(),
            "mask_sha256": causal_gate.sha256_json(masks[name].tolist()),
            "records": records,
            "objective_summary": causal_gate._objective_summary(records),
            "execution_role": (
                "gradient_and_candidate_evaluation"
                if backward
                else "terminal_forward_only_candidate_evaluation"
            ),
        }
    selection = causal_gate._select_executed_mask(evaluations)
    selected_name = selection["selected_mask_name"]
    selected_mask = masks[selected_name]
    selected_rows = causal_gate._selected_head_rows(
        selected_mask,
        scores_by_mask[selected_name],
    )
    protocol_hash = "3" * 64
    post = {name: True for name in causal_gate._TRAINING_POST_AUTHENTICATION_NAMES}
    evidence = {name: True for name in causal_gate._TRAINING_EVIDENCE_NAMES}
    result = {
        "schema_version": 1,
        "experiment": causal_gate._TRAINING_EXPERIMENT,
        "status": causal_gate._TRAINING_STATUS,
        "artifacts": causal_gate._expected_training_artifacts(
            protocol,
            protocol_hash,
        ),
        "framework_contract": protocol["framework_contract"],
        "record_split": protocol["record_split"],
        "training_data_access": protocol["training_data_access"],
        "training": protocol["training"],
        "objective": protocol["objective"],
        "budget_contract": protocol["budget_contract"],
        "attention_expectations_per_sequence": protocol["budget_contract"][
            "attention_expectations_per_sequence"
        ],
        "IHT_step_results": steps,
        "executed_mask_evaluations": evaluations,
        "mask_selection": selection,
        "mask_churn": {
            "M0_to_M1": int(np.count_nonzero(masks["M0"] != masks["M1"])),
            "M1_to_M2": int(np.count_nonzero(masks["M1"] != masks["M2"])),
        },
        "selected_mask_name": selected_name,
        "selected_heads": selected_rows,
        "attention_head_mask": selected_mask.tolist(),
        "attention_head_mask_sha256": causal_gate.sha256_json(selected_mask.tolist()),
        "selected_head_count": 51,
        "evidence_checks": evidence,
        "evidence_passed": True,
        "native_screen_eligible": True,
        "decision": "freeze_exactly_one_native_internal_development_screen",
        "post_training_authentication": post,
        "performance": {
            "elapsed_seconds": 20.0,
            "executed_record_seconds": {
                name: [row["elapsed_seconds"] for row in evaluations[name]["records"]]
                for name in ("M0", "M1", "M2")
            },
        },
        "limitations": protocol["limitations"],
    }
    return protocol, result


def _replay_fixed(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return causal_gate._streaming_attention_replay(
        query,
        key,
        value,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
    )


def test_w16_c8_k4_s2_replay_matches_independent_reference():
    query, key, value = _toy_qkv()

    actual, trace = _replay_fixed(query, key, value)
    expected = _independent_streaming_reference(
        query.numpy(),
        key.numpy(),
        value.numpy(),
    )

    assert actual.shape == query.shape
    torch.testing.assert_close(
        actual,
        torch.from_numpy(expected),
        rtol=2e-5,
        atol=2e-6,
    )
    assert trace["visible_positions"].shape == (1, 2, 32, 20)
    assert trace["visible_weights"].shape == (1, 2, 32, 20)
    assert trace["selected_older_entries"].shape == (1, 2, 32)
    assert torch.equal(
        trace["visible_positions"][0, 0, 0],
        torch.tensor([0] + [-1] * 19),
    )
    torch.testing.assert_close(
        trace["visible_weights"].sum(dim=-1),
        torch.ones((1, 2, 32)),
        rtol=2e-6,
        atol=2e-6,
    )


def test_w16_c8_k4_s2_replay_matches_native_stream_and_counters():
    library = Path("build/libengram_attention.so")
    if not library.is_file():
        pytest.skip("native streaming-attention library has not been built")
    query, key, value = _toy_qkv()
    replay, trace = _replay_fixed(query, key, value)
    native_query = query[0].permute(1, 0, 2).numpy()
    native_key = key[0].permute(1, 0, 2).numpy()
    native_value = value[0].permute(1, 0, 2).numpy()

    with NativeStreamingAttention(
        query_heads=2,
        key_value_heads=2,
        head_dimension=4,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        library=library,
    ) as native:
        native_output, metrics = native.stream(
            native_query,
            native_key,
            native_value,
        )

    np.testing.assert_allclose(
        replay[0].permute(1, 0, 2).numpy(),
        native_output,
        rtol=3e-5,
        atol=3e-6,
    )
    assert metrics.tokens_seen == 32
    assert metrics.local_entries == 16
    assert metrics.active_older_entries == 16
    assert metrics.eviction_events == 16
    assert metrics.older_candidate_entries_scored == 200
    assert metrics.older_selected_entries == 116
    assert metrics.sink_insertions == 4
    assert 12 <= metrics.heavy_hitter_updates <= 28
    assert metrics.candidate_key_bytes == 3_200
    assert metrics.selected_value_bytes == 1_856
    assert metrics.local_kv_bytes == 25_088
    assert int(trace["eviction_events"].sum().item()) == 32
    assert int(trace["sink_insertions"].item()) == 4
    assert int(trace["heavy_hitter_updates"].item()) == metrics.heavy_hitter_updates


def test_native_identity_oracle_beats_scalar_reduction_on_cancellation():
    library = Path("build/libengram_attention.so")
    if not library.is_file():
        pytest.skip("native streaming-attention library has not been built")
    query_np, key_np, value_np = _adversarial_cancellation_qkv()
    query = torch.from_numpy(query_np)
    key = torch.from_numpy(key_np)
    value = torch.from_numpy(value_np)

    _scalar_output, scalar_trace = _replay_fixed(query, key, value)
    indices, native_weights, diagnostics = causal_gate._native_identity_schedule(
        query,
        key,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        attention_library=library,
    )
    indices_np = indices.detach().cpu().numpy()
    weights_np = native_weights.detach().cpu().numpy()
    scalar_old = {
        int(position)
        for position in scalar_trace["visible_positions"][0, 0, 24].tolist()
        if 0 <= int(position) <= 8
    }
    native_old = {
        int(position)
        for position in indices_np[0, 0, 24].tolist()
        if 0 <= int(position) <= 8
    }

    # This normalized near-cancellation fixture is deliberately sensitive to
    # reduction order: the legacy Torch sum and the compiled C++ float loop
    # select different older records. The oracle must follow this DSO, not a
    # platform-specific hard-coded set.
    assert scalar_old != native_old
    assert native_old == set(np.flatnonzero(weights_np[0, 0, 24, :9] > 0.0).tolist())
    expected_counts = np.asarray(
        [
            min(position + 1, 16) + min(4, min(8, max(0, position - 16 + 1)))
            for position in range(128)
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(
        diagnostics["expected_visible_counts"],
        expected_counts,
    )
    np.testing.assert_array_equal(
        (indices_np >= 0).sum(axis=-1),
        np.broadcast_to(expected_counts, (1, 1, 128)),
    )
    assert diagnostics["observed_visible_count_minimum"] == 1
    assert diagnostics["observed_visible_count_maximum"] == 20
    assert diagnostics["maximum_row_sum_error"] <= 2e-6
    assert diagnostics["minimum_positive_weight"] > 0.0

    native_query = query_np[0].transpose(1, 0, 2)
    native_key = key_np[0].transpose(1, 0, 2)
    native_value = value_np[0].transpose(1, 0, 2)
    with NativeStreamingAttention(
        query_heads=1,
        key_value_heads=1,
        head_dimension=128,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        library=library,
    ) as native:
        direct, _metrics = native.stream(
            native_query,
            native_key,
            native_value,
        )
    reconstructed = np.einsum(
        "bhts,bhsd->bhtd",
        weights_np,
        value_np,
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        reconstructed[0].transpose(1, 0, 2),
        direct,
        rtol=3e-5,
        atol=3e-6,
    )


def test_native_identity_oracle_rejects_underflowed_selected_weight():
    library = Path("build/libengram_attention.so")
    if not library.is_file():
        pytest.skip("native streaming-attention library has not been built")
    query = torch.zeros((1, 1, 128, 128), dtype=torch.float32)
    query[0, 0, 16] = 1.0
    key = torch.full_like(query, 10.0)
    key[0, 0, 0] = -10.0

    with pytest.raises(ValueError, match="support|underflow|positive"):
        causal_gate._native_identity_schedule(
            query,
            key,
            local_window=16,
            older_candidates=8,
            older_top_k=4,
            sink_tokens=2,
            attention_library=library,
        )


def test_differentiable_gathered_attention_batches_heads_and_gradients():
    generator = torch.Generator().manual_seed(20260728)
    query = torch.randn(
        (1, 3, 6, 4),
        generator=generator,
        requires_grad=True,
    )
    key = torch.randn(
        (1, 3, 6, 4),
        generator=generator,
        requires_grad=True,
    )
    value = torch.randn(
        (1, 3, 6, 4),
        generator=generator,
        requires_grad=True,
    )
    indices = torch.full((1, 3, 6, 4), -1, dtype=torch.long)
    expected_rows = []
    for head in range(3):
        head_rows = []
        for position in range(6):
            visible = list(range(max(0, position - 3), position + 1))
            indices[0, head, position, : len(visible)] = torch.tensor(visible)
            scores = key[0, head, visible] @ query[0, head, position] / math.sqrt(4.0)
            head_rows.append(torch.softmax(scores, dim=0) @ value[0, head, visible])
        expected_rows.append(torch.stack(head_rows))
    expected = torch.stack(expected_rows).unsqueeze(0)

    actual = causal_gate._differentiable_gathered_attention(
        query,
        key,
        value,
        indices,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    actual[:, :, 2:].square().mean().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert int(torch.count_nonzero(tensor.grad).item()) > 0


def test_native_sparse_straight_through_is_exact_forward_and_has_gradients():
    library = Path("build/libengram_attention.so")
    if not library.is_file():
        pytest.skip("native streaming-attention library has not been built")
    query_np, key_np, value_np = _adversarial_cancellation_qkv()
    query = torch.from_numpy(query_np).requires_grad_(True)
    key = torch.from_numpy(key_np).requires_grad_(True)
    value = torch.from_numpy(value_np).requires_grad_(True)

    actual, diagnostics = causal_gate._native_sparse_straight_through(
        query,
        key,
        value,
        attention_library=library,
    )
    with NativeStreamingAttention(
        query_heads=1,
        key_value_heads=1,
        head_dimension=128,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        library=library,
    ) as native:
        direct, _metrics = native.stream(
            query_np[0].transpose(1, 0, 2),
            key_np[0].transpose(1, 0, 2),
            value_np[0].transpose(1, 0, 2),
        )
    direct_tensor = torch.from_numpy(direct.transpose(1, 0, 2)[None, ...])

    assert torch.equal(actual.detach().cpu(), direct_tensor)
    assert diagnostics["mode"] == (
        "native_exact_sparse_forward_gathered_surrogate_backward"
    )
    loss = actual[:, :, 24:].square().mean()
    loss.backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert int(torch.count_nonzero(tensor.grad).item()) > 0


def test_generalized_replay_rejects_sink_capacity_edge():
    query, key, value = _toy_qkv()
    with pytest.raises(ValueError):
        causal_gate._streaming_attention_replay(
            query,
            key,
            value,
            local_window=16,
            older_candidates=8,
            older_top_k=4,
            sink_tokens=8,
        )


def test_full_context_and_gate_mixing_have_exact_endpoints():
    query, key, value = _toy_qkv()
    base, _trace = _replay_fixed(query, key, value)
    expected_full = _independent_full_causal_reference(
        query.numpy(),
        key.numpy(),
        value.numpy(),
    )
    full = torch.from_numpy(expected_full)

    zeros = torch.zeros(query.shape[1], dtype=query.dtype)
    ones = torch.ones(query.shape[1], dtype=query.dtype)
    assert torch.equal(
        causal_gate._mix_head_outputs(base, full, zeros),
        base,
    )
    assert torch.equal(
        causal_gate._mix_head_outputs(base, full, ones),
        full,
    )
    quarter = torch.full((query.shape[1],), 0.25, dtype=query.dtype)
    torch.testing.assert_close(
        causal_gate._mix_head_outputs(base, full, quarter),
        base + 0.25 * (full - base),
        rtol=0.0,
        atol=0.0,
    )


def test_post_eviction_replay_and_mixed_gate_are_differentiable():
    query, key, value = tuple(
        tensor.clone().requires_grad_(True) for tensor in _toy_qkv()
    )
    sparse, _trace = _replay_fixed(query, key, value)
    full = _differentiable_full_causal_reference(query, key, value)
    gate_logits = torch.tensor(
        [-0.75, 0.5],
        dtype=query.dtype,
        requires_grad=True,
    )
    mixed = causal_gate._mix_head_outputs(
        sparse,
        full,
        torch.sigmoid(gate_logits),
    )

    # Restrict supervision to positions after the first cache eviction. This
    # exercises gradients through both the local and promoted-cache paths.
    post_eviction = mixed[:, :, 20:, :]
    loss = post_eviction.square().mean() + 0.01 * post_eviction.mean()
    loss.backward()

    for tensor in (query, key, value, gate_logits):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert int(torch.count_nonzero(tensor.grad).item()) > 0


def test_patched_attention_preserves_bfloat16_shape_and_float_gate_gradients(
    monkeypatch: pytest.MonkeyPatch,
):
    observed: dict[str, object] = {}

    def fake_sparse(
        query,
        key,
        value,
        *,
        attention_library,
        policy,
    ):
        observed["sparse_dtype"] = query.dtype
        observed["sparse_shape"] = tuple(query.shape)
        observed["policy"] = policy
        observed["attention_library"] = attention_library
        return 0.25 * query + 0.25 * key + 0.25 * value, {}

    def fake_full(query, key, value, *, attention_library):
        observed["full_dtype"] = query.dtype
        assert attention_library == Path("/test/libattention.so")
        return 0.5 * query + 0.5 * key + 0.25 * value, {}

    monkeypatch.setattr(
        causal_gate,
        "_native_sparse_straight_through",
        fake_sparse,
    )
    monkeypatch.setattr(
        causal_gate,
        "_native_full_straight_through",
        fake_full,
    )
    gates = torch.full(
        (16, 16),
        0.5,
        dtype=torch.float32,
        requires_grad=True,
    )
    state = {
        "gates": gates,
        "apply_rotary_pos_emb": lambda query, key, _cos, _sin: (
            query,
            key,
        ),
        "attention_library": Path("/test/libattention.so"),
        "diagnostics": [],
    }
    identity = torch.nn.Identity()
    module = SimpleNamespace(
        head_dim=1,
        layer_idx=0,
        config=SimpleNamespace(clip_qkv=None),
        q_proj=identity,
        k_proj=identity,
        v_proj=identity,
        q_norm=identity,
        k_norm=identity,
        o_proj=identity,
        _engram_causal_gate_state=state,
    )
    hidden = torch.linspace(
        -1.0,
        1.0,
        128 * 16,
        dtype=torch.bfloat16,
    ).reshape(1, 128, 16)
    hidden.requires_grad_(True)

    output, attention_weights = causal_gate._gated_attention_forward(
        module,
        hidden,
        (torch.empty(0), torch.empty(0)),
        attention_mask=torch.where(
            torch.ones((1, 1, 128, 128), dtype=torch.bool).tril(),
            0.0,
            -torch.inf,
        ),
        position_ids=torch.arange(128).reshape(1, -1),
        use_cache=False,
    )

    assert output.shape == hidden.shape
    assert output.dtype == torch.bfloat16
    assert attention_weights is None
    assert observed == {
        "sparse_dtype": torch.float32,
        "sparse_shape": (1, 16, 128, 1),
        "policy": {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
        "attention_library": Path("/test/libattention.so"),
        "full_dtype": torch.float32,
    }
    assert len(state["diagnostics"]) == 1
    loss = output[:, 20:].float().square().mean()
    loss.backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert int(torch.count_nonzero(hidden.grad).item()) > 0
    assert gates.grad is not None
    assert gates.grad.dtype == torch.float32
    assert torch.isfinite(gates.grad).all()
    assert int(torch.count_nonzero(gates.grad[0]).item()) > 0
    assert int(torch.count_nonzero(gates.grad[1:]).item()) == 0

    leaking_mask = torch.where(
        torch.ones((1, 1, 128, 128), dtype=torch.bool).tril(),
        0.0,
        -torch.inf,
    )
    leaking_mask[0, 0, 0, 127] = 0.0
    with pytest.raises(ValueError, match="padding or leakage"):
        causal_gate._gated_attention_forward(
            module,
            hidden.detach(),
            (torch.empty(0), torch.empty(0)),
            attention_mask=leaking_mask,
            position_ids=torch.arange(128).reshape(1, -1),
            use_cache=False,
        )


def test_distillation_loss_gives_each_position_band_equal_weight():
    teacher_logits = torch.zeros((1, 4, 3), dtype=torch.float32)
    student_logits = teacher_logits.clone()
    student_logits[0, 0, 0] = 4.0
    student_logits.requires_grad_(True)
    teacher_hidden = torch.ones((1, 4, 2), dtype=torch.float32)
    student_hidden = teacher_hidden.clone().requires_grad_(True)
    targets = torch.zeros((1, 4), dtype=torch.long)
    bands = (("one_position", 0, 1), ("three_positions", 1, 4))

    total, components = causal_gate._distillation_loss(
        student_logits,
        teacher_logits,
        student_hidden,
        teacher_hidden,
        targets,
        bands=bands,
    )

    teacher_log_probability = torch.log_softmax(teacher_logits, dim=-1)
    student_log_probability = torch.log_softmax(student_logits, dim=-1)
    per_position_kl = (
        teacher_log_probability.exp()
        * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1)
    equal_band_mean = torch.stack(
        (
            per_position_kl[:, 0:1].mean(),
            per_position_kl[:, 1:4].mean(),
        )
    ).mean()
    naive_position_mean = per_position_kl.mean()

    torch.testing.assert_close(
        components["kl"],
        equal_band_mean,
        rtol=1e-6,
        atol=1e-7,
    )
    assert not torch.isclose(components["kl"], naive_position_mean)
    assert set(components) == {
        "kl",
        "hidden_relative_l2",
        "positive_nll_delta",
        "top1_margin_deficit",
        "total",
        "bands",
    }
    assert set(components["bands"]) == {
        "one_position_kl",
        "three_positions_kl",
        "one_position_hidden_relative_l2",
        "three_positions_hidden_relative_l2",
        "one_position_positive_nll_delta",
        "three_positions_positive_nll_delta",
        "one_position_top1_margin_deficit",
        "three_positions_top1_margin_deficit",
    }
    torch.testing.assert_close(
        total,
        components["kl"] / 0.05,
        rtol=1e-6,
        atol=1e-7,
    )
    total.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert int(torch.count_nonzero(student_logits.grad).item()) > 0

    with pytest.raises(ValueError, match="overlap"):
        causal_gate._distillation_loss(
            student_logits.detach(),
            teacher_logits,
            student_hidden.detach(),
            teacher_hidden,
            targets,
            bands=((0, 3), (2, 4)),
        )


def test_selection_slices_cannot_read_internal_screen_records():
    positions_per_sequence = 128
    source = np.arange(
        8 * positions_per_sequence,
        dtype=np.float64,
    ).reshape(-1, 1)
    source[2 * positions_per_sequence :] = np.nan

    selected = causal_gate._selection_slices_only(
        source,
        [0, 1],
        positions_per_sequence=positions_per_sequence,
    )

    assert tuple(selected) == (0, 1)
    assert selected[0].shape == (positions_per_sequence, 1)
    assert selected[1].shape == (positions_per_sequence, 1)
    assert selected[0].base is None
    assert selected[1].base is None
    assert not np.shares_memory(selected[0], source)
    assert not np.shares_memory(selected[1], source)
    assert np.isfinite(selected[0]).all()
    assert np.isfinite(selected[1]).all()
    np.testing.assert_array_equal(
        selected[0][:, 0],
        np.arange(positions_per_sequence, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        selected[1][:, 0],
        np.arange(
            positions_per_sequence,
            2 * positions_per_sequence,
            dtype=np.float64,
        ),
    )
    for prohibited in ([0, 2], [2, 3], [0, 1, 2]):
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            causal_gate._selection_slices_only(
                source,
                prohibited,
                positions_per_sequence=positions_per_sequence,
            )


def test_frozen_protocol_binds_training_data_objective_and_source(
    tmp_path: Path,
):
    context = _fake_protocol_context()
    prerequisite_hashes = {
        "trace_protocol_sha256": "1" * 64,
        "failed_head_mask_sha256": "2" * 64,
        "failed_screen_protocol_sha256": "3" * 64,
        "failed_screen_result_sha256": "4" * 64,
        "headwise_library_sha256": "5" * 64,
    }
    source_hash = "6" * 64
    source_inventory = {
        "src/engram/evaluation/olmoe_causal_head_gate.py": "7" * 64,
    }
    framework = {
        "torch_version": "test",
        "transformers_version": "test",
        "transformers_olmoe_modeling_path": "/test/modeling_olmoe.py",
        "transformers_olmoe_modeling_sha256": "8" * 64,
    }
    attention_path = Path("/test/libengram_attention.so")
    attention_hash = "a" * 64
    protocol = causal_gate._build_gate_protocol(
        context,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        framework_contract=framework,
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
        device="cpu",
        threads=12,
    )
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)

    causal_gate._validate_protocol_shape(protocol)
    causal_gate._validate_gate_protocol(
        protocol,
        context,
        protocol_sha256=protocol_hash,
        supplied_sha256=protocol_hash,
        prerequisite_hashes=prerequisite_hashes,
        source_sha256=source_hash,
        source_inventory=source_inventory,
        framework_contract=framework,
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
    )
    assert protocol["training_data_access"] == {
        "selection_records": context["split"]["selection"],
        "selection_sequence_indices": [0, 1],
        "gradient_sequence_indices_per_step": [0, 1],
        "iht_steps": 2,
        "terminal_evaluation_sequence_indices": [0, 1],
        "internal_screen_records": context["split"]["internal_holdout"],
        "internal_screen_sequence_order": [3, 4, 7, 2, 5, 6],
        "prohibited_internal_screen_sequence_indices": [2, 3, 4, 5, 6, 7],
        "internal_screen_records_used": False,
    }
    assert protocol["base_attention_policy"] == {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    assert protocol["training"]["hard_head_count_after_each_IHT_step"] == 51
    assert protocol["budget_contract"]["next_head_boundary"]["within_budget"] is False

    digest_drift = "0" * 64
    with pytest.raises(ValueError, match="frozen protocol"):
        causal_gate._validate_gate_protocol(
            protocol,
            context,
            protocol_sha256=protocol_hash,
            supplied_sha256=digest_drift,
            prerequisite_hashes=prerequisite_hashes,
            source_sha256=source_hash,
            source_inventory=source_inventory,
            framework_contract=framework,
            attention_library_path=attention_path,
            attention_library_sha256=attention_hash,
        )

    context_drift = deepcopy(context)
    context_drift["input_ids"][0][0] = 999
    with pytest.raises(ValueError, match="frozen protocol"):
        causal_gate._validate_gate_protocol(
            protocol,
            context_drift,
            protocol_sha256=protocol_hash,
            supplied_sha256=protocol_hash,
            prerequisite_hashes=prerequisite_hashes,
            source_sha256=source_hash,
            source_inventory=source_inventory,
            framework_contract=framework,
            attention_library_path=attention_path,
            attention_library_sha256=attention_hash,
        )

    source_drift = deepcopy(source_inventory)
    source_drift["src/engram/evaluation/olmoe_causal_head_gate.py"] = "9" * 64
    with pytest.raises(ValueError, match="frozen protocol"):
        causal_gate._validate_gate_protocol(
            protocol,
            context,
            protocol_sha256=protocol_hash,
            supplied_sha256=protocol_hash,
            prerequisite_hashes=prerequisite_hashes,
            source_sha256=source_hash,
            source_inventory=source_drift,
            framework_contract=framework,
            attention_library_path=attention_path,
            attention_library_sha256=attention_hash,
        )


def test_protocol_leakage_and_freeze_boundaries_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _fake_protocol_context()
    attention_path = Path("/test/libengram_attention.so")
    attention_hash = "a" * 64
    protocol = causal_gate._build_gate_protocol(
        context,
        prerequisite_hashes={},
        source_sha256="1" * 64,
        source_inventory={},
        framework_contract={},
        attention_library_path=attention_path,
        attention_library_sha256=attention_hash,
        device="cpu",
        threads=12,
    )
    leaking = deepcopy(protocol)
    leaking["training_data_access"]["internal_screen_records_used"] = True
    with pytest.raises(ValueError, match="protocol shape"):
        causal_gate._validate_protocol_shape(leaking)

    leaking = deepcopy(protocol)
    leaking["training_data_access"]["prohibited_internal_screen_sequence_indices"] = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    with pytest.raises(ValueError, match="protocol shape|leaks screen"):
        causal_gate._validate_protocol_shape(leaking)

    existing = tmp_path / "existing-protocol.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="target already exists"):
        causal_gate.freeze_native_olmoe_causal_head_gate_protocol(
            out=existing,
            attention_library=attention_path,
            attention_library_sha256=attention_hash,
            manifest_sha256="not-consulted",
        )
    assert existing.read_text(encoding="utf-8") == "preserve"

    monkeypatch.setattr(
        causal_gate,
        "_authenticate_attention_library",
        lambda _path, _digest: (attention_path, attention_hash),
    )
    missing_prerequisites = tmp_path / "must-not-be-created.json"
    with pytest.raises(ValueError, match="prerequisite arguments"):
        causal_gate.freeze_native_olmoe_causal_head_gate_protocol(
            out=missing_prerequisites,
            attention_library=attention_path,
            attention_library_sha256=attention_hash,
            manifest_sha256="not-consulted",
        )
    assert not missing_prerequisites.exists()


def test_freeze_uses_authenticated_context_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _fake_protocol_context()
    prerequisite_hashes = {
        "trace_protocol_sha256": "1" * 64,
        "trace_metadata_sha256": "2" * 64,
        "trace_arrays_sha256": "3" * 64,
        "failed_head_mask_sha256": "4" * 64,
        "failed_screen_protocol_sha256": "5" * 64,
        "failed_screen_result_sha256": "6" * 64,
        "headwise_library_sha256": "7" * 64,
    }
    source_inventory = {
        "src/engram/evaluation/olmoe_causal_head_gate.py": "8" * 64,
    }
    framework = {
        "torch_version": "test",
        "transformers_version": "test",
        "transformers_olmoe_modeling_path": "/test/modeling_olmoe.py",
        "transformers_olmoe_modeling_sha256": "9" * 64,
    }
    attention_path = Path("/test/libengram_attention.so")
    attention_hash = "a" * 64
    authentication_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        causal_gate,
        "_authenticate_attention_library",
        lambda _path, _digest: (attention_path, attention_hash),
    )
    monkeypatch.setattr(
        causal_gate.headwise,
        "_common_context",
        lambda **_kwargs: ({}, context),
    )

    def authenticate(_context, **kwargs):
        assert _context is context
        authentication_calls.append(kwargs)
        return prerequisite_hashes, {"paths": {}}

    monkeypatch.setattr(
        causal_gate,
        "_authenticate_failed_headwise_boundary",
        authenticate,
    )
    monkeypatch.setattr(
        causal_gate,
        "_current_source_inventory",
        lambda _context: source_inventory,
    )
    monkeypatch.setattr(
        causal_gate,
        "_framework_contract",
        lambda: framework,
    )
    prerequisite_arguments = {
        name: f"argument:{name}" for name in causal_gate._PREREQUISITE_ARGUMENT_NAMES
    }
    output_path = tmp_path / "frozen.json"

    protocol = causal_gate.freeze_native_olmoe_causal_head_gate_protocol(
        out=output_path,
        attention_library=attention_path,
        attention_library_sha256=attention_hash,
        manifest_sha256="manifest",
        **prerequisite_arguments,
    )

    assert len(authentication_calls) == 1
    assert authentication_calls[0] == prerequisite_arguments
    assert output_path.is_file()
    assert causal_gate._read_json(output_path, "test protocol") == protocol
    causal_gate._validate_protocol_shape(protocol)

    def reject_authentication(_context, **_kwargs):
        assert _context is context
        raise ValueError("simulated authentication failure")

    monkeypatch.setattr(
        causal_gate,
        "_authenticate_failed_headwise_boundary",
        reject_authentication,
    )
    rejected_path = tmp_path / "rejected.json"
    with pytest.raises(ValueError, match="authentication failure"):
        causal_gate.freeze_native_olmoe_causal_head_gate_protocol(
            out=rejected_path,
            attention_library=attention_path,
            attention_library_sha256=attention_hash,
            manifest_sha256="manifest",
            **prerequisite_arguments,
        )
    assert not rejected_path.exists()


def test_training_result_recomputes_and_rejects_tampered_iht_chain():
    protocol, result = _synthetic_training_chain()
    protocol_hash = "3" * 64

    selected = causal_gate._validate_training_result(
        result,
        result_sha256="a" * 64,
        supplied_sha256="a" * 64,
        protocol=protocol,
        protocol_sha256=protocol_hash,
    )

    assert len(selected) == 51
    assert selected == [
        (int(row["layer"]), int(row["head"])) for row in result["selected_heads"]
    ]

    deleted = object()
    cases = (
        (
            "gradient",
            (
                "IHT_step_results",
                0,
                "record_gradients",
                0,
                "gradient",
                0,
                0,
            ),
            999.0,
        ),
        (
            "projected score",
            ("IHT_step_results", 1, "projected_score", 0, 0),
            999.0,
        ),
        (
            "mask hash",
            ("IHT_step_results", 0, "output_mask_sha256"),
            "0" * 64,
        ),
        ("final choice", ("selected_mask_name",), "M1"),
        (
            "record id",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "record_id",
            ),
            "wrong-record",
        ),
        (
            "loss component",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "loss",
                "kl",
            ),
            999.0,
        ),
        (
            "loss total",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "loss",
                "total",
            ),
            999.0,
        ),
        (
            "layer order",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "layer",
            ),
            1,
        ),
        (
            "float layer index",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "layer",
            ),
            0.0,
        ),
        (
            "native mode",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "mode",
            ),
            "wrong-mode",
        ),
        (
            "native hash",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "exact_forward_sha256",
            ),
            "not-a-sha256",
        ),
        (
            "native DSO",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "schedule",
                "attention_library_sha256",
            ),
            "0" * 64,
        ),
        (
            "native metrics",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "schedule",
                "native_metrics",
                "tokens_seen",
            ),
            127,
        ),
        (
            "native timing",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_timing",
                "native_identity_schedule_seconds",
            ),
            999.0,
        ),
        (
            "implausible branch total",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "total_elapsed_seconds",
            ),
            1.0e100,
        ),
        (
            "missing native evidence",
            (
                "executed_mask_evaluations",
                "M0",
                "records",
                0,
                "native_oracle_layers",
                0,
                "sparse",
                "schedule",
                "indices_sha256",
            ),
            deleted,
        ),
        (
            "execution role",
            ("executed_mask_evaluations", "M2", "execution_role"),
            "gradient_and_candidate_evaluation",
        ),
        (
            "evidence key",
            ("evidence_checks", "exact_two_IHT_steps"),
            deleted,
        ),
        (
            "non-Boolean evidence",
            ("evidence_checks", "exact_two_IHT_steps"),
            1,
        ),
        (
            "post authentication",
            ("post_training_authentication", "package"),
            deleted,
        ),
        (
            "artifact",
            ("artifacts", "training_attention_library_sha256"),
            "0" * 64,
        ),
        (
            "framework",
            ("framework_contract", "synthetic_framework"),
            "tampered",
        ),
        (
            "record split",
            ("record_split", "split_identity"),
            "tampered",
        ),
        (
            "zero performance elapsed",
            ("performance", "elapsed_seconds"),
            0.0,
        ),
    )
    for _label, path, replacement in cases:
        tampered = deepcopy(result)
        parent = tampered
        for part in path[:-1]:
            parent = parent[part]
        if replacement is deleted:
            del parent[path[-1]]
        else:
            parent[path[-1]] = replacement
        with pytest.raises(ValueError, match="causal head-gate"):
            causal_gate._validate_training_result(
                tampered,
                result_sha256="a" * 64,
                supplied_sha256="a" * 64,
                protocol=protocol,
                protocol_sha256=protocol_hash,
            )

    negative_hidden = deepcopy(result)
    negative_loss = negative_hidden["executed_mask_evaluations"]["M0"]["records"][0][
        "loss"
    ]
    negative_loss["hidden_relative_l2"] = -0.1
    for band_name in (
        "positions_16_31",
        "positions_32_63",
        "positions_64_95",
        "positions_96_127",
    ):
        negative_loss["bands"][f"{band_name}_hidden_relative_l2"] = -0.1
    negative_loss["total"] = 9.0
    negative_evaluations = negative_hidden["executed_mask_evaluations"]
    negative_evaluations["M0"]["objective_summary"] = causal_gate._objective_summary(
        negative_evaluations["M0"]["records"]
    )
    negative_hidden["mask_selection"] = causal_gate._select_executed_mask(
        negative_evaluations
    )

    numeric_string_gradient = deepcopy(result)
    numeric_string_gradient["IHT_step_results"][0]["record_gradients"][0]["gradient"][
        0
    ][0] = "1.0"
    numeric_string_gradient["executed_mask_evaluations"]["M0"]["records"][0][
        "gradient"
    ][0][0] = "1.0"

    for adversarial in (
        negative_hidden,
        numeric_string_gradient,
    ):
        with pytest.raises(ValueError, match="causal head-gate"):
            causal_gate._validate_training_result(
                adversarial,
                result_sha256="a" * 64,
                supplied_sha256="a" * 64,
                protocol=protocol,
                protocol_sha256=protocol_hash,
            )


def test_mask_selection_rejects_any_per_record_M0_regression():
    def records(first: float, second: float):
        return [
            {"sequence_index": 0, "loss": {"total": first}},
            {"sequence_index": 1, "loss": {"total": second}},
        ]

    evaluations = {
        "M0": {"records": records(1.0, 10.0)},
        # Better maximum and mean, but record zero regresses.
        "M1": {"records": records(1.5, 2.0)},
        "M2": {"records": records(2.0, 3.0)},
    }

    selection = causal_gate._select_executed_mask(evaluations)

    assert selection["selected_mask_name"] == "M1"
    assert selection["screen_eligible"] is False
    assert selection["per_record_deltas"] == [
        {
            "sequence_index": 0,
            "selected_minus_M0_composite_objective": 0.5,
            "regressed": True,
        },
        {
            "sequence_index": 1,
            "selected_minus_M0_composite_objective": -8.0,
            "regressed": False,
        },
    ]


def test_exact_top_51_projection_is_deterministic_under_ties():
    scores = np.zeros((16, 16), dtype=np.float64)
    first = causal_gate._project_top_k(scores, count=51)
    second = causal_gate._project_top_k(scores, count=51)

    expected = np.zeros((16, 16), dtype=np.bool_)
    expected.reshape(-1)[:51] = True
    assert first.dtype == np.bool_
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert int(np.count_nonzero(first)) == 51

    scores[15, 15] = 1.0
    scores[7, 7] = 1.0
    projected = causal_gate._project_top_k(scores, count=51)
    assert projected[7, 7]
    assert projected[15, 15]
    assert int(np.count_nonzero(projected)) == 51

    invalid = scores.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError):
        causal_gate._project_top_k(invalid, count=51)
    with pytest.raises(ValueError, match="count"):
        causal_gate._project_top_k(scores, count=257)

    torch_mask = causal_gate._project_top_k(
        torch.zeros((16, 16), dtype=torch.float32),
        count=51,
    )
    assert isinstance(torch_mask, torch.Tensor)
    assert torch.equal(torch_mask.cpu(), torch.from_numpy(expected))


def test_projected_gate_step_is_exact_stable_iht():
    empty = np.zeros((16, 16), dtype=np.bool_)
    tied_gradient = np.ones((16, 16), dtype=np.float64)

    first_scores, first_mask, first_rms = causal_gate._projected_gate_step(
        empty,
        tied_gradient,
    )
    expected = np.zeros((16, 16), dtype=np.bool_)
    expected.reshape(-1)[:51] = True
    assert first_rms == pytest.approx(1.0)
    assert np.array_equal(first_mask, expected)
    assert int(first_mask.sum()) == 51
    assert np.isfinite(first_scores).all()

    scaled_scores, scaled_mask, scaled_rms = causal_gate._projected_gate_step(
        empty,
        tied_gradient * 7.0,
    )
    assert scaled_rms == pytest.approx(7.0)
    assert np.array_equal(scaled_mask, first_mask)
    np.testing.assert_allclose(first_scores, scaled_scores, atol=1e-11)

    carried_scores, carried_mask, carried_rms = causal_gate._projected_gate_step(
        first_mask,
        tied_gradient,
    )
    assert carried_rms == pytest.approx(1.0)
    assert np.array_equal(carried_mask, first_mask)
    assert carried_scores[first_mask].min() > carried_scores[~first_mask].max()

    invalid_mask = empty.copy()
    invalid_mask[0, 0] = True
    with pytest.raises(ValueError, match="input mask"):
        causal_gate._projected_gate_step(invalid_mask, tied_gradient)
    with pytest.raises(ValueError, match="RMS"):
        causal_gate._projected_gate_step(empty, np.zeros((16, 16)))
    invalid_gradient = tied_gradient.copy()
    invalid_gradient[0, 0] = np.nan
    with pytest.raises(ValueError, match="mask/gradient"):
        causal_gate._projected_gate_step(empty, invalid_gradient)

    asymmetric = np.zeros((16, 16), dtype=np.float64)
    asymmetric[0, 0] = -2.0
    asymmetric[0, 1] = 2.0
    _scores, negative_gradient_mask, _rms = causal_gate._projected_gate_step(
        empty, asymmetric
    )
    _scores, positive_gradient_mask, _rms = causal_gate._projected_gate_step(
        empty, -asymmetric
    )
    assert negative_gradient_mask[0, 0]
    assert not negative_gradient_mask[0, 1]
    assert positive_gradient_mask[0, 1]
    assert not positive_gradient_mask[0, 0]
