from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_full_visible_simplex_oracle as full


def _basis_arrays(
    *,
    prefix: tuple[int, ...] = (1,),
    padded_older: bool = False,
    query_position: int = 96,
) -> dict[str, np.ndarray]:
    heads = 2
    dimension = 2
    hidden = heads * dimension
    regular_entries = 20
    episodic_entries = 8

    regular_template = np.arange(
        heads * regular_entries * dimension,
        dtype=np.float32,
    ).reshape(heads, regular_entries, dimension) / np.float32(8.0) - np.float32(4.0)
    slot_template = (
        np.arange(
            heads * episodic_entries * dimension,
            dtype=np.float32,
        ).reshape(heads, episodic_entries, dimension)
        % np.float32(13.0)
    ) / np.float32(2.0) - np.float32(3.0)
    regular_values = np.broadcast_to(
        regular_template,
        prefix + regular_template.shape,
    ).copy()
    slot_values = np.broadcast_to(
        slot_template,
        prefix + slot_template.shape,
    ).copy()

    regular_weights = np.arange(1, regular_entries + 1, dtype=np.float32)
    if padded_older:
        regular_weights[-1] = np.float32(0.0)
        regular_values[..., -1, :] = np.float32(0.0)
    regular_weights *= np.float32(0.6) / np.sum(
        regular_weights,
        dtype=np.float32,
    )
    slot_weights = np.arange(1, episodic_entries + 1, dtype=np.float32)
    slot_weights *= np.float32(0.4) / np.sum(
        slot_weights,
        dtype=np.float32,
    )
    regular_entry_mass = np.broadcast_to(
        regular_weights,
        prefix + (heads, regular_entries),
    ).copy()
    slot_mass = np.broadcast_to(
        slot_weights,
        prefix + (heads, episodic_entries),
    ).copy()
    regular_mass = np.sum(
        regular_entry_mass,
        axis=-1,
        dtype=np.float32,
    )
    episodic_mass = np.sum(slot_mass, axis=-1, dtype=np.float32)

    kinds = np.concatenate(
        (
            np.full(16, 1, dtype=np.uint8),
            np.full(4, 2, dtype=np.uint8),
        )
    )
    if padded_older:
        kinds[-1] = np.uint8(0)
    entry_kind = np.broadcast_to(
        kinds,
        prefix + (heads, regular_entries),
    ).copy()
    positions = np.concatenate(
        (
            np.arange(
                query_position - 15,
                query_position + 1,
                dtype=np.uint64,
            ),
            np.arange(8, 12, dtype=np.uint64),
        )
    )
    if padded_older:
        positions[-1] = full._INVALID_POSITION
    entry_positions = np.broadcast_to(
        positions,
        prefix + (heads, regular_entries),
    ).copy()

    regular_component = np.einsum(
        "...he,...hed->...hd",
        regular_entry_mass,
        regular_values,
        optimize=True,
    ).astype(np.float32)
    episodic_component = np.einsum(
        "...hs,...hsd->...hd",
        slot_mass,
        slot_values,
        optimize=True,
    ).astype(np.float32)
    base_heads = regular_component + episodic_component
    base = base_heads.reshape(prefix + (hidden,))
    return {
        "base_attention_output": np.ascontiguousarray(base),
        "regular_component": np.ascontiguousarray(
            regular_component.reshape(prefix + (hidden,))
        ),
        "episodic_component": np.ascontiguousarray(
            episodic_component.reshape(prefix + (hidden,))
        ),
        "regular_mass": np.ascontiguousarray(regular_mass),
        "episodic_mass": np.ascontiguousarray(episodic_mass),
        "base_projected": np.ascontiguousarray(base.copy()),
        "target_residual": np.ones_like(base, dtype=np.float32),
        "regular_entry_mass": np.ascontiguousarray(regular_entry_mass),
        "regular_entry_values": np.ascontiguousarray(regular_values),
        "regular_entry_valid_kind": np.ascontiguousarray(entry_kind),
        "regular_entry_positions": np.ascontiguousarray(entry_positions),
        "slot_mass": np.ascontiguousarray(slot_mass),
        "slot_values": np.ascontiguousarray(slot_values),
    }


def _capture_arrays(
    *,
    padded_older: bool = False,
    query_position: int = 96,
) -> dict[str, np.ndarray]:
    arrays = _basis_arrays(
        prefix=(1, 1),
        padded_older=padded_older,
        query_position=query_position,
    )
    arrays["episodic_source_positions"] = np.arange(
        32,
        40,
        dtype=np.uint64,
    ).reshape(1, 8)
    return arrays


def test_build_authoritative_28_and_optimistic_29_way_bases() -> None:
    arrays = _basis_arrays()
    constructible = full.build_full_visible_basis(
        arrays,
        query_heads=2,
        include_exact_native_anchor=False,
    )
    optimistic = full.build_full_visible_basis(
        arrays,
        query_heads=2,
        include_exact_native_anchor=True,
    )
    assert constructible.components.shape == (1, 2, 28, 2)
    assert optimistic.components.shape == (1, 2, 29, 2)
    assert not constructible.diagnostic_only
    assert not optimistic.diagnostic_only
    np.testing.assert_allclose(
        np.sum(constructible.base_coefficients, axis=-1),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        optimistic.base_coefficients[..., 0],
        np.ones((1, 2), dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.einsum(
            "nhc,nhcd->nhd",
            constructible.base_coefficients,
            constructible.components,
            optimize=True,
        ),
        constructible.base_heads,
        rtol=0.0,
        atol=7.5e-5,
    )


def test_invalid_padding_duplicates_a_real_value_without_expanding_hull() -> None:
    arrays = _basis_arrays(padded_older=True)
    basis = full.build_full_visible_basis(
        arrays,
        query_heads=2,
        include_exact_native_anchor=False,
    )
    assert basis.invalid_regular_entries == 2
    np.testing.assert_array_equal(
        basis.base_coefficients[..., 19],
        np.zeros((1, 2), dtype=np.float64),
    )
    np.testing.assert_array_equal(
        basis.components[..., 19, :],
        basis.components[..., 0, :],
    )


def test_nested_10_way_selected_set_is_subset_of_16_way() -> None:
    arrays = _basis_arrays()
    ten = full.build_nested_visible_basis(
        arrays,
        component_count=10,
        query_heads=2,
    )
    sixteen = full.build_nested_visible_basis(
        arrays,
        component_count=16,
        query_heads=2,
    )
    assert ten.components.shape[-2] == 10
    assert sixteen.components.shape[-2] == 16
    assert ten.diagnostic_only and sixteen.diagnostic_only
    for head in range(2):
        assert set(ten.component_source_indices[0, head, 1:]).issubset(
            set(sixteen.component_source_indices[0, head, 1:])
        )
    np.testing.assert_allclose(
        np.einsum(
            "nhc,nhcd->nhd",
            ten.base_coefficients,
            ten.components,
            optimize=True,
        ),
        ten.base_heads,
        rtol=0.0,
        atol=7.5e-5,
    )


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    (
        (
            "regular_entry_mass",
            lambda value: value.__setitem__(
                (0, 0, 0),
                value[0, 0, 0] + np.float32(0.1),
            ),
            "masses",
        ),
        (
            "regular_entry_valid_kind",
            lambda value: value.__setitem__((0, 0, 0), np.uint8(2)),
            "kind",
        ),
        (
            "regular_entry_positions",
            lambda value: value.__setitem__((0, 0, 1), value[0, 0, 0]),
            "positions",
        ),
        (
            "slot_values",
            lambda value: value.__setitem__((0, 0, 0, 0), np.float32(1.0001)),
            "BF16",
        ),
    ),
)
def test_basis_rejects_corrupt_trace(
    name: str,
    mutate,
    message: str,
) -> None:
    arrays = _basis_arrays()
    arrays[name] = arrays[name].copy()
    mutate(arrays[name])
    with pytest.raises(ValueError, match=message):
        full.build_full_visible_basis(
            arrays,
            query_heads=2,
            include_exact_native_anchor=False,
        )


def test_trace_summary_validates_causality_and_duplicate_exclusion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(full, "_READ_POSITIONS", (96,))
    monkeypatch.setattr(full, "_LAYERS", 1)
    monkeypatch.setattr(full, "_QUERY_HEADS", 2)
    monkeypatch.setattr(full, "_HEAD_DIMENSION", 2)
    monkeypatch.setattr(full, "_HIDDEN_SIZE", 4)
    arrays = _capture_arrays()
    summary = full._trace_summary(arrays, [96])
    assert summary["visible_entries_per_head"] == {
        "minimum": 28,
        "mean": 28.0,
        "maximum": 28,
    }
    assert summary["slot_values_exact_bf16_decodes"]

    duplicate = {name: value.copy() for name, value in arrays.items()}
    duplicate["regular_entry_positions"][0, 0, 0, 16] = np.uint64(32)
    with pytest.raises(ValueError, match="duplicate episodic"):
        full._trace_summary(duplicate, [96])

    future = {name: value.copy() for name, value in arrays.items()}
    future["regular_entry_positions"][0, 0, 0, 15] = np.uint64(97)
    with pytest.raises(ValueError, match="not causal"):
        full._trace_summary(future, [96])


def _patch_small_solver_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(full, "_RECORDS", 1)
    monkeypatch.setattr(full, "_READ_POSITIONS", (96,))
    monkeypatch.setattr(full, "_LAYERS", 1)
    monkeypatch.setattr(full, "_QUERY_HEADS", 2)
    monkeypatch.setattr(full, "_HEAD_DIMENSION", 2)
    monkeypatch.setattr(full, "_HIDDEN_SIZE", 4)


def test_array_solver_recovers_known_visible_value_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_solver_contract(monkeypatch)
    arrays = _basis_arrays(prefix=(1, 1, 1))
    basis = full.build_full_visible_basis(
        {
            name: value.reshape((-1,) + value.shape[3:])
            if name in full._BASE_TRACE_KEYS
            else value.reshape((-1,) + value.shape[3:])
            for name, value in arrays.items()
        },
        query_heads=2,
        include_exact_native_anchor=False,
    )
    desired = basis.components[:, :, 0].reshape(1, 4)
    arrays["target_residual"] = (
        desired.reshape(1, 1, 1, 4) - arrays["base_attention_output"]
    ).astype(np.float32)
    projection = np.eye(4, dtype=np.float32)[None]
    result = full.run_full_visible_oracle_from_arrays(
        arrays,
        projection,
        include_exact_native_anchor=False,
        row_batch_size=1,
        maximum_active_set_iterations=32,
        fallback_maximum_iterations=128,
        relative_gap_tolerance=1.0e-8,
        absolute_gap_tolerance=1.0e-10,
    )
    assert result.component_count == 28
    assert not result.exact_native_anchor_included
    assert result.deterministic_replay_exact
    assert result.authenticated_base_projection_max_abs == 0.0
    assert np.max(result.direct_error_energy) <= 1.0e-9
    np.testing.assert_allclose(
        result.objective,
        result.direct_error_energy,
        rtol=0.0,
        atol=1.0e-9,
    )


def test_array_solver_rejects_mismatched_authenticated_base_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_small_solver_contract(monkeypatch)
    arrays = _basis_arrays(prefix=(1, 1, 1))
    arrays["base_projected"] = arrays["base_projected"] + np.float32(0.01)
    with pytest.raises(ValueError, match="does not match Wo"):
        full.run_full_visible_oracle_from_arrays(
            arrays,
            np.eye(4, dtype=np.float32)[None],
            include_exact_native_anchor=False,
            row_batch_size=1,
        )


def test_native_regular_entry_adapter_uses_fixed_capability_and_fields() -> None:
    expected = {
        "entry_mass": np.ones((1, 2, 20), dtype=np.float32),
        "entry_values": np.ones((1, 2, 20, 2), dtype=np.float32),
        "valid_kind": np.ones((1, 2, 20), dtype=np.uint8),
        "positions": np.ones((1, 2, 20), dtype=np.uint64),
    }
    runtime = SimpleNamespace(
        regular_entry_trace_available=True,
        last_regular_entry_trace=lambda: SimpleNamespace(**expected),
    )
    captured = full.NativeRegularEntryTraceAdapter().capture(runtime)
    assert set(captured) == set(full._REGULAR_ENTRY_TRACE_KEYS)
    np.testing.assert_array_equal(
        captured["regular_entry_mass"],
        expected["entry_mass"],
    )
    with pytest.raises(ValueError, match="lacks"):
        full.NativeRegularEntryTraceAdapter().capture(
            SimpleNamespace(regular_entry_trace_available=False)
        )


def _predecessor_fixture() -> tuple[dict[str, object], dict[str, object]]:
    resource = {
        "fixed_attention_state_bytes": 10534912,
        "fixed_combined_attention_and_episodic_traffic_bytes": 714866688,
        "fixed_fraction_of_dense_full_context_KV": 0.33030523255813954,
        "gamma_zero_earns_read_savings": False,
        "oracle_shadow_trace_and_projection_evaluator_only": True,
        "predictor_weights_features_and_execution_not_counted": True,
    }
    binding = {"path": "/tmp/cached_v2_protocol.json", "sha256": "a" * 64}
    protocol: dict[str, object] = {
        "schema_version": 1,
        "experiment": ("olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_protocol"),
        "status": "frozen_before_authenticated_cached_v2_solve",
        "resource_contract": resource,
        "confirmation_split_opened": False,
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": (
            "olmoe_q7_retrieval_episodic_slot_simplex_cached_v2_train_screen"
        ),
        "status": "train_episodic_slot_simplex_cached_v2_gate_failed",
        "protocol": binding,
        "resource_contract": resource,
        "decision": {
            "failure_is_decisive": True,
            "certified_optimistic_gate_passed": False,
            "train_slot_simplex_capacity_gate_passed": False,
        },
        "confirmation_split_opened": False,
    }
    return protocol, result


def test_predecessor_validator_accepts_actual_schema_v1_and_rejects_v2() -> None:
    protocol, result = _predecessor_fixture()
    binding = result["protocol"]
    assert isinstance(binding, dict)
    full._validate_predecessor_failure(
        protocol,
        result,
        protocol_binding=binding,
    )
    protocol["schema_version"] = 2
    with pytest.raises(ValueError, match="not authoritative"):
        full._validate_predecessor_failure(
            protocol,
            result,
            protocol_binding=binding,
        )


def test_checked_file_rejects_confirmation_lexically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "confirmation.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="confirmation scope"):
        full._checked_file(path, digest, "forbidden input")


def test_trace_shard_and_manifest_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_small_solver_contract(monkeypatch)
    arrays = _capture_arrays()
    shard_dir = full._prepare_shard_directory(tmp_path / "capture")
    digest = "b" * 64
    summary = full._trace_summary(arrays, [96])
    descriptor = full.write_full_visible_trace_shard(
        shard_dir,
        record_index=0,
        record_id="train-00",
        arrays=arrays,
        query_positions=[96],
        source_record_sha256=digest,
        output_evidence_sha256=digest,
        reset_output_evidence_sha256=digest,
        reset_trace_sha256=summary["trace_sha256"],
        schedule_rows_sha256=digest,
    )
    loaded = full.validate_full_visible_trace_shard(
        shard_dir / descriptor["file"],
        descriptor,
    )
    np.testing.assert_array_equal(
        loaded["regular_entry_values"],
        arrays["regular_entry_values"],
    )
    protocol = {"path": "/tmp/protocol.json", "sha256": "c" * 64}
    manifest = full.write_full_visible_trace_manifest(
        shard_dir,
        protocol=protocol,
        shards=[descriptor],
    )
    stacked, manifest_value = full.load_stacked_full_visible_trace(
        manifest["path"],
        manifest["sha256"],
        protocol=protocol,
    )
    assert manifest_value["confirmation_split_opened"] is False
    assert stacked["base_attention_output"].shape == (1, 1, 1, 4)


def test_blockwise_qk_shard_and_manifest_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_small_solver_contract(monkeypatch)
    qk = np.arange(
        1 * 1 * 2 * full._C28_QK_TRACE_ENTRIES * full._QK_PARTIAL_BANDS,
        dtype=np.float32,
    ).reshape(
        1,
        1,
        2,
        full._C28_QK_TRACE_ENTRIES,
        full._QK_PARTIAL_BANDS,
    )
    shard_dir = full._prepare_shard_directory(tmp_path / "qk-capture")
    digest = "b" * 64
    descriptor = full.write_full_visible_qk_trace_shard(
        shard_dir,
        record_index=0,
        record_id="train-00",
        qk_partials=qk,
        reset_qk_partials=qk.copy(),
        query_positions=[96],
        source_record_sha256=digest,
        output_evidence_sha256=digest,
        reset_output_evidence_sha256=digest,
        schedule_rows_sha256=digest,
    )
    loaded = full.validate_full_visible_qk_trace_shard(
        shard_dir / descriptor["file"],
        descriptor,
    )
    np.testing.assert_array_equal(loaded, qk)
    protocol = {"path": "/tmp/protocol.json", "sha256": "c" * 64}
    manifest = full.write_full_visible_qk_trace_manifest(
        shard_dir,
        protocol=protocol,
        shards=[descriptor],
    )
    stacked, manifest_value = full.load_stacked_full_visible_qk_trace(
        manifest["path"],
        manifest["sha256"],
        protocol=protocol,
    )
    assert manifest_value["confirmation_split_opened"] is False
    np.testing.assert_array_equal(stacked, qk[None])


def test_qualification_failure_is_inconclusive_not_decisive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        full,
        "run_full_visible_oracle_from_arrays",
        lambda *args, **kwargs: object(),
    )

    def summarize(
        _solved: object,
        *,
        arm: str,
        progression_authority: bool,
    ) -> dict[str, object]:
        del progression_authority
        if arm.startswith("constructible"):
            return {
                "feasible_gate_passed": False,
                "qualification": {"passed": True},
                "optimistic_recovery_upper_bound_metrics": {"passed": False},
                "optimistic_gate_passed": False,
            }
        return {
            "feasible_gate_passed": False,
            "qualification": {"passed": False},
            "optimistic_recovery_upper_bound_metrics": {"passed": False},
            "optimistic_gate_passed": False,
        }

    monkeypatch.setattr(full, "_summarize_oracle_arm", summarize)
    result = full.run_cached_full_visible_screen(
        {},
        np.empty((0,), dtype=np.float32),
        include_nested_diagnostics=False,
    )
    assert result["status"] == "train_full_visible_simplex_gate_inconclusive"
    assert result["decision"]["failure_is_decisive"] is False


def _manifest_audit_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    record = {"record_id": "train-00", "input_ids": [1, 2, 3]}
    source_sha256 = full.sha256_json(record)
    output_sha256 = "d" * 64
    schedule_sha256 = "e" * 64
    base_hashes = {
        name: f"{index + 1:064x}" for index, name in enumerate(full._BASE_TRACE_KEYS)
    }
    slot_hashes = {
        "slot_mass": "a" * 64,
        "slot_values": "b" * 64,
    }
    tensor_hashes = {name: "c" * 64 for name in full._CAPTURE_TRACE_KEYS}
    tensor_hashes.update(base_hashes)
    tensor_hashes.update(slot_hashes)
    common = {
        "record_index": 0,
        "record_id": record["record_id"],
        "source_record_sha256": source_sha256,
        "output_evidence_sha256": output_sha256,
        "reset_output_evidence_sha256": output_sha256,
    }
    head_manifest = {
        "confirmation_split_opened": False,
        "shards": [{**common, "tensor_sha256": base_hashes}],
    }
    slot_manifest = {
        "confirmation_split_opened": False,
        "shards": [{**common, "tensor_sha256": slot_hashes}],
    }
    head_result = {
        "confirmation_split_opened": False,
        "base_output_authentication": [
            {
                "record_index": 0,
                "record_id": record["record_id"],
                "source_record_sha256": source_sha256,
                "observed_output_evidence_sha256": output_sha256,
                "reset_output_evidence_sha256": output_sha256,
            }
        ],
    }

    def write_json(name: str, value: object) -> dict[str, str]:
        path = tmp_path / name
        path.write_text(
            json.dumps(value, sort_keys=True),
            encoding="utf-8",
        )
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    frozen: dict[str, object] = {
        "historical_bindings": {
            "train_split": {
                "path": str(train_path),
                "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
            },
            "inherited_head_mass_manifest": write_json(
                "head-manifest.json",
                head_manifest,
            ),
            "slot_trace_manifest": write_json(
                "slot-manifest.json",
                slot_manifest,
            ),
            "inherited_head_mass_result": write_json(
                "head-result.json",
                head_result,
            ),
        },
        "schedule_contract": {
            "per_record_rows_sha256": [schedule_sha256],
        },
    }
    manifest: dict[str, object] = {
        "shards": [
            {
                **common,
                "schedule_rows_sha256": schedule_sha256,
                "tensor_sha256": tensor_hashes,
            }
        ]
    }
    return frozen, manifest


@pytest.mark.parametrize(
    "mutation",
    ("record", "source", "schedule", "base_tensor", "slot_tensor"),
)
def test_manifest_audit_rejects_unbound_record_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    monkeypatch.setattr(full, "_RECORDS", 1)
    frozen, manifest = _manifest_audit_fixture(tmp_path)
    assert all(full._audit_manifest_record_bindings(frozen, manifest).values())
    changed = deepcopy(manifest)
    descriptor = changed["shards"][0]
    if mutation == "record":
        descriptor["record_id"] = "wrong-record"
    elif mutation == "source":
        descriptor["source_record_sha256"] = "f" * 64
    elif mutation == "schedule":
        descriptor["schedule_rows_sha256"] = "f" * 64
    elif mutation == "base_tensor":
        descriptor["tensor_sha256"]["base_attention_output"] = "f" * 64
    else:
        descriptor["tensor_sha256"]["slot_values"] = "f" * 64
    with pytest.raises(ValueError, match="not authenticated"):
        full._audit_manifest_record_bindings(frozen, changed)
