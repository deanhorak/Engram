from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_residual_capacity as capacity
from engram.utils import atomic_json, sha256_file


_BETA0_LOSSES = [
    1.3091617822647095,
    1.1883000135421753,
    1.3092560768127441,
    0.9684914350509644,
    1.2390215396881104,
    1.196218729019165,
    1.2578880786895752,
    1.3273429870605469,
]


def _bias_failure_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = capacity.bias._validated_bias_candidates()
    mask = capacity.bias._fixed_mask_descriptor()
    resource = {"within_total_traffic_budget": True, "marker": "K256"}
    outcomes: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        selected = index == 0
        mean = capacity._EXPECTED_GAMMA_HALF_MEAN if selected else 2.0 + index
        maximum = capacity._EXPECTED_GAMMA_HALF_WORST if selected else 2.5 + index
        outcomes[candidate["candidate_id"]] = {
            "candidate": candidate,
            "head_mask": mask,
            "resource_contract": resource,
            "population_resource_checks": {
                "counter_stream": True,
                "traffic": True,
            },
            "population_resource_passed": True,
            "loss_gate": {
                "passed": False,
                "summaries": {
                    "candidate": {
                        "mean_answer_cross_entropy": mean,
                        "maximum_answer_cross_entropy": maximum,
                    }
                },
            },
            "pre_replay_passed": False,
            "reset_replay": (
                {
                    "executed": True,
                    "native_sequence_forwards": 1,
                    "passed": True,
                }
                if selected
                else {
                    "executed": False,
                    "native_sequence_forwards": 0,
                }
            ),
            "passed": False,
        }
    frozen = {
        "candidates": candidates,
        "fixed_arm": {
            "head_mask": mask,
            "resource_contract": resource,
            "historical_K256_attribution": {
                "record_answer_cross_entropy": list(_BETA0_LOSSES),
            },
        },
    }
    candidate_ids = [row["candidate_id"] for row in candidates]
    result = {
        "schema_version": capacity.bias._SCHEMA_VERSION,
        "experiment": capacity.bias._RESULT_EXPERIMENT,
        "status": "train_episodic_logit_bias_gate_failed",
        "protocol": {
            "path": "/fixture/bias-protocol.json",
            "sha256": capacity._EXPECTED_BIAS_PROTOCOL_SHA256,
        },
        "scope": {
            "split": "train",
            "dense_teacher_forwards": 0,
            "fixed_K": 256,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "logit_bias_sweep": {
            "candidate_order": candidate_ids,
            "executed_candidates": candidate_ids,
            "skipped_candidates": [],
            "candidate_outcomes": outcomes,
            "selected_candidate_id": "gamma_1_2",
            "selected_candidate": candidates[0],
            "selection_role": "best_failed_candidate_for_diagnostic_replay",
            "selection_key": [
                capacity._EXPECTED_GAMMA_HALF_WORST,
                capacity._EXPECTED_GAMMA_HALF_MEAN,
                0,
            ],
            "passed": False,
        },
        "decision": {
            "train_progression_gate_passed": False,
            "semantic_gate_passed": False,
            "development_authorized": False,
            "confirmation_authorized": False,
        },
        "post_run_authentication": {
            key: True for key in capacity._EXPECTED_POST_AUTHENTICATION_KEYS
        },
        "confirmation_split_opened": False,
    }
    return frozen, result


def test_source_inventory_is_complete_union():
    assert capacity._SOURCE_FILES == tuple(
        dict.fromkeys(
            (
                *capacity.bias._SOURCE_FILES,
                "src/engram/evaluation/olmoe_retrieval_episodic_residual_capacity.py",
            )
        )
    )
    assert set(capacity.bias._SOURCE_FILES).issubset(capacity._SOURCE_FILES)
    assert "src/engram/runtime/olmoe_native.py" in capacity._SOURCE_FILES
    assert "native/src/olmoe_token_runtime.cpp" in capacity._SOURCE_FILES
    inventory = capacity._source_inventory()
    assert list(inventory) == list(capacity._SOURCE_FILES)
    assert all(
        capacity.bias.rank.retrieval._is_sha256(value) for value in inventory.values()
    )


def test_bias_total_failure_and_beta0_selection_are_exact():
    frozen, result = _bias_failure_fixture()
    failure = capacity._validate_bias_total_failure(
        result,
        protocol_path=Path("/fixture/bias-protocol.json"),
        protocol_sha256=capacity._EXPECTED_BIAS_PROTOCOL_SHA256,
        frozen_bias=frozen,
    )
    assert failure["all_candidates_executed"] is True
    assert failure["all_candidates_failed"] is True
    assert failure["systems_clean"] is True
    assert failure["selected_candidate_id"] == "gamma_1_2"
    choice = capacity._select_beta0_base(frozen, failure)
    assert choice["selected_base"] == "historical_beta0_K256"
    assert choice["beta_float32_bits"] == "0x00000000"
    assert choice["historical_mean_answer_cross_entropy"] == pytest.approx(
        1.2244600802659988,
        abs=0.0,
    )
    assert choice["historical_worst_answer_cross_entropy"] == _BETA0_LOSSES[-1]
    assert (
        choice["historical_mean_answer_cross_entropy"]
        < choice["diagnostic_gamma_1_2_mean_answer_cross_entropy"]
    )
    assert (
        choice["historical_worst_answer_cross_entropy"]
        < choice["diagnostic_gamma_1_2_worst_answer_cross_entropy"]
    )
    assert choice["base_tuning_permitted"] is False
    assert choice["fixed_before_trace_execution"] is True


@pytest.mark.parametrize(
    "mutation",
    [
        "skipped",
        "candidate_pass",
        "resource",
        "selection",
        "replay",
        "post_key",
        "development",
        "confirmation",
    ],
)
def test_bias_total_failure_rejects_tampering(mutation: str):
    frozen, result = _bias_failure_fixture()
    value = deepcopy(result)
    if mutation == "skipped":
        value["logit_bias_sweep"]["skipped_candidates"] = ["gamma_1_8"]
    elif mutation == "candidate_pass":
        value["logit_bias_sweep"]["candidate_outcomes"]["gamma_1_4"]["passed"] = True
    elif mutation == "resource":
        value["logit_bias_sweep"]["candidate_outcomes"]["gamma_3_16"][
            "population_resource_checks"
        ]["traffic"] = False
    elif mutation == "selection":
        value["logit_bias_sweep"]["selected_candidate_id"] = "gamma_1_4"
    elif mutation == "replay":
        value["logit_bias_sweep"]["candidate_outcomes"]["gamma_1_2"]["reset_replay"][
            "passed"
        ] = False
    elif mutation == "post_key":
        value["post_run_authentication"].pop("logit_bias_protocol")
    elif mutation == "development":
        value["decision"]["development_authorized"] = True
    else:
        value["confirmation_split_opened"] = True
    with pytest.raises(ValueError, match="bias"):
        capacity._validate_bias_total_failure(
            value,
            protocol_path=Path("/fixture/bias-protocol.json"),
            protocol_sha256=capacity._EXPECTED_BIAS_PROTOCOL_SHA256,
            frozen_bias=frozen,
        )


def test_beta0_selection_rejects_loss_or_diagnostic_drift():
    frozen, result = _bias_failure_fixture()
    failure = capacity._validate_bias_total_failure(
        result,
        protocol_path=Path("/fixture/bias-protocol.json"),
        protocol_sha256=capacity._EXPECTED_BIAS_PROTOCOL_SHA256,
        frozen_bias=frozen,
    )
    changed = deepcopy(frozen)
    changed["fixed_arm"]["historical_K256_attribution"]["record_answer_cross_entropy"][
        0
    ] += 1e-6
    with pytest.raises(ValueError, match="selection basis"):
        capacity._select_beta0_base(changed, failure)
    changed_failure = deepcopy(failure)
    changed_failure["selection_key"][1] += 1e-6
    with pytest.raises(ValueError, match="selection basis"):
        capacity._select_beta0_base(frozen, changed_failure)


class _ParityRuntime:
    attention_metrics_available = True
    episodic_metrics_available = True
    episodic_policy = dict(capacity._EPISODIC_POLICY)
    episodic_head_mask = tuple(
        tuple(True for _head in range(16)) for _layer in range(16)
    )
    episodic_logit_bias = 0.0

    def __init__(self, *, shadow: bool) -> None:
        self.position = 0
        self.shadow = shadow
        self.episodic_open_abi = "shadow_trace_v1" if shadow else "v2"
        self.shadow_trace_available = shadow
        self.shadow_attention_policy = dict(capacity._SHADOW_POLICY) if shadow else None
        self.reset_calls = 0
        self.close_calls = 0

    def reset(self) -> None:
        self.position = 0
        self.reset_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _parity_evidence(marker: str, elapsed: float) -> dict[str, Any]:
    return {
        "record_index": 0,
        "record_id": "train-0",
        "answer_cross_entropy": 1.25,
        "top1_tokens": [1, 2],
        "hidden_sha256": marker,
        "logits_sha256": marker,
        "counter_stream_sha256": marker,
        "episodic_call_stream_sha256": marker,
        "schedule_rows_sha256": "schedule",
        "counter_stream": [{"passed": True}],
        "counter_stream_passed": True,
        "final_metrics": {"cache_position": capacity._POSITIONS},
        "final_position": capacity._POSITIONS,
        "elapsed_seconds": elapsed,
    }


def test_parity_mock_requires_base_shadow_reset_and_trace_exact(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(capacity, "_HIDDEN_SIZE", 4)
    runtimes: list[_ParityRuntime] = []
    executions: dict[int, int] = {}

    def factory(shadow: bool):
        def make(_context):
            runtime = _ParityRuntime(shadow=shadow)
            runtimes.append(runtime)
            return runtime

        return make

    def execute(runtime, **_kwargs):
        raw = (
            runtime._runtime
            if isinstance(runtime, capacity._TraceCaptureRuntime)
            else runtime
        )
        executions[id(raw)] = executions.get(id(raw), 0) + 1
        raw.position = capacity._POSITIONS
        evidence = _parity_evidence("a" * 64, float(executions[id(raw)]))
        if not raw.shadow:
            return evidence, None, None
        arrays = {
            name: np.ones(
                (
                    len(capacity._READ_POSITIONS),
                    capacity._LAYERS,
                    capacity._HIDDEN_SIZE,
                ),
                dtype=np.float32,
            )
            * (index + 1)
            for index, name in enumerate(capacity._TRACE_KEYS)
        }
        return evidence, arrays, list(capacity._READ_POSITIONS)

    monkeypatch.setattr(capacity, "_execute_record", execute)
    result = capacity._run_trace_parity(
        context={},
        record={},
        schedule={},
        resource={},
        base_factory=factory(False),
        trace_factory=factory(True),
    )
    assert result["passed"] is True
    assert all(result["checks"].values())
    assert result["first_trace"] == result["reset_trace"]
    assert result["first_trace"]["nonzero_residual_read_rows"] == 32
    assert [runtime.reset_calls for runtime in runtimes] == [1, 1]
    assert [runtime.close_calls for runtime in runtimes] == [1, 1]


def test_parity_mock_rejects_shadow_output_or_trace_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(capacity, "_HIDDEN_SIZE", 4)
    calls = 0

    def factory(shadow: bool):
        return lambda _context: _ParityRuntime(shadow=shadow)

    def execute(runtime, **_kwargs):
        nonlocal calls
        calls += 1
        raw = (
            runtime._runtime
            if isinstance(runtime, capacity._TraceCaptureRuntime)
            else runtime
        )
        raw.position = capacity._POSITIONS
        marker = "b" * 64 if raw.shadow else "a" * 64
        evidence = _parity_evidence(marker, float(calls))
        if not raw.shadow:
            return evidence, None, None
        arrays = {
            name: np.ones((32, 16, 4), dtype=np.float32)
            for name in capacity._TRACE_KEYS
        }
        return evidence, arrays, list(capacity._READ_POSITIONS)

    monkeypatch.setattr(capacity, "_execute_record", execute)
    with pytest.raises(ValueError, match="trace parity failed"):
        capacity._run_trace_parity(
            context={},
            record={},
            schedule={},
            resource={},
            base_factory=factory(False),
            trace_factory=factory(True),
        )


def test_parity_post_authentication_requires_exact_fresh_twenty_key_map(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = {key: True for key in capacity._EXPECTED_BASE_POST_AUTHENTICATION_KEYS}
    assert len(expected) == 20
    monkeypatch.setattr(
        capacity,
        "_base_post_authentication",
        lambda _context, *, checkpoint: dict(expected),
    )
    assert (
        capacity._validated_base_post_authentication(
            expected,
            context={},
            checkpoint={},
        )
        == expected
    )
    missing = dict(expected)
    missing.pop("trace_library")
    with pytest.raises(ValueError, match="post-authentication changed"):
        capacity._validated_base_post_authentication(
            missing,
            context={},
            checkpoint={},
        )
    extra = {**expected, "unbound_all_true_key": True}
    with pytest.raises(ValueError, match="post-authentication changed"):
        capacity._validated_base_post_authentication(
            extra,
            context={},
            checkpoint={},
        )
    stale = dict(expected)
    stale["historical_K256_result"] = False
    with pytest.raises(ValueError, match="post-authentication changed"):
        capacity._validated_base_post_authentication(
            stale,
            context={},
            checkpoint={},
        )


def test_parity_trace_summary_requires_explicit_position_layer_hidden_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(capacity, "_HIDDEN_SIZE", 4)
    summary = capacity._trace_summary(
        _trace_arrays(4),
        capacity._READ_POSITIONS,
    )
    assert capacity._trace_report_summaries_valid(summary, summary) is True
    missing = deepcopy(summary)
    missing.pop("layout")
    assert capacity._trace_report_summaries_valid(missing, missing) is False
    changed = deepcopy(summary)
    changed["layout"] = "layer_position_hidden"
    assert capacity._trace_report_summaries_valid(changed, changed) is False


def _low_rank_targets(rank_value: int = 2, *, hidden: int = 8) -> np.ndarray:
    targets = np.zeros((8, 32, 16, hidden), dtype=np.float32)
    for sequence in range(8):
        for position in range(32):
            for layer in range(16):
                for component in range(rank_value):
                    targets[sequence, position, layer, component] = (
                        1.0
                        + 0.2 * sequence
                        + 0.03 * position * (component + 1)
                        + 0.01 * layer * (component + 2)
                    )
    return np.ascontiguousarray(targets)


def test_oracle_rank2_recovers_shared_subspace_and_replays_exactly():
    targets = _low_rank_targets(2)
    result = capacity._capacity_screen(targets)
    assert result["passed"] is True
    assert result["selected_rank"] == 2
    assert result["selection_role"] == "smallest_passing_rank"
    assert result["rank_outcomes"]["2"]["global"]["recovery"] == pytest.approx(
        1.0,
        abs=1e-12,
    )
    assert all(
        row["recovery"] == pytest.approx(1.0, abs=1e-12)
        for row in result["rank_outcomes"]["2"]["heldout_sequences"]
    )
    assert result["selected_metric_replay"]["passed"] is True
    assert (
        result["selected_metric_replay"]["reference_sha256"]
        == result["selected_metric_replay"]["recomputed_sha256"]
    )
    assert result["rank_zero_mean_baseline"]["rank"] == 0


def test_fold_fit_does_not_observe_heldout_sequence():
    targets = _low_rank_targets(2)
    mean, basis, indices = capacity._fit_fold_layer_subspace(
        targets,
        heldout=3,
        layer=7,
        rank_value=2,
    )
    changed = targets.copy()
    changed[3] = np.float32(10_000.0)
    changed_mean, changed_basis, changed_indices = capacity._fit_fold_layer_subspace(
        changed,
        heldout=3,
        layer=7,
        rank_value=2,
    )
    np.testing.assert_array_equal(mean, changed_mean)
    np.testing.assert_allclose(
        basis.T @ basis,
        changed_basis.T @ changed_basis,
        rtol=0.0,
        atol=1e-12,
    )
    assert indices == changed_indices == [0, 1, 2, 4, 5, 6, 7]


def test_random_sequence_specific_residual_fails_capacity_gate():
    rng = np.random.default_rng(20260729)
    targets = np.ascontiguousarray(rng.normal(size=(8, 32, 16, 64)).astype(np.float32))
    result = capacity._capacity_screen(targets)
    assert result["passed"] is False
    assert result["selection_role"] == "best_failed_rank_for_diagnostic_replay"
    assert result["selected_rank"] in capacity._RANKS
    assert result["selected_metric_replay"]["passed"] is True
    assert any(not row["gate"]["passed"] for row in result["rank_outcomes"].values())


def _selection_outcome(
    rank_value: int,
    *,
    recovery: float,
    sequence_error: float,
) -> dict[str, Any]:
    error_squared = 1.0 - recovery
    return {
        "rank": rank_value,
        "global": {
            "target_squared_frobenius": 1.0,
            "error_squared_frobenius": error_squared,
            "error_ratio": float(np.sqrt(max(error_squared, 0.0))),
            "recovery": recovery,
        },
        "heldout_sequences": [
            {
                "record_index": index,
                "target_squared_frobenius": 1.0,
                "error_squared_frobenius": sequence_error**2,
                "error_ratio": sequence_error,
                "recovery": 1.0 - sequence_error**2,
            }
            for index in range(8)
        ],
        "layers": [
            {
                "layer": layer,
                "target_squared_frobenius": 1.0,
                "error_squared_frobenius": error_squared,
                "error_ratio": float(np.sqrt(max(error_squared, 0.0))),
                "recovery": recovery,
            }
            for layer in range(16)
        ],
        "block_entry_positions": [
            {
                "position": position,
                "target_squared_frobenius": 1.0,
                "error_squared_frobenius": error_squared,
                "error_ratio": float(np.sqrt(max(error_squared, 0.0))),
                "recovery": recovery,
            }
            for position in capacity._BLOCK_ENTRY_POSITIONS
        ],
        "folds": [],
    }


def test_capacity_replay_uses_identical_full_population_qr_width(
    monkeypatch: pytest.MonkeyPatch,
):
    full_ranks = (0, *capacity._RANKS)
    calls: list[tuple[int, ...]] = []
    population = {
        0: _selection_outcome(0, recovery=0.10, sequence_error=0.90),
        2: _selection_outcome(2, recovery=0.70, sequence_error=0.50),
        4: _selection_outcome(4, recovery=0.80, sequence_error=0.40),
        8: _selection_outcome(8, recovery=0.90, sequence_error=0.30),
    }

    def metrics(_targets, ranks):
        requested = tuple(ranks)
        calls.append(requested)
        result = {
            rank_value: deepcopy(population[rank_value]) for rank_value in requested
        }
        # This emulates the observed floating-bit QR-width drift. The former
        # selected-rank-only replay path would receive (2,), change the metric
        # SHA, and fail deterministic replay.
        if requested != full_ranks and 2 in result:
            result[2]["global"]["error_squared_frobenius"] += 2.27e-13
        return result

    monkeypatch.setattr(capacity, "_capacity_metrics", metrics)
    result = capacity._capacity_screen(_low_rank_targets(2))
    assert result["selected_rank"] == 2
    assert result["selected_metric_replay"]["passed"] is True
    assert calls == [full_ranks, full_ranks]
    assert (
        result["selected_metric_replay"]["reference_sha256"]
        == result["selected_metric_replay"]["recomputed_sha256"]
    )


def test_gate_selects_smallest_pass_and_failure_uses_lexicographic_key():
    passing = {
        2: _selection_outcome(2, recovery=0.49, sequence_error=0.7),
        4: _selection_outcome(4, recovery=0.60, sequence_error=0.6),
        8: _selection_outcome(8, recovery=0.80, sequence_error=0.4),
    }
    selected = capacity._select_capacity_outcome(passing)
    assert selected["passed"] is True
    assert selected["selected_rank"] == 4
    assert selected["selection_role"] == "smallest_passing_rank"

    failed = {
        2: _selection_outcome(2, recovery=0.10, sequence_error=0.90),
        4: _selection_outcome(4, recovery=0.20, sequence_error=0.80),
        8: _selection_outcome(8, recovery=0.20, sequence_error=0.80),
    }
    failed[4]["global"]["error_ratio"] = 0.75
    failed[8]["global"]["error_ratio"] = 0.70
    selected = capacity._select_capacity_outcome(failed)
    assert selected["passed"] is False
    assert selected["selected_rank"] == 8
    assert selected["selection_key"] == [0.8, 0.7, 8]
    assert selected["selection_role"] == ("best_failed_rank_for_diagnostic_replay")


def test_capacity_validators_reject_shape_dtype_nan_and_rank_population():
    targets = _low_rank_targets(2)
    with pytest.raises(ValueError, match="target tensor"):
        capacity._capacity_metrics(targets[:, :-1], (2,))
    with pytest.raises(ValueError, match="target tensor"):
        capacity._capacity_metrics(targets.astype(np.float64), (2,))
    changed = targets.copy()
    changed[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="target tensor"):
        capacity._capacity_metrics(changed, (2,))
    with pytest.raises(ValueError, match="rank population"):
        capacity._select_capacity_outcome(
            {2: _selection_outcome(2, recovery=0.5, sequence_error=0.5)}
        )


def _trace_arrays(hidden: int) -> dict[str, np.ndarray]:
    shape = (len(capacity._READ_POSITIONS), capacity._LAYERS, hidden)
    return {
        "input_norm": np.ones(shape, dtype=np.float32),
        "base_projected": np.full(shape, 2.0, dtype=np.float32),
        "target_residual": np.full(shape, 3.0, dtype=np.float32),
    }


def test_safetensors_shard_round_trip_tamper_and_symlink_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(capacity, "_HIDDEN_SIZE", 4)
    arrays = _trace_arrays(4)
    descriptor = capacity._write_trace_shard(
        tmp_path,
        record={"record_index": 0, "record_id": "train-0"},
        arrays=arrays,
        positions=capacity._READ_POSITIONS,
        source_record_sha256="a" * 64,
        output_sha256="b" * 64,
        reset_output_sha256="c" * 64,
        reset_trace_sha256=capacity._trace_array_digest(arrays),
    )
    path = tmp_path / descriptor["file"]
    loaded = capacity._validate_trace_shard(path, descriptor)
    for name in capacity._TRACE_KEYS:
        np.testing.assert_array_equal(loaded[name], arrays[name])

    symlink = tmp_path / "trace-link.safetensors"
    symlink.symlink_to(path)
    linked_descriptor = dict(descriptor)
    linked_descriptor["file"] = symlink.name
    with pytest.raises(ValueError, match="descriptor"):
        capacity._validate_trace_shard(symlink, linked_descriptor)

    with path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="descriptor"):
        capacity._validate_trace_shard(path, descriptor)


def test_shard_directory_rejects_direct_symlink_and_nonempty_target(
    tmp_path: Path,
):
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked"
    linked_directory.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(ValueError, match="directory is invalid"):
        capacity._prepare_shard_directory(linked_directory)
    (real_directory / "existing").write_text("occupied", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        capacity._prepare_shard_directory(real_directory)


def test_shard_rejects_tensor_hash_and_zero_residual(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(capacity, "_HIDDEN_SIZE", 4)
    arrays = _trace_arrays(4)
    descriptor = capacity._write_trace_shard(
        tmp_path,
        record={"record_index": 0, "record_id": "train-0"},
        arrays=arrays,
        positions=capacity._READ_POSITIONS,
        source_record_sha256="a" * 64,
        output_sha256="b" * 64,
        reset_output_sha256="c" * 64,
        reset_trace_sha256=capacity._trace_array_digest(arrays),
    )
    changed = deepcopy(descriptor)
    changed["tensor_sha256"]["target_residual"] = "0" * 64
    with pytest.raises(ValueError, match="tensor hash"):
        capacity._validate_trace_shard(tmp_path / descriptor["file"], changed)
    zero = _trace_arrays(4)
    zero["target_residual"][2] = 0.0
    with pytest.raises(ValueError, match="zero residual"):
        capacity._trace_summary(zero, capacity._READ_POSITIONS)


def _protocol_fixtures(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    monkeypatch.setattr(
        capacity.bias.rank.fixed,
        "_derive_schedule",
        lambda input_ids, _anchors: {"rows_sha256": f"{int(input_ids[0]):064x}"},
    )
    monkeypatch.setattr(
        capacity,
        "_source_inventory",
        lambda: {"capacity.py": "f" * 64},
    )
    context = {
        "protocol": {
            "package": {"path": "package", "manifest_sha256": "1" * 64},
            "corpus": {"splits": {"train": {"sha256": "2" * 64}}},
            "source_model": {"config_sha256": "3" * 64},
            "libraries": {"attention": {"sha256": "4" * 64}},
        },
        "model": {"layers": 16, "hidden_size": 2048},
        "train_records": [
            {"record_index": index, "input_ids": [index]} for index in range(8)
        ],
        "bias_protocol_path": Path("/fixture/bias-protocol.json"),
        "bias_protocol_sha256": capacity._EXPECTED_BIAS_PROTOCOL_SHA256,
        "bias_result_path": Path("/fixture/bias-result.json"),
        "bias_result_sha256": capacity._EXPECTED_BIAS_RESULT_SHA256,
        "trace_library_path": Path("/fixture/trace.so"),
        "trace_library_sha256": "5" * 64,
        "base_choice": {"selected_base": "historical_beta0_K256"},
    }
    training = {"checkpoint": "fixture"}
    frozen_bias = {
        "training_checkpoint": {
            "path": "checkpoint.json",
            "sha256": "6" * 64,
            "training_sha256": "7" * 64,
        },
        "fixed_arm": {
            "resource_contract": {"traffic": 10},
            "historical_K256_attribution": {"attribution_only": True},
        },
        "tokenizer_fact_anchor_ids": {"A": [1], "B": [2]},
        "authenticated_confirmation_descriptor": {
            "file": "confirmation.jsonl",
            "sha256": "8" * 64,
        },
    }
    failure = {"all_candidates_failed": True}
    parity = {
        "path": "/fixture/parity.json",
        "sha256": "9" * 64,
        "status": capacity._PARITY_STATUS,
    }
    return context, training, frozen_bias, failure, parity


def test_protocol_reconstruction_is_exact_and_train_only(
    monkeypatch: pytest.MonkeyPatch,
):
    context, training, frozen_bias, failure, parity = _protocol_fixtures(monkeypatch)
    first = capacity._build_protocol(
        context=context,
        training=training,
        frozen_bias=frozen_bias,
        failure=failure,
        parity=parity,
    )
    second = capacity._build_protocol(
        context=context,
        training=training,
        frozen_bias=frozen_bias,
        failure=failure,
        parity=parity,
    )
    assert first == second
    assert first["capacity_method"]["ranks"] == [2, 4, 8]
    assert first["policies"]["base"] == capacity._BASE_POLICY
    assert first["policies"]["shadow"] == capacity._SHADOW_POLICY
    assert first["trace_schema"]["positions"] == list(capacity._READ_POSITIONS)
    assert (
        first["trace_schema"]["W128_exact_full_context_only_for_position_horizon"]
        == 128
    )
    assert first["scope"]["capacity_evidence_only"] is True
    assert first["scope"]["learned_predictor"] is False
    assert first["scope"]["confirmation_file_access_permitted"] is False
    assert first["confirmation_split_opened"] is False
    assert first["source_sha256"] == {"capacity.py": "f" * 64}


def test_authenticate_protocol_reconstructs_and_rejects_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context, training, frozen_bias, failure, parity = _protocol_fixtures(monkeypatch)
    value = capacity._build_protocol(
        context=context,
        training=training,
        frozen_bias=frozen_bias,
        failure=failure,
        parity=parity,
    )
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, value)
    monkeypatch.setattr(
        capacity,
        "_authenticate_bias_inputs",
        lambda **_kwargs: (context, training, frozen_bias, failure),
    )
    monkeypatch.setattr(
        capacity,
        "_validate_parity_report",
        lambda **_kwargs: parity,
    )
    loaded_context, loaded_training, loaded = capacity._authenticate_protocol(
        protocol_path,
        sha256_file(protocol_path),
    )
    assert loaded == value
    assert loaded_training == training
    assert loaded_context["capacity_protocol_path"] == protocol_path.resolve()

    tampered = deepcopy(value)
    tampered["capacity_method"]["ranks"] = [2, 8]
    tampered_path = tmp_path / "tampered.json"
    atomic_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="frozen protocol changed"):
        capacity._authenticate_protocol(
            tampered_path,
            sha256_file(tampered_path),
        )


def test_root_hashes_are_exact_constants():
    assert (
        capacity._EXPECTED_BIAS_PROTOCOL_SHA256
        == "025ff45e41966faf033338ffcac0c3fc1f93b40ed7676c36f189ba57485e8be7"
    )
    assert (
        capacity._EXPECTED_BIAS_RESULT_SHA256
        == "19d08ce9eb4b673d423e9781a491e25ec03bdec09467a43e7be1881874ef2287"
    )
    assert capacity._RANKS == (2, 4, 8)
    assert capacity._BASE_POLICY == {
        "local_window": 16,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
    assert capacity._SHADOW_POLICY == {
        "local_window": 128,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }
