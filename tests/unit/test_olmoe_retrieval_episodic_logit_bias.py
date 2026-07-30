from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_logit_bias as bias


_EXPECTED_BIAS_BITS = [
    ("gamma_1_2", "0x3f000000", "0xbf317218"),
    ("gamma_1_4", "0x3e800000", "0xbfb17218"),
    ("gamma_3_16", "0x3e400000", "0xbfd644dc"),
    ("gamma_1_8", "0x3e000000", "0xc0051592"),
]


def test_bias_grid_order_and_float32_bits_are_exact():
    candidates = bias._validated_bias_candidates()
    assert [
        (
            row["candidate_id"],
            row["gamma_float32_bits"],
            row["beta_float32_bits"],
        )
        for row in candidates
    ] == _EXPECTED_BIAS_BITS
    assert [row["order"] for row in candidates] == [0, 1, 2, 3]
    assert [bias._float32_bits(row["gamma_float32"]) for row in candidates] == [
        row[1] for row in _EXPECTED_BIAS_BITS
    ]
    assert [bias._float32_bits(row["beta_float32"]) for row in candidates] == [
        row[2] for row in _EXPECTED_BIAS_BITS
    ]
    assert bias._float32_bits(0.0) == "0x00000000"

    mask = bias._all_ones_mask()
    assert mask.dtype == np.bool_
    assert mask.shape == (16, 16)
    assert int(mask.sum()) == 256

    tampered = deepcopy(candidates)
    tampered[1]["beta_float32_bits"] = "0xbfb17219"
    with pytest.raises(ValueError, match="candidate changed"):
        bias._validated_bias_candidates(tampered)


class _ParityRuntime:
    attention_metrics_available = True
    episodic_metrics_available = True
    episodic_policy = {"slots": 32, "span_size": 8}

    def __init__(self, route: str) -> None:
        self.route = route
        self.position = 0
        self.executions = 0
        self.reset_calls = 0
        self.closed = False
        if route == "v1":
            self.episodic_head_mask = None
            self.episodic_logit_bias = 0.0
            self.episodic_open_abi = "v1"
        else:
            self.episodic_head_mask = tuple(
                tuple(True for _head in range(16)) for _layer in range(16)
            )
            self.episodic_logit_bias = 0.0
            self.episodic_open_abi = "v2"

    def reset(self) -> None:
        self.position = 0
        self.reset_calls += 1

    def close(self) -> None:
        self.closed = True


def _parity_evidence(marker: str, *, elapsed_ns: int) -> dict[str, Any]:
    return {
        "record_index": 0,
        "record_id": "train-0",
        "top1_tokens": [7, 8],
        "output_sha256": marker,
        "hidden_sha256": marker,
        "logits_sha256": marker,
        "counter_stream_sha256": "counter",
        "episodic_call_stream_sha256": "calls",
        "schedule_rows_sha256": "schedule-0",
        "counter_stream_passed": True,
        "final_metrics": {
            "attention_state_bytes": 10,
            "episodic_entries_read": 20,
            "elapsed_ns": elapsed_ns,
        },
        "final_position": bias._POSITIONS,
    }


def _run_mock_parity(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tamper_v2_first: bool,
) -> tuple[dict[str, Any] | None, list[_ParityRuntime]]:
    runtimes: list[_ParityRuntime] = []

    def legacy_factory(_context):
        runtime = _ParityRuntime("v1")
        runtimes.append(runtime)
        return runtime

    def v2_factory(_context, beta):
        assert bias._float32_bits(beta) == "0x00000000"
        runtime = _ParityRuntime("v2")
        runtimes.append(runtime)
        return runtime

    def execute(runtime, **_kwargs):
        runtime.executions += 1
        runtime.position = bias._POSITIONS
        marker = "same"
        if tamper_v2_first and runtime.route == "v2" and runtime.executions == 1:
            marker = "tampered"
        return _parity_evidence(marker, elapsed_ns=runtime.executions)

    monkeypatch.setattr(bias, "_execute_parity_record", execute)
    result = bias._run_zero_bias_parity(
        context={},
        record={"record_index": 0, "record_id": "train-0"},
        schedule={"rows_sha256": "schedule-0"},
        resource={},
        legacy_factory=legacy_factory,
        v2_factory=v2_factory,
    )
    return result, runtimes


def test_beta_zero_v1_v2_parity_and_reset_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    result, runtimes = _run_mock_parity(
        monkeypatch,
        tamper_v2_first=False,
    )
    assert result is not None
    assert result["passed"] is True
    assert result["native_sequence_forwards"] == 4
    assert result["native_token_steps"] == 4 * bias._POSITIONS
    assert all(result["checks"].values())
    assert [runtime.route for runtime in runtimes] == ["v1", "v2"]
    assert [runtime.executions for runtime in runtimes] == [2, 2]
    assert [runtime.reset_calls for runtime in runtimes] == [1, 1]
    assert all(runtime.closed for runtime in runtimes)


def test_beta_zero_parity_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    runtimes: list[_ParityRuntime] = []

    def factory(route):
        def make(_context, *_beta):
            runtime = _ParityRuntime(route)
            runtimes.append(runtime)
            return runtime

        return make

    def execute(runtime, **_kwargs):
        runtime.executions += 1
        runtime.position = bias._POSITIONS
        marker = "tampered" if runtime.route == "v2" else "same"
        return _parity_evidence(marker, elapsed_ns=runtime.executions)

    monkeypatch.setattr(bias, "_execute_parity_record", execute)
    with pytest.raises(ValueError, match="V1/V2 parity failed"):
        bias._run_zero_bias_parity(
            context={},
            record={"record_index": 0, "record_id": "train-0"},
            schedule={"rows_sha256": "schedule-0"},
            resource={},
            legacy_factory=factory("v1"),
            v2_factory=factory("v2"),
        )
    assert [runtime.executions for runtime in runtimes] == [2, 2]
    assert all(runtime.closed for runtime in runtimes)


class _SweepRuntime:
    def __init__(self, beta: float) -> None:
        self.beta = beta
        self.candidate_id: str | None = None
        self.replay_calls = 0
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _records() -> list[dict[str, Any]]:
    return [
        {
            "record_index": index,
            "record_id": f"train-{index}",
            "input_ids": [index],
        }
        for index in range(bias._RECORDS)
    ]


def _protocol(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = bias._validated_bias_candidates()
    return {
        "candidate_order": [row["candidate_id"] for row in candidates],
        "candidates": candidates,
        "fixed_arm": {"resource_contract": {"fixed": True}},
        "tokenizer_fact_anchor_ids": {},
        "schedule_contract": {
            "per_record_rows_sha256": [
                f"schedule-{record['record_index']}" for record in records
            ]
        },
    }


def _install_sweep_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    specifications: dict[str, dict[str, Any]],
    replay_failures: set[str] | None = None,
) -> list[str]:
    replay_failures = set() if replay_failures is None else replay_failures
    replayed: list[str] = []

    monkeypatch.setattr(
        bias.rank.fixed,
        "_derive_schedule",
        lambda input_ids, _anchors: {"rows_sha256": f"schedule-{input_ids[0]}"},
    )

    def population(runtime, *, candidate, resource, **_kwargs):
        candidate_id = candidate["candidate_id"]
        runtime.candidate_id = candidate_id
        specification = specifications[candidate_id]
        systems = bool(specification.get("systems", True))
        gate = bool(specification.get("gate", False))
        return {
            "candidate": dict(candidate),
            "resource_contract": resource,
            "sequence_evidence": [],
            "loss_gate": {
                "passed": gate,
                "summaries": {
                    "candidate": {
                        "maximum_answer_cross_entropy": float(specification["maximum"]),
                        "mean_answer_cross_entropy": float(specification["mean"]),
                    }
                },
            },
            "population_resource_passed": systems,
            "pre_replay_passed": gate and systems,
            "reset_replay": {
                "executed": False,
                "native_sequence_forwards": 0,
            },
            "passed": False,
        }

    def attach(outcome, runtime, **_kwargs):
        candidate_id = outcome["candidate"]["candidate_id"]
        runtime.replay_calls += 1
        replayed.append(candidate_id)
        passed = candidate_id not in replay_failures
        outcome["reset_replay"] = {
            "executed": True,
            "native_sequence_forwards": 1,
            "passed": passed,
        }
        outcome["passed"] = outcome["pre_replay_passed"] and passed

    monkeypatch.setattr(bias, "_candidate_population", population)
    monkeypatch.setattr(bias, "_attach_reset_replay", attach)
    return replayed


def _run_sweep(
    monkeypatch: pytest.MonkeyPatch,
    specifications: dict[str, dict[str, Any]],
    *,
    replay_failures: set[str] | None = None,
) -> tuple[dict[str, Any], list[_SweepRuntime], list[str]]:
    records = _records()
    runtimes: list[_SweepRuntime] = []
    replayed = _install_sweep_stubs(
        monkeypatch,
        specifications=specifications,
        replay_failures=replay_failures,
    )

    def factory(_context, beta):
        runtime = _SweepRuntime(beta)
        runtimes.append(runtime)
        return runtime

    result = bias._run_bias_sweep(
        context={},
        records=records,
        protocol=_protocol(records),
        baselines={},
        runtime_factory=factory,
    )
    return result, runtimes, replayed


def test_ordered_sweep_selects_first_replay_qualified_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    ids = [row["candidate_id"] for row in bias._BIAS_CANDIDATES]
    specifications = {
        ids[0]: {"maximum": 2.0, "mean": 1.5, "gate": False},
        ids[1]: {"maximum": 1.0, "mean": 0.8, "gate": True},
        ids[2]: {"maximum": 0.9, "mean": 0.7, "gate": True},
        ids[3]: {"maximum": 0.8, "mean": 0.6, "gate": True},
    }
    result, runtimes, replayed = _run_sweep(
        monkeypatch,
        specifications,
    )
    assert result["passed"] is True
    assert result["selected_candidate_id"] == ids[1]
    assert result["selection_role"] == "first_replay_qualified_strict_pass"
    assert result["executed_candidates"] == ids[:2]
    assert result["skipped_candidates"] == ids[2:]
    assert result["population_native_sequence_forwards"] == 16
    assert result["reset_replay_native_sequence_forwards"] == 1
    assert result["total_native_sequence_forwards"] == 17
    assert replayed == [ids[1]]
    assert [runtime.candidate_id for runtime in runtimes] == ids[:2]
    assert [bias._float32_bits(runtime.beta) for runtime in runtimes] == [
        row["beta_float32_bits"] for row in bias._BIAS_CANDIDATES[:2]
    ]
    assert all(runtime.closed for runtime in runtimes)


def test_total_failure_replays_lexicographic_best_only(
    monkeypatch: pytest.MonkeyPatch,
):
    ids = [row["candidate_id"] for row in bias._BIAS_CANDIDATES]
    specifications = {
        ids[0]: {"maximum": 1.8, "mean": 1.2, "gate": False},
        ids[1]: {"maximum": 1.5, "mean": 1.4, "gate": False},
        ids[2]: {"maximum": 1.5, "mean": 1.3, "gate": False},
        ids[3]: {"maximum": 1.5, "mean": 1.3, "gate": False},
    }
    result, runtimes, replayed = _run_sweep(
        monkeypatch,
        specifications,
    )
    assert result["passed"] is False
    assert result["selected_candidate_id"] == ids[2]
    assert result["selection_role"] == ("best_failed_candidate_for_diagnostic_replay")
    assert result["selection_key"] == [1.5, 1.3, 2]
    assert result["executed_candidates"] == ids
    assert result["skipped_candidates"] == []
    assert result["population_native_sequence_forwards"] == 32
    assert result["reset_replay_native_sequence_forwards"] == 1
    assert result["total_native_sequence_forwards"] == 33
    assert replayed == [ids[2]]
    assert [runtime.replay_calls for runtime in runtimes] == [0, 0, 1, 0]
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.parametrize(
    ("failure", "message", "expected_runtime_count"),
    [
        ("systems", "systems contract failed", 1),
        ("passing_replay", "reset replay failed", 1),
        ("diagnostic_replay", "reset replay failed", 4),
    ],
)
def test_systems_or_replay_failure_aborts_and_closes_runtimes(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
    expected_runtime_count: int,
):
    records = _records()
    ids = [row["candidate_id"] for row in bias._BIAS_CANDIDATES]
    specifications = {
        candidate_id: {
            "maximum": 2.0 + index,
            "mean": 1.0 + index,
            "gate": False,
        }
        for index, candidate_id in enumerate(ids)
    }
    replay_failures: set[str] = set()
    if failure == "systems":
        specifications[ids[0]]["systems"] = False
    elif failure == "passing_replay":
        specifications[ids[0]]["gate"] = True
        replay_failures.add(ids[0])
    else:
        replay_failures.add(ids[0])

    runtimes: list[_SweepRuntime] = []
    replayed = _install_sweep_stubs(
        monkeypatch,
        specifications=specifications,
        replay_failures=replay_failures,
    )

    def factory(_context, beta):
        runtime = _SweepRuntime(beta)
        runtimes.append(runtime)
        return runtime

    with pytest.raises(ValueError, match=message):
        bias._run_bias_sweep(
            context={},
            records=records,
            protocol=_protocol(records),
            baselines={},
            runtime_factory=factory,
        )
    assert len(runtimes) == expected_runtime_count
    assert all(runtime.closed for runtime in runtimes)
    assert replayed == ([] if failure == "systems" else [ids[0]])


def _rank_failure_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    protocol_path = (tmp_path / "rank-protocol.json").resolve()
    protocol_sha256 = "a" * 64
    historical_k256 = {"fixture": "authenticated-K256"}
    descriptors: list[dict[str, Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    for index, k in enumerate(bias.rank._CANDIDATE_K):
        head_mask = {
            "K": k,
            "mask_sha256": (
                bias._EXPECTED_K165_MASK_SHA256 if k == 165 else f"mask-{k}"
            ),
        }
        resource = (
            dict(bias._EXPECTED_K165_RESOURCE)
            if k == 165
            else {"combined_state_bytes": 1_000 + k}
        )
        descriptor = {
            "K": k,
            "head_mask": head_mask,
            "resource_contract": resource,
        }
        maximum = 2.0 + index / 10.0
        mean = 1.0 + index / 10.0
        outcome = {
            "K": k,
            "head_mask": head_mask,
            "resource_contract": resource,
            "population_resource_passed": True,
            "loss_gate": {
                "passed": False,
                "summaries": {
                    "candidate": {
                        "maximum_answer_cross_entropy": maximum,
                        "mean_answer_cross_entropy": mean,
                    }
                },
            },
            "pre_replay_passed": False,
            "reset_replay": {
                "executed": k == 165,
                "passed": k == 165,
            },
            "passed": False,
        }
        descriptors.append(descriptor)
        outcomes[f"K{k}"] = outcome
    selected_summary = outcomes["K165"]["loss_gate"]["summaries"]["candidate"]
    frozen_rank = {
        "candidates": descriptors,
        "all_head_K256_attribution": historical_k256,
    }
    value = {
        "schema_version": bias.rank._SCHEMA_VERSION,
        "experiment": bias.rank._RESULT_EXPERIMENT,
        "status": "train_episodic_rank_sweep_gate_failed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "scope": {
            "dense_teacher_forwards": 0,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "rank_sweep": {
            "candidate_order": list(bias.rank._CANDIDATE_K),
            "executed_candidates": list(bias.rank._CANDIDATE_K),
            "skipped_candidates": [],
            "passed": False,
            "selected_K": 165,
            "selection_role": "best_failed_candidate_for_diagnostic_replay",
            "selection_key": [
                selected_summary["maximum_answer_cross_entropy"],
                selected_summary["mean_answer_cross_entropy"],
                165,
            ],
            "candidate_outcomes": outcomes,
        },
        "K256_attribution": historical_k256,
        "decision": {
            "passed": False,
            "semantic_gate_passed": False,
            "confirmation_authorized": False,
        },
        "post_run_authentication": {
            "rank_sweep_protocol": True,
            "confirmation_not_opened": True,
        },
        "confirmation_split_opened": False,
    }
    return value, protocol_path, protocol_sha256, frozen_rank


def _mutate_rank_failure(value: dict[str, Any], mutation: str) -> None:
    if mutation == "status":
        value["status"] = "train_episodic_rank_sweep_gate_passed"
    elif mutation == "passed":
        value["rank_sweep"]["passed"] = True
    elif mutation == "incomplete_executed":
        value["rank_sweep"]["executed_candidates"] = [64, 96, 128]
    elif mutation == "skipped":
        value["rank_sweep"]["skipped_candidates"] = [165]
    elif mutation == "resource":
        value["rank_sweep"]["candidate_outcomes"]["K96"][
            "population_resource_passed"
        ] = False
    elif mutation == "postauth":
        value["post_run_authentication"]["rank_sweep_protocol"] = False
    elif mutation == "development":
        value["scope"]["development_outcomes_used"] = True
    elif mutation == "confirmation":
        value["decision"]["confirmation_authorized"] = True
    elif mutation == "confirmation_opened":
        value["scope"]["confirmation_split_opened"] = True
        value["confirmation_split_opened"] = True
    elif mutation == "protocol_binding":
        value["protocol"]["sha256"] = "b" * 64
    else:  # pragma: no cover - protects the table itself
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "status",
        "passed",
        "incomplete_executed",
        "skipped",
        "resource",
        "postauth",
        "development",
        "confirmation",
        "confirmation_opened",
        "protocol_binding",
    ],
)
def test_rank_failure_validator_rejects_changed_invariants(
    tmp_path: Path,
    mutation: str,
):
    value, protocol_path, protocol_sha256, frozen_rank = _rank_failure_fixture(tmp_path)
    accepted = bias._validate_rank_failure_value(
        value,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        frozen_rank=frozen_rank,
    )
    assert accepted["all_candidates_failed"] is True
    assert accepted["selected_K"] == 165
    assert accepted["selected_reset_replay_passed"] is True
    assert accepted["systems_clean"] is True

    tampered = deepcopy(value)
    _mutate_rank_failure(tampered, mutation)
    with pytest.raises(ValueError, match="ranked"):
        bias._validate_rank_failure_value(
            tampered,
            protocol_path=protocol_path,
            protocol_sha256=protocol_sha256,
            frozen_rank=frozen_rank,
        )


class _TrainOnlyContext(dict):
    def __getitem__(self, key):
        if key in {
            "confirmation_records",
            "confirmation_path",
            "development_outcomes",
            "development_result",
        }:
            raise AssertionError(f"forbidden experiment input: {key}")
        return super().__getitem__(key)


def _parity_freeze_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    _TrainOnlyContext,
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    source_inventory = {"logit-bias.py": "c" * 64}
    monkeypatch.setattr(bias, "_source_inventory", lambda: source_inventory)
    context = _TrainOnlyContext(
        {
            "rank_protocol_path": (tmp_path / "rank-protocol.json").resolve(),
            "rank_protocol_sha256": "a" * 64,
            "rank_result_path": (tmp_path / "rank-result.json").resolve(),
            "rank_result_sha256": "b" * 64,
            "rank_failure": {"fixture": "rank-total-failure"},
            "rank_protocol": {"tokenizer_fact_anchor_ids": {}},
            "historical_episodic_library_path": (tmp_path / "legacy-v1.so").resolve(),
            "historical_episodic_library_sha256": "d" * 64,
            "logit_bias_library_path": (tmp_path / "proposed-v2.so").resolve(),
            "logit_bias_library_sha256": "e" * 64,
            "package_path": (tmp_path / "model.engram").resolve(),
            "protocol": {"package": {"manifest_sha256": "2" * 64}},
            "train_records": _records(),
            "confirmation_descriptor": {
                "file": "sealed-confirmation.jsonl",
                "sha256": "f" * 64,
                "records": 8,
            },
        }
    )
    training = {"fixture": "training-checkpoint"}
    checkpoint = {
        "path": str((tmp_path / "checkpoint.json").resolve()),
        "sha256": "1" * 64,
    }
    frozen_rank = {"tokenizer_fact_anchor_ids": {}}
    rank_failure = {
        "selected_head_mask": {"K": 165},
        "selected_resource_contract": {"K": 165},
        "selected_loss_summary": {"candidate": {"mean": 1.0}},
    }
    fixed = {
        "resource_contract": {"fixed": "K256"},
        "historical": {"fixture": "historical-K256"},
    }
    monkeypatch.setattr(
        bias.rank.fixed,
        "_checkpoint_references",
        lambda observed: (
            {"baselines": {"M0": {}, "M2": {}}}
            if observed is training
            else (_ for _ in ()).throw(AssertionError("training changed"))
        ),
    )
    monkeypatch.setattr(
        bias.rank.fixed,
        "_derive_schedule",
        lambda input_ids, _anchors: {"rows_sha256": f"schedule-{input_ids[0]}"},
    )
    monkeypatch.setattr(
        bias.episodic,
        "_counter_checks",
        lambda *_args, **_kwargs: {"synthetic_counter_contract": True},
    )
    return (
        context,
        training,
        checkpoint,
        frozen_rank,
        rank_failure,
        fixed,
    )


def _parity_report(
    context: _TrainOnlyContext,
    fixed: dict[str, Any],
    source_inventory: dict[str, str],
) -> dict[str, Any]:
    checks = {
        "legacy_reset_replay_exact": True,
        "v2_reset_replay_exact": True,
        "v1_v2_first_outputs_and_counters_exact": True,
        "v1_v2_replay_outputs_and_counters_exact": True,
        "passed": True,
    }
    evidence = {
        "record_index": 0,
        "record_id": context["train_records"][0]["record_id"],
        "top1_tokens": [7] * bias._POSITIONS,
        "output_sha256": "3" * 64,
        "hidden_sha256": "4" * 64,
        "logits_sha256": "5" * 64,
        "counter_stream_sha256": "6" * 64,
        "episodic_call_stream_sha256": "7" * 64,
        "schedule_rows_sha256": "schedule-0",
        "counter_stream_passed": True,
        "final_metrics": {"synthetic_counter": 1, "elapsed_ns": 10},
        "final_position": bias._POSITIONS,
    }
    return {
        "schema_version": bias._SCHEMA_VERSION,
        "experiment": bias._PARITY_EXPERIMENT,
        "status": bias._PARITY_STATUS,
        "rank_protocol": {
            "path": str(context["rank_protocol_path"]),
            "sha256": context["rank_protocol_sha256"],
        },
        "rank_result": {
            "path": str(context["rank_result_path"]),
            "sha256": context["rank_result_sha256"],
            "authenticated_failure": context["rank_failure"],
        },
        "legacy_v1_library": {
            "path": str(context["historical_episodic_library_path"]),
            "sha256": context["historical_episodic_library_sha256"],
            "required_symbol": bias._REQUIRED_V1_SYMBOL,
        },
        "proposed_v2_library": {
            "path": str(context["logit_bias_library_path"]),
            "sha256": context["logit_bias_library_sha256"],
            "required_symbol": bias._REQUIRED_V2_SYMBOL,
        },
        "fixed_arm": {
            "head_mask": bias._fixed_mask_descriptor(),
            "resource_contract": fixed["resource_contract"],
            "historical_K256_attribution": fixed["historical"],
        },
        "package": {
            "path": str(context["package_path"]),
            "manifest_sha256": context["protocol"]["package"]["manifest_sha256"],
        },
        "beta_zero": {
            "value": 0.0,
            "float32_bits": "0x00000000",
            "legacy_route": "episodic V1 all-head open",
            "new_route": "episodic headwise V2 all-ones open",
        },
        "scope": {
            "split": "train",
            "record_index": 0,
            "positions": bias._POSITIONS,
            "dense_teacher_forwards": 0,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "parity": {
            "legacy_v1_first": deepcopy(evidence),
            "legacy_v1_reset_replay": deepcopy(evidence),
            "explicit_beta_zero_v2_first": deepcopy(evidence),
            "explicit_beta_zero_v2_reset_replay": deepcopy(evidence),
            "passed": True,
            "native_sequence_forwards": 4,
            "native_token_steps": 4 * bias._POSITIONS,
            "checks": checks,
        },
        "schedule_rows_sha256": "schedule-0",
        "source_sha256": source_inventory,
        "post_run_authentication": {
            key: True for key in bias._EXPECTED_POST_AUTHENTICATION_KEYS
        },
        "confirmation_split_opened": False,
    }


def _write_parity_report(path: Path, value: dict[str, Any]) -> str:
    bias.atomic_json(path, value)
    return bias.sha256_file(path)


def _install_fixed_input_authentication(
    monkeypatch: pytest.MonkeyPatch,
    values: tuple[
        _TrainOnlyContext,
        dict[str, Any],
        dict[str, str],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
) -> None:
    def authenticate(**kwargs):
        assert kwargs["rank_protocol_sha256"] == bias._EXPECTED_RANK_PROTOCOL_SHA256
        assert kwargs["rank_result_sha256"] == bias._EXPECTED_RANK_RESULT_SHA256
        assert kwargs["logit_bias_library_sha256"] == "e" * 64
        return values

    monkeypatch.setattr(bias, "_authenticate_fixed_inputs", authenticate)


def test_parity_report_validates_and_freezes_without_confirmation_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    values = _parity_freeze_fixture(tmp_path, monkeypatch)
    context, _training, _checkpoint, _frozen, _failure, fixed = values
    source_inventory = bias._source_inventory()
    parity_path = tmp_path / "parity.json"
    parity_sha256 = _write_parity_report(
        parity_path,
        _parity_report(context, fixed, source_inventory),
    )
    parity = bias._validate_parity_report(
        path=parity_path,
        expected_sha256=parity_sha256,
        context=context,
        fixed=fixed,
    )
    assert parity["outputs_counters_and_reset_exact"] is True
    assert parity["beta_float32_bits"] == "0x00000000"

    _install_fixed_input_authentication(monkeypatch, values)
    output = tmp_path / "protocol.json"
    frozen = bias.freeze_episodic_logit_bias_protocol(
        rank_protocol=context["rank_protocol_path"],
        rank_protocol_sha256=bias._EXPECTED_RANK_PROTOCOL_SHA256,
        rank_result=context["rank_result_path"],
        rank_result_sha256=bias._EXPECTED_RANK_RESULT_SHA256,
        logit_bias_library=context["logit_bias_library_path"],
        logit_bias_library_sha256=context["logit_bias_library_sha256"],
        parity_report=parity_path,
        parity_report_sha256=parity_sha256,
        out=output,
    )
    protocol = frozen["protocol"]
    assert output.is_file()
    assert protocol["candidate_order"] == [
        row["candidate_id"] for row in bias._BIAS_CANDIDATES
    ]
    assert protocol["beta_zero_parity"] == parity
    assert protocol["train_scope"]["confirmation_file_access_permitted"] is False
    assert protocol["confirmation_split_opened"] is False


def _tamper_parity_report(value: dict[str, Any], mutation: str) -> None:
    if mutation == "parity":
        value["parity"]["checks"]["v1_v2_first_outputs_and_counters_exact"] = False
    elif mutation == "binding":
        value["rank_result"]["sha256"] = "9" * 64
    elif mutation == "source":
        value["source_sha256"] = {"logit-bias.py": "8" * 64}
    elif mutation == "missing_postauth":
        value["post_run_authentication"].pop("source_shards")
    else:  # pragma: no cover - protects the table itself
        raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    ["parity", "binding", "source", "missing_postauth"],
)
def test_parity_validator_and_freeze_reject_tampered_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    values = _parity_freeze_fixture(tmp_path, monkeypatch)
    context, _training, _checkpoint, _frozen, _failure, fixed = values
    report = _parity_report(context, fixed, bias._source_inventory())
    _tamper_parity_report(report, mutation)
    parity_path = tmp_path / f"parity-{mutation}.json"
    parity_sha256 = _write_parity_report(parity_path, report)

    with pytest.raises(ValueError, match="parity report is invalid"):
        bias._validate_parity_report(
            path=parity_path,
            expected_sha256=parity_sha256,
            context=context,
            fixed=fixed,
        )

    _install_fixed_input_authentication(monkeypatch, values)
    output = tmp_path / f"protocol-{mutation}.json"
    with pytest.raises(ValueError, match="parity report is invalid"):
        bias.freeze_episodic_logit_bias_protocol(
            rank_protocol=context["rank_protocol_path"],
            rank_protocol_sha256=bias._EXPECTED_RANK_PROTOCOL_SHA256,
            rank_result=context["rank_result_path"],
            rank_result_sha256=bias._EXPECTED_RANK_RESULT_SHA256,
            logit_bias_library=context["logit_bias_library_path"],
            logit_bias_library_sha256=context["logit_bias_library_sha256"],
            parity_report=parity_path,
            parity_report_sha256=parity_sha256,
            out=output,
        )
    assert not output.exists()
