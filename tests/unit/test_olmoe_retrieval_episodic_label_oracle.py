from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_label_oracle as label_oracle
from engram.utils import sha256_file, sha256_json


_ANCHORS = {
    "A": (2, 3, 4, 5),
    "B": (6, 7, 8, 9),
    "C": (10, 11, 12, 13),
    "D": (14, 15, 16, 17),
}


def _record(index: int, order: str) -> dict[str, Any]:
    input_ids = [1] * 129
    codes = [
        [20 + ((index * 7 + label_index * 8 + offset) % 40) for offset in range(8)]
        for label_index in range(4)
    ]
    for depth, label in enumerate(order):
        label_index = label_oracle.retrieval._LABELS.index(label)
        anchor = label_oracle.retrieval._FACT_ANCHORS[depth]
        input_ids[anchor : anchor + 4] = _ANCHORS[label]
        source = label_oracle.retrieval._PASSKEY_SOURCE_STARTS[depth]
        input_ids[source : source + 8] = codes[label_index]
    input_ids[97:] = [value for code in codes for value in code]
    provisional = {
        "record_index": index,
        "record_id": f"train-{index}",
        "identity_sha256": "",
        "input_ids": input_ids,
        "answer_prediction_positions": list(range(96, 128)),
    }
    provisional["identity_sha256"] = sha256_json(
        {key: value for key, value in provisional.items() if key != "identity_sha256"}
    )
    return provisional


def _records() -> list[dict[str, Any]]:
    return [
        _record(index, order)
        for index, order in enumerate(label_oracle.retrieval._FACT_ORDERS)
    ]


def _model() -> dict[str, int]:
    return {
        "layers": 16,
        "query_heads": 16,
        "key_value_heads": 16,
        "head_dimension": 1,
        "hidden_size": 4,
        "vocab_size": 64,
    }


def _context(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": _model(),
        "q7_expectations": {
            "scheduled_bytes_per_position": 10,
            "scheduled_bytes_per_sequence": 1_280,
        },
        "train_records": records,
    }


def _protocol(
    records: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    schedules = [
        label_oracle._derive_schedule(record["input_ids"], _ANCHORS)
        for record in records
    ]
    return {
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in _ANCHORS.items()
        },
        "schedule_contract": {
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ]
        },
        "resource_contract": label_oracle._resource_contract(
            context["model"],
            context["q7_expectations"],
        ),
    }


class _PrefixTrap(Sequence[int]):
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, slice):
            stop = len(self.values) if key.stop is None else key.stop
            if stop > 97:
                raise AssertionError("schedule derivation inspected future input")
            return self.values[key]
        if key > 96:
            raise AssertionError("schedule derivation inspected future input")
        return self.values[key]


class _FakeRuntime:
    attention_metrics_available = True
    episodic_policy = {"slots": 36, "span_size": 9}

    def __init__(
        self,
        *,
        context: dict[str, Any],
        records: list[dict[str, Any]],
        resource: dict[str, Any],
        tamper_counter: bool = False,
        tamper_replay: bool = False,
    ) -> None:
        self.context = context
        self.records = records
        self.resource = resource
        self.tamper_counter = tamper_counter
        self.tamper_replay = tamper_replay
        self._position = 0
        self._run = 0
        self._writes: list[int] = []
        self._reads: list[int] = []
        self._last_logits = np.zeros(64, dtype=np.float32)
        self.closed = False

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0
        self._run += 1
        self._writes = []
        self._reads = []

    def forward_episodic(
        self,
        token_ids: list[int],
        write_slots: list[int],
        read_spans: list[int],
    ) -> SimpleNamespace:
        assert len(token_ids) == len(write_slots) == len(read_spans) == 1
        position = self._position
        if write_slots[0] >= 0:
            self._writes.append(write_slots[0])
        if read_spans[0] >= 0:
            self._reads.append(read_spans[0])
        self._position += 1
        model = self.context["model"]
        base = label_oracle.episodic.sustained._attention_expectations(
            model,
            label_oracle.retrieval._BASE_POLICY,
            positions=self._position,
        )
        metrics = {
            name: int(value)
            for name, value in base.items()
            if name != "attention_logical_read_fraction"
            and not name.endswith("_minimum")
            and not name.endswith("_maximum")
        }
        metrics["attention_heavy_hitter_updates"] = int(
            base["attention_heavy_hitter_updates_minimum"]
        )
        metrics["q7_scheduled_bytes"] = (
            self._position
            * self.context["q7_expectations"]["scheduled_bytes_per_position"]
        )
        layers = int(model["layers"])
        hidden = int(model["hidden_size"])
        heads = int(model["query_heads"])
        writes = len(self._writes)
        reads = len(self._reads)
        metrics.update(
            {
                "episodic_slots_written": writes * layers,
                "episodic_read_events": reads * layers,
                "episodic_active_slots": len(set(self._writes)) * layers,
                "episodic_entries_read": reads * 9 * heads * layers,
                "episodic_write_bytes": writes * layers * 2 * hidden * 2,
                "episodic_key_read_bytes": reads * 9 * layers * hidden * 2,
                "episodic_value_read_bytes": reads * 9 * layers * hidden * 2,
                "episodic_duplicate_older_entries_suppressed": 0,
                "episodic_state_bytes": int(self.resource["combined_state_bytes"]),
                "episodic_scratch_bytes": int(self.resource["combined_scratch_bytes"]),
            }
        )
        metrics["attention_state_bytes"] = int(self.resource["combined_state_bytes"])
        metrics["attention_scratch_bytes"] = int(
            self.resource["combined_scratch_bytes"]
        )
        metrics["attention_logical_read_bytes"] = (
            int(base["attention_logical_read_bytes"])
            + metrics["episodic_key_read_bytes"]
            + metrics["episodic_value_read_bytes"]
        )
        if self.tamper_counter and position >= 96:
            metrics["episodic_key_read_bytes"] += 1
        metrics["elapsed_ns"] = self._position
        metrics["q7_elapsed_ns"] = self._position
        self._last_logits = np.zeros(64, dtype=np.float32)
        next_token = 0
        if position >= 96:
            target = self.records[self._run % 8]["input_ids"][position + 1]
            self._last_logits[target] = 8.0
            if self.tamper_replay and self._run >= 8:
                self._last_logits[(target + 1) % 64] = 0.25
            next_token = int(target)
        return SimpleNamespace(next_token=next_token, metrics=metrics)

    def last_diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(self.context["model"]["hidden_size"], dtype=np.float32),
            self._last_logits.copy(),
        )

    def close(self) -> None:
        self.closed = True


def _training(records: list[dict[str, Any]]) -> dict[str, Any]:
    m0_mask = np.zeros((16, 16), dtype=np.bool_)
    m2_mask = np.zeros((16, 16), dtype=np.bool_)
    m2_mask.reshape(-1)[:51] = True
    return {
        "masks": {
            "M0": {
                "mask": m0_mask.tolist(),
                "records": [
                    {
                        "record_index": index,
                        "record_id": record["record_id"],
                        "loss": {"answer_cross_entropy": 2.0 + index / 10.0},
                    }
                    for index, record in enumerate(records)
                ],
            },
            "M2": {
                "mask": m2_mask.tolist(),
                "records": [
                    {
                        "record_index": index,
                        "record_id": record["record_id"],
                        "loss": {"answer_cross_entropy": 1.0 + index / 10.0},
                    }
                    for index, record in enumerate(records)
                ],
            },
        }
    }


def test_schedule_is_causal_and_writes_label_then_payload_postings():
    record = _record(4, "ACDB")
    schedule = label_oracle._derive_schedule(
        _PrefixTrap(record["input_ids"]),
        _ANCHORS,
    )
    rows = schedule["rows"]
    assert [rows[position]["write_slot"] for position in (5, *range(8, 16))] == (
        list(range(0, 9))
    )
    assert [rows[position]["write_slot"] for position in (25, *range(28, 36))] == list(
        range(18, 27)
    )
    assert [rows[position]["write_slot"] for position in (45, *range(48, 56))] == list(
        range(27, 36)
    )
    assert [rows[position]["write_slot"] for position in (65, *range(68, 76))] == list(
        range(9, 18)
    )
    assert rows[96]["read_span"] == 0
    assert rows[104]["read_span"] == 1
    assert rows[112]["read_span"] == 2
    assert rows[120]["read_span"] == 3
    assert rows[127]["read_span"] == 3

    mutated = list(record["input_ids"])
    mutated[97:] = [63] * 32
    changed = label_oracle._derive_schedule(mutated, _ANCHORS)
    assert changed["rows_sha256"] == schedule["rows_sha256"]
    assert changed["causal_prefix_sha256"] == schedule["causal_prefix_sha256"]

    resource = label_oracle._resource_contract(
        _model(),
        {
            "scheduled_bytes_per_position": 10,
            "scheduled_bytes_per_sequence": 1_280,
        },
    )
    assert label_oracle._schedule_counters(
        schedule,
        positions=128,
        model=_model(),
        resource=resource,
    ) == {
        "episodic_slots_written": 576,
        "episodic_read_events": 512,
        "episodic_active_slots": 576,
        "episodic_entries_read": 73_728,
        "episodic_write_bytes": 9_216,
        "episodic_key_read_bytes": 36_864,
        "episodic_value_read_bytes": 36_864,
        "episodic_state_bytes": 108_032,
        "episodic_scratch_bytes": 4_992,
    }
    assert (
        label_oracle._maximum_duplicate_suppressions(
            positions=128,
            model=_model(),
            schedule=schedule,
        )
        == 65_536
    )


def test_real_resource_contract_is_exact_and_below_boundary():
    contract = label_oracle._resource_contract(
        {
            "layers": 16,
            "query_heads": 16,
            "key_value_heads": 16,
            "head_dimension": 128,
            "hidden_size": 2_048,
        },
        {
            "scheduled_bytes_per_position": 1,
            "scheduled_bytes_per_sequence": 128,
        },
    )
    assert contract["cache_payload"]["capacity_slots"] == 36
    assert contract["cache_payload"]["span_size"] == 9
    assert contract["cache_payload"]["payload_state_bytes"] == 4_718_592
    assert contract["cache_payload"]["position_state_bytes"] == 4_608
    assert contract["episodic_write_bytes_per_sequence"] == 4_718_592
    assert contract["episodic_key_read_bytes_per_sequence"] == 18_874_368
    assert contract["episodic_value_read_bytes_per_sequence"] == 18_874_368
    assert contract["combined_attention_and_episodic_read_bytes"] == 714_866_688
    assert contract["combined_attention_and_episodic_traffic_bytes"] == 719_585_280
    assert contract["combined_state_bytes"] == 11_059_712
    assert contract["episodic_joint_softmax_scratch_bytes"] == 1_152
    assert contract["combined_scratch_bytes"] == 4_992
    assert contract["combined_read_fraction_of_dense"] == pytest.approx(
        0.33030523255813954
    )
    assert contract["combined_traffic_fraction_of_dense"] == pytest.approx(
        0.3324854651162791
    )
    assert contract["within_read_budget"] is True
    assert contract["within_total_traffic_budget"] is True


def test_loss_gate_requires_strict_summary_gains_and_no_regression():
    records = _records()
    m2_baseline = [1.0 + index / 10.0 for index in range(8)]
    m0_control = [2.0 + index / 10.0 for index in range(8)]
    passing = [
        {
            "record_index": index,
            "record_id": record["record_id"],
            "answer_cross_entropy": value - 0.1,
        }
        for index, (record, value) in enumerate(zip(records, m2_baseline, strict=True))
    ]
    result = label_oracle._loss_gate(
        records=records,
        m2_baseline=m2_baseline,
        m0_control=m0_control,
        evidence=passing,
    )
    assert result["passed"] is True
    assert all(result["global_M2_gate_checks"].values())
    assert all(result["same_policy_control_checks"].values())

    passing[0]["answer_cross_entropy"] = m2_baseline[0] + 0.01
    failed = label_oracle._loss_gate(
        records=records,
        m2_baseline=m2_baseline,
        m0_control=m0_control,
        evidence=passing,
    )
    assert failed["global_M2_gate_checks"]["no_record_regression"] is False
    assert failed["passed"] is False

    equal = [
        {
            "record_index": index,
            "record_id": record["record_id"],
            "answer_cross_entropy": value,
        }
        for index, (record, value) in enumerate(zip(records, m2_baseline, strict=True))
    ]
    failed = label_oracle._loss_gate(
        records=records,
        m2_baseline=m2_baseline,
        m0_control=m0_control,
        evidence=equal,
    )
    assert (
        failed["global_M2_gate_checks"][
            "maximum_answer_cross_entropy_strictly_improved"
        ]
        is False
    )
    assert (
        failed["global_M2_gate_checks"]["mean_answer_cross_entropy_strictly_improved"]
        is False
    )
    assert failed["global_M2_gate_checks"]["no_record_regression"] is True
    assert failed["passed"] is False


def test_mock_candidate_passes_loss_resources_counters_and_reset_replay():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    runtimes: list[_FakeRuntime] = []

    def factory(_context):
        runtime = _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
        )
        runtimes.append(runtime)
        return runtime

    result = label_oracle._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=label_oracle._checkpoint_baselines(_training(records)),
        runtime_factory=factory,
    )
    assert result["passed"] is True
    assert result["loss_gate"]["passed"] is True
    assert result["resource_passed"] is True
    assert all(result["resource_checks"].values())
    assert result["reset_replay"]["passed"] is True
    assert result["native_sequence_forwards"] == 9
    assert len(result["sequence_evidence"]) == 8
    assert len(runtimes) == 1
    assert runtimes[0].closed is True


def test_mock_counter_or_replay_tamper_fails_closed():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    baselines = label_oracle._checkpoint_baselines(_training(records))

    counter = label_oracle._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=baselines,
        runtime_factory=lambda _context: _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
            tamper_counter=True,
        ),
    )
    assert counter["loss_gate"]["passed"] is True
    assert counter["resource_checks"]["all_sequence_counter_streams"] is False
    assert counter["passed"] is False

    replay = label_oracle._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=baselines,
        runtime_factory=lambda _context: _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
            tamper_replay=True,
        ),
    )
    assert replay["loss_gate"]["passed"] is True
    assert replay["reset_replay"]["logits_sha256"] is False
    assert replay["reset_replay"]["passed"] is False
    assert replay["passed"] is False


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


def test_freeze_and_authenticate_without_confirmation_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = _records()
    context = _TrainOnlyContext(
        {
            **_context(records),
            "protocol_path": (tmp_path / "base-protocol.json").resolve(),
            "protocol_sha256": "a" * 64,
            "tokenizer_path": tmp_path / "tokenizer.json",
            "confirmation_descriptor": {
                "file": "sealed-confirmation.jsonl",
                "sha256": "d" * 64,
                "record_identity_sha256": "e" * 64,
                "records": 8,
                "tokens_per_record": 129,
                "prediction_positions_per_record": 128,
                "answer_prediction_positions_per_record": 32,
            },
        }
    )
    context["protocol_path"].write_text("{}\n", encoding="utf-8")
    training = _training(records)
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text("{}\n", encoding="utf-8")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": checkpoint_sha256,
        "mode": "resumed",
    }
    library_path = tmp_path / "libepisodic.so"
    library_path.write_bytes(b"episodic")
    library_sha256 = sha256_file(library_path)
    context["episodic_library_path"] = library_path.resolve()
    context["episodic_library_sha256"] = library_sha256
    monkeypatch.setattr(
        label_oracle.episodic,
        "_authenticate_base_inputs",
        lambda *_args: (
            context,
            training,
            {"screen_eligible": True, "selected_mask_name": "M2"},
            checkpoint,
        ),
    )
    monkeypatch.setattr(
        label_oracle,
        "_fact_anchor_ids",
        lambda _path: _ANCHORS,
    )
    monkeypatch.setattr(
        label_oracle,
        "_source_inventory",
        lambda: {"label_oracle.py": "b" * 64},
    )

    output = tmp_path / "label-protocol.json"
    frozen = label_oracle.freeze_episodic_label_oracle_protocol(
        base_protocol=context["protocol_path"],
        base_protocol_sha256=context["protocol_sha256"],
        training_checkpoint=checkpoint_path,
        training_checkpoint_sha256=checkpoint_sha256,
        episodic_library=library_path,
        episodic_library_sha256=library_sha256,
        out=output,
    )
    protocol = frozen["protocol"]
    assert protocol["experiment"] == label_oracle._PROTOCOL_EXPERIMENT
    assert protocol["train_scope"]["dense_teacher_forwards"] == 0
    assert protocol["train_scope"]["development_outcomes_used"] is False
    assert protocol["train_scope"]["confirmation_file_access_permitted"] is False
    assert protocol["runtime_abi"]["episodic_policy"] == {
        "slots": 36,
        "span_size": 9,
    }
    assert protocol["confirmation_split_opened"] is False

    authenticated_context, loaded_training, loaded_protocol = (
        label_oracle._authenticate_protocol(output, frozen["sha256"])
    )
    assert authenticated_context["train_records"] == records
    assert loaded_training is training
    assert loaded_protocol == protocol

    with pytest.raises(ValueError, match="target already exists"):
        label_oracle.freeze_episodic_label_oracle_protocol(
            base_protocol=context["protocol_path"],
            base_protocol_sha256=context["protocol_sha256"],
            training_checkpoint=checkpoint_path,
            training_checkpoint_sha256=checkpoint_sha256,
            episodic_library=library_path,
            episodic_library_sha256=library_sha256,
            out=output,
        )

    monkeypatch.setattr(
        label_oracle,
        "_source_inventory",
        lambda: {"label_oracle.py": "c" * 64},
    )
    with pytest.raises(ValueError, match="contract changed"):
        label_oracle._authenticate_protocol(output, frozen["sha256"])
