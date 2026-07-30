from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_oracle as episodic
from engram.utils import sha256_file, sha256_json


_ANCHORS = {
    "A": (2, 3, 4, 5),
    "B": (6, 7, 8, 9),
    "C": (10, 11, 12, 13),
    "D": (14, 15, 16, 17),
}


def _record(index: int, order: str) -> dict[str, Any]:
    input_ids = [1] * 129
    codes: list[list[int]] = []
    for label_index in range(4):
        codes.append(
            [
                20 + ((index * 7 + label_index * 8 + offset) % 40)
                for offset in range(8)
            ]
        )
    depth_by_label = [0] * 4
    for depth, label in enumerate(order):
        label_index = episodic.retrieval._LABELS.index(label)
        depth_by_label[label_index] = depth
        anchor = episodic.retrieval._FACT_ANCHORS[depth]
        input_ids[anchor : anchor + 4] = _ANCHORS[label]
        source = episodic.retrieval._PASSKEY_SOURCE_STARTS[depth]
        input_ids[source : source + 8] = codes[label_index]
    input_ids[97:] = [value for code in codes for value in code]
    provisional = {
        "record_index": index,
        "record_id": f"train-{index}",
        "identity_sha256": "",
        "input_ids": input_ids,
        "answer_prediction_positions": list(range(96, 128)),
        "answer_source_depths": [
            episodic.retrieval._SOURCE_DEPTH_NAMES[
                depth_by_label[offset // 8]
            ]
            for offset in range(32)
        ],
    }
    provisional["identity_sha256"] = sha256_json(
        {key: value for key, value in provisional.items() if key != "identity_sha256"}
    )
    return provisional


def _records() -> list[dict[str, Any]]:
    return [
        _record(index, order)
        for index, order in enumerate(episodic.retrieval._FACT_ORDERS)
    ]


def _teacher(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        targets = np.asarray(record["input_ids"][97:], dtype=np.int64)
        logits = np.zeros((32, 64), dtype=np.float32)
        logits[np.arange(32), targets] = 6.0
        hidden = np.stack(
            [
                np.asarray(
                    [
                        float(record["record_index"] + 1),
                        float(offset + 1) / 32.0,
                        0.25,
                        -0.5,
                    ],
                    dtype=np.float32,
                )
                for offset in range(32)
            ]
        )
        result.append(
            {
                "record_index": record["record_index"],
                "logits": logits,
                "hidden": hidden,
                "targets": targets,
            }
        )
    return result


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
        episodic._derive_schedule(record["input_ids"], _ANCHORS)
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
        "resource_contract": episodic._resource_contract(
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
                raise AssertionError("schedule derivation inspected a future token")
            return self.values[key]
        if key > 96:
            raise AssertionError("schedule derivation inspected a future token")
        return self.values[key]


class _FakeEpisodicRuntime:
    attention_metrics_available = True
    episodic_policy = {"slots": 32, "span_size": 8}

    def __init__(
        self,
        *,
        context: dict[str, Any],
        teacher: list[dict[str, Any]],
        resource: dict[str, Any],
        tamper_counter: bool = False,
        tamper_replay: bool = False,
    ) -> None:
        self.context = context
        self.teacher = teacher
        self.resource = resource
        self.tamper_counter = tamper_counter
        self.tamper_replay = tamper_replay
        self._position = 0
        self._run = 0
        self._writes: list[int] = []
        self._read_spans: list[int] = []
        self._next_token = 0
        self.closed = False
        self.calls: list[dict[str, Any]] = []

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0
        self._run += 1
        self._writes = []
        self._read_spans = []

    def _episodic_metrics(self) -> dict[str, int]:
        layers = int(self.context["model"]["layers"])
        hidden = int(self.context["model"]["hidden_size"])
        heads = int(self.context["model"]["query_heads"])
        writes = len(self._writes)
        reads = len(self._read_spans)
        return {
            "episodic_slots_written": writes * layers,
            "episodic_read_events": reads * layers,
            "episodic_active_slots": len(set(self._writes)) * layers,
            "episodic_entries_read": reads * 8 * heads * layers,
            "episodic_write_bytes": writes * layers * 2 * hidden * 2,
            "episodic_key_read_bytes": reads * 8 * layers * hidden * 2,
            "episodic_value_read_bytes": reads * 8 * layers * hidden * 2,
            "episodic_duplicate_older_entries_suppressed": 0,
            "episodic_state_bytes": int(self.resource["combined_state_bytes"]),
            "episodic_scratch_bytes": int(
                self.resource["combined_scratch_bytes"]
            ),
        }

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
            self._read_spans.append(read_spans[0])
        self.calls.append(
            {
                "run": self._run,
                "position": position,
                "write_slots": list(write_slots),
                "read_spans": list(read_spans),
            }
        )
        self._position += 1
        base = episodic.sustained._attention_expectations(
            self.context["model"],
            episodic.retrieval._BASE_POLICY,
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
        metrics.update(self._episodic_metrics())
        metrics["attention_state_bytes"] = int(
            self.resource["combined_state_bytes"]
        )
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
        if position >= 96:
            logits = self.teacher[self._run % 8]["logits"][position - 96]
            self._next_token = int(np.argmax(logits))
        else:
            self._next_token = 0
        return SimpleNamespace(next_token=self._next_token, metrics=metrics)

    def last_diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
        offset = self._position - 1 - 96
        row = self.teacher[self._run % 8]
        hidden = row["hidden"][offset].copy()
        logits = row["logits"][offset].copy()
        if self.tamper_replay and self._run >= 8:
            logits[0] += 0.25
        return hidden.astype(np.float32), logits.astype(np.float32)

    def close(self) -> None:
        self.closed = True


def _checkpoint_training(records: list[dict[str, Any]]) -> dict[str, Any]:
    mask = np.zeros((16, 16), dtype=np.bool_)
    mask.reshape(-1)[:51] = True
    return {
        "masks": {
            "M2": {
                "mask": mask.tolist(),
                "records": [
                    {
                        "record_index": index,
                        "record_id": record["record_id"],
                        "loss": {"answer_cross_entropy": 1.0 + index / 10.0},
                    }
                    for index, record in enumerate(records)
                ],
            }
        }
    }


def test_schedule_uses_only_causal_prefix_and_has_exact_span_operations():
    record = _record(4, "ACDB")
    schedule = episodic._derive_schedule(
        _PrefixTrap(record["input_ids"]),
        _ANCHORS,
    )
    rows = schedule["rows"]
    writes = [
        (row["position"], row["write_slot"])
        for row in rows
        if row["write_slot"] >= 0
    ]
    assert len(writes) == 32
    assert sorted(slot for _position, slot in writes) == list(range(32))
    assert [position for position, _slot in writes] == [
        start + offset
        for start in (8, 28, 48, 68)
        for offset in range(8)
    ]
    assert rows[96]["read_span"] == 0
    assert rows[104]["read_span"] == 1
    assert rows[112]["read_span"] == 2
    assert rows[120]["read_span"] == 3
    assert rows[127]["read_span"] == 3

    mutated = list(record["input_ids"])
    mutated[97:] = [63] * 32
    changed = episodic._derive_schedule(mutated, _ANCHORS)
    assert changed["rows_sha256"] == schedule["rows_sha256"]
    assert changed["causal_prefix_sha256"] == schedule["causal_prefix_sha256"]

    counters = episodic._schedule_counters(
        schedule,
        positions=128,
        model=_model(),
        resource=episodic._resource_contract(
            _model(),
            {
                "scheduled_bytes_per_position": 10,
                "scheduled_bytes_per_sequence": 1_280,
            },
        ),
    )
    assert counters == {
        "episodic_slots_written": 512,
        "episodic_read_events": 512,
        "episodic_active_slots": 512,
        "episodic_entries_read": 65_536,
        "episodic_write_bytes": 8_192,
        "episodic_key_read_bytes": 32_768,
        "episodic_value_read_bytes": 32_768,
        "episodic_state_bytes": 106_496,
        "episodic_scratch_bytes": 4_864,
    }


def test_real_model_resource_contract_is_exact_and_below_boundary():
    model = {
        "layers": 16,
        "query_heads": 16,
        "key_value_heads": 16,
        "head_dimension": 128,
        "hidden_size": 2_048,
    }
    q7 = {
        "scheduled_bytes_per_position": 1,
        "scheduled_bytes_per_sequence": 128,
    }
    contract = episodic._resource_contract(model, q7)
    assert contract["cache_payload"]["dtype"] == "bfloat16"
    assert contract["cache_payload"]["payload_state_bytes"] == 4_194_304
    assert contract["cache_payload"]["position_state_bytes"] == 4_096
    assert contract["episodic_write_bytes_per_sequence"] == 4_194_304
    assert contract["episodic_key_read_bytes_per_sequence"] == 16_777_216
    assert contract["episodic_value_read_bytes_per_sequence"] == 16_777_216
    assert contract["combined_attention_and_episodic_read_bytes"] == 710_672_384
    assert (
        contract["combined_attention_and_episodic_traffic_bytes"]
        == 714_866_688
    )
    assert contract["combined_state_bytes"] == 10_534_912
    assert contract["episodic_joint_softmax_scratch_bytes"] == 1_024
    assert contract["combined_scratch_bytes"] == 4_864
    assert contract["combined_read_fraction_of_dense"] == pytest.approx(
        0.3283672480620155
    )
    assert contract["combined_traffic_fraction_of_dense"] == pytest.approx(
        0.33030523255813954
    )
    assert contract["within_read_budget"] is True
    assert contract["within_total_traffic_budget"] is True


def test_mock_native_candidate_passes_semantics_counters_budget_and_replay():
    records = _records()
    teacher = _teacher(records)
    context = _context(records)
    protocol = _protocol(records, context)
    runtimes: list[_FakeEpisodicRuntime] = []

    def factory(_context):
        runtime = _FakeEpisodicRuntime(
            context=context,
            teacher=teacher,
            resource=protocol["resource_contract"],
        )
        runtimes.append(runtime)
        return runtime

    result = episodic._evaluate_candidate(
        context=context,
        records=records,
        teacher=teacher,
        protocol=protocol,
        runtime_factory=factory,
    )
    assert result["passed"] is True
    assert result["quality_passed"] is True
    assert result["resource_passed"] is True
    assert result["overall_answer_positions"]["prediction_positions"] == 256
    assert all(
        metrics["prediction_positions"] == 64
        for metrics in result["source_depths"].values()
    )
    assert result["reset_replay"]["passed"] is True
    assert all(result["resource_checks"].values())
    assert len(result["sequence_evidence"]) == 8
    assert len(runtimes) == 1
    assert runtimes[0].closed is True
    assert len(runtimes[0].calls) == 9 * 128


def test_mock_native_counter_or_replay_tamper_fails_closed():
    records = _records()
    teacher = _teacher(records)
    context = _context(records)
    protocol = _protocol(records, context)

    counter = episodic._evaluate_candidate(
        context=context,
        records=records,
        teacher=teacher,
        protocol=protocol,
        runtime_factory=lambda _context: _FakeEpisodicRuntime(
            context=context,
            teacher=teacher,
            resource=protocol["resource_contract"],
            tamper_counter=True,
        ),
    )
    assert counter["quality_passed"] is True
    assert counter["resource_checks"]["all_sequence_counter_streams"] is False
    assert counter["resource_passed"] is False
    assert counter["passed"] is False

    replay = episodic._evaluate_candidate(
        context=context,
        records=records,
        teacher=teacher,
        protocol=protocol,
        runtime_factory=lambda _context: _FakeEpisodicRuntime(
            context=context,
            teacher=teacher,
            resource=protocol["resource_contract"],
            tamper_replay=True,
        ),
    )
    assert replay["quality_passed"] is True
    assert replay["reset_replay"]["logits_sha256"] is False
    assert replay["reset_replay"]["passed"] is False
    assert replay["resource_passed"] is False
    assert replay["passed"] is False


@pytest.mark.parametrize(
    ("field", "failed"),
    [
        ("teacher_to_native_kl", 0.0500001),
        ("teacher_top1_agreement", 0.8999999),
        ("target_nll_delta", 0.0500001),
        ("final_hidden_relative_l2", 0.1000001),
    ],
)
def test_semantic_thresholds_are_exact(field: str, failed: float):
    boundary = {
        "teacher_to_native_kl": 0.05,
        "teacher_top1_agreement": 0.90,
        "target_nll_delta": 0.05,
        "final_hidden_relative_l2": 0.10,
    }
    assert all(episodic._quality_checks(boundary).values())
    boundary[field] = failed
    assert not all(episodic._quality_checks(boundary).values())


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


def test_freeze_and_authenticate_distinct_protocol_without_confirmation_access(
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
    training = _checkpoint_training(records)
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
        episodic,
        "_authenticate_base_inputs",
        lambda *_args: (
            context,
            training,
            {"screen_eligible": True, "selected_mask_name": "M2"},
            checkpoint,
        ),
    )
    monkeypatch.setattr(
        episodic,
        "_fact_anchor_ids",
        lambda _path: _ANCHORS,
    )
    monkeypatch.setattr(
        episodic,
        "_source_inventory",
        lambda: {"episodic.py": "b" * 64},
    )

    output = tmp_path / "episodic-protocol.json"
    frozen = episodic.freeze_episodic_oracle_protocol(
        base_protocol=context["protocol_path"],
        base_protocol_sha256=context["protocol_sha256"],
        training_checkpoint=checkpoint_path,
        training_checkpoint_sha256=checkpoint_sha256,
        episodic_library=library_path,
        episodic_library_sha256=library_sha256,
        out=output,
    )
    assert output.is_file()
    assert frozen["protocol"]["experiment"] == episodic._PROTOCOL_EXPERIMENT
    assert frozen["protocol"]["train_scope"]["development_outcomes_used"] is False
    assert (
        frozen["protocol"]["train_scope"]["confirmation_file_access_permitted"]
        is False
    )
    assert frozen["protocol"]["confirmation_split_opened"] is False

    authenticated_context, loaded_training, loaded_protocol = (
        episodic._authenticate_protocol(output, frozen["sha256"])
    )
    assert authenticated_context["train_records"] == records
    assert loaded_training is training
    assert loaded_protocol == frozen["protocol"]

    with pytest.raises(ValueError, match="target already exists"):
        episodic.freeze_episodic_oracle_protocol(
            base_protocol=context["protocol_path"],
            base_protocol_sha256=context["protocol_sha256"],
            training_checkpoint=checkpoint_path,
            training_checkpoint_sha256=checkpoint_sha256,
            episodic_library=library_path,
            episodic_library_sha256=library_sha256,
            out=output,
        )

    monkeypatch.setattr(
        episodic,
        "_source_inventory",
        lambda: {"episodic.py": "c" * 64},
    )
    with pytest.raises(ValueError, match="contract changed"):
        episodic._authenticate_protocol(output, frozen["sha256"])


def test_runtime_without_prospective_episodic_abi_is_rejected():
    records = _records()
    context = _context(records)
    schedule = episodic._derive_schedule(records[0]["input_ids"], _ANCHORS)
    runtime = SimpleNamespace(
        position=0,
        attention_metrics_available=True,
        episodic_policy=None,
        reset=lambda: None,
    )
    with pytest.raises(ValueError, match="capability is unavailable"):
        episodic._execute_episodic_record(
            runtime,
            record=records[0],
            context=context,
            schedule=schedule,
            resource=episodic._resource_contract(
                context["model"],
                context["q7_expectations"],
            ),
        )
