from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_head_mask_oracle as head_mask
from engram.utils import sha256_file, sha256_json


_ANCHORS = {
    "A": (2, 3, 4, 5),
    "B": (6, 7, 8, 9),
    "C": (10, 11, 12, 13),
    "D": (14, 15, 16, 17),
}
_M2_PAIRS = (
    (0, 0),
    (0, 10),
    (0, 11),
    (1, 4),
    (1, 6),
    (1, 11),
    (2, 13),
    (3, 6),
    (3, 7),
    (3, 12),
    (3, 15),
    (5, 3),
    (5, 4),
    (5, 7),
    (5, 9),
    (5, 10),
    (5, 13),
    (5, 14),
    (6, 1),
    (6, 7),
    (6, 8),
    (6, 9),
    (6, 10),
    (6, 11),
    (6, 13),
    (7, 0),
    (7, 4),
    (7, 6),
    (7, 11),
    (8, 1),
    (9, 1),
    (9, 8),
    (9, 10),
    (9, 12),
    (9, 13),
    (9, 14),
    (10, 4),
    (10, 6),
    (10, 7),
    (10, 13),
    (11, 8),
    (12, 3),
    (12, 4),
    (12, 10),
    (12, 14),
    (12, 15),
    (13, 2),
    (13, 8),
    (13, 11),
    (14, 8),
    (14, 13),
)


def _m2_mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=np.bool_)
    for layer, head in _M2_PAIRS:
        mask[layer, head] = True
    return mask


def _projected_scores(mask: np.ndarray) -> np.ndarray:
    scores = np.full((16, 16), -1.0, dtype=np.float64)
    for rank, (layer, head) in enumerate(_M2_PAIRS):
        scores[layer, head] = 1000.0 - rank
    needed = head_mask._EXPECTED_POSITIVE_PROJECTED_SCORES - int(mask.sum())
    for index in np.flatnonzero(~mask.reshape(-1))[:needed]:
        scores.reshape(-1)[index] = 100.0 - float(index) / 1000.0
    return scores


def _record(index: int, order: str) -> dict[str, Any]:
    input_ids = [1] * 129
    codes = [
        [20 + ((index * 7 + label_index * 8 + offset) % 40) for offset in range(8)]
        for label_index in range(4)
    ]
    for depth, label in enumerate(order):
        label_index = head_mask.retrieval._LABELS.index(label)
        anchor = head_mask.retrieval._FACT_ANCHORS[depth]
        input_ids[anchor : anchor + 4] = _ANCHORS[label]
        source = head_mask.retrieval._PASSKEY_SOURCE_STARTS[depth]
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
        for index, order in enumerate(head_mask.retrieval._FACT_ORDERS)
    ]


def _model() -> dict[str, int]:
    return {
        "layers": 16,
        "query_heads": 16,
        "key_value_heads": 16,
        "head_dimension": 1,
        "hidden_size": 16,
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


def _training(records: list[dict[str, Any]]) -> dict[str, Any]:
    m0 = np.zeros((16, 16), dtype=np.bool_)
    m2 = _m2_mask()
    return {
        "masks": {
            "M0": {
                "mask": m0.tolist(),
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
                "mask": m2.tolist(),
                "projected_scores": _projected_scores(m2).tolist(),
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


def _protocol(
    records: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, Any]:
    schedules = [
        head_mask._derive_schedule(record["input_ids"], _ANCHORS) for record in records
    ]
    mask = _m2_mask()
    return {
        "fixed_M2_head_mask": head_mask._mask_descriptor(mask),
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in _ANCHORS.items()
        },
        "schedule_contract": {
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ]
        },
        "resource_contract": head_mask._resource_contract(
            context["model"],
            context["q7_expectations"],
            mask,
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
    episodic_policy = {"slots": 32, "span_size": 8}

    def __init__(
        self,
        *,
        context: dict[str, Any],
        records: list[dict[str, Any]],
        resource: dict[str, Any],
        mask: np.ndarray,
        tamper_counter: bool = False,
        tamper_replay: bool = False,
    ) -> None:
        self.context = context
        self.records = records
        self.resource = resource
        self.episodic_head_mask = tuple(
            tuple(bool(value) for value in row) for row in mask
        )
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
        base = head_mask.episodic.sustained._attention_expectations(
            model,
            head_mask.retrieval._BASE_POLICY,
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
        active_layers = len(head_mask._EXPECTED_ACTIVE_LAYERS)
        selected_pairs = head_mask._EXPECTED_SELECTED_PAIRS
        writes = len(self._writes)
        reads = len(self._reads)
        head_dimension = int(model["head_dimension"])
        key_value_width = int(model["key_value_heads"]) * head_dimension
        metrics.update(
            {
                "episodic_slots_written": writes * active_layers,
                "episodic_read_events": reads * active_layers,
                "episodic_active_slots": len(set(self._writes)) * active_layers,
                "episodic_entries_read": (reads * 8 * selected_pairs),
                "episodic_write_bytes": (
                    writes * active_layers * 2 * key_value_width * 2
                ),
                "episodic_key_read_bytes": (
                    reads * 8 * selected_pairs * head_dimension * 2
                ),
                "episodic_value_read_bytes": (
                    reads * 8 * selected_pairs * head_dimension * 2
                ),
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
            metrics["episodic_entries_read"] += 1
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


def test_fixed_m2_mask_and_projected_order_are_exact():
    mask = _m2_mask()
    validated = head_mask._validate_fixed_m2_mask(mask)
    assert np.array_equal(validated, mask)
    assert head_mask._runtime_mask_matches(SimpleNamespace(), mask) is False
    assert (
        head_mask._runtime_mask_matches(
            SimpleNamespace(
                episodic_head_mask=tuple(
                    tuple(bool(value) for value in row) for row in mask
                )
            ),
            mask,
        )
        is True
    )
    assert sha256_json(mask.tolist()) == head_mask._EXPECTED_M2_MASK_SHA256
    assert [int(row.sum()) for row in mask] == list(
        head_mask._EXPECTED_LAYER_HEAD_COUNTS
    )
    ordering = head_mask._projected_score_ordering(
        _projected_scores(mask),
        mask,
    )
    assert ordering["positive_score_count"] == 165
    assert ordering["future_rank_sweep_boundaries_not_executed"] == [
        32,
        64,
        96,
        128,
        165,
    ]
    assert all(row["selected_in_fixed_M2"] for row in ordering["ordering"][:51])

    tampered = mask.copy()
    tampered[0, 0] = False
    tampered[0, 1] = True
    with pytest.raises(ValueError, match="contract changed"):
        head_mask._validate_fixed_m2_mask(tampered)


def test_original_payload_schedule_is_causal_and_resource_contract_is_exact():
    record = _record(4, "ACDB")
    schedule = head_mask._derive_schedule(
        _PrefixTrap(record["input_ids"]),
        _ANCHORS,
    )
    writes = [
        (row["position"], row["write_slot"])
        for row in schedule["rows"]
        if row["write_slot"] >= 0
    ]
    assert [position for position, _slot in writes] == [
        start + offset for start in (8, 28, 48, 68) for offset in range(8)
    ]
    assert sorted(slot for _position, slot in writes) == list(range(32))
    assert schedule["rows"][96]["read_span"] == 0
    assert schedule["rows"][104]["read_span"] == 1
    assert schedule["rows"][112]["read_span"] == 2
    assert schedule["rows"][120]["read_span"] == 3

    mutated = list(record["input_ids"])
    mutated[97:] = [63] * 32
    changed = head_mask._derive_schedule(mutated, _ANCHORS)
    assert changed["rows_sha256"] == schedule["rows_sha256"]
    assert changed["causal_prefix_sha256"] == schedule["causal_prefix_sha256"]

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
    resource = head_mask._resource_contract(model, q7, _m2_mask())
    assert resource["cache_payload"]["active_layers"] == 14
    assert resource["cache_payload"]["payload_state_bytes"] == 3_670_016
    assert resource["cache_payload"]["position_state_bytes"] == 3_584
    assert resource["episodic_write_bytes_per_sequence"] == 3_670_016
    assert resource["episodic_key_read_bytes_per_sequence"] == 3_342_336
    assert resource["episodic_value_read_bytes_per_sequence"] == 3_342_336
    assert resource["combined_attention_and_episodic_read_bytes"] == 683_802_624
    assert resource["combined_attention_and_episodic_traffic_bytes"] == 687_472_640
    assert resource["combined_state_bytes"] == 10_010_112
    assert resource["combined_scratch_bytes"] == 4_736
    assert resource["within_read_budget"] is True
    assert resource["within_total_traffic_budget"] is True
    assert head_mask._schedule_counters(
        schedule,
        positions=128,
        model=model,
        mask=_m2_mask(),
        resource=resource,
    ) == {
        "episodic_slots_written": 448,
        "episodic_read_events": 448,
        "episodic_active_slots": 448,
        "episodic_entries_read": 13_056,
        "episodic_write_bytes": 3_670_016,
        "episodic_key_read_bytes": 3_342_336,
        "episodic_value_read_bytes": 3_342_336,
        "episodic_state_bytes": 10_010_112,
        "episodic_scratch_bytes": 4_736,
    }


def test_loss_gate_is_strict_against_m2_and_m0_is_attribution_only():
    records = _records()
    state = head_mask._checkpoint_references(_training(records))
    evidence = [
        {
            "record_index": index,
            "record_id": record["record_id"],
            "answer_cross_entropy": 0.9 + index / 10.0,
        }
        for index, record in enumerate(records)
    ]
    result = head_mask._loss_gate(
        records=records,
        baselines=state["baselines"],
        evidence=evidence,
    )
    assert result["passed"] is True
    assert all(result["gate_checks"].values())
    assert "same_policy_M0_attribution" in result["summaries"]

    evidence[0]["answer_cross_entropy"] = 1.01
    failed = head_mask._loss_gate(
        records=records,
        baselines=state["baselines"],
        evidence=evidence,
    )
    assert failed["gate_checks"]["no_record_regression"] is False
    assert failed["passed"] is False

    equal = [
        {
            "record_index": index,
            "record_id": record["record_id"],
            "answer_cross_entropy": 1.0 + index / 10.0,
        }
        for index, record in enumerate(records)
    ]
    failed = head_mask._loss_gate(
        records=records,
        baselines=state["baselines"],
        evidence=equal,
    )
    assert (
        failed["gate_checks"]["maximum_answer_cross_entropy_strictly_improved"] is False
    )
    assert failed["gate_checks"]["mean_answer_cross_entropy_strictly_improved"] is False
    assert failed["gate_checks"]["no_record_regression"] is True
    assert failed["passed"] is False


def test_mock_candidate_passes_loss_resources_counters_and_reset_replay():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    state = head_mask._checkpoint_references(_training(records))
    runtimes: list[_FakeRuntime] = []

    def factory(_context, mask):
        runtime = _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
            mask=mask,
        )
        runtimes.append(runtime)
        return runtime

    result = head_mask._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=state["baselines"],
        runtime_factory=factory,
    )
    assert result["passed"] is True
    assert result["loss_gate"]["passed"] is True
    assert result["resource_passed"] is True
    assert all(result["resource_checks"].values())
    assert result["reset_replay"]["passed"] is True
    assert result["fixed_M2_head_mask"]["mask_sha256"] == (
        head_mask._EXPECTED_M2_MASK_SHA256
    )
    assert result["native_sequence_forwards"] == 9
    assert len(result["sequence_evidence"]) == 8
    assert len(runtimes) == 1
    assert runtimes[0].closed is True


def test_mock_counter_or_replay_tamper_fails_closed():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    state = head_mask._checkpoint_references(_training(records))

    counter = head_mask._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=state["baselines"],
        runtime_factory=lambda _context, mask: _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
            mask=mask,
            tamper_counter=True,
        ),
    )
    assert counter["loss_gate"]["passed"] is True
    assert counter["resource_checks"]["all_sequence_counter_streams"] is False
    assert counter["passed"] is False

    replay = head_mask._evaluate_candidate(
        context=context,
        records=records,
        protocol=protocol,
        baselines=state["baselines"],
        runtime_factory=lambda _context, mask: _FakeRuntime(
            context=context,
            records=records,
            resource=protocol["resource_contract"],
            mask=mask,
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
    library_path = tmp_path / "libepisodic-headwise.so"
    library_path.write_bytes(b"episodic-headwise")
    library_sha256 = sha256_file(library_path)
    context["episodic_library_path"] = library_path.resolve()
    context["episodic_library_sha256"] = library_sha256
    monkeypatch.setattr(
        head_mask.episodic,
        "_authenticate_base_inputs",
        lambda *_args: (
            context,
            training,
            {"screen_eligible": True, "selected_mask_name": "M2"},
            checkpoint,
        ),
    )
    monkeypatch.setattr(
        head_mask,
        "_fact_anchor_ids",
        lambda _path: _ANCHORS,
    )
    monkeypatch.setattr(
        head_mask,
        "_source_inventory",
        lambda: {"head-mask-oracle.py": "b" * 64},
    )

    output = tmp_path / "head-mask-protocol.json"
    frozen = head_mask.freeze_episodic_head_mask_oracle_protocol(
        base_protocol=context["protocol_path"],
        base_protocol_sha256=context["protocol_sha256"],
        training_checkpoint=checkpoint_path,
        training_checkpoint_sha256=checkpoint_sha256,
        headwise_episodic_library=library_path,
        headwise_episodic_library_sha256=library_sha256,
        out=output,
    )
    protocol = frozen["protocol"]
    assert protocol["experiment"] == head_mask._PROTOCOL_EXPERIMENT
    assert protocol["fixed_M2_head_mask"]["mask_sha256"] == (
        head_mask._EXPECTED_M2_MASK_SHA256
    )
    assert protocol["train_scope"]["dense_teacher_forwards"] == 0
    assert protocol["train_scope"]["mask_fitting_or_selection"] is False
    assert protocol["train_scope"]["development_outcomes_used"] is False
    assert protocol["train_scope"]["confirmation_file_access_permitted"] is False
    assert protocol["runtime_abi"]["required_open_symbol"] == (
        "engram_olmoe_token_open_episodic_headwise_v1"
    )
    assert protocol["all_head_payload_attribution"]["consumed"] is False
    assert protocol["reused_checkpoint_evidence"]["M0"]["role"] == (
        "same_policy_W16_C8_K4_S2_no_episodic_cache_attribution"
    )
    assert protocol["confirmation_split_opened"] is False

    authenticated_context, loaded_training, loaded_protocol = (
        head_mask._authenticate_protocol(output, frozen["sha256"])
    )
    assert authenticated_context["train_records"] == records
    assert loaded_training is training
    assert loaded_protocol == protocol

    with pytest.raises(ValueError, match="target already exists"):
        head_mask.freeze_episodic_head_mask_oracle_protocol(
            base_protocol=context["protocol_path"],
            base_protocol_sha256=context["protocol_sha256"],
            training_checkpoint=checkpoint_path,
            training_checkpoint_sha256=checkpoint_sha256,
            headwise_episodic_library=library_path,
            headwise_episodic_library_sha256=library_sha256,
            out=output,
        )

    monkeypatch.setattr(
        head_mask,
        "_source_inventory",
        lambda: {"head-mask-oracle.py": "c" * 64},
    )
    with pytest.raises(ValueError, match="contract changed"):
        head_mask._authenticate_protocol(output, frozen["sha256"])
