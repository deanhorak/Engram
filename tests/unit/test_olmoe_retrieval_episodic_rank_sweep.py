from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_rank_sweep as sweep
from engram.utils import sha256_file, sha256_json


_ANCHORS = {
    "A": (2, 3, 4, 5),
    "B": (6, 7, 8, 9),
    "C": (10, 11, 12, 13),
    "D": (14, 15, 16, 17),
}
_ORDER = (
    116,
    97,
    112,
    60,
    152,
    63,
    0,
    206,
    157,
    184,
    27,
    216,
    22,
    83,
    173,
    11,
    90,
    202,
    10,
    89,
    55,
    84,
    145,
    45,
    87,
    156,
    195,
    232,
    104,
    54,
    106,
    109,
    93,
    129,
    158,
    123,
    207,
    164,
    105,
    107,
    219,
    237,
    166,
    210,
    196,
    154,
    103,
    94,
    167,
    20,
    118,
    125,
    47,
    139,
    34,
    98,
    62,
    136,
    42,
    110,
    150,
    17,
    8,
    7,
    176,
    143,
    101,
    212,
    108,
    99,
    131,
    29,
    69,
    162,
    77,
    75,
    111,
    18,
    80,
    124,
    171,
    209,
    24,
    128,
    35,
    144,
    175,
    52,
    178,
    5,
    198,
    71,
    25,
    67,
    32,
    88,
    155,
    113,
    194,
    78,
    188,
    235,
    213,
    181,
    161,
    133,
    114,
    160,
    208,
    197,
    38,
    211,
    59,
    201,
    193,
    61,
    233,
    250,
    225,
    66,
    249,
    95,
    253,
    53,
    242,
    203,
    163,
    153,
    58,
    170,
    204,
    224,
    91,
    240,
    252,
    214,
    251,
    192,
    137,
    245,
    28,
    226,
    134,
    15,
    247,
    14,
    141,
    239,
    148,
    13,
    227,
    1,
    241,
    117,
    231,
    229,
    96,
    220,
    238,
    243,
    190,
    119,
    26,
    236,
    180,
    33,
    102,
    40,
    36,
    234,
    248,
    254,
    222,
    255,
    230,
    221,
    138,
    3,
    41,
    246,
    169,
    126,
    244,
    140,
    215,
    217,
    183,
    12,
    74,
    179,
    21,
    218,
    4,
    228,
    2,
    191,
    57,
    223,
    205,
    186,
    146,
    168,
    56,
    172,
    76,
    85,
    199,
    115,
    187,
    200,
    121,
    142,
    19,
    46,
    147,
    177,
    135,
    23,
    68,
    151,
    82,
    72,
    122,
    73,
    132,
    174,
    9,
    159,
    31,
    43,
    37,
    65,
    86,
    182,
    16,
    100,
    92,
    130,
    70,
    39,
    64,
    185,
    189,
    165,
    44,
    149,
    50,
    6,
    81,
    49,
    79,
    48,
    51,
    120,
    30,
    127,
)


def _scores() -> np.ndarray:
    values = np.empty(256, dtype=np.float64)
    for rank, index in enumerate(_ORDER, start=1):
        values[index] = float(166 - rank) if rank <= 165 else float(165 - rank)
    return values.reshape(16, 16)


def _m2_mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=np.bool_)
    for index in _ORDER[:51]:
        mask[index // 16, index % 16] = True
    return mask


def _ordering_payload() -> dict[str, Any]:
    return sweep.fixed._projected_score_ordering(_scores(), _m2_mask())


def _record(index: int, order: str) -> dict[str, Any]:
    input_ids = [1] * 129
    codes = [
        [20 + ((index * 7 + label_index * 8 + offset) % 40) for offset in range(8)]
        for label_index in range(4)
    ]
    for depth, label in enumerate(order):
        label_index = sweep.retrieval._LABELS.index(label)
        anchor = sweep.retrieval._FACT_ANCHORS[depth]
        input_ids[anchor : anchor + 4] = _ANCHORS[label]
        source = sweep.retrieval._PASSKEY_SOURCE_STARTS[depth]
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
        for index, order in enumerate(sweep.retrieval._FACT_ORDERS)
    ]


def _model() -> dict[str, int]:
    return {
        "layers": 16,
        "query_heads": 16,
        "key_value_heads": 16,
        "head_dimension": 128,
        "hidden_size": 2_048,
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
                "projected_scores": _scores().tolist(),
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
    ordering = sweep._validate_frozen_ordering(_ordering_payload())
    schedules = [
        sweep.fixed._derive_schedule(record["input_ids"], _ANCHORS)
        for record in records
    ]
    candidates = []
    for k in sweep._CANDIDATE_K:
        mask = sweep._rank_prefix_mask(ordering, k)
        candidates.append(
            {
                "K": k,
                "head_mask": sweep._mask_descriptor(mask, k),
                "resource_contract": sweep._resource_contract(
                    context["model"],
                    context["q7_expectations"],
                    mask,
                    k,
                ),
            }
        )
    return {
        "candidate_order": list(sweep._CANDIDATE_K),
        "candidates": candidates,
        "tokenizer_fact_anchor_ids": {
            label: list(values) for label, values in _ANCHORS.items()
        },
        "schedule_contract": {
            "per_record_rows_sha256": [
                schedule["rows_sha256"] for schedule in schedules
            ]
        },
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
        margin: float,
        tamper_counter: bool = False,
        tamper_replay: bool = False,
    ) -> None:
        self.context = context
        self.records = records
        self.resource = resource
        self.k = int(mask.sum())
        self.margin = margin
        self.tamper_counter = tamper_counter
        self.tamper_replay = tamper_replay
        self.episodic_head_mask = tuple(
            tuple(bool(value) for value in row) for row in mask
        )
        self._position = 0
        self._run = 0
        self._writes: list[int] = []
        self._reads: list[int] = []
        self._last_logits = np.zeros(64, dtype=np.float32)
        self.forward_calls = 0
        self.reset_calls = 0
        self.closed = False

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0
        self._run += 1
        self._writes = []
        self._reads = []
        self.reset_calls += 1

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
        self.forward_calls += 1
        model = self.context["model"]
        base = sweep.episodic.sustained._attention_expectations(
            model,
            sweep.retrieval._BASE_POLICY,
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
        active_layers = int(
            np.count_nonzero(np.any(np.asarray(self.episodic_head_mask), axis=1))
        )
        writes = len(self._writes)
        reads = len(self._reads)
        head_dimension = int(model["head_dimension"])
        key_value_width = int(model["key_value_heads"]) * head_dimension
        metrics.update(
            {
                "episodic_slots_written": writes * active_layers,
                "episodic_read_events": reads * active_layers,
                "episodic_active_slots": len(set(self._writes)) * active_layers,
                "episodic_entries_read": reads * 8 * self.k,
                "episodic_write_bytes": (
                    writes * active_layers * 2 * key_value_width * 2
                ),
                "episodic_key_read_bytes": (reads * 8 * self.k * head_dimension * 2),
                "episodic_value_read_bytes": (reads * 8 * self.k * head_dimension * 2),
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
            self._last_logits[target] = self.margin
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


def test_rank_masks_schedule_and_all_resource_contracts_are_exact():
    ordering = sweep._validate_frozen_ordering(_ordering_payload())
    record = _record(4, "ACDB")
    schedule = sweep.fixed._derive_schedule(
        _PrefixTrap(record["input_ids"]),
        _ANCHORS,
    )
    assert schedule["last_input_index_observed_during_derivation"] == 96
    assert schedule["future_answer_tokens_observed_during_derivation"] is False
    assert sum(row["write_slot"] >= 0 for row in schedule["rows"]) == 32
    assert sum(row["read_span"] >= 0 for row in schedule["rows"]) == 32

    context = _context(_records())
    for k in sweep._CANDIDATE_K:
        mask = sweep._rank_prefix_mask(ordering, k)
        assert int(mask.sum()) == k
        assert sha256_json(mask.tolist()) == sweep._EXPECTED_MASK_SHA256[k]
        resource = sweep._resource_contract(
            context["model"],
            context["q7_expectations"],
            mask,
            k,
        )
        assert {
            name: resource[name] for name in sweep._EXPECTED_RESOURCES[k]
        } == sweep._EXPECTED_RESOURCES[k]
        counters = sweep._schedule_counters(
            schedule,
            positions=128,
            model=context["model"],
            mask=mask,
            k=k,
            resource=resource,
        )
        assert counters["episodic_entries_read"] == 32 * 8 * k
        assert (
            counters["episodic_state_bytes"]
            == (sweep._EXPECTED_RESOURCES[k]["combined_state_bytes"])
        )


def test_ordered_sweep_stops_at_smallest_passing_and_replays_retained_runtime():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    baselines = sweep.fixed._checkpoint_references(_training(records))["baselines"]
    margins = {64: 1.0, 96: 6.0}
    runtimes: list[_FakeRuntime] = []

    def factory(_context, mask):
        k = int(mask.sum())
        descriptor = next(row for row in protocol["candidates"] if row["K"] == k)
        runtime = _FakeRuntime(
            context=context,
            records=records,
            resource=descriptor["resource_contract"],
            mask=mask,
            margin=margins[k],
        )
        runtimes.append(runtime)
        return runtime

    result = sweep._run_sweep(
        context=context,
        records=records,
        protocol=protocol,
        baselines=baselines,
        runtime_factory=factory,
    )
    assert result["passed"] is True
    assert result["selected_K"] == 96
    assert result["selection_role"] == "smallest_passing_candidate"
    assert result["executed_candidates"] == [64, 96]
    assert result["skipped_candidates"] == [128, 165]
    assert result["population_native_sequence_forwards"] == 16
    assert result["reset_replay_native_sequence_forwards"] == 1
    assert result["total_native_sequence_forwards"] == 17
    assert [row["executed"] for row in result["execution_manifest"]] == [
        True,
        True,
        False,
        False,
    ]
    assert len(runtimes) == 2
    assert all(runtime.closed for runtime in runtimes)
    selected_runtime = next(runtime for runtime in runtimes if runtime.k == 96)
    assert selected_runtime.forward_calls == 9 * 128
    assert selected_runtime.reset_calls == 8
    assert result["candidate_outcomes"]["K96"]["reset_replay"]["passed"] is True


def test_total_failure_executes_all_and_replays_lexicographic_best_only():
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    baselines = sweep.fixed._checkpoint_references(_training(records))["baselines"]
    margins = {64: 1.0, 96: 1.5, 128: 2.0, 165: 1.8}
    runtimes: list[_FakeRuntime] = []

    def factory(_context, mask):
        k = int(mask.sum())
        descriptor = next(row for row in protocol["candidates"] if row["K"] == k)
        runtime = _FakeRuntime(
            context=context,
            records=records,
            resource=descriptor["resource_contract"],
            mask=mask,
            margin=margins[k],
        )
        runtimes.append(runtime)
        return runtime

    result = sweep._run_sweep(
        context=context,
        records=records,
        protocol=protocol,
        baselines=baselines,
        runtime_factory=factory,
    )
    assert result["passed"] is False
    assert result["selected_K"] == 128
    assert result["selection_role"] == ("best_failed_candidate_for_diagnostic_replay")
    assert result["executed_candidates"] == [64, 96, 128, 165]
    assert result["skipped_candidates"] == []
    assert result["population_native_sequence_forwards"] == 32
    assert result["reset_replay_native_sequence_forwards"] == 1
    assert result["total_native_sequence_forwards"] == 33
    assert len(runtimes) == 4
    assert all(runtime.closed for runtime in runtimes)
    assert {runtime.k: runtime.forward_calls for runtime in runtimes} == {
        64: 8 * 128,
        96: 8 * 128,
        128: 9 * 128,
        165: 8 * 128,
    }
    selected_runtime = next(runtime for runtime in runtimes if runtime.k == 128)
    assert selected_runtime.reset_calls == 8
    replayed = [
        row["K"] for row in result["execution_manifest"] if row["reset_replay_executed"]
    ]
    assert replayed == [128]


@pytest.mark.parametrize("failure", ["counter", "replay"])
def test_system_or_replay_failure_aborts_without_selecting_or_skipping(
    failure: str,
):
    records = _records()
    context = _context(records)
    protocol = _protocol(records, context)
    baselines = sweep.fixed._checkpoint_references(_training(records))["baselines"]
    runtimes: list[_FakeRuntime] = []

    def factory(_context, mask):
        k = int(mask.sum())
        descriptor = next(row for row in protocol["candidates"] if row["K"] == k)
        runtime = _FakeRuntime(
            context=context,
            records=records,
            resource=descriptor["resource_contract"],
            mask=mask,
            margin=6.0,
            tamper_counter=failure == "counter",
            tamper_replay=failure == "replay",
        )
        runtimes.append(runtime)
        return runtime

    match = "systems contract failed" if failure == "counter" else "reset replay failed"
    with pytest.raises(ValueError, match=match):
        sweep._run_sweep(
            context=context,
            records=records,
            protocol=protocol,
            baselines=baselines,
            runtime_factory=factory,
        )
    assert len(runtimes) == 1
    assert runtimes[0].k == 64
    assert runtimes[0].closed is True


def test_k51_prerequisite_requires_systems_clean_loss_gate_failure(
    tmp_path: Path,
):
    protocol_path = (tmp_path / "k51-protocol.json").resolve()
    protocol_path.write_text("{}\n", encoding="utf-8")
    result = {
        "schema_version": sweep.fixed._SCHEMA_VERSION,
        "experiment": sweep.fixed._RESULT_EXPERIMENT,
        "status": "train_episodic_head_mask_gate_failed",
        "protocol": {
            "path": str(protocol_path),
            "sha256": "a" * 64,
        },
        "scope": {
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "episodic_head_mask_candidate": {
            "resource_passed": True,
            "loss_gate": {
                "passed": False,
                "summaries": {"candidate": {"mean": 2.0}},
                "matrix": [
                    {
                        "record_index": index,
                        "record_id": f"train-{index}",
                        "candidate_answer_cross_entropy": 2.0 + index / 10.0,
                    }
                    for index in range(8)
                ],
            },
            "passed": False,
            "fixed_M2_head_mask": {
                "mask_sha256": sweep.fixed._EXPECTED_M2_MASK_SHA256,
            },
        },
        "decision": {
            "passed": False,
            "semantic_gate_passed": False,
            "confirmation_authorized": False,
        },
        "post_run_authentication": {
            name: True
            for name in (
                "base_protocol",
                "package",
                "corpus_manifest",
                "train_split",
                "confirmation_not_opened",
                "source_config",
                "source_index",
                "source_shards",
                "expert_proxy",
                "headwise_episodic_library",
                "head_mask_protocol",
                "head_mask_source_inventory",
                "training_checkpoint",
                "layered_library",
                "headwise_library",
                "attention_library",
            )
        },
        "confirmation_split_opened": False,
    }
    result_path = tmp_path / "k51-result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    descriptor = sweep._validate_k51_result(
        path=result_path,
        expected_sha256=sha256_file(result_path),
        protocol_path=protocol_path,
        protocol_sha256="a" * 64,
    )
    assert descriptor["systems_clean"] is True
    assert descriptor["loss_gate_passed"] is False
    assert descriptor["attribution_only"] is True

    result["episodic_head_mask_candidate"]["resource_passed"] = False
    invalid_path = tmp_path / "invalid-k51-result.json"
    invalid_path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="prerequisite is invalid"):
        sweep._validate_k51_result(
            path=invalid_path,
            expected_sha256=sha256_file(invalid_path),
            protocol_path=protocol_path,
            protocol_sha256="a" * 64,
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


def test_freeze_and_reauthenticate_binds_k51_without_confirmation_access(
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
    checkpoint = {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "mode": "resumed",
    }
    library_path = tmp_path / "headwise.so"
    library_path.write_bytes(b"headwise")
    context["episodic_library_path"] = library_path.resolve()
    context["episodic_library_sha256"] = sha256_file(library_path)
    k51_protocol_path = tmp_path / "k51-protocol.json"
    k51_protocol_path.write_text("{}\n", encoding="utf-8")
    k51_result_path = tmp_path / "k51-result.json"
    k51_result_path.write_text("{}\n", encoding="utf-8")
    all_head_protocol_path = tmp_path / "all-head-protocol.json"
    all_head_protocol_path.write_text("{}\n", encoding="utf-8")
    all_head_result_path = tmp_path / "all-head-result.json"
    all_head_result_path.write_text("{}\n", encoding="utf-8")
    historical_library_path = tmp_path / "historical-all-head.so"
    historical_library_path.write_bytes(b"historical")
    frozen_k51 = {
        "authenticated_M2_projected_score_ordering": _ordering_payload(),
    }
    prerequisite = {
        "path": str(k51_result_path.resolve()),
        "sha256": sha256_file(k51_result_path),
        "status": "train_episodic_head_mask_gate_failed",
        "systems_clean": True,
        "loss_gate_passed": False,
        "loss_summaries": {},
        "loss_matrix_sha256": "f" * 64,
        "attribution_only": True,
    }
    all_head = {
        "protocol": {
            "path": str(all_head_protocol_path.resolve()),
            "sha256": sha256_file(all_head_protocol_path),
        },
        "result": {
            "path": str(all_head_result_path.resolve()),
            "sha256": sha256_file(all_head_result_path),
        },
        "historical_episodic_library": {
            "path": str(historical_library_path.resolve()),
            "sha256": sha256_file(historical_library_path),
        },
        "status": "train_episodic_oracle_gate_failed",
        "systems_clean": True,
        "record_ids": [record["record_id"] for record in records],
        "record_answer_cross_entropy": [1.5] * 8,
        "sequence_evidence_sha256": "9" * 64,
        "attribution_only": True,
        "strictly_better_than_K51_on_all_train_records": True,
    }
    monkeypatch.setattr(
        sweep,
        "_authenticate_sweep_inputs",
        lambda **_kwargs: (
            context,
            training,
            checkpoint,
            frozen_k51,
            prerequisite,
            all_head,
        ),
    )
    monkeypatch.setattr(
        sweep.fixed,
        "_fact_anchor_ids",
        lambda _path: _ANCHORS,
    )
    monkeypatch.setattr(
        sweep,
        "_source_inventory",
        lambda: {"rank-sweep.py": "b" * 64},
    )

    output = tmp_path / "rank-protocol.json"
    frozen = sweep.freeze_episodic_rank_sweep_protocol(
        base_protocol=context["protocol_path"],
        base_protocol_sha256=context["protocol_sha256"],
        training_checkpoint=checkpoint_path,
        training_checkpoint_sha256=checkpoint["sha256"],
        headwise_episodic_library=library_path,
        headwise_episodic_library_sha256=sha256_file(library_path),
        k51_protocol=k51_protocol_path,
        k51_protocol_sha256=sha256_file(k51_protocol_path),
        k51_result=k51_result_path,
        k51_result_sha256=sha256_file(k51_result_path),
        all_head_protocol=all_head_protocol_path,
        all_head_protocol_sha256=sha256_file(all_head_protocol_path),
        all_head_result=all_head_result_path,
        all_head_result_sha256=sha256_file(all_head_result_path),
        out=output,
    )
    protocol = frozen["protocol"]
    assert protocol["candidate_order"] == [64, 96, 128, 165]
    assert [row["K"] for row in protocol["candidates"]] == [64, 96, 128, 165]
    assert protocol["K51_prerequisite"]["result"]["attribution_only"] is True
    assert protocol["all_head_K256_attribution"]["attribution_only"] is True
    assert protocol["attribution_contract"]["K32"]["ruled_out"] is False
    revision = protocol["attribution_contract"]["K32"]["candidate_set_revision"]
    assert revision["K51_prospective_boundaries_included_K32"] is True
    assert revision["current_precommit_omits_K32"] is True
    assert "not ruled out" in revision["interpretation_limit"]
    assert protocol["train_scope"]["dense_teacher_forwards"] == 0
    assert protocol["train_scope"]["candidate_masks_fitted_by_this_experiment"] is False
    assert protocol["train_scope"]["candidate_selection_uses_train_outcomes"] is True
    assert protocol["train_scope"]["development_outcomes_used"] is False
    assert protocol["train_scope"]["confirmation_file_access_permitted"] is False
    assert protocol["confirmation_split_opened"] is False

    loaded_context, loaded_training, loaded_protocol = sweep._authenticate_protocol(
        output, frozen["sha256"]
    )
    assert loaded_context["train_records"] == records
    assert loaded_training is training
    assert loaded_protocol == protocol

    monkeypatch.setattr(
        sweep,
        "_source_inventory",
        lambda: {"rank-sweep.py": "c" * 64},
    )
    with pytest.raises(ValueError, match="contract changed"):
        sweep._authenticate_protocol(output, frozen["sha256"])
