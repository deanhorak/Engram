from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_prefix_selector as prefix
from engram.utils import sha256_file


_ANCHORS = {
    "A": (10, 11, 12, 13),
    "B": (20, 21, 22, 23),
    "C": (30, 31, 32, 33),
    "D": (40, 41, 42, 43),
}


def _input_ids(order: str) -> list[int]:
    values = [1] * prefix.retrieval._TOKENS_PER_RECORD
    for start, label in zip(
        prefix.retrieval._FACT_ANCHORS,
        order,
        strict=True,
    ):
        values[start : start + 4] = _ANCHORS[label]
    return values


def _records() -> list[dict[str, Any]]:
    return [
        {
            "record_index": index,
            "record_id": f"train-{index}",
            "input_ids": _input_ids(order),
        }
        for index, order in enumerate(prefix.retrieval._FACT_ORDERS)
    ]


def _exact_mask(start: int) -> np.ndarray:
    flat = np.zeros(prefix.retrieval._LAYERS * prefix.retrieval._HEADS, dtype=np.bool_)
    flat[start : start + prefix.retrieval._RESCUED_HEADS] = True
    return flat.reshape(prefix.retrieval._LAYERS, prefix.retrieval._HEADS)


def _synthetic_partition_state() -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
]:
    base = _exact_mask(0)
    later_mask = _exact_mask(40)
    earlier_mask = _exact_mask(100)
    later_gradient = np.where(later_mask, -3.0, 1.0)
    earlier_gradient = np.where(earlier_mask, -3.0, 1.0)
    later_indices = set(prefix._EXPECTED_D_LATER_INDICES)
    gradients = tuple(
        later_gradient.copy() if index in later_indices else earlier_gradient.copy()
        for index in range(prefix._RECORDS)
    )
    return base, base.copy(), gradients


def _synthetic_training(
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base, global_mask, gradients = _synthetic_partition_state()
    training = {
        "masks": {
            "M1": {
                "mask": base.tolist(),
                "records": [
                    {
                        "record_index": index,
                        "record_id": record["record_id"],
                        "gradient": gradients[index].tolist(),
                    }
                    for index, record in enumerate(records)
                ],
            },
            "M2": {
                "mask": global_mask.tolist(),
                "records": [
                    {
                        "record_index": index,
                        "record_id": record["record_id"],
                        "loss": {"answer_cross_entropy": 2.0},
                    }
                    for index, record in enumerate(records)
                ],
            },
        },
    }
    state = prefix._training_state(training, records)
    return training, state


def _evaluations(
    records: list[dict[str, Any]],
    *,
    assigned: list[float],
    other: float = 20.0,
) -> dict[str, list[dict[str, Any]]]:
    later = set(prefix._EXPECTED_D_LATER_INDICES)
    result: dict[str, list[dict[str, Any]]] = {}
    for name in prefix._PROTOTYPE_NAMES:
        rows: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            assigned_name = prefix._D_LATER if index in later else prefix._D_EARLIER
            rows.append(
                {
                    "record_index": index,
                    "record_id": record["record_id"],
                    "answer_cross_entropy": (
                        assigned[index] if name == assigned_name else other
                    ),
                    "proof": f"{name}-{index}",
                }
            )
        result[name] = rows
    return result


class _CausalPrefixOnly(Sequence[int]):
    """Sequence double that raises if code inspects an answer token."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, key):
        if isinstance(key, slice):
            stop = len(self._values) if key.stop is None else key.stop
            if stop > prefix._PREFIX_INPUT_COUNT:
                raise AssertionError("future answer token was inspected")
            return self._values[key]
        if key >= prefix._PREFIX_INPUT_COUNT:
            raise AssertionError("future answer token was inspected")
        return self._values[key]


def test_prefix_rule_stops_at_row_96_and_ignores_future_mutation():
    values = _input_ids("ACDB")
    expected = ("A", "C", "D", "B")
    assert (
        prefix._fact_order_from_causal_prefix(
            _CausalPrefixOnly(values),
            _ANCHORS,
        )
        == expected
    )

    mutated = list(values)
    mutated[prefix._PREFIX_INPUT_COUNT :] = [999] * (
        len(mutated) - prefix._PREFIX_INPUT_COUNT
    )
    assert prefix._fact_order_from_causal_prefix(mutated, _ANCHORS) == expected

    corrupted = list(values)
    corrupted[prefix.retrieval._FACT_ANCHORS[2] + 1] = 999
    with pytest.raises(ValueError, match="not identifiable"):
        prefix._fact_order_from_causal_prefix(corrupted, _ANCHORS)


def test_prefix_partition_uses_raw_input_ids_not_record_metadata():
    records = _records()
    for record in records:
        assert set(record) == {"record_index", "record_id", "input_ids"}
    result = prefix._prefix_partition(records, _ANCHORS)
    assert result["clusters"] == {
        prefix._D_LATER: list(prefix._EXPECTED_D_LATER_INDICES),
        prefix._D_EARLIER: list(prefix._EXPECTED_D_EARLIER_INDICES),
    }
    assert result["last_input_index_observed"] == 96
    assert result["future_answer_tokens_observed"] is False


def test_balanced_partition_is_exhaustive_deterministic_and_exact_51(
    monkeypatch: pytest.MonkeyPatch,
):
    base, global_mask, gradients = _synthetic_partition_state()
    first = prefix._derive_balanced_partition(base, global_mask, gradients)
    second = prefix._derive_balanced_partition(base, global_mask, gradients)

    assert first["candidate_partition_count"] == 35
    assert first["clusters"] == (
        prefix._EXPECTED_D_LATER_INDICES,
        prefix._EXPECTED_D_EARLIER_INDICES,
    )
    assert second["clusters"] == first["clusters"]
    assert second["objective"] == first["objective"]
    assert [prototype["mask_sha256"] for prototype in second["prototypes"]] == [
        prototype["mask_sha256"] for prototype in first["prototypes"]
    ]
    assert all(
        int(prototype["mask"].sum()) == prefix.retrieval._RESCUED_HEADS
        for prototype in first["prototypes"]
    )

    monkeypatch.setattr(
        prefix,
        "_D_LATER_MASK_SHA256",
        first["prototypes"][0]["mask_sha256"],
    )
    monkeypatch.setattr(
        prefix,
        "_D_EARLIER_MASK_SHA256",
        first["prototypes"][1]["mask_sha256"],
    )
    bound = prefix._bind_and_validate_prefix_prototypes(
        first,
        prefix._prefix_partition(_records(), _ANCHORS),
        global_mask,
    )
    assert set(bound) == set(prefix._PROTOTYPE_NAMES)
    assert all(
        item["report"]["selected_head_count"] == prefix.retrieval._RESCUED_HEADS
        for item in bound.values()
    )

    monkeypatch.setattr(prefix, "_D_LATER_MASK_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen mask changed"):
        prefix._bind_and_validate_prefix_prototypes(
            first,
            prefix._prefix_partition(_records(), _ANCHORS),
            global_mask,
        )


def test_transfer_gate_requires_strict_summary_gains_and_no_regression():
    records = _records()
    allocation = prefix._prefix_partition(records, _ANCHORS)

    passed = prefix._transfer_gate(
        records=records,
        baseline=[2.0] * 8,
        prefix=allocation,
        evaluations=_evaluations(records, assigned=[1.0] * 8),
    )
    assert passed["passed"] is True
    assert passed["native_sequence_forwards"] == 16
    assert len(passed["matrix"]) == 8
    assert all(passed["gate_checks"].values())

    baseline = [10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    regressed = prefix._transfer_gate(
        records=records,
        baseline=baseline,
        prefix=allocation,
        evaluations=_evaluations(
            records,
            assigned=[9.0, 2.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    )
    assert regressed["gate_checks"]["assigned_worst_strictly_improved"] is True
    assert regressed["gate_checks"]["assigned_mean_strictly_improved"] is True
    assert regressed["gate_checks"]["no_record_regression"] is False
    assert regressed["passed"] is False

    unchanged = prefix._transfer_gate(
        records=records,
        baseline=[2.0] * 8,
        prefix=allocation,
        evaluations=_evaluations(records, assigned=[2.0] * 8),
    )
    assert unchanged["gate_checks"]["no_record_regression"] is True
    assert unchanged["gate_checks"]["assigned_worst_strictly_improved"] is False
    assert unchanged["gate_checks"]["assigned_mean_strictly_improved"] is False
    assert unchanged["passed"] is False


class _TrainingOnlyContext(dict):
    def __getitem__(self, key):
        if key in {
            "development_records",
            "development_outcomes",
            "confirmation_records",
            "confirmation_path",
        }:
            raise AssertionError(f"forbidden split access: {key}")
        return super().__getitem__(key)


def test_authenticated_screen_writes_atomically_without_sealed_split_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = _records()
    training, state = _synthetic_training(records)
    partition = prefix._derive_balanced_partition(
        state["base_mask"],
        state["global_mask"],
        state["gradients"],
    )
    monkeypatch.setattr(
        prefix,
        "_D_LATER_MASK_SHA256",
        partition["prototypes"][0]["mask_sha256"],
    )
    monkeypatch.setattr(
        prefix,
        "_D_EARLIER_MASK_SHA256",
        partition["prototypes"][1]["mask_sha256"],
    )

    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}\n", encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text('{"checkpoint":true}\n', encoding="utf-8")
    checkpoint_sha256 = sha256_file(checkpoint_path)
    context = _TrainingOnlyContext(
        {
            "protocol_path": protocol_path.resolve(),
            "protocol_sha256": "a" * 64,
            "tokenizer_path": tmp_path / "tokenizer.json",
            "train_records": records,
            "confirmation_records": object(),
            "development_records": object(),
        }
    )
    authenticate_calls = 0

    def authenticate(_protocol, _sha256):
        nonlocal authenticate_calls
        authenticate_calls += 1
        return context

    monkeypatch.setattr(
        prefix.retrieval,
        "_authenticate_fit_screen",
        authenticate,
    )
    monkeypatch.setattr(
        prefix.retrieval,
        "_load_training_checkpoint",
        lambda _path, _sha256, *, context: (
            training,
            {"screen_eligible": True, "selected_mask_name": "M2"},
            [],
            {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha256,
                "mode": "resumed",
            },
        ),
    )
    monkeypatch.setattr(prefix, "_fact_anchor_ids", lambda _path: _ANCHORS)
    monkeypatch.setattr(
        prefix,
        "_evaluate_prototype_transfer",
        lambda *, context, records, prototypes: _evaluations(
            list(records),
            assigned=[1.0] * 8,
        ),
    )
    monkeypatch.setattr(
        prefix.retrieval,
        "_fit_post_authentication",
        lambda _context: {
            "confirmation_not_opened": True,
            "development_authentication_only": True,
        },
    )

    output = tmp_path / "prefix-result.json"
    report = prefix.screen_retrieval_prefix_selector(
        protocol=protocol_path,
        protocol_sha256="a" * 64,
        training_checkpoint=checkpoint_path,
        training_checkpoint_sha256=checkpoint_sha256,
        out=output,
    )
    assert output.is_file()
    assert authenticate_calls == 1
    assert report["status"] == "train_prefix_gate_passed"
    assert report["method"]["development_outcomes_used"] is False
    assert report["confirmation_split_opened"] is False
    assert report["train_transfer"]["native_sequence_forwards"] == 16

    monkeypatch.setattr(
        prefix.retrieval,
        "_authenticate_fit_screen",
        lambda *_args: pytest.fail("existing output must fail before authentication"),
    )
    with pytest.raises(ValueError, match="target already exists"):
        prefix.screen_retrieval_prefix_selector(
            protocol=protocol_path,
            protocol_sha256="a" * 64,
            training_checkpoint=checkpoint_path,
            training_checkpoint_sha256=checkpoint_sha256,
            out=output,
        )
