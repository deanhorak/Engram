from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

import engram.evaluation.olmoe_retrieval_head_selector as selector


class _RoundTripTokenizer:
    """Small tokenizer double with the exact interfaces used by the corpus."""

    def __init__(self) -> None:
        self._decode: dict[int, str] = {}
        self._fixed: dict[str, list[int]] = {}
        next_identifier = 10
        fixed_segments = (
            (selector._OPENING_TEXT, 4),
            *((text, 7) for text in selector._FILLER_TEXT),
            (selector._FINAL_CONTEXT_TEXT, 12),
            (selector._QUERY_TEXT, selector._QUERY_TOKENS),
            *((f" Key {label} has code", 4) for label in selector._LABELS),
            (".", 1),
        )
        for text, count in fixed_segments:
            identifiers = list(range(next_identifier, next_identifier + count))
            next_identifier += count
            self._fixed[text] = identifiers
            widths = [
                len(text) * (offset + 1) // count - len(text) * offset // count
                for offset in range(count)
            ]
            cursor = 0
            for identifier, width in zip(identifiers, widths, strict=True):
                self._decode[identifier] = text[cursor : cursor + width]
                cursor += width

        self._numeric = {
            f" {number}": 1_000 + offset
            for offset, number in enumerate(range(100, 900))
        }
        self._decode.update(
            {identifier: text for text, identifier in self._numeric.items()}
        )
        self._vocabulary = {
            f"token-{identifier}": identifier for identifier in self._decode
        }

    def _consume(
        self,
        text: str,
        cursor: int,
        segment: str,
        identifiers: list[int],
    ) -> int:
        assert text.startswith(segment, cursor)
        identifiers.extend(self._fixed[segment])
        return cursor + len(segment)

    def _encode_record(self, text: str) -> list[int]:
        identifiers: list[int] = []
        cursor = self._consume(
            text,
            0,
            selector._OPENING_TEXT,
            identifiers,
        )
        for depth in range(selector._PASSKEYS_PER_RECORD):
            label = next(
                label
                for label in selector._LABELS
                if text.startswith(f" Key {label} has code", cursor)
            )
            cursor = self._consume(
                text,
                cursor,
                f" Key {label} has code",
                identifiers,
            )
            for _ in range(selector._PASSKEY_TOKENS):
                code = next(
                    code for code in self._numeric if text.startswith(code, cursor)
                )
                identifiers.append(self._numeric[code])
                cursor += len(code)
            cursor = self._consume(text, cursor, ".", identifiers)
            if depth < len(selector._FILLER_TEXT):
                cursor = self._consume(
                    text,
                    cursor,
                    selector._FILLER_TEXT[depth],
                    identifiers,
                )
        cursor = self._consume(
            text,
            cursor,
            selector._FINAL_CONTEXT_TEXT,
            identifiers,
        )
        cursor = self._consume(
            text,
            cursor,
            selector._QUERY_TEXT,
            identifiers,
        )
        while cursor < len(text):
            code = next(code for code in self._numeric if text.startswith(code, cursor))
            identifiers.append(self._numeric[code])
            cursor += len(code)
        return identifiers

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> SimpleNamespace:
        assert add_special_tokens is False
        if text in self._fixed:
            identifiers = self._fixed[text]
        elif text in self._numeric:
            identifiers = [self._numeric[text]]
        else:
            identifiers = self._encode_record(text)
        return SimpleNamespace(ids=list(identifiers))

    def decode(
        self,
        identifiers: list[int],
        *,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is False
        return "".join(self._decode[value] for value in identifiers)

    def get_vocab(self, *, with_added_tokens: bool) -> dict[str, int]:
        assert with_added_tokens is False
        return dict(self._vocabulary)


@pytest.fixture
def tokenizer() -> _RoundTripTokenizer:
    return _RoundTripTokenizer()


@pytest.fixture
def token_pool(tokenizer: _RoundTripTokenizer) -> list[int]:
    return selector._token_pool(tokenizer)


def _records(
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
) -> dict[str, list[dict[str, object]]]:
    partitions = selector._partition_code_tokens(
        token_pool,
        seed=selector._SEED,
    )
    return {
        split: [
            selector._generate_record(
                tokenizer,
                token_pool=token_pool,
                split=split,
                index=index,
                seed=selector._SEED,
                code_token_ids=partitions[(split, index)],
            )
            for index in range(selector._RECORDS_PER_SPLIT)
        ]
        for split in selector._SPLITS
    }


def _write_split(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def test_code_token_partition_is_deterministic_and_globally_disjoint(
    token_pool: list[int],
):
    first = selector._partition_code_tokens(token_pool, seed=selector._SEED)
    repeated = selector._partition_code_tokens(token_pool, seed=selector._SEED)
    changed = selector._partition_code_tokens(token_pool, seed=selector._SEED + 1)

    assert first == repeated
    assert first != changed
    assert list(first) == [
        (split, index)
        for split in selector._SPLITS
        for index in range(selector._RECORDS_PER_SPLIT)
    ]
    assert all(
        len(values) == selector._ANSWER_POSITIONS
        and len(set(values)) == selector._ANSWER_POSITIONS
        for values in first.values()
    )
    flattened = [value for values in first.values() for value in values]
    assert len(flattened) == 3 * 8 * 32
    assert len(set(flattened)) == len(flattened)

    split_values = {
        split: {
            value
            for index in range(selector._RECORDS_PER_SPLIT)
            for value in first[(split, index)]
        }
        for split in selector._SPLITS
    }
    assert split_values["train"].isdisjoint(split_values["development"])
    assert split_values["train"].isdisjoint(split_values["confirmation"])
    assert split_values["development"].isdisjoint(split_values["confirmation"])

    with pytest.raises(ValueError, match="pool"):
        selector._partition_code_tokens(
            [*token_pool[:767], token_pool[0]],
            seed=selector._SEED,
        )


def test_generated_record_has_exact_answer_only_layout_and_round_trip(
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
):
    partitions = selector._partition_code_tokens(
        token_pool,
        seed=selector._SEED,
    )
    record = selector._generate_record(
        tokenizer,
        token_pool=token_pool,
        split="train",
        index=5,
        seed=selector._SEED,
        code_token_ids=partitions[("train", 5)],
    )

    assert len(record["input_ids"]) == 129
    assert record["prediction_positions"] == 128
    assert record["answer_prediction_positions"] == list(range(96, 128))
    assert len(record["answer_source_depths"]) == 32
    assert record["passkey_answer_token_starts"] == [97, 105, 113, 121]

    flat_passkeys = [
        value for passkey in record["passkey_token_ids"] for value in passkey
    ]
    targets = [
        record["input_ids"][position + 1]
        for position in record["answer_prediction_positions"]
    ]
    assert targets == flat_passkeys
    assert record["input_ids"][97:] == flat_passkeys
    assert tokenizer.encode(record["text"]).ids == record["input_ids"]

    # The latest source code ends at position 75.  Thus none of the passkey
    # tokens is visible in the W16 local keys for the first answer at row 96.
    assert max(record["passkey_source_token_starts"]) + 7 == 75
    assert max(record["passkey_source_token_starts"]) + 7 < 96 - 15
    assert (
        selector._validate_record(
            deepcopy(record),
            split="train",
            index=5,
        )
        == record
    )


def test_label_to_depth_rotation_is_balanced_and_validated(
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
):
    rows = _records(tokenizer, token_pool)["train"]
    expected_starts = set(selector._PASSKEY_SOURCE_STARTS)

    for row in rows:
        assert set(row["passkey_source_token_starts"]) == expected_starts
        for label, start in enumerate(row["passkey_source_token_starts"]):
            depth = selector._SOURCE_DEPTH_NAMES[
                selector._PASSKEY_SOURCE_STARTS.index(start)
            ]
            begin = label * selector._PASSKEY_TOKENS
            stop = begin + selector._PASSKEY_TOKENS
            assert row["answer_source_depths"][begin:stop] == [depth] * 8
        selector._validate_record(
            deepcopy(row),
            split="train",
            index=row["record_index"],
        )

    # Across eight records, each answer label appears twice at each depth.
    for label in range(selector._PASSKEYS_PER_RECORD):
        starts = [row["passkey_source_token_starts"][label] for row in rows]
        assert {start: starts.count(start) for start in expected_starts} == {
            start: 2 for start in expected_starts
        }

    tampered = deepcopy(rows[1])
    (
        tampered["passkey_source_token_starts"][0],
        tampered["passkey_source_token_starts"][1],
    ) = (
        tampered["passkey_source_token_starts"][1],
        tampered["passkey_source_token_starts"][0],
    )
    with pytest.raises(ValueError, match="contract"):
        selector._validate_record(tampered, split="train", index=1)

    source_tampered = deepcopy(rows[1])
    first_source = source_tampered["passkey_source_token_starts"][0]
    source_tampered["input_ids"][first_source] += 1
    source_tampered["identity_sha256"] = selector._record_identity(source_tampered)
    with pytest.raises(ValueError, match="contract"):
        selector._validate_record(source_tampered, split="train", index=1)

    seed_tampered = deepcopy(rows[1])
    seed_tampered["seed"] += 1
    with pytest.raises(ValueError, match="contract"):
        selector._validate_record(seed_tampered, split="train", index=1)


def test_read_split_enforces_eight_ordered_unique_records(
    tmp_path: Path,
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
):
    rows = _records(tokenizer, token_pool)["development"]
    path = tmp_path / "development.jsonl"
    _write_split(path, rows)

    assert selector._read_split(path, split="development") == rows

    wrong_count = tmp_path / "wrong-count.jsonl"
    _write_split(wrong_count, rows[:-1])
    with pytest.raises(ValueError, match="wrong record count"):
        selector._read_split(wrong_count, split="development")

    reordered = tmp_path / "reordered.jsonl"
    _write_split(reordered, [rows[1], rows[0], *rows[2:]])
    with pytest.raises(ValueError, match="contract"):
        selector._read_split(reordered, split="development")


def test_complete_8_8_8_corpus_has_isolated_records_and_code_tokens(
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
):
    split_rows = _records(tokenizer, token_pool)

    assert {split: len(rows) for split, rows in split_rows.items()} == {
        "train": 8,
        "development": 8,
        "confirmation": 8,
    }
    identities = {
        split: {row["identity_sha256"] for row in rows}
        for split, rows in split_rows.items()
    }
    code_tokens = {
        split: {
            token
            for row in rows
            for passkey in row["passkey_token_ids"]
            for token in passkey
        }
        for split, rows in split_rows.items()
    }
    for left_index, left in enumerate(selector._SPLITS):
        for right in selector._SPLITS[left_index + 1 :]:
            assert identities[left].isdisjoint(identities[right])
            assert code_tokens[left].isdisjoint(code_tokens[right])

    assert len(set().union(*identities.values())) == 24
    assert len(set().union(*code_tokens.values())) == 768


@pytest.mark.parametrize(
    ("override", "value"),
    (
        ("train_records", 7),
        ("development_records", 9),
        ("confirmation_records", 7),
        ("tokens", 127),
        ("answer_tokens", 31),
        ("seed", selector._SEED + 1),
        ("workers", selector._WORKERS - 1),
    ),
)
def test_freeze_rejects_any_change_to_the_8_8_8_contract(
    tmp_path: Path,
    override: str,
    value: int,
):
    arguments = {
        "package": tmp_path / "package",
        "manifest_sha256": "0" * 64,
        "layered_library": tmp_path / "layered.so",
        "headwise_library": tmp_path / "headwise.so",
        "attention_library": tmp_path / "attention.so",
        "proxy_qualifier": tmp_path / "qualifier.json",
        "out": tmp_path / "protocol.json",
        override: value,
    }
    with pytest.raises(ValueError, match="frozen 8/8/8"):
        selector.freeze_retrieval_head_selector_protocol(**arguments)


def test_retrieval_projection_is_exactly_51_heads_and_stable():
    empty = np.zeros((selector._LAYERS, selector._HEADS), dtype=np.bool_)
    tied_gradient = np.ones_like(empty, dtype=np.float64)

    scores, mask, rms = selector.causal_gate._projected_gate_step(
        empty,
        tied_gradient,
    )
    expected = np.zeros_like(empty)
    expected.reshape(-1)[: selector._RESCUED_HEADS] = True

    assert rms == pytest.approx(1.0)
    assert int(mask.sum()) == selector._RESCUED_HEADS == 51
    assert np.array_equal(mask, expected)
    assert np.isfinite(scores).all()

    _scores, repeated, _rms = selector.causal_gate._projected_gate_step(
        mask,
        tied_gradient,
    )
    assert np.array_equal(repeated, mask)


def test_training_mask_selection_uses_all_eight_answer_only_records():
    def records(values: list[float]) -> list[dict[str, object]]:
        return [
            {
                "record_index": index,
                "loss": {"answer_cross_entropy": value},
            }
            for index, value in enumerate(values)
        ]

    evaluations = {
        "M0": {"records": records([10.0] * 8)},
        "M1": {"records": records([9.0] * 8)},
        # Identical objective summaries must prefer the earlier IHT mask.
        "M2": {"records": records([9.0] * 8)},
    }
    selection = selector._select_training_mask(evaluations)

    assert selection["selected_mask_name"] == "M1"
    assert selection["screen_eligible"] is True
    assert len(selection["per_record_deltas"]) == 8
    assert all(not row["regressed"] for row in selection["per_record_deltas"])

    regressed = deepcopy(evaluations)
    regressed["M0"]["records"] = records([1.0, *([10.0] * 7)])
    regressed["M1"]["records"] = records([1.5, *([2.0] * 7)])
    regressed["M2"]["records"] = records([2.0, *([3.0] * 7)])
    selection = selector._select_training_mask(regressed)
    assert selection["selected_mask_name"] == "M1"
    assert selection["screen_eligible"] is False
    assert selection["per_record_deltas"][0]["regressed"] is True

    incomplete = deepcopy(evaluations)
    incomplete["M2"]["records"] = incomplete["M2"]["records"][:7]
    with pytest.raises(ValueError, match="record"):
        selector._select_training_mask(incomplete)

    perfect = deepcopy(evaluations)
    perfect["M1"]["records"] = records([0.0] * 8)
    perfect["M2"]["records"] = records([1.0] * 8)
    perfect_selection = selector._select_training_mask(perfect)
    assert perfect_selection["selected_mask_name"] == "M1"
    assert perfect_selection["screen_eligible"] is True


def test_selected_head_validation_requires_exact_ranked_unique_51():
    rows = [
        {
            "rank": rank,
            "layer": flat_index // selector._HEADS,
            "head": flat_index % selector._HEADS,
        }
        for rank, flat_index in enumerate(
            range(selector._RESCUED_HEADS),
            start=1,
        )
    ]
    assert selector._validate_selected_heads(rows) == [
        (flat_index // 16, flat_index % 16) for flat_index in range(51)
    ]

    with pytest.raises(ValueError, match="population"):
        selector._validate_selected_heads(rows[:-1])

    duplicate = deepcopy(rows)
    duplicate[-1]["layer"] = duplicate[0]["layer"]
    duplicate[-1]["head"] = duplicate[0]["head"]
    with pytest.raises(ValueError, match="unique"):
        selector._validate_selected_heads(duplicate)

    wrong_rank = deepcopy(rows)
    wrong_rank[-1]["rank"] = 52
    with pytest.raises(ValueError, match="coordinate"):
        selector._validate_selected_heads(wrong_rank)

    out_of_range = deepcopy(rows)
    out_of_range[-1]["layer"] = selector._LAYERS
    with pytest.raises(ValueError, match="coordinate"):
        selector._validate_selected_heads(out_of_range)


def _native_replay_evidence() -> dict[str, object]:
    return {
        "hidden_sha256": "1" * 64,
        "logits_sha256": "2" * 64,
        "final_metrics": {
            "positions": selector._PREDICTION_POSITIONS,
            "attention_selected_heads": selector._RESCUED_HEADS,
            "elapsed_ns": 1_000,
            "q7_elapsed_ns": 500,
        },
        "answer_cross_entropy": 0.25,
    }


def test_native_replay_checks_ignore_only_timing_metrics():
    reference = _native_replay_evidence()
    replay = deepcopy(reference)
    replay["final_metrics"]["elapsed_ns"] = 9_999
    replay["final_metrics"]["q7_elapsed_ns"] = 8_888

    checks = selector._native_replay_checks(replay, reference)

    assert checks == {
        "hidden_sha256": True,
        "logits_sha256": True,
        "deterministic_final_metrics": True,
        "answer_cross_entropy": True,
    }


@pytest.mark.parametrize(
    ("mutate", "failed_check"),
    (
        (
            lambda evidence: evidence.__setitem__("hidden_sha256", "3" * 64),
            "hidden_sha256",
        ),
        (
            lambda evidence: evidence.__setitem__("logits_sha256", "4" * 64),
            "logits_sha256",
        ),
        (
            lambda evidence: evidence["final_metrics"].__setitem__(
                "positions",
                selector._PREDICTION_POSITIONS - 1,
            ),
            "deterministic_final_metrics",
        ),
        (
            lambda evidence: evidence.__setitem__(
                "answer_cross_entropy",
                0.2501,
            ),
            "answer_cross_entropy",
        ),
    ),
)
def test_native_replay_checks_fail_closed_on_deterministic_corruption(
    mutate,
    failed_check: str,
):
    reference = _native_replay_evidence()
    replay = deepcopy(reference)
    mutate(replay)

    checks = selector._native_replay_checks(replay, reference)

    assert checks[failed_check] is False
    assert not all(checks.values())


def _training_checkpoint_context(tmp_path: Path) -> dict[str, object]:
    return {
        "protocol_path": (tmp_path / "protocol.json").resolve(),
        "protocol_sha256": "a" * 64,
        "protocol": {
            "source_sha256": {
                "src/engram/evaluation/olmoe_retrieval_head_selector.py": "b" * 64,
            },
        },
        "train_records": [
            {
                "record_index": index,
                "record_id": f"train-{index}",
                "identity_sha256": f"{index + 1:x}" * 64,
            }
            for index in range(selector._RECORDS_PER_SPLIT)
        ],
    }


def _training_checkpoint_payload(
    context: dict[str, object],
) -> dict[str, object]:
    m0_mask = np.zeros(
        (selector._LAYERS, selector._HEADS),
        dtype=np.bool_,
    )
    m0_gradient = np.ones_like(m0_mask, dtype=np.float64)
    m1_scores, m1_mask, m0_rms = selector.causal_gate._projected_gate_step(
        m0_mask,
        m0_gradient,
    )
    m1_gradient = np.ones_like(m0_gradient)
    m2_scores, m2_mask, m1_rms = selector.causal_gate._projected_gate_step(
        m1_mask,
        m1_gradient,
    )
    masks = {
        "M0": m0_mask,
        "M1": m1_mask,
        "M2": m2_mask,
    }
    scores = {"M1": m1_scores, "M2": m2_scores}
    gradients = {"M0": m0_gradient, "M1": m1_gradient}
    rms_values = {"M0": m0_rms, "M1": m1_rms}
    losses = {"M0": 2.0, "M1": 1.0, "M2": 1.0}
    entries: dict[str, object] = {}
    train_records = context["train_records"]
    assert isinstance(train_records, list)
    for name in selector._MASK_NAMES:
        mask = masks[name]
        mask_sha256 = selector.sha256_json(mask.tolist())
        rows = [
            {
                "record_index": record["record_index"],
                "record_id": record["record_id"],
                "mask_sha256": mask_sha256,
                "selected_head_count": int(mask.sum()),
                "loss": {
                    "answer_cross_entropy": losses[name],
                },
                "gradient": (None if name == "M2" else gradients[name].tolist()),
                "native_q7": {
                    "record_index": record["record_index"],
                    "record_id": record["record_id"],
                    "answer_cross_entropy": losses[name],
                    "answer_positions_copied": selector._ANSWER_POSITIONS,
                    "final_position": selector._PREDICTION_POSITIONS,
                    "hidden_sha256": "c" * 64,
                    "logits_sha256": "d" * 64,
                    "counter_checks": {"attention": True, "q7": True},
                },
            }
            for record in train_records
        ]
        entries[name] = {
            "mask": mask.tolist(),
            "mask_sha256": mask_sha256,
            "selected_head_count": int(mask.sum()),
            "selected_heads": (
                []
                if name == "M0"
                else selector.causal_gate._selected_head_rows(
                    mask,
                    scores[name],
                )
            ),
            "projected_scores": (None if name == "M0" else scores[name].tolist()),
            "average_gradient": (None if name == "M2" else gradients[name].tolist()),
            "gradient_rms": rms_values.get(name),
            "records": rows,
        }
    evaluations = {
        name: {"records": entries[name]["records"]} for name in selector._MASK_NAMES
    }
    selection = selector._select_training_mask(evaluations)
    selected_name = selection["selected_mask_name"]
    selected_rows = selector.causal_gate._selected_head_rows(
        masks[selected_name],
        scores[selected_name],
    )
    snapshot = _valid_proxy_snapshot()
    return {
        "masks": entries,
        "selection": selection,
        "selected_heads": selected_rows,
        "expert_proxy": snapshot,
        "expert_proxy_checks": selector._proxy_execution_checks(snapshot),
        "model_parameters_frozen": True,
        "answer_positions_only": True,
        "elapsed_seconds": 12.5,
    }


def test_training_checkpoint_payload_validates_complete_projection_chain(
    tmp_path: Path,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)

    selection, selected_heads = selector._validate_training_payload(
        training,
        context=context,
    )

    assert selection == training["selection"]
    assert selection["selected_mask_name"] == "M1"
    assert selection["screen_eligible"] is True
    assert len(selected_heads) == selector._RESCUED_HEADS
    assert selected_heads == [
        (row["layer"], row["head"]) for row in training["selected_heads"]
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda training: training["masks"]["M0"]["mask"][0].__setitem__(
                0,
                True,
            ),
            "mask contract",
        ),
        (
            lambda training: training["masks"]["M0"]["average_gradient"][0].__setitem__(
                0, 2.0
            ),
            "average gradient changed",
        ),
        (
            lambda training: training["selection"].__setitem__(
                "selected_mask_name",
                "M2",
            ),
            "training selection changed",
        ),
        (
            lambda training: training["masks"]["M1"]["records"].__setitem__(
                slice(0, 2),
                list(
                    reversed(
                        training["masks"]["M1"]["records"][:2],
                    )
                ),
            ),
            "records are invalid",
        ),
        (
            lambda training: training["selected_heads"][0].__setitem__(
                "head",
                15,
            ),
            "selected-head ranking changed",
        ),
        (
            lambda training: training["expert_proxy"].__setitem__(
                "workers",
                selector._WORKERS - 1,
            ),
            "expert proxy contract failed",
        ),
    ),
)
def test_training_checkpoint_payload_fails_closed_on_deep_tampering(
    tmp_path: Path,
    mutate,
    message: str,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)
    mutate(training)

    with pytest.raises(ValueError, match=message):
        selector._validate_training_payload(training, context=context)


def test_training_checkpoint_round_trip_is_authenticated_and_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)
    post_authentication = {
        "protocol": True,
        "train_split": True,
        "development_split": True,
        "confirmation_not_opened": True,
    }
    output = tmp_path / "development_result.json"
    checkpoint_path = selector._training_checkpoint_path(output)
    assert checkpoint_path == (tmp_path / "development_result.training_checkpoint.json")

    descriptor = selector._write_training_checkpoint(
        checkpoint_path,
        context=context,
        training=training,
        post_training_authentication=post_authentication,
    )
    monkeypatch.setattr(
        selector,
        "_fit_post_authentication",
        lambda _context: post_authentication,
    )

    loaded, selection, selected_heads, resumed = selector._load_training_checkpoint(
        checkpoint_path,
        descriptor["sha256"],
        context=context,
    )

    assert loaded == training
    assert selection == training["selection"]
    assert selected_heads == [
        (row["layer"], row["head"]) for row in training["selected_heads"]
    ]
    assert resumed == {
        "path": str(checkpoint_path),
        "sha256": descriptor["sha256"],
        "mode": "resumed",
    }
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["training_sha256"] == selector.sha256_json(training)
    assert checkpoint["confirmation_split_opened"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda checkpoint: checkpoint["protocol"].__setitem__(
                "sha256",
                "e" * 64,
            ),
            "contract changed",
        ),
        (
            lambda checkpoint: checkpoint["source_sha256"].__setitem__(
                "src/engram/evaluation/olmoe_retrieval_head_selector.py",
                "e" * 64,
            ),
            "contract changed",
        ),
        (
            lambda checkpoint: checkpoint.__setitem__(
                "training_sha256",
                "f" * 64,
            ),
            "contract changed",
        ),
        (
            lambda checkpoint: checkpoint.__setitem__(
                "confirmation_split_opened",
                True,
            ),
            "contract changed",
        ),
    ),
)
def test_training_checkpoint_loader_rejects_authenticated_contract_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)
    authentication = {"all_artifacts": True}
    checkpoint_path = tmp_path / "checkpoint.json"
    selector._write_training_checkpoint(
        checkpoint_path,
        context=context,
        training=training,
        post_training_authentication=authentication,
    )
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    mutate(checkpoint)
    checkpoint_path.write_text(
        json.dumps(checkpoint),
        encoding="utf-8",
    )
    digest = selector.sha256_file(checkpoint_path)
    monkeypatch.setattr(
        selector,
        "_fit_post_authentication",
        lambda _context: authentication,
    )

    with pytest.raises(ValueError, match=message):
        selector._load_training_checkpoint(
            checkpoint_path,
            digest,
            context=context,
        )


def test_training_checkpoint_loader_rejects_wrong_file_digest(
    tmp_path: Path,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)
    checkpoint_path = tmp_path / "checkpoint.json"
    selector._write_training_checkpoint(
        checkpoint_path,
        context=context,
        training=training,
        post_training_authentication={"all_artifacts": True},
    )

    with pytest.raises(ValueError, match="authentication failed"):
        selector._load_training_checkpoint(
            checkpoint_path,
            "0" * 64,
            context=context,
        )


def test_training_checkpoint_requires_current_complete_artifact_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _training_checkpoint_context(tmp_path)
    training = _training_checkpoint_payload(context)
    checkpoint_path = tmp_path / "checkpoint.json"
    with pytest.raises(ValueError, match="post-training authentication failed"):
        selector._write_training_checkpoint(
            checkpoint_path,
            context=context,
            training=training,
            post_training_authentication={
                "confirmation_not_opened": False,
            },
        )
    assert not checkpoint_path.exists()

    descriptor = selector._write_training_checkpoint(
        checkpoint_path,
        context=context,
        training=training,
        post_training_authentication={
            "confirmation_not_opened": True,
        },
    )
    monkeypatch.setattr(
        selector,
        "_fit_post_authentication",
        lambda _context: {
            "confirmation_not_opened": True,
            "new_artifact_check": True,
        },
    )
    with pytest.raises(
        ValueError,
        match="post-training authentication changed",
    ):
        selector._load_training_checkpoint(
            checkpoint_path,
            descriptor["sha256"],
            context=context,
        )


def test_fresh_fit_persists_checkpoint_before_teacher_capture_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _training_checkpoint_context(tmp_path)
    context["development_records"] = []
    training = _training_checkpoint_payload(context)
    selection, selected_heads = selector._validate_training_payload(
        training,
        context=context,
    )
    authentication = {
        "confirmation_not_opened": True,
        "all_artifacts": True,
    }
    output = tmp_path / "development_result.json"
    checkpoint_path = selector._training_checkpoint_path(output)
    model = object()

    monkeypatch.setattr(
        selector,
        "_authenticate_fit_screen",
        lambda _protocol, _sha256: context,
    )
    monkeypatch.setattr(
        selector,
        "_fit_post_authentication",
        lambda _context: authentication,
    )
    monkeypatch.setattr(
        selector,
        "_load_frozen_surrogate",
        lambda _context: model,
    )
    monkeypatch.setattr(
        selector,
        "_fit_training_selector",
        lambda loaded, *, context: (
            training,
            selection,
            selected_heads,
        ),
    )

    def fail_teacher_capture(loaded, records):
        assert loaded is model
        assert records == []
        assert checkpoint_path.is_file()
        raise RuntimeError("deliberate teacher failure after checkpoint")

    monkeypatch.setattr(
        selector,
        "_capture_dense_teacher",
        fail_teacher_capture,
    )

    with pytest.raises(RuntimeError, match="deliberate teacher failure"):
        selector.fit_and_screen_retrieval_head_selector(
            protocol=context["protocol_path"],
            protocol_sha256=context["protocol_sha256"],
            out=output,
        )

    assert checkpoint_path.is_file()
    assert not output.exists()
    digest = selector.sha256_file(checkpoint_path)
    (
        loaded_training,
        loaded_selection,
        loaded_heads,
        descriptor,
    ) = selector._load_training_checkpoint(
        checkpoint_path,
        digest,
        context=context,
    )
    assert loaded_training == training
    assert loaded_selection == selection
    assert loaded_heads == selected_heads
    assert descriptor["mode"] == "resumed"


def test_resume_ineligible_checkpoint_skips_training_and_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    context = _training_checkpoint_context(tmp_path)
    context["development_records"] = []
    training = _training_checkpoint_payload(context)
    losses = {"M0": 2.0, "M1": 3.0, "M2": 4.0}
    for name, value in losses.items():
        for row in training["masks"][name]["records"]:
            row["loss"]["answer_cross_entropy"] = value
            row["native_q7"]["answer_cross_entropy"] = value
    evaluations = {
        name: {"records": training["masks"][name]["records"]}
        for name in selector._MASK_NAMES
    }
    selection = selector._select_training_mask(evaluations)
    assert selection["selected_mask_name"] == "M1"
    assert selection["screen_eligible"] is False
    training["selection"] = selection
    training["selected_heads"] = deepcopy(training["masks"]["M1"]["selected_heads"])
    selector._validate_training_payload(training, context=context)

    authentication = {
        "confirmation_not_opened": True,
        "all_artifacts": True,
    }
    checkpoint_path = tmp_path / "training_checkpoint.json"
    checkpoint = selector._write_training_checkpoint(
        checkpoint_path,
        context=context,
        training=training,
        post_training_authentication=authentication,
    )
    output = tmp_path / "development_result.json"

    monkeypatch.setattr(
        selector,
        "_authenticate_fit_screen",
        lambda _protocol, _sha256: context,
    )
    monkeypatch.setattr(
        selector,
        "_fit_post_authentication",
        lambda _context: authentication,
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("resume unexpectedly executed training or model load")

    monkeypatch.setattr(selector, "_fit_training_selector", unexpected)
    monkeypatch.setattr(
        selector,
        "_native_only_training_evaluation",
        unexpected,
    )
    monkeypatch.setattr(selector, "_load_frozen_surrogate", unexpected)
    monkeypatch.setattr(selector, "_capture_dense_teacher", unexpected)

    report = selector.fit_and_screen_retrieval_head_selector(
        protocol=context["protocol_path"],
        protocol_sha256=context["protocol_sha256"],
        out=output,
        resume_training_checkpoint=checkpoint_path,
        resume_training_checkpoint_sha256=checkpoint["sha256"],
    )

    assert output.is_file()
    assert report["status"] == "training_selector_failed"
    assert report["decision"] == {
        "status": "training_selector_failed",
        "passed": False,
        "confirmation_authorized": False,
        "next_step": (
            "stop this static retrieval selector without opening "
            "confirmation; investigate prefix-conditioned allocation"
        ),
    }
    assert report["training"] == training
    assert report["training_checkpoint"] == {
        "path": str(checkpoint_path),
        "sha256": checkpoint["sha256"],
        "mode": "resumed",
    }
    assert report["teacher_retrieval_evidence"] is None
    assert report["development"] is None
    assert report["confirmation_split_opened"] is False


def test_native_value_surrogate_backward_preserves_values_and_routes_gradient():
    native = torch.tensor(
        [[1.0, -3.0], [2.5, 7.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    surrogate = torch.tensor(
        [[9.0, 8.0], [7.0, 6.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.tensor([[0.5, 2.0], [-1.0, 3.0]], dtype=torch.float32)

    output = selector._native_forward_surrogate_backward(native, surrogate)
    assert torch.equal(output, native)
    (output * weights).sum().backward()

    assert native.grad is None
    assert torch.equal(surrogate.grad, weights)

    with pytest.raises(ValueError, match="straight-through"):
        selector._native_forward_surrogate_backward(
            native,
            surrogate[:, :1],
        )
    non_finite = surrogate.detach().clone()
    non_finite[0, 0] = torch.nan
    with pytest.raises(ValueError, match="straight-through"):
        selector._native_forward_surrogate_backward(native, non_finite)


def test_answer_cross_entropy_is_exactly_the_32_answer_rows():
    generator = torch.Generator().manual_seed(selector._SEED)
    logits = torch.randn(
        (2, selector._ANSWER_POSITIONS, 11),
        generator=generator,
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = (
        torch.arange(
            2 * selector._ANSWER_POSITIONS,
            dtype=torch.long,
        ).reshape(2, selector._ANSWER_POSITIONS)
        % 11
    )

    observed = selector._answer_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11),
        targets.reshape(-1),
    )
    assert torch.equal(observed, expected)
    observed.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape
    assert torch.isfinite(logits.grad).all()

    with pytest.raises(ValueError, match="answer-loss"):
        selector._answer_cross_entropy(logits[:, :-1], targets[:, :-1])
    invalid_targets = targets.clone()
    invalid_targets[0, 0] = logits.shape[-1]
    with pytest.raises(ValueError, match="answer-loss"):
        selector._answer_cross_entropy(logits, invalid_targets)


def _valid_proxy_snapshot() -> dict[str, object]:
    expected_calls = (
        selector._IHT_STEPS * selector._RECORDS_PER_SPLIT * selector._LAYERS
    )
    return {
        "workers": selector._WORKERS,
        "patched_layers": selector._LAYERS,
        "restored_layers": selector._LAYERS,
        "serial_forward_calls": expected_calls,
        "parallel_backward_calls": expected_calls,
        "expert_backward_tasks": expected_calls,
        "serial_forward_seconds": 1.0,
        "parallel_backward_task_seconds": 2.0,
        "ordered_reduction_seconds": 0.5,
        "context_active": False,
        "executor_shutdown": True,
    }


def test_proxy_execution_checks_require_the_complete_exact_lifecycle():
    snapshot = _valid_proxy_snapshot()
    checks = selector._proxy_execution_checks(snapshot)

    assert snapshot["workers"] == 12
    assert snapshot["patched_layers"] == snapshot["restored_layers"] == 16
    assert (
        snapshot["serial_forward_calls"] == snapshot["parallel_backward_calls"] == 256
    )
    assert checks == {
        "workers": True,
        "patched_layers": True,
        "restored_layers": True,
        "serial_forward_calls": True,
        "parallel_backward_calls": True,
        "expert_backward_tasks": True,
        "serial_forward_seconds": True,
        "parallel_backward_task_seconds": True,
        "ordered_reduction_seconds": True,
        "context_inactive": True,
        "executor_shutdown": True,
    }
    assert all(checks.values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("workers", 11, "workers"),
        ("patched_layers", 15, "patched_layers"),
        ("restored_layers", 15, "restored_layers"),
        ("serial_forward_calls", 255, "serial_forward_calls"),
        ("parallel_backward_calls", 257, "parallel_backward_calls"),
        ("context_active", True, "context_inactive"),
        ("executor_shutdown", False, "executor_shutdown"),
    ),
)
def test_proxy_execution_checks_fail_closed_on_call_or_lifecycle_corruption(
    field: str,
    value: object,
    failed_check: str,
):
    snapshot = _valid_proxy_snapshot()
    snapshot[field] = value

    checks = selector._proxy_execution_checks(snapshot)

    assert checks[failed_check] is False
    assert not all(checks.values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    (
        ("expert_backward_tasks", True, "expert_backward_tasks"),
        ("expert_backward_tasks", 255, "expert_backward_tasks"),
        ("expert_backward_tasks", 16_385, "expert_backward_tasks"),
        ("serial_forward_seconds", 0.0, "serial_forward_seconds"),
        (
            "parallel_backward_task_seconds",
            float("nan"),
            "parallel_backward_task_seconds",
        ),
        ("ordered_reduction_seconds", -0.1, "ordered_reduction_seconds"),
    ),
)
def test_proxy_execution_checks_enforce_task_and_timing_bounds(
    field: str,
    value: object,
    failed_check: str,
):
    snapshot = _valid_proxy_snapshot()
    snapshot[field] = value

    checks = selector._proxy_execution_checks(snapshot)

    assert checks[failed_check] is False
    assert not all(checks.values())


def test_execute_native_record_runs_128_single_tokens_and_copies_answer_rows(
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
    monkeypatch: pytest.MonkeyPatch,
):
    record = _records(tokenizer, token_pool)["train"][0]
    context = {
        "model": {"hidden_size": 3, "vocab_size": 2_000},
        "q7_expectations": {"q7": "expected"},
    }
    selected_heads = [(0, 0), (3, 7)]
    expected_counters = {"attention": "expected"}
    expectation_calls: list[tuple[object, object, int]] = []
    counter_calls: list[tuple[object, object, object, int]] = []

    def fake_expectations(model, heads, *, positions):
        expectation_calls.append((model, heads, positions))
        return expected_counters

    def fake_counter_checks(metrics, expected, q7, *, position):
        counter_calls.append((metrics, expected, q7, position))
        return {"attention": True, "q7": True}

    monkeypatch.setattr(
        selector.headwise,
        "_headwise_expectations",
        fake_expectations,
    )
    monkeypatch.setattr(
        selector.headwise,
        "_counter_checks",
        fake_counter_checks,
    )

    class _FakeRuntime:
        def __init__(self, initial_position: int) -> None:
            self.position = initial_position
            self.attention_metrics_available = True
            self.reset_calls = 0
            self.forward_tokens: list[int] = []
            self.diagnostic_rows: list[int] = []
            self._last_row = -1
            self._hidden = np.empty(3, dtype=np.float32)
            self._logits = np.empty(2_000, dtype=np.float32)

        def reset(self) -> None:
            self.reset_calls += 1
            self.position = 0

        def forward(self, tokens: list[int]) -> SimpleNamespace:
            assert len(tokens) == 1
            self._last_row = self.position
            self.forward_tokens.append(tokens[0])
            self.position += 1
            self._hidden = np.full(3, self._last_row, dtype=np.float32)
            self._logits = np.full(2_000, -1.0, dtype=np.float32)
            self._logits[self._last_row] = 1.0
            return SimpleNamespace(
                metrics={"positions": self.position},
                next_token=self._last_row,
            )

        def last_diagnostics(self) -> tuple[np.ndarray, np.ndarray]:
            self.diagnostic_rows.append(self._last_row)
            return self._hidden.copy(), self._logits.copy()

    runtime = _FakeRuntime(initial_position=9)
    logits, hidden, evidence = selector._execute_native_record(
        runtime,
        record=record,
        context=context,
        selected_heads=selected_heads,
    )

    assert runtime.reset_calls == 1
    assert runtime.position == 128
    assert runtime.forward_tokens == record["input_ids"][:-1]
    assert len(runtime.forward_tokens) == 128
    assert runtime.diagnostic_rows == list(range(96, 128))
    assert logits.shape == (32, 2_000)
    assert hidden.shape == (32, 3)
    np.testing.assert_array_equal(hidden[:, 0], np.arange(96, 128))
    assert evidence["answer_positions_copied"] == 32
    assert evidence["final_position"] == 128
    assert evidence["counter_checks"] == {"attention": True, "q7": True}
    assert expectation_calls == [(context["model"], selected_heads, 128)]
    assert counter_calls == [
        (
            {"positions": 128},
            expected_counters,
            context["q7_expectations"],
            128,
        )
    ]

    monkeypatch.setattr(
        selector.headwise,
        "_counter_checks",
        lambda *_args, **_kwargs: {"attention": False},
    )
    with pytest.raises(ValueError, match="counter contract"):
        selector._execute_native_record(
            _FakeRuntime(initial_position=0),
            record=record,
            context=context,
            selected_heads=selected_heads,
        )


def test_fit_authentication_never_opens_confirmation(
    tmp_path: Path,
    tokenizer: _RoundTripTokenizer,
    token_pool: list[int],
    monkeypatch: pytest.MonkeyPatch,
):
    split_rows = _records(tokenizer, token_pool)
    digest = "a" * 64
    descriptors = {}
    for split, rows in split_rows.items():
        path = tmp_path / f"{split}.jsonl"
        _write_split(path, rows)
        descriptors[split] = {
            "file": path.name,
            "sha256": digest,
            "records": 8,
            "tokens_per_record": 129,
            "prediction_positions_per_record": 128,
            "answer_prediction_positions_per_record": 32,
            "record_identity_sha256": selector.sha256_json(
                [row["identity_sha256"] for row in rows]
            ),
        }

    package_path = tmp_path / "package"
    model_path = tmp_path / "source-model"
    package_path.mkdir()
    model_path.mkdir()
    libraries = {}
    for name in ("layered", "headwise", "attention"):
        path = tmp_path / f"{name}.so"
        path.write_bytes(name.encode())
        libraries[name] = {"path": str(path), "sha256": digest}
    proxy_path = tmp_path / "proxy.json"
    proxy_path.write_text("{}", encoding="utf-8")
    framework = {"framework": "frozen"}
    source_inventory = {"selector.py": digest}
    model = {"layers": 16, "query_heads": 16}
    q7_expectations = {"q7_bytes": 1}
    budget = {"selected_head_count": 51}
    dispatch = {"dispatcher": "frozen"}
    manifest = {
        "model": {"config_path": "config.json"},
        "transformer": {"path": "non-mlp.bin"},
        "mlp": {"path": "q7.bin"},
        "tokenizer": {"path": "tokenizer"},
        "source": {"path": str(model_path), "revision": "revision"},
    }
    corpus_manifest = {
        "schema_version": selector._SCHEMA_VERSION,
        "experiment": "olmoe_q7_synthetic_passkey_corpus",
        "generator_seed": selector._SEED,
        "tokenizer_sha256": digest,
        "splits": descriptors,
    }
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "schema_version": selector._SCHEMA_VERSION,
        "experiment": selector._PROTOCOL_EXPERIMENT,
        "status": selector._PROTOCOL_STATUS,
        "seed": selector._SEED,
        "training": {
            "workers": selector._WORKERS,
            "iht_steps": selector._IHT_STEPS,
            "masks": list(selector._MASK_NAMES),
            "answer_prediction_positions": list(range(96, 128)),
        },
        "development_screen": {"records": 8},
        "budget": budget,
        "framework_contract": framework,
        "source_sha256": source_inventory,
        "package": {
            "path": str(package_path),
            "manifest_sha256": digest,
            "model": model,
            "tokenizer_sha256": digest,
            "q7_expectations_per_sequence": q7_expectations,
        },
        "source_model": {
            "path": str(model_path),
            "revision": "revision",
            "config_sha256": digest,
            "index_sha256": digest,
            "shard_sha256": {"model.safetensors": digest},
        },
        "libraries": libraries,
        "expert_proxy_qualifier": {
            "path": str(proxy_path),
            "sha256": digest,
            "result_expert_proxy_source_sha256": digest,
            "transformers_expert_dispatch_contract": dispatch,
        },
        "corpus": {
            "manifest_file": "corpus_manifest.json",
            "manifest_sha256": digest,
            "splits": descriptors,
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    opened: list[str] = []
    confirmation_path = (tmp_path / "confirmation.jsonl").resolve()

    def guarded_read_split(path: Path, *, split: str):
        opened.append(split)
        if split == "confirmation":
            raise AssertionError("fit-screen opened sealed confirmation")
        return deepcopy(split_rows[split])

    monkeypatch.setattr(selector, "_read_split", guarded_read_split)
    monkeypatch.setattr(
        selector.causal_gate,
        "_framework_contract",
        lambda: framework,
    )
    monkeypatch.setattr(selector, "_source_inventory", lambda: source_inventory)
    monkeypatch.setattr(
        selector,
        "validate_olmoe_native_package",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        selector.sustained,
        "_model_descriptor",
        lambda *_args: model,
    )
    monkeypatch.setattr(
        selector.sustained,
        "_q7_expectations",
        lambda *_args: q7_expectations,
    )
    monkeypatch.setattr(
        selector,
        "_budget_contract",
        lambda *_args: budget,
    )
    monkeypatch.setattr(
        selector,
        "_validate_proxy_qualifier",
        lambda _path: {
            "artifacts": {
                "expert_proxy_source_sha256": digest,
                "transformers_expert_dispatch_contract": dispatch,
            }
        },
    )
    monkeypatch.setattr(
        selector.proxy_record,
        "_transformers_expert_dispatch_contract",
        lambda: dispatch,
    )
    monkeypatch.setattr(
        selector,
        "audit_olmoe_source",
        lambda _path: SimpleNamespace(
            decision="proceed_to_router_trace",
            resolved_revision="revision",
            config_sha256=digest,
            index_sha256=digest,
        ),
    )
    monkeypatch.setattr(
        selector,
        "_source_shard_inventory",
        lambda _path: {"model.safetensors": digest},
    )

    def fake_read_json(path: str | Path, label: str):
        resolved = Path(path).resolve()
        if resolved == confirmation_path:
            raise AssertionError("fit-screen read sealed confirmation")
        if label == "retrieval selector protocol":
            return protocol
        if label == "retrieval corpus manifest":
            return corpus_manifest
        if label == "packaged OLMoE config":
            return {}
        raise AssertionError(f"unexpected JSON read: {label}")

    def fake_sha256(path: str | Path) -> str:
        if Path(path).resolve() == confirmation_path:
            raise AssertionError("fit-screen hashed sealed confirmation")
        return digest

    monkeypatch.setattr(selector, "_read_json", fake_read_json)
    monkeypatch.setattr(selector, "sha256_file", fake_sha256)
    tokenizers = ModuleType("tokenizers")

    class _TokenizerFactory:
        @staticmethod
        def from_file(_path: str) -> _RoundTripTokenizer:
            return tokenizer

    tokenizers.Tokenizer = _TokenizerFactory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", tokenizers)

    context = selector._authenticate_fit_screen(protocol_path, digest)

    assert opened == ["train", "development"]
    assert context["train_records"] == split_rows["train"]
    assert context["development_records"] == split_rows["development"]
    assert context["confirmation_descriptor_authenticated_without_file_access"] is True
    assert "confirmation" not in context["split_paths"]
