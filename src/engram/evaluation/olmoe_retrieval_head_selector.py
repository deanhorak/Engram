"""Prospective retrieval-targeted OLMoE head selector.

This evaluator is deliberately separate from the frozen natural-prose
causal-head-gate experiment.  It creates a new deterministic token-level
passkey corpus with independent train, development, and sealed-confirmation
splits; fits exactly 51 rescued layer/head pairs using answer-position-only
causal distillation; and evaluates one selected mask through the complete
native Q7 runtime on development data.

The confirmation split is created and hashed by ``freeze`` but is never opened
by ``fit-screen``.  A development pass can only authorize a later, separately
implemented confirmation command.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

import engram.evaluation.olmoe_causal_head_gate as causal_gate
import engram.evaluation.olmoe_causal_head_gate_proxy_record as proxy_record
import engram.evaluation.olmoe_native_headwise as headwise
import engram.evaluation.olmoe_native_layer_rescue as layer_rescue
import engram.evaluation.olmoe_native_sustained as sustained
from engram.compiler.olmoe_native import validate_olmoe_native_package
from engram.evaluation.olmoe_expert_proxy import (
    frozen_olmoe_expert_backward_proxy,
)
from engram.evaluation.olmoe_native_causal import _position_metrics
from engram.models.olmoe import audit_olmoe_source
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.tracing.olmoe import _prepare_transformers_imports
from engram.utils import atomic_json, sha256_file, sha256_json


_PROTOCOL_EXPERIMENT = "olmoe_q7_retrieval_head_selector_protocol"
_RESULT_EXPERIMENT = "olmoe_q7_retrieval_head_selector_development"
_TRAINING_CHECKPOINT_EXPERIMENT = "olmoe_q7_retrieval_head_selector_training_checkpoint"
_TRAINING_CHECKPOINT_STATUS = "training_complete_before_development"
_PROTOCOL_STATUS = "frozen_before_retrieval_teacher_or_candidate_execution"
_SCHEMA_VERSION = 1
_SEED = 20260728
_WORKERS = 12
_RECORDS_PER_SPLIT = 8
_PREDICTION_POSITIONS = 128
_TOKENS_PER_RECORD = _PREDICTION_POSITIONS + 1
_ANSWER_POSITIONS = 32
_ANSWER_START = _PREDICTION_POSITIONS - _ANSWER_POSITIONS
_PASSKEYS_PER_RECORD = 4
_PASSKEY_TOKENS = 8
_CONTEXT_TOKENS = 89
_QUERY_TOKENS = 8
_FACT_ANCHORS = (4, 24, 44, 64)
_PASSKEY_SOURCE_STARTS = (8, 28, 48, 68)
_SOURCE_DEPTH_NAMES = ("earliest", "early", "middle", "late")
_LABELS = ("A", "B", "C", "D")
_FACT_ORDERS = (
    "ABCD",
    "BADC",
    "CDAB",
    "DCBA",
    "ACDB",
    "BDCA",
    "CABD",
    "DBAC",
)
_OPENING_TEXT = "Read all entries:"
_FILLER_TEXT = (
    " Keep every code for the final question",
    " The requested order appears at the end",
    " Use only values attached to each key",
)
_FINAL_CONTEXT_TEXT = " Hold all codes in memory until asked to repeat them exactly."
_QUERY_TEXT = " Repeat codes A B C D exactly:"
_CODE_TOKEN_DOMAIN = b"engram-olmoe-retrieval-passkey-v1\0code-token\0"
_IHT_STEPS = 2
_LAYERS = 16
_HEADS = 16
_RESCUED_HEADS = 51
_MASK_NAMES = ("M0", "M1", "M2")
_SPLITS = ("train", "development", "confirmation")
_BASE_POLICY = {
    "local_window": 16,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_FULL_POLICY = {
    "local_window": 128,
    "older_candidates": 8,
    "older_top_k": 4,
    "sink_tokens": 2,
}
_THRESHOLDS = {
    "maximum_mean_kl": 0.05,
    "minimum_top1_agreement": 0.90,
    "maximum_target_nll_delta": 0.05,
    "maximum_hidden_relative_l2": 0.10,
}
_EXPECTED_ATTENTION_LOGICAL_READ_BYTES = 973_384_704
_EXPECTED_ATTENTION_READ_FRACTION = 0.44975387218386625
_EXPECTED_FIFTY_TWO_HEAD_READ_FRACTION = 0.4524379996366279
_EXPECTED_ATTENTION_STATE_BYTES = 12_284_864
_EXPECTED_Q7_FRACTION = 0.22786458333333334
_MINIMUM_TEACHER_RETRIEVAL_LOG_PROBABILITY_ADVANTAGE = 0.0
_SOURCE_FILES = (
    "src/engram/evaluation/olmoe_retrieval_head_selector.py",
    "src/engram/evaluation/olmoe_causal_head_gate.py",
    "src/engram/evaluation/olmoe_causal_head_gate_proxy_record.py",
    "src/engram/evaluation/olmoe_expert_proxy.py",
    "src/engram/evaluation/olmoe_native_headwise.py",
    "src/engram/evaluation/olmoe_native_layer_rescue.py",
    "src/engram/evaluation/olmoe_native_sustained.py",
    "src/engram/evaluation/olmoe_native_causal.py",
    "src/engram/compiler/olmoe_native.py",
    "src/engram/runtime/olmoe_native.py",
    "src/engram/runtime/native_attention.py",
)
_CODE_TOKEN_TEXT = re.compile(r" [0-9]{2,4}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _new_output(path: str | Path, label: str) -> Path:
    result = Path(path).expanduser().resolve()
    if result.exists() or result.is_symlink():
        raise ValueError(f"{label} target already exists")
    return result


def _safe_relative_path(parent: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} path is unsafe")
    result = (parent / relative).resolve()
    if result.parent != parent.resolve():
        raise ValueError(f"{label} path escapes the protocol directory")
    return result


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"retrieval corpus target already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _single_token_id(tokenizer: Any, text: str, label: str) -> int:
    encoding = tokenizer.encode(text, add_special_tokens=False)
    identifiers = list(encoding.ids)
    if len(identifiers) != 1:
        raise ValueError(f"{label} must encode to exactly one token")
    identifier = identifiers[0]
    if (
        isinstance(identifier, bool)
        or not isinstance(identifier, int)
        or identifier < 0
    ):
        raise ValueError(f"{label} token identifier is invalid")
    return identifier


def _progress(message: str) -> None:
    print(f"[retrieval-selector] {message}", file=sys.stderr, flush=True)


def _token_pool(tokenizer: Any) -> list[int]:
    vocabulary = tokenizer.get_vocab(with_added_tokens=False)
    if not isinstance(vocabulary, dict):
        raise ValueError("retrieval tokenizer vocabulary is invalid")
    values: list[int] = []
    for identifier in sorted(set(vocabulary.values())):
        if (
            isinstance(identifier, bool)
            or not isinstance(identifier, int)
            or identifier < 0
        ):
            continue
        text = tokenizer.decode([identifier], skip_special_tokens=False)
        if (
            isinstance(text, str)
            and _CODE_TOKEN_TEXT.fullmatch(text)
            and list(tokenizer.encode(text, add_special_tokens=False).ids)
            == [identifier]
        ):
            values.append(identifier)
    required = len(_SPLITS) * _RECORDS_PER_SPLIT * _ANSWER_POSITIONS
    if len(values) < required:
        raise ValueError("retrieval tokenizer has too few one-token numeric codes")
    return values


def _partition_code_tokens(
    token_pool: Sequence[int],
    *,
    seed: int,
) -> dict[tuple[str, int], list[int]]:
    required = len(_SPLITS) * _RECORDS_PER_SPLIT * _ANSWER_POSITIONS
    if len(set(token_pool)) != len(token_pool) or len(token_pool) < required:
        raise ValueError("retrieval code-token pool is invalid")
    ranked = sorted(
        (int(value) for value in token_pool),
        key=lambda value: (
            hashlib.sha256(
                _CODE_TOKEN_DOMAIN
                + int(seed).to_bytes(8, "big", signed=False)
                + value.to_bytes(4, "big", signed=False)
            ).digest(),
            value,
        ),
    )[:required]
    result: dict[tuple[str, int], list[int]] = {}
    offset = 0
    for split in _SPLITS:
        for index in range(_RECORDS_PER_SPLIT):
            result[(split, index)] = ranked[offset : offset + _ANSWER_POSITIONS]
            offset += _ANSWER_POSITIONS
    if offset != required or len(set(ranked)) != required:
        raise AssertionError("retrieval code-token partition is inconsistent")
    return result


def _record_seed(seed: int, split: str, index: int) -> int:
    payload = f"{seed}:{split}:{index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _record_identity(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {name: value for name, value in record.items() if name != "identity_sha256"}
    )


def _generate_record(
    tokenizer: Any,
    *,
    token_pool: Sequence[int],
    split: str,
    index: int,
    seed: int,
    code_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    if split not in _SPLITS or not 0 <= index < _RECORDS_PER_SPLIT:
        raise ValueError("retrieval record split/index is invalid")
    fixed_segments = (
        _OPENING_TEXT,
        *_FILLER_TEXT,
        _FINAL_CONTEXT_TEXT,
        _QUERY_TEXT,
    )
    fixed_lengths = (4, 7, 7, 7, 12, _QUERY_TOKENS)
    for text, expected in zip(fixed_segments, fixed_lengths, strict=True):
        if len(tokenizer.encode(text, add_special_tokens=False).ids) != expected:
            raise ValueError("retrieval scaffold does not have the frozen token length")
    if code_token_ids is None:
        rng = random.Random(_record_seed(seed, split, index))
        codes = rng.sample(
            list(token_pool),
            _PASSKEYS_PER_RECORD * _PASSKEY_TOKENS,
        )
    else:
        codes = [int(value) for value in code_token_ids]
        if (
            len(codes) != _ANSWER_POSITIONS
            or len(set(codes)) != len(codes)
            or any(value not in token_pool for value in codes)
        ):
            raise ValueError("retrieval record code-token partition is invalid")
    passkeys = [
        codes[passkey_index * _PASSKEY_TOKENS : (passkey_index + 1) * _PASSKEY_TOKENS]
        for passkey_index in range(_PASSKEYS_PER_RECORD)
    ]
    code_text = [
        [
            tokenizer.decode([identifier], skip_special_tokens=False)
            for identifier in passkey
        ]
        for passkey in passkeys
    ]
    if any(
        not _CODE_TOKEN_TEXT.fullmatch(text)
        for passkey in code_text
        for text in passkey
    ):
        raise ValueError("retrieval passkey text is not a singleton numeric code")
    fact_order = _FACT_ORDERS[index]
    label_by_depth = [_LABELS.index(label) for label in fact_order]
    depth_by_label = [0] * _PASSKEYS_PER_RECORD
    pieces = [_OPENING_TEXT]
    for depth, passkey_index in enumerate(label_by_depth):
        depth_by_label[passkey_index] = depth
        label = _LABELS[passkey_index]
        pieces.append(
            f" Key {label} has code" + "".join(code_text[passkey_index]) + "."
        )
        if depth < len(_FILLER_TEXT):
            pieces.append(_FILLER_TEXT[depth])
    pieces.extend((_FINAL_CONTEXT_TEXT, _QUERY_TEXT))
    context_text = "".join(pieces)
    context_ids = list(tokenizer.encode(context_text, add_special_tokens=False).ids)
    answer_text = "".join(text for code in code_text for text in code)
    text = context_text + answer_text
    input_ids = list(tokenizer.encode(text, add_special_tokens=False).ids)
    answer = [identifier for code in passkeys for identifier in code]
    if (
        len(input_ids) != _TOKENS_PER_RECORD
        or len(answer) != _ANSWER_POSITIONS
        or len(context_ids) != _ANSWER_START + 1
        or input_ids[: _ANSWER_START + 1] != context_ids
        or input_ids[_ANSWER_START + 1 :] != answer
    ):
        raise ValueError("retrieval record token accounting is invalid")
    for depth, passkey_index in enumerate(label_by_depth):
        source_start = _PASSKEY_SOURCE_STARTS[depth]
        if (
            input_ids[source_start : source_start + _PASSKEY_TOKENS]
            != passkeys[passkey_index]
        ):
            raise ValueError("retrieval source passkey span is invalid")
    answer_prediction_positions = list(range(_ANSWER_START, _PREDICTION_POSITIONS))
    answer_source_depths = [
        _SOURCE_DEPTH_NAMES[depth_by_label[offset // _PASSKEY_TOKENS]]
        for offset in range(_ANSWER_POSITIONS)
    ]
    decoded = tokenizer.decode(input_ids, skip_special_tokens=False)
    if (
        decoded != text
        or list(
            tokenizer.encode(
                decoded,
                add_special_tokens=False,
            ).ids
        )
        != input_ids
    ):
        raise ValueError("retrieval record is not tokenizer round-trip exact")
    provisional = {
        "schema_version": _SCHEMA_VERSION,
        "source_kind": "engram_synthetic_retrieval_passkey_v1",
        "record_id": f"olmoe-retrieval-passkey-v1-{split}-{index:02d}",
        "split": split,
        "record_index": index,
        "seed": _record_seed(seed, split, index),
        "fact_order": list(fact_order),
        "input_ids": input_ids,
        "prediction_positions": _PREDICTION_POSITIONS,
        "answer_prediction_positions": answer_prediction_positions,
        "answer_source_depths": answer_source_depths,
        "passkey_source_token_starts": [
            _PASSKEY_SOURCE_STARTS[depth_by_label[label]]
            for label in range(_PASSKEYS_PER_RECORD)
        ],
        "passkey_answer_token_starts": [
            _ANSWER_START + 1 + index * _PASSKEY_TOKENS
            for index in range(_PASSKEYS_PER_RECORD)
        ],
        "passkey_token_ids": passkeys,
        "text": text,
    }
    provisional["identity_sha256"] = _record_identity(provisional)
    return provisional


def _validate_record(record: Any, *, split: str, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("retrieval corpus record must be an object")
    required = {
        "schema_version",
        "source_kind",
        "record_id",
        "split",
        "record_index",
        "seed",
        "fact_order",
        "input_ids",
        "prediction_positions",
        "answer_prediction_positions",
        "answer_source_depths",
        "passkey_source_token_starts",
        "passkey_answer_token_starts",
        "passkey_token_ids",
        "text",
        "identity_sha256",
    }
    if set(record) != required:
        raise ValueError("retrieval corpus record fields are invalid")
    input_ids = record["input_ids"]
    answer_positions = record["answer_prediction_positions"]
    depths = record["answer_source_depths"]
    passkeys = record["passkey_token_ids"]
    fact_order = record["fact_order"]
    expected_order = list(_FACT_ORDERS[index])
    depth_by_label = {label: depth for depth, label in enumerate(expected_order)}
    expected_source_starts = [
        _PASSKEY_SOURCE_STARTS[depth_by_label[label]] for label in _LABELS
    ]
    expected_depths = [
        _SOURCE_DEPTH_NAMES[depth_by_label[_LABELS[offset // _PASSKEY_TOKENS]]]
        for offset in range(_ANSWER_POSITIONS)
    ]
    if (
        record["schema_version"] != _SCHEMA_VERSION
        or record["source_kind"] != "engram_synthetic_retrieval_passkey_v1"
        or record["split"] != split
        or record["record_index"] != index
        or record["record_id"] != f"olmoe-retrieval-passkey-v1-{split}-{index:02d}"
        or record["seed"] != _record_seed(_SEED, split, index)
        or fact_order != expected_order
        or record["prediction_positions"] != _PREDICTION_POSITIONS
        or not isinstance(record["text"], str)
        or not record["text"]
        or not isinstance(input_ids, list)
        or len(input_ids) != _TOKENS_PER_RECORD
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in input_ids
        )
        or answer_positions != list(range(_ANSWER_START, _PREDICTION_POSITIONS))
        or depths != expected_depths
        or record["passkey_source_token_starts"] != expected_source_starts
        or record["passkey_answer_token_starts"]
        != [
            _ANSWER_START + 1 + value * _PASSKEY_TOKENS
            for value in range(_PASSKEYS_PER_RECORD)
        ]
        or not isinstance(passkeys, list)
        or len(passkeys) != _PASSKEYS_PER_RECORD
        or any(
            not isinstance(code, list)
            or len(code) != _PASSKEY_TOKENS
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in code
            )
            for code in passkeys
        )
        or [value for code in passkeys for value in code]
        != input_ids[_ANSWER_START + 1 :]
        or any(
            input_ids[start : start + _PASSKEY_TOKENS] != passkeys[label]
            for label, start in enumerate(expected_source_starts)
        )
        or any(input_ids.count(value) != 2 for code in passkeys for value in code)
        or max(expected_source_starts) + _PASSKEY_TOKENS - 1
        >= _ANSWER_START - _BASE_POLICY["local_window"] + 1
        or record["identity_sha256"] != _record_identity(record)
    ):
        raise ValueError("retrieval corpus record contract is invalid")
    return record


def _read_split(path: Path, *, split: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read retrieval {split} split: {exc}") from exc
    if len(lines) != _RECORDS_PER_SPLIT or any(not line for line in lines):
        raise ValueError(f"retrieval {split} split has the wrong record count")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"retrieval {split} record {index} is invalid JSON"
            ) from exc
        rows.append(_validate_record(value, split=split, index=index))
    identities = [row["identity_sha256"] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"retrieval {split} records are not unique")
    return rows


def _split_descriptor(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "records": len(records),
        "tokens_per_record": _TOKENS_PER_RECORD,
        "prediction_positions_per_record": _PREDICTION_POSITIONS,
        "answer_prediction_positions_per_record": _ANSWER_POSITIONS,
        "record_identity_sha256": sha256_json(
            [row["identity_sha256"] for row in records]
        ),
    }


def _source_inventory() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    return {relative: sha256_file(repository / relative) for relative in _SOURCE_FILES}


def _source_shard_inventory(model_path: Path) -> dict[str, str]:
    index = _read_json(
        model_path / "model.safetensors.index.json",
        "OLMoE source index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("OLMoE source index has no weight map")
    names = sorted(set(weight_map.values()))
    if not names or not all(isinstance(name, str) and name for name in names):
        raise ValueError("OLMoE source shard inventory is invalid")

    def describe(name: str) -> tuple[str, str]:
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("OLMoE source shard path is invalid")
        path = model_path / relative
        if not path.is_file():
            raise ValueError("OLMoE source shard path is invalid")
        return name, sha256_file(path)

    with ThreadPoolExecutor(max_workers=min(6, len(names))) as executor:
        return dict(executor.map(describe, names))


def _validate_proxy_qualifier(path: Path) -> dict[str, Any]:
    value = _read_json(path, "expert-proxy qualifier")
    if (
        value.get("schema_version") != 1
        or value.get("experiment")
        != "olmoe_causal_head_gate_frozen_expert_proxy_full_record"
        or value.get("status") != "proxy_record_exact_and_materially_faster"
        or value.get("exact_parity_passed") is not True
        or value.get("evidence_passed") is not True
        or value.get("authorized_for_larger_development_fits") is not True
        or value.get("execution_contract", {}).get("proxy_workers") != _WORKERS
        or value.get("execution_contract", {}).get("proxy_backward_is_expert_parallel")
        is not True
        or not isinstance(value.get("post_run_authentication"), dict)
        or not value["post_run_authentication"]
        or not all(value["post_run_authentication"].values())
    ):
        raise ValueError("expert-proxy qualifier does not authorize retrieval fitting")
    return value


def _budget_contract(
    model: Mapping[str, int],
    q7_expectations: Mapping[str, int],
) -> dict[str, Any]:
    canonical_heads = [
        (flat // _HEADS, flat % _HEADS) for flat in range(_RESCUED_HEADS)
    ]
    candidate = headwise._headwise_expectations(dict(model), canonical_heads)
    full = headwise._headwise_expectations(
        dict(model),
        [(layer, head) for layer in range(_LAYERS) for head in range(_HEADS)],
    )
    fifty_two = headwise._headwise_expectations(
        dict(model),
        [(flat // _HEADS, flat % _HEADS) for flat in range(_RESCUED_HEADS + 1)],
    )
    q7 = layer_rescue._q7_traffic_contract(
        dict(model),
        dict(q7_expectations),
    )
    if (
        candidate["attention_logical_read_bytes"]
        != _EXPECTED_ATTENTION_LOGICAL_READ_BYTES
        or candidate["attention_logical_read_fraction"]
        != _EXPECTED_ATTENTION_READ_FRACTION
        or candidate["attention_state_bytes"] != _EXPECTED_ATTENTION_STATE_BYTES
        or q7["q7_fraction_of_all_expert_ideal_q4"] != _EXPECTED_Q7_FRACTION
        or fifty_two["attention_logical_read_fraction"]
        != _EXPECTED_FIFTY_TWO_HEAD_READ_FRACTION
        or fifty_two["attention_logical_read_fraction"] <= 0.45
    ):
        raise ValueError("retrieval selector resource contract changed")
    return {
        "selected_head_count": _RESCUED_HEADS,
        "candidate_attention_expectations_per_sequence": candidate,
        "full_control_attention_expectations_per_sequence": full,
        "q7_traffic_contract_per_sequence": q7,
        "maximum_attention_logical_read_fraction": 0.45,
        "maximum_q7_traffic_fraction": 0.45,
        "next_head_boundary": {
            "selected_head_count": _RESCUED_HEADS + 1,
            "attention_logical_read_fraction": (
                fifty_two["attention_logical_read_fraction"]
            ),
            "within_budget": False,
        },
    }


def freeze_retrieval_head_selector_protocol(
    *,
    package: str | Path,
    manifest_sha256: str,
    layered_library: str | Path,
    headwise_library: str | Path,
    attention_library: str | Path,
    proxy_qualifier: str | Path,
    out: str | Path,
    train_records: int = _RECORDS_PER_SPLIT,
    development_records: int = _RECORDS_PER_SPLIT,
    confirmation_records: int = _RECORDS_PER_SPLIT,
    tokens: int = _PREDICTION_POSITIONS,
    answer_tokens: int = _ANSWER_POSITIONS,
    seed: int = _SEED,
    workers: int = _WORKERS,
) -> dict[str, Any]:
    """Create and freeze new passkey corpora plus the execution protocol."""

    _progress("authenticating package, libraries, source model, and proxy")
    if (
        (train_records, development_records, confirmation_records)
        != (_RECORDS_PER_SPLIT,) * 3
        or tokens != _PREDICTION_POSITIONS
        or answer_tokens != _ANSWER_POSITIONS
        or seed != _SEED
        or workers != _WORKERS
    ):
        raise ValueError("retrieval selector requires the frozen 8/8/8 contract")
    output = _new_output(out, "retrieval protocol")
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = {split: output.parent / f"{split}.jsonl" for split in _SPLITS}
    manifest_path = output.parent / "corpus_manifest.json"
    for path in [*paths.values(), manifest_path]:
        if path.exists() or path.is_symlink():
            raise ValueError(f"retrieval freeze target already exists: {path.name}")
    package_path = Path(package).expanduser().resolve()
    layered_path = Path(layered_library).expanduser().resolve()
    headwise_path = Path(headwise_library).expanduser().resolve()
    attention_path = Path(attention_library).expanduser().resolve()
    proxy_path = Path(proxy_qualifier).expanduser().resolve()
    for path, label in (
        (layered_path, "layered library"),
        (headwise_path, "head-wise library"),
        (attention_path, "attention library"),
        (proxy_path, "proxy qualifier"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"retrieval {label} is invalid")
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=manifest_sha256,
    )
    proxy_value = _validate_proxy_qualifier(proxy_path)
    repository = Path(__file__).resolve().parents[3]
    proxy_source_sha256 = sha256_file(
        repository / "src/engram/evaluation/olmoe_expert_proxy.py"
    )
    dispatch_contract = proxy_record._transformers_expert_dispatch_contract()
    proxy_artifacts = proxy_value.get("artifacts", {})
    if proxy_source_sha256 != proxy_artifacts.get(
        "expert_proxy_source_sha256"
    ) or dispatch_contract != proxy_artifacts.get(
        "transformers_expert_dispatch_contract"
    ):
        raise ValueError("retrieval expert-proxy qualification binding changed")
    model_path = Path(manifest["source"]["path"]).resolve()
    audit = audit_olmoe_source(model_path)
    if (
        audit.decision != "proceed_to_router_trace"
        or audit.resolved_revision != manifest["source"]["revision"]
    ):
        raise ValueError("retrieval selector OLMoE source audit failed")
    config_path = package_path / manifest["model"]["config_path"]
    config = _read_json(config_path, "packaged OLMoE config")
    model = sustained._model_descriptor(manifest, config)
    if model["layers"] != _LAYERS or model["query_heads"] != _HEADS:
        raise ValueError("retrieval selector requires the frozen OLMoE topology")
    q7_expectations = sustained._q7_expectations(model)
    budget = _budget_contract(model, q7_expectations)
    source_inventory = _source_inventory()
    source_shards = _source_shard_inventory(model_path)
    _progress("source authentication complete; generating sealed 8/8/8 corpus")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval corpus generation"
        ) from exc
    tokenizer_path = package_path / manifest["tokenizer"]["path"] / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pool = _token_pool(tokenizer)
    code_partitions = _partition_code_tokens(pool, seed=seed)
    split_records = {
        split: [
            _generate_record(
                tokenizer,
                token_pool=pool,
                split=split,
                index=index,
                seed=seed,
                code_token_ids=code_partitions[(split, index)],
            )
            for index in range(_RECORDS_PER_SPLIT)
        ]
        for split in _SPLITS
    }
    all_identities = [
        row["identity_sha256"] for split in _SPLITS for row in split_records[split]
    ]
    if len(set(all_identities)) != _RECORDS_PER_SPLIT * len(_SPLITS):
        raise ValueError("retrieval split identities overlap")
    for split, rows in split_records.items():
        _atomic_jsonl(paths[split], rows)
    corpus_manifest = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": "olmoe_q7_synthetic_passkey_corpus",
        "generator_seed": seed,
        "generator_domain_hex": _CODE_TOKEN_DOMAIN.hex(),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "tokenizer_vocabulary_size": len(tokenizer.get_vocab(with_added_tokens=True)),
        "add_special_tokens": False,
        "numeric_candidate_pool_size": len(pool),
        "numeric_candidate_pool_sha256": sha256_json(pool),
        "selected_code_token_sha256": sha256_json(
            [
                code_partitions[(split, index)]
                for split in _SPLITS
                for index in range(_RECORDS_PER_SPLIT)
            ]
        ),
        "fact_orders": list(_FACT_ORDERS),
        "token_layout": {
            "tokens_per_record": _TOKENS_PER_RECORD,
            "model_input": [0, _PREDICTION_POSITIONS],
            "all_causal_targets": [1, _TOKENS_PER_RECORD],
            "answer_targets": [_ANSWER_START + 1, _TOKENS_PER_RECORD],
            "answer_prediction_positions": [
                _ANSWER_START,
                _PREDICTION_POSITIONS,
            ],
            "passkey_source_starts": list(_PASSKEY_SOURCE_STARTS),
        },
        "splits": {
            split: _split_descriptor(paths[split], split_records[split])
            for split in _SPLITS
        },
        "split_isolation": {
            "record_identities_pairwise_disjoint": True,
            "all_768_code_tokens_globally_unique": True,
            "split_code_token_intersections_empty": True,
            "confirmation_created_before_teacher_or_candidate_execution": True,
            "fit_screen_must_not_open_confirmation": True,
        },
    }
    atomic_json(manifest_path, corpus_manifest)
    protocol = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _PROTOCOL_EXPERIMENT,
        "status": _PROTOCOL_STATUS,
        "seed": seed,
        "package": {
            "path": str(package_path),
            "manifest_sha256": manifest_sha256.lower(),
            "model": model,
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "q7_expectations_per_sequence": q7_expectations,
        },
        "source_model": {
            "path": str(model_path),
            "revision": audit.resolved_revision,
            "config_sha256": audit.config_sha256,
            "index_sha256": audit.index_sha256,
            "shard_sha256": source_shards,
        },
        "libraries": {
            "layered": {
                "path": str(layered_path),
                "sha256": sha256_file(layered_path),
            },
            "headwise": {
                "path": str(headwise_path),
                "sha256": sha256_file(headwise_path),
            },
            "attention": {
                "path": str(attention_path),
                "sha256": sha256_file(attention_path),
            },
        },
        "expert_proxy_qualifier": {
            "path": str(proxy_path),
            "sha256": sha256_file(proxy_path),
            "result_expert_proxy_source_sha256": proxy_value["artifacts"][
                "expert_proxy_source_sha256"
            ],
            "transformers_expert_dispatch_contract": dispatch_contract,
            "workers": _WORKERS,
        },
        "corpus": {
            "manifest_file": manifest_path.name,
            "manifest_sha256": sha256_file(manifest_path),
            "splits": corpus_manifest["splits"],
        },
        "training": {
            "device": "cpu",
            "workers": _WORKERS,
            "iht_steps": _IHT_STEPS,
            "masks": list(_MASK_NAMES),
            "answer_prediction_positions": list(
                range(_ANSWER_START, _PREDICTION_POSITIONS)
            ),
            "loss": {
                "name": "mean_ground_truth_answer_cross_entropy",
                "record_weighting": "equal",
                "target_tokens_per_record": _ANSWER_POSITIONS,
                "position_scope": "answer_positions_96_127_only",
                "forward_values": "complete_packaged_native_Q7_runtime",
                "backward_surrogate": (
                    "frozen_BF16_shell_with_exact_native_attention_and_"
                    "qualified_expert_parallel_backward"
                ),
                "straight_through_boundary": ("native_forward_surrogate_backward"),
            },
            "selection_rule": (
                "among M1 and M2 minimize worst train-record objective, then "
                "mean objective, then prefer M1; require strict worst and mean "
                "improvement versus M0 and no train-record regression"
            ),
        },
        "development_screen": {
            "records": _RECORDS_PER_SPLIT,
            "answer_positions_per_record": _ANSWER_POSITIONS,
            "source_depth_strata": list(_SOURCE_DEPTH_NAMES),
            "thresholds": dict(_THRESHOLDS),
            "teacher_retrieval_minimum_log_probability_advantage": (
                _MINIMUM_TEACHER_RETRIEVAL_LOG_PROBABILITY_ADVANTAGE
            ),
            "full_W128_control_required": True,
            "candidate_and_each_depth_must_pass": True,
            "confirmation_must_remain_unopened": True,
        },
        "budget": budget,
        "attention_policies": {
            "base": dict(_BASE_POLICY),
            "rescued": dict(_FULL_POLICY),
        },
        "source_sha256": source_inventory,
        "framework_contract": causal_gate._framework_contract(),
        "decision_rule": {
            "development_pass": (
                "authorize a separately frozen confirmation protocol only"
            ),
            "semantic_failure": (
                "stop the static retrieval-targeted selector without opening "
                "confirmation; move next to prefix-conditioned allocation"
            ),
            "evidence_failure": "stop and diagnose before any semantic decision",
        },
        "limitations": [
            "The corpus is synthetic token-level passkey retrieval, not natural prose.",
            "The confirmation split is sealed but this module implements no confirmation command.",
            "Training loss values execute the complete packaged Q7 runtime; gradients use the frozen BF16 model as a straight-through surrogate.",
            "Logical byte counters are algorithmic traffic, not measured DRAM traffic.",
        ],
    }
    atomic_json(output, protocol)
    _progress(f"frozen protocol written to {output}")
    return protocol


def _authenticate_fit_screen(
    protocol_path: str | Path,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Authenticate training/development inputs without opening confirmation."""

    path = Path(protocol_path).expanduser().resolve()
    protocol = _read_json(path, "retrieval selector protocol")
    observed_protocol_sha256 = sha256_file(path)
    if (
        not _is_sha256(protocol_sha256)
        or observed_protocol_sha256 != protocol_sha256.lower()
        or protocol.get("schema_version") != _SCHEMA_VERSION
        or protocol.get("experiment") != _PROTOCOL_EXPERIMENT
        or protocol.get("status") != _PROTOCOL_STATUS
        or protocol.get("seed") != _SEED
        or protocol.get("training", {}).get("workers") != _WORKERS
        or protocol.get("training", {}).get("iht_steps") != _IHT_STEPS
        or protocol.get("training", {}).get("masks") != list(_MASK_NAMES)
        or protocol.get("training", {}).get("answer_prediction_positions")
        != list(range(_ANSWER_START, _PREDICTION_POSITIONS))
        or protocol.get("development_screen", {}).get("records") != _RECORDS_PER_SPLIT
        or protocol.get("budget", {}).get("selected_head_count") != _RESCUED_HEADS
        or protocol.get("framework_contract") != causal_gate._framework_contract()
        or protocol.get("source_sha256") != _source_inventory()
    ):
        raise ValueError("retrieval selector protocol authentication failed")
    package_contract = protocol.get("package")
    source_contract = protocol.get("source_model")
    libraries = protocol.get("libraries")
    proxy_contract = protocol.get("expert_proxy_qualifier")
    corpus_contract = protocol.get("corpus")
    if not all(
        isinstance(value, dict)
        for value in (
            package_contract,
            source_contract,
            libraries,
            proxy_contract,
            corpus_contract,
        )
    ):
        raise ValueError("retrieval selector protocol bindings are invalid")
    package_path = Path(package_contract["path"]).expanduser().resolve()
    manifest = validate_olmoe_native_package(
        package_path,
        expected_manifest_sha256=package_contract["manifest_sha256"],
    )
    config_path = package_path / manifest["model"]["config_path"]
    non_mlp_path = package_path / manifest["transformer"]["path"]
    q7_path = package_path / manifest["mlp"]["path"]
    tokenizer_path = package_path / manifest["tokenizer"]["path"] / "tokenizer.json"
    config = _read_json(config_path, "packaged OLMoE config")
    model = sustained._model_descriptor(manifest, config)
    q7_expectations = sustained._q7_expectations(model)
    if (
        model != package_contract.get("model")
        or q7_expectations != package_contract.get("q7_expectations_per_sequence")
        or sha256_file(tokenizer_path) != package_contract.get("tokenizer_sha256")
        or _budget_contract(model, q7_expectations) != protocol.get("budget")
    ):
        raise ValueError("retrieval selector package contract changed")
    library_paths: dict[str, Path] = {}
    for name in ("layered", "headwise", "attention"):
        descriptor = libraries.get(name)
        if not isinstance(descriptor, dict):
            raise ValueError("retrieval selector library descriptor is invalid")
        library_path = Path(descriptor.get("path", "")).expanduser().resolve()
        if (
            library_path.is_symlink()
            or not library_path.is_file()
            or sha256_file(library_path) != descriptor.get("sha256")
        ):
            raise ValueError(f"retrieval selector {name} library changed")
        library_paths[name] = library_path
    proxy_path = Path(proxy_contract.get("path", "")).expanduser().resolve()
    proxy_value = _validate_proxy_qualifier(proxy_path)
    dispatch_contract = proxy_record._transformers_expert_dispatch_contract()
    repository = Path(__file__).resolve().parents[3]
    current_proxy_source = sha256_file(
        repository / "src/engram/evaluation/olmoe_expert_proxy.py"
    )
    if (
        sha256_file(proxy_path) != proxy_contract.get("sha256")
        or current_proxy_source
        != proxy_contract.get("result_expert_proxy_source_sha256")
        or current_proxy_source
        != proxy_value.get("artifacts", {}).get("expert_proxy_source_sha256")
        or dispatch_contract
        != proxy_contract.get("transformers_expert_dispatch_contract")
        or dispatch_contract
        != proxy_value.get("artifacts", {}).get("transformers_expert_dispatch_contract")
    ):
        raise ValueError("retrieval selector expert-proxy binding changed")
    model_path = Path(source_contract.get("path", "")).expanduser().resolve()
    audit = audit_olmoe_source(model_path)
    if (
        audit.decision != "proceed_to_router_trace"
        or audit.resolved_revision != source_contract.get("revision")
        or audit.config_sha256 != source_contract.get("config_sha256")
        or audit.index_sha256 != source_contract.get("index_sha256")
        or _source_shard_inventory(model_path) != source_contract.get("shard_sha256")
        or model_path != Path(manifest["source"]["path"]).resolve()
    ):
        raise ValueError("retrieval selector source model changed")
    manifest_file = _safe_relative_path(
        path.parent,
        corpus_contract.get("manifest_file"),
        "retrieval corpus manifest",
    )
    corpus_manifest = _read_json(manifest_file, "retrieval corpus manifest")
    split_contracts = corpus_contract.get("splits")
    if (
        sha256_file(manifest_file) != corpus_contract.get("manifest_sha256")
        or corpus_manifest.get("schema_version") != _SCHEMA_VERSION
        or corpus_manifest.get("experiment") != "olmoe_q7_synthetic_passkey_corpus"
        or corpus_manifest.get("generator_seed") != _SEED
        or corpus_manifest.get("tokenizer_sha256") != sha256_file(tokenizer_path)
        or corpus_manifest.get("splits") != split_contracts
        or not isinstance(split_contracts, dict)
        or set(split_contracts) != set(_SPLITS)
    ):
        raise ValueError("retrieval corpus manifest authentication failed")
    records: dict[str, list[dict[str, Any]]] = {}
    split_paths: dict[str, Path] = {}
    # The confirmation descriptor is authenticated transitively through the
    # already-frozen protocol and manifest.  Its file is deliberately not
    # opened or hashed in this command.
    for split in ("train", "development"):
        descriptor = split_contracts[split]
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("records") != _RECORDS_PER_SPLIT
            or descriptor.get("tokens_per_record") != _TOKENS_PER_RECORD
            or descriptor.get("prediction_positions_per_record")
            != _PREDICTION_POSITIONS
            or descriptor.get("answer_prediction_positions_per_record")
            != _ANSWER_POSITIONS
        ):
            raise ValueError(f"retrieval {split} descriptor is invalid")
        split_path = _safe_relative_path(
            path.parent,
            descriptor.get("file"),
            f"retrieval {split}",
        )
        if sha256_file(split_path) != descriptor.get("sha256"):
            raise ValueError(f"retrieval {split} split changed")
        rows = _read_split(split_path, split=split)
        if sha256_json([row["identity_sha256"] for row in rows]) != descriptor.get(
            "record_identity_sha256"
        ):
            raise ValueError(f"retrieval {split} identity changed")
        records[split] = rows
        split_paths[split] = split_path
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pool = _token_pool(tokenizer)
    partitions = _partition_code_tokens(pool, seed=_SEED)
    for split, rows in records.items():
        for index, row in enumerate(rows):
            expected = _generate_record(
                tokenizer,
                token_pool=pool,
                split=split,
                index=index,
                seed=_SEED,
                code_token_ids=partitions[(split, index)],
            )
            if row != expected:
                raise ValueError(
                    f"retrieval {split} record {index} failed reconstruction"
                )
    train_codes = {
        value
        for row in records["train"]
        for code in row["passkey_token_ids"]
        for value in code
    }
    development_codes = {
        value
        for row in records["development"]
        for code in row["passkey_token_ids"]
        for value in code
    }
    if (
        train_codes.intersection(development_codes)
        or len(train_codes) != _RECORDS_PER_SPLIT * _ANSWER_POSITIONS
        or len(development_codes) != _RECORDS_PER_SPLIT * _ANSWER_POSITIONS
    ):
        raise ValueError("retrieval train/development code isolation failed")
    return {
        "protocol": protocol,
        "protocol_path": path,
        "protocol_sha256": observed_protocol_sha256,
        "manifest": manifest,
        "package_path": package_path,
        "config_path": config_path,
        "non_mlp_path": non_mlp_path,
        "q7_path": q7_path,
        "tokenizer_path": tokenizer_path,
        "model_path": model_path,
        "model": model,
        "q7_expectations": q7_expectations,
        "library_paths": library_paths,
        "proxy_path": proxy_path,
        "manifest_path": manifest_file,
        "split_paths": split_paths,
        "train_records": records["train"],
        "development_records": records["development"],
        "confirmation_descriptor_authenticated_without_file_access": True,
    }


def _validate_selected_heads(rows: Any) -> list[tuple[int, int]]:
    if not isinstance(rows, list) or len(rows) != _RESCUED_HEADS:
        raise ValueError("retrieval selected-head population is invalid")
    result: list[tuple[int, int]] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError("retrieval selected-head row is invalid")
        layer = row.get("layer")
        head = row.get("head")
        if (
            row.get("rank") != rank
            or isinstance(layer, bool)
            or not isinstance(layer, int)
            or isinstance(head, bool)
            or not isinstance(head, int)
            or not 0 <= layer < _LAYERS
            or not 0 <= head < _HEADS
        ):
            raise ValueError("retrieval selected-head coordinate is invalid")
        result.append((layer, head))
    if len(set(result)) != _RESCUED_HEADS:
        raise ValueError("retrieval selected-head coordinates are not unique")
    return result


def _training_objective_summary(
    records: Any,
) -> dict[str, float]:
    if (
        not isinstance(records, list)
        or len(records) != _RECORDS_PER_SPLIT
        or [row.get("record_index") for row in records]
        != list(range(_RECORDS_PER_SPLIT))
    ):
        raise ValueError("retrieval training objective records are invalid")
    values = np.asarray(
        [row.get("loss", {}).get("answer_cross_entropy") for row in records],
        dtype=np.float64,
    )
    if values.shape != (_RECORDS_PER_SPLIT,) or not np.isfinite(values).all():
        raise ValueError("retrieval training objective is non-finite")
    return {
        "maximum_per_record_answer_cross_entropy": float(values.max()),
        "mean_per_record_answer_cross_entropy": float(values.mean()),
    }


def _select_training_mask(
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(evaluations) != set(_MASK_NAMES):
        raise ValueError("retrieval training mask population is invalid")
    summaries = {
        name: _training_objective_summary(evaluations[name].get("records"))
        for name in _MASK_NAMES
    }
    selected = min(
        ("M1", "M2"),
        key=lambda name: (
            summaries[name]["maximum_per_record_answer_cross_entropy"],
            summaries[name]["mean_per_record_answer_cross_entropy"],
            0 if name == "M1" else 1,
        ),
    )
    baseline = summaries["M0"]
    chosen = summaries[selected]
    baseline_by_record = {
        row["record_index"]: float(row["loss"]["answer_cross_entropy"])
        for row in evaluations["M0"]["records"]
    }
    deltas = [
        {
            "record_index": row["record_index"],
            "selected_minus_M0_answer_cross_entropy": (
                float(row["loss"]["answer_cross_entropy"])
                - baseline_by_record[row["record_index"]]
            ),
            "regressed": (
                float(row["loss"]["answer_cross_entropy"])
                > baseline_by_record[row["record_index"]]
            ),
        }
        for row in evaluations[selected]["records"]
    ]
    eligible = (
        chosen["maximum_per_record_answer_cross_entropy"]
        < baseline["maximum_per_record_answer_cross_entropy"]
        and chosen["mean_per_record_answer_cross_entropy"]
        < baseline["mean_per_record_answer_cross_entropy"]
        and not any(row["regressed"] for row in deltas)
    )
    return {
        "selected_mask_name": selected,
        "summaries": summaries,
        "screen_eligible": eligible,
        "per_record_deltas": deltas,
        "selection_key": [
            chosen["maximum_per_record_answer_cross_entropy"],
            chosen["mean_per_record_answer_cross_entropy"],
            0 if selected == "M1" else 1,
        ],
    }


def _native_forward_surrogate_backward(native: Any, surrogate: Any) -> Any:
    """Return exact native values and route gradients only to the surrogate."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    if (
        not isinstance(native, torch.Tensor)
        or not isinstance(surrogate, torch.Tensor)
        or native.shape != surrogate.shape
        or native.device != surrogate.device
        or native.dtype != surrogate.dtype
        or not native.is_floating_point()
        or not surrogate.is_floating_point()
        or not bool(torch.isfinite(native).all())
        or not bool(torch.isfinite(surrogate).all())
    ):
        raise ValueError("retrieval straight-through tensors are invalid")

    class _NativeValue(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, native_value: Any, surrogate_value: Any) -> Any:
            del ctx, surrogate_value
            return native_value.clone()

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[None, Any]:
            del ctx
            return None, gradient

    return _NativeValue.apply(native, surrogate)


def _answer_cross_entropy(logits: Any, targets: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    if (
        not isinstance(logits, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or logits.ndim != 3
        or logits.shape[:2] != targets.shape
        or logits.shape[1] != _ANSWER_POSITIONS
        or targets.dtype != torch.long
        or logits.device != targets.device
        or not logits.is_floating_point()
        or not bool(torch.isfinite(logits).all())
        or bool((targets < 0).any())
        or bool((targets >= logits.shape[-1]).any())
    ):
        raise ValueError("retrieval answer-loss tensors are invalid")
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        reduction="mean",
    )


def _mask_pairs(mask: np.ndarray) -> list[tuple[int, int]]:
    value = np.asarray(mask)
    if (
        value.shape != (_LAYERS, _HEADS)
        or value.dtype != np.bool_
        or int(value.sum()) not in {0, _RESCUED_HEADS, _LAYERS * _HEADS}
    ):
        raise ValueError("retrieval head mask is invalid")
    return [(int(layer), int(head)) for layer, head in np.argwhere(value)]


def _execute_native_record(
    runtime: Any,
    *,
    record: Mapping[str, Any],
    context: Mapping[str, Any],
    selected_heads: Sequence[tuple[int, int]],
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Execute one record token-at-a-time and copy answer diagnostics only."""

    if runtime.position != 0:
        runtime.reset()
    if runtime.position != 0 or not runtime.attention_metrics_available:
        raise ValueError("retrieval native runtime did not reset cleanly")
    expected = headwise._headwise_expectations(
        context["model"],
        list(selected_heads),
        positions=_PREDICTION_POSITIONS,
    )
    hidden_rows: list[np.ndarray] = []
    logit_rows: list[np.ndarray] = []
    hidden_digest = hashlib.sha256()
    logit_digest = hashlib.sha256()
    final_metrics: dict[str, int] | None = None
    started = time.perf_counter()
    for position, token_id in enumerate(record["input_ids"][:-1]):
        result = runtime.forward([token_id])
        if runtime.position != position + 1:
            raise ValueError("retrieval native cache position did not advance")
        final_metrics = dict(result.metrics)
        if position in record["answer_prediction_positions"]:
            hidden, logits = runtime.last_diagnostics()
            if (
                hidden.shape != (int(context["model"]["hidden_size"]),)
                or logits.shape != (int(context["model"]["vocab_size"]),)
                or hidden.dtype != np.float32
                or logits.dtype != np.float32
                or not np.isfinite(hidden).all()
                or not np.isfinite(logits).all()
                or int(np.argmax(logits)) != result.next_token
            ):
                raise ValueError("retrieval native diagnostics are invalid")
            hidden_rows.append(hidden)
            logit_rows.append(logits)
            hidden_digest.update(hidden.tobytes())
            logit_digest.update(logits.tobytes())
        if progress_label is not None and (position + 1) % 32 == 0:
            _progress(
                f"{progress_label}: native Q7 position "
                f"{position + 1}/{_PREDICTION_POSITIONS}"
            )
    if (
        final_metrics is None
        or len(hidden_rows) != _ANSWER_POSITIONS
        or len(logit_rows) != _ANSWER_POSITIONS
        or runtime.position != _PREDICTION_POSITIONS
    ):
        raise ValueError("retrieval native record execution is incomplete")
    counter_checks = headwise._counter_checks(
        final_metrics,
        expected,
        context["q7_expectations"],
        position=_PREDICTION_POSITIONS,
    )
    if not counter_checks or not all(counter_checks.values()):
        raise ValueError("retrieval native counter contract failed")
    native_hidden = np.stack(hidden_rows).astype(np.float32, copy=False)
    native_logits = np.stack(logit_rows).astype(np.float32, copy=False)
    targets = np.asarray(
        record["input_ids"][_ANSWER_START + 1 :],
        dtype=np.int64,
    )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    with torch.inference_mode():
        loss = float(
            _answer_cross_entropy(
                torch.from_numpy(native_logits).unsqueeze(0),
                torch.from_numpy(targets).unsqueeze(0),
            ).item()
        )
    evidence = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "answer_cross_entropy": loss,
        "answer_positions_copied": _ANSWER_POSITIONS,
        "hidden_sha256": hidden_digest.hexdigest(),
        "logits_sha256": logit_digest.hexdigest(),
        "final_position": runtime.position,
        "final_metrics": final_metrics,
        "counter_checks": counter_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return native_logits, native_hidden, evidence


def _open_native_runtime(
    context: Mapping[str, Any],
    selected_heads: Sequence[tuple[int, int]],
) -> Any:
    return OLMoENativeTokenRuntime(
        context["config_path"],
        context["non_mlp_path"],
        context["q7_path"],
        context["library_paths"]["headwise"],
        threads=_WORKERS,
        attention_head_policies=headwise._head_policies(list(selected_heads)),
    )


def _run_surrogate_record(
    loaded: Any,
    gate_state: MutableMapping[str, Any],
    *,
    record: Mapping[str, Any],
    mask: np.ndarray,
    native_logits: np.ndarray,
) -> dict[str, Any]:
    """Backpropagate exact-native answer CE through the frozen HF surrogate."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    if (
        mask.shape != (_LAYERS, _HEADS)
        or mask.dtype != np.bool_
        or native_logits.shape
        != (
            _ANSWER_POSITIONS,
            int(loaded.config.vocab_size),
        )
    ):
        raise ValueError("retrieval surrogate record request is invalid")
    gates = torch.tensor(
        mask.astype(np.float32),
        dtype=torch.float32,
        device="cpu",
        requires_grad=True,
    )
    gate_state["gates"] = gates
    gate_state["diagnostics"] = []
    tokens = torch.tensor(
        [record["input_ids"][:-1]],
        dtype=torch.long,
        device="cpu",
    )
    targets = torch.tensor(
        [record["input_ids"][_ANSWER_START + 1 :]],
        dtype=torch.long,
        device="cpu",
    )
    positions = torch.tensor(
        record["answer_prediction_positions"],
        dtype=torch.long,
        device="cpu",
    )
    native_tensor = torch.from_numpy(native_logits).unsqueeze(0)
    loaded.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.enable_grad():
        output = loaded(
            input_ids=tokens,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
            logits_to_keep=positions,
        )
        surrogate_logits = output.logits.float()
        if surrogate_logits.shape != native_tensor.shape:
            raise ValueError("retrieval surrogate/native logit shape differs")
        exact_logits = _native_forward_surrogate_backward(
            native_tensor,
            surrogate_logits,
        )
        loss = _answer_cross_entropy(exact_logits, targets)
        loss.backward()
    if gates.grad is None:
        raise ValueError("retrieval surrogate produced no gate gradient")
    gradient = (
        gates.grad.detach()
        .float()
        .cpu()
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )
    if gradient.shape != (_LAYERS, _HEADS) or not np.isfinite(gradient).all():
        raise ValueError("retrieval surrogate gate gradient is invalid")
    diagnostics = list(gate_state["diagnostics"])
    timing = causal_gate._diagnostic_timing_summary(diagnostics)
    result = {
        "record_index": int(record["record_index"]),
        "record_id": record["record_id"],
        "mask_sha256": sha256_json(mask.tolist()),
        "selected_head_count": int(mask.sum()),
        "loss": {"answer_cross_entropy": float(loss.detach().item())},
        "gradient": gradient.tolist(),
        "native_oracle_layers": diagnostics,
        "native_oracle_timing": timing,
        "elapsed_seconds": time.perf_counter() - started,
    }
    del (
        gates,
        tokens,
        targets,
        positions,
        native_tensor,
        output,
        surrogate_logits,
        exact_logits,
        loss,
    )
    gc.collect()
    return result


def _training_mask_step(
    loaded: Any,
    gate_state: MutableMapping[str, Any],
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    mask_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    selected_heads = _mask_pairs(mask)
    rows: list[dict[str, Any]] = []
    gradients: list[np.ndarray] = []
    with _open_native_runtime(context, selected_heads) as runtime:
        for record in records:
            record_label = (
                f"{mask_name} train record "
                f"{int(record['record_index']) + 1}/{_RECORDS_PER_SPLIT}"
            )
            _progress(f"{record_label}: starting native Q7 forward")
            native_logits, _native_hidden, native_evidence = _execute_native_record(
                runtime,
                record=record,
                context=context,
                selected_heads=selected_heads,
                progress_label=record_label,
            )
            _progress(f"{record_label}: starting surrogate backward")
            row = _run_surrogate_record(
                loaded,
                gate_state,
                record=record,
                mask=mask,
                native_logits=native_logits,
            )
            if (
                row["loss"]["answer_cross_entropy"]
                != native_evidence["answer_cross_entropy"]
            ):
                raise ValueError("retrieval straight-through loss changed native value")
            row["native_q7"] = native_evidence
            rows.append(row)
            gradients.append(np.asarray(row["gradient"], dtype=np.float64))
            _progress(f"{record_label}: complete")
    average = np.mean(np.stack(gradients), axis=0, dtype=np.float64)
    if average.shape != (_LAYERS, _HEADS) or not np.isfinite(average).all():
        raise ValueError("retrieval average gate gradient is invalid")
    return rows, average


def _proxy_execution_checks(snapshot: Mapping[str, Any]) -> dict[str, bool]:
    expected_calls = _IHT_STEPS * _RECORDS_PER_SPLIT * _LAYERS
    tasks = snapshot.get("expert_backward_tasks")
    return {
        "workers": snapshot.get("workers") == _WORKERS,
        "patched_layers": snapshot.get("patched_layers") == _LAYERS,
        "restored_layers": snapshot.get("restored_layers") == _LAYERS,
        "serial_forward_calls": (
            snapshot.get("serial_forward_calls") == expected_calls
        ),
        "parallel_backward_calls": (
            snapshot.get("parallel_backward_calls") == expected_calls
        ),
        "expert_backward_tasks": (
            not isinstance(tasks, bool)
            and isinstance(tasks, int)
            and expected_calls <= tasks <= expected_calls * 64
        ),
        "serial_forward_seconds": (
            isinstance(snapshot.get("serial_forward_seconds"), (int, float))
            and bool(np.isfinite(float(snapshot["serial_forward_seconds"])))
            and float(snapshot["serial_forward_seconds"]) > 0.0
        ),
        "parallel_backward_task_seconds": (
            isinstance(
                snapshot.get("parallel_backward_task_seconds"),
                (int, float),
            )
            and bool(np.isfinite(float(snapshot["parallel_backward_task_seconds"])))
            and float(snapshot["parallel_backward_task_seconds"]) > 0.0
        ),
        "ordered_reduction_seconds": (
            isinstance(
                snapshot.get("ordered_reduction_seconds"),
                (int, float),
            )
            and bool(np.isfinite(float(snapshot["ordered_reduction_seconds"])))
            and float(snapshot["ordered_reduction_seconds"]) >= 0.0
        ),
        "context_inactive": snapshot.get("context_active") is False,
        "executor_shutdown": snapshot.get("executor_shutdown") is True,
    }


def _native_only_training_evaluation(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    selected_heads = _mask_pairs(mask)
    rows: list[dict[str, Any]] = []
    with _open_native_runtime(context, selected_heads) as runtime:
        for record in records:
            record_label = (
                "M2 train record "
                f"{int(record['record_index']) + 1}/{_RECORDS_PER_SPLIT}"
            )
            _progress(f"{record_label}: starting native Q7 evaluation")
            _logits, _hidden, evidence = _execute_native_record(
                runtime,
                record=record,
                context=context,
                selected_heads=selected_heads,
                progress_label=record_label,
            )
            rows.append(
                {
                    "record_index": int(record["record_index"]),
                    "record_id": record["record_id"],
                    "mask_sha256": sha256_json(mask.tolist()),
                    "selected_head_count": int(mask.sum()),
                    "loss": {"answer_cross_entropy": evidence["answer_cross_entropy"]},
                    "gradient": None,
                    "native_q7": evidence,
                }
            )
            _progress(f"{record_label}: complete")
    return rows


def _capture_dense_teacher(
    loaded: Any,
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Capture untouched dense-teacher answer rows for development only."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval teacher capture"
        ) from exc
    positions = torch.tensor(
        list(range(_ANSWER_START, _PREDICTION_POSITIONS)),
        dtype=torch.long,
        device="cpu",
    )
    captures: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    logit_digest = hashlib.sha256()
    hidden_digest = hashlib.sha256()
    started = time.perf_counter()
    for record in records:
        _progress(
            "development dense teacher record "
            f"{int(record['record_index']) + 1}/{_RECORDS_PER_SPLIT}"
        )
        tokens = torch.tensor(
            [record["input_ids"][:-1]],
            dtype=torch.long,
            device="cpu",
        )
        with torch.inference_mode():
            output = loaded(
                input_ids=tokens,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
                logits_to_keep=positions,
            )
        logits = output.logits[0].float().cpu().numpy().astype(np.float32, copy=True)
        hidden = (
            output.hidden_states[-1][0, positions]
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=True)
        )
        targets = np.asarray(
            record["input_ids"][_ANSWER_START + 1 :],
            dtype=np.int64,
        )
        if (
            logits.shape != (_ANSWER_POSITIONS, int(loaded.config.vocab_size))
            or hidden.shape != (_ANSWER_POSITIONS, int(loaded.config.hidden_size))
            or targets.shape != (_ANSWER_POSITIONS,)
            or not np.isfinite(logits).all()
            or not np.isfinite(hidden).all()
        ):
            raise ValueError("retrieval dense-teacher capture is invalid")
        logit_digest.update(logits.tobytes())
        hidden_digest.update(hidden.tobytes())
        counterfactual = np.roll(targets, _PASSKEY_TOKENS)
        with torch.inference_mode():
            log_probability = torch.log_softmax(
                torch.from_numpy(logits),
                dim=-1,
            )
            actual = (
                log_probability.gather(
                    1,
                    torch.from_numpy(targets).reshape(-1, 1),
                )
                .reshape(-1)
                .numpy()
            )
            other = (
                log_probability.gather(
                    1,
                    torch.from_numpy(counterfactual).reshape(-1, 1),
                )
                .reshape(-1)
                .numpy()
            )
        for answer_offset, (actual_value, other_value) in enumerate(
            zip(actual, other, strict=True)
        ):
            evidence_rows.append(
                {
                    "record_index": int(record["record_index"]),
                    "answer_offset": answer_offset,
                    "source_depth": record["answer_source_depths"][answer_offset],
                    "actual_minus_counterfactual_log_probability": float(
                        actual_value - other_value
                    ),
                }
            )
        captures.append(
            {
                "record_index": int(record["record_index"]),
                "logits": logits,
                "hidden": hidden,
                "targets": targets,
            }
        )
        del tokens, output
        gc.collect()

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        values = np.asarray(
            [row["actual_minus_counterfactual_log_probability"] for row in rows],
            dtype=np.float64,
        )
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("retrieval teacher evidence is invalid")
        mean = float(values.mean())
        return {
            "answer_positions": int(values.size),
            "mean_actual_minus_counterfactual_log_probability": mean,
            "passed": (mean > _MINIMUM_TEACHER_RETRIEVAL_LOG_PROBABILITY_ADVANTAGE),
        }

    overall = summarize(evidence_rows)
    depths = {
        depth: summarize([row for row in evidence_rows if row["source_depth"] == depth])
        for depth in _SOURCE_DEPTH_NAMES
    }
    evidence = {
        "comparison": (
            "ground-truth answer token versus the token at the same offset "
            "in the preceding cyclic passkey"
        ),
        "overall": overall,
        "source_depths": depths,
        "passed": overall["passed"]
        and all(value["passed"] for value in depths.values()),
        "logits_sha256": logit_digest.hexdigest(),
        "hidden_sha256": hidden_digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return captures, evidence


def _quality_checks(metrics: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "mean_kl": (
            float(metrics["teacher_to_native_kl"]) <= _THRESHOLDS["maximum_mean_kl"]
        ),
        "top1_agreement": (
            float(metrics["teacher_top1_agreement"])
            >= _THRESHOLDS["minimum_top1_agreement"]
        ),
        "target_nll_delta": (
            float(metrics["target_nll_delta"])
            <= _THRESHOLDS["maximum_target_nll_delta"]
        ),
        "hidden_relative_l2": (
            float(metrics["final_hidden_relative_l2"])
            <= _THRESHOLDS["maximum_hidden_relative_l2"]
        ),
    }


def _native_replay_checks(
    replay: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> dict[str, bool]:
    """Compare reset/replay evidence while excluding timing-only counters."""

    replay_metrics = replay.get("final_metrics")
    reference_metrics = reference.get("final_metrics")
    deterministic_metrics_match = (
        isinstance(replay_metrics, dict)
        and isinstance(reference_metrics, dict)
        and sustained._deterministic_metrics(replay_metrics)
        == sustained._deterministic_metrics(reference_metrics)
    )
    return {
        "hidden_sha256": replay.get("hidden_sha256") == reference.get("hidden_sha256"),
        "logits_sha256": replay.get("logits_sha256") == reference.get("logits_sha256"),
        "deterministic_final_metrics": deterministic_metrics_match,
        "answer_cross_entropy": replay.get("answer_cross_entropy")
        == reference.get("answer_cross_entropy"),
    }


def _evaluate_native_development(
    *,
    context: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    teacher: Sequence[Mapping[str, Any]],
    selected_heads: Sequence[tuple[int, int]],
    role: str,
) -> dict[str, Any]:
    if (
        len(records) != _RECORDS_PER_SPLIT
        or len(teacher) != _RECORDS_PER_SPLIT
        or [row["record_index"] for row in teacher] != list(range(_RECORDS_PER_SPLIT))
    ):
        raise ValueError("retrieval development population is invalid")
    expectations = headwise._headwise_expectations(
        context["model"],
        list(selected_heads),
        positions=_PREDICTION_POSITIONS,
    )
    rows: list[dict[str, Any]] = []
    sequence_evidence: list[dict[str, Any]] = []
    replay_reference: dict[str, Any] | None = None
    started = time.perf_counter()
    with _open_native_runtime(context, selected_heads) as runtime:
        for record, reference in zip(records, teacher, strict=True):
            record_label = (
                f"{role} development record "
                f"{int(record['record_index']) + 1}/{_RECORDS_PER_SPLIT}"
            )
            _progress(f"{record_label}: starting")
            native_logits, native_hidden, evidence = _execute_native_record(
                runtime,
                record=record,
                context=context,
                selected_heads=selected_heads,
                progress_label=record_label,
            )
            if reference["record_index"] != record["record_index"]:
                raise ValueError("retrieval teacher/native record order differs")
            for answer_offset, position in enumerate(
                record["answer_prediction_positions"]
            ):
                metric = _position_metrics(
                    reference["logits"][answer_offset],
                    native_logits[answer_offset],
                    reference["hidden"][answer_offset],
                    native_hidden[answer_offset],
                    int(reference["targets"][answer_offset]),
                )
                rows.append(
                    {
                        "record_index": int(record["record_index"]),
                        "position": int(position),
                        "answer_offset": answer_offset,
                        "source_depth": record["answer_source_depths"][answer_offset],
                        **metric,
                    }
                )
            sequence_evidence.append(evidence)
            if replay_reference is None:
                replay_reference = evidence
            _progress(f"{record_label}: complete")
        if replay_reference is None:
            raise ValueError("retrieval development replay reference is missing")
        replay_logits, replay_hidden, replay = _execute_native_record(
            runtime,
            record=records[0],
            context=context,
            selected_heads=selected_heads,
            progress_label=f"{role} reset replay",
        )
        del replay_logits, replay_hidden
    replay_checks = _native_replay_checks(replay, replay_reference)
    if not all(replay_checks.values()):
        raise ValueError("retrieval native reset/replay parity failed")
    overall = layer_rescue._aggregate(rows)
    depth_metrics = {
        depth: layer_rescue._aggregate(
            [row for row in rows if row["source_depth"] == depth]
        )
        for depth in _SOURCE_DEPTH_NAMES
    }
    checks = {
        "overall": _quality_checks(overall),
        "source_depths": {
            depth: _quality_checks(value) for depth, value in depth_metrics.items()
        },
    }
    passed = all(checks["overall"].values()) and all(
        all(value.values()) for value in checks["source_depths"].values()
    )
    return {
        "role": role,
        "selected_head_count": len(selected_heads),
        "selected_heads": [
            {"layer": layer, "head": head} for layer, head in selected_heads
        ],
        "attention_expectations_per_sequence": expectations,
        "overall_answer_positions": overall,
        "source_depths": depth_metrics,
        "quality_checks": checks,
        "quality_passed": passed,
        "position_rows": rows,
        "sequence_evidence": sequence_evidence,
        "reset_replay_checks": replay_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _fit_post_authentication(
    context: Mapping[str, Any],
) -> dict[str, bool]:
    protocol = context["protocol"]
    source = protocol["source_model"]
    proxy_contract = protocol["expert_proxy_qualifier"]
    checks = {
        "protocol": (
            sha256_file(context["protocol_path"]) == context["protocol_sha256"]
        ),
        "package": (
            validate_olmoe_native_package(
                context["package_path"],
                expected_manifest_sha256=protocol["package"]["manifest_sha256"],
            )
            == context["manifest"]
        ),
        "corpus_manifest": (
            sha256_file(context["manifest_path"])
            == protocol["corpus"]["manifest_sha256"]
        ),
        "train_split": (
            sha256_file(context["split_paths"]["train"])
            == protocol["corpus"]["splits"]["train"]["sha256"]
        ),
        "development_split": (
            sha256_file(context["split_paths"]["development"])
            == protocol["corpus"]["splits"]["development"]["sha256"]
        ),
        "confirmation_not_opened": context[
            "confirmation_descriptor_authenticated_without_file_access"
        ],
        "source_inventory": protocol["source_sha256"] == _source_inventory(),
        "framework_contract": (
            protocol["framework_contract"] == causal_gate._framework_contract()
        ),
        "source_config": (
            sha256_file(context["model_path"] / "config.json")
            == source["config_sha256"]
        ),
        "source_index": (
            sha256_file(context["model_path"] / "model.safetensors.index.json")
            == source["index_sha256"]
        ),
        "source_shards": (
            _source_shard_inventory(context["model_path"]) == source["shard_sha256"]
        ),
        "proxy_qualifier": (
            sha256_file(context["proxy_path"]) == proxy_contract["sha256"]
        ),
        "transformers_expert_dispatch": (
            proxy_record._transformers_expert_dispatch_contract()
            == proxy_contract["transformers_expert_dispatch_contract"]
        ),
    }
    for name, path in context["library_paths"].items():
        checks[f"{name}_library"] = (
            sha256_file(path) == protocol["libraries"][name]["sha256"]
        )
    return checks


def _training_checkpoint_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.training_checkpoint.json")


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
        if raw.dtype == np.bool_:
            raise ValueError(f"{label} matrix is invalid")
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} matrix is invalid") from exc
    if matrix.shape != (_LAYERS, _HEADS) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} matrix is invalid")
    return matrix


def _boolean_mask(value: Any, label: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != _LAYERS
        or any(
            not isinstance(row, list)
            or len(row) != _HEADS
            or any(not isinstance(item, bool) for item in row)
            for row in value
        )
    ):
        raise ValueError(f"{label} mask is invalid")
    return np.asarray(value, dtype=np.bool_)


def _validate_checkpoint_training_rows(
    *,
    name: str,
    rows: Any,
    expected_records: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
) -> list[np.ndarray]:
    if (
        not isinstance(rows, list)
        or len(rows) != _RECORDS_PER_SPLIT
        or len(expected_records) != _RECORDS_PER_SPLIT
        or any(not isinstance(row, Mapping) for row in rows)
        or [row.get("record_index") for row in rows] != list(range(_RECORDS_PER_SPLIT))
    ):
        raise ValueError(f"retrieval checkpoint {name} records are invalid")
    expected_mask_sha256 = sha256_json(mask.tolist())
    gradients: list[np.ndarray] = []
    for row, expected in zip(rows, expected_records, strict=True):
        loss = row.get("loss")
        native = row.get("native_q7")
        cross_entropy = (
            loss.get("answer_cross_entropy") if isinstance(loss, Mapping) else None
        )
        if (
            not isinstance(row, Mapping)
            or row.get("record_index") != expected["record_index"]
            or row.get("record_id") != expected["record_id"]
            or row.get("mask_sha256") != expected_mask_sha256
            or row.get("selected_head_count") != int(mask.sum())
            or isinstance(cross_entropy, bool)
            or not isinstance(cross_entropy, (int, float))
            or not np.isfinite(float(cross_entropy))
            or not isinstance(native, Mapping)
            or native.get("record_index") != expected["record_index"]
            or native.get("record_id") != expected["record_id"]
            or native.get("answer_cross_entropy") != cross_entropy
            or native.get("answer_positions_copied") != _ANSWER_POSITIONS
            or native.get("final_position") != _PREDICTION_POSITIONS
            or not _is_sha256(native.get("hidden_sha256"))
            or not _is_sha256(native.get("logits_sha256"))
            or not isinstance(native.get("counter_checks"), Mapping)
            or not native["counter_checks"]
            or not all(value is True for value in native["counter_checks"].values())
        ):
            raise ValueError(f"retrieval checkpoint {name} record evidence is invalid")
        gradient = row.get("gradient")
        if name == "M2":
            if gradient is not None:
                raise ValueError("retrieval checkpoint M2 must not contain gradients")
        else:
            gradients.append(
                _finite_matrix(
                    gradient,
                    f"retrieval checkpoint {name} record gradient",
                )
            )
    return gradients


def _validate_training_payload(
    training: Any,
    *,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    expected_training_keys = {
        "masks",
        "selection",
        "selected_heads",
        "expert_proxy",
        "expert_proxy_checks",
        "model_parameters_frozen",
        "answer_positions_only",
        "elapsed_seconds",
    }
    if (
        not isinstance(training, dict)
        or set(training) != expected_training_keys
        or training.get("model_parameters_frozen") is not True
        or training.get("answer_positions_only") is not True
        or isinstance(training.get("elapsed_seconds"), bool)
        or not isinstance(training.get("elapsed_seconds"), (int, float))
        or not np.isfinite(float(training["elapsed_seconds"]))
        or float(training["elapsed_seconds"]) <= 0.0
    ):
        raise ValueError("retrieval checkpoint training payload is invalid")
    entries = training.get("masks")
    if not isinstance(entries, dict) or set(entries) != set(_MASK_NAMES):
        raise ValueError("retrieval checkpoint mask population is invalid")
    expected_entry_keys = {
        "mask",
        "mask_sha256",
        "selected_head_count",
        "selected_heads",
        "projected_scores",
        "average_gradient",
        "gradient_rms",
        "records",
    }
    masks: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    gradients: dict[str, np.ndarray] = {}
    for name in _MASK_NAMES:
        entry = entries[name]
        if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
            raise ValueError(f"retrieval checkpoint {name} mask entry is invalid")
        mask = _boolean_mask(
            entry["mask"],
            f"retrieval checkpoint {name}",
        )
        expected_count = 0 if name == "M0" else _RESCUED_HEADS
        if (
            int(mask.sum()) != expected_count
            or entry.get("selected_head_count") != expected_count
            or entry.get("mask_sha256") != sha256_json(mask.tolist())
        ):
            raise ValueError(f"retrieval checkpoint {name} mask contract failed")
        masks[name] = mask
        if name == "M0":
            if (
                entry.get("selected_heads") != []
                or entry.get("projected_scores") is not None
            ):
                raise ValueError("retrieval checkpoint M0 projection is invalid")
        else:
            scores[name] = _finite_matrix(
                entry.get("projected_scores"),
                f"retrieval checkpoint {name} projected scores",
            )
            expected_rows = causal_gate._selected_head_rows(
                mask,
                scores[name],
            )
            if entry.get("selected_heads") != expected_rows:
                raise ValueError(f"retrieval checkpoint {name} ranking is invalid")
        if name == "M2":
            if (
                entry.get("average_gradient") is not None
                or entry.get("gradient_rms") is not None
            ):
                raise ValueError("retrieval checkpoint M2 gradient state is invalid")
        else:
            gradients[name] = _finite_matrix(
                entry.get("average_gradient"),
                f"retrieval checkpoint {name} average gradient",
            )
            rms = entry.get("gradient_rms")
            if (
                isinstance(rms, bool)
                or not isinstance(rms, (int, float))
                or not np.isfinite(float(rms))
                or float(rms) <= 0.0
            ):
                raise ValueError(f"retrieval checkpoint {name} gradient RMS is invalid")
        row_gradients = _validate_checkpoint_training_rows(
            name=name,
            rows=entry.get("records"),
            expected_records=context["train_records"],
            mask=mask,
        )
        if name != "M2":
            average = np.mean(
                np.stack(row_gradients),
                axis=0,
                dtype=np.float64,
            )
            if not np.array_equal(average, gradients[name]):
                raise ValueError(
                    f"retrieval checkpoint {name} average gradient changed"
                )
    m1_scores, m1_mask, m0_rms = causal_gate._projected_gate_step(
        masks["M0"],
        gradients["M0"],
    )
    m2_scores, m2_mask, m1_rms = causal_gate._projected_gate_step(
        masks["M1"],
        gradients["M1"],
    )
    if (
        not np.array_equal(m1_scores, scores["M1"])
        or not np.array_equal(m1_mask, masks["M1"])
        or m0_rms != entries["M0"]["gradient_rms"]
        or not np.array_equal(m2_scores, scores["M2"])
        or not np.array_equal(m2_mask, masks["M2"])
        or m1_rms != entries["M1"]["gradient_rms"]
    ):
        raise ValueError("retrieval checkpoint IHT projection chain changed")
    evaluations = {name: {"records": entries[name]["records"]} for name in _MASK_NAMES}
    selection = _select_training_mask(evaluations)
    if training.get("selection") != selection:
        raise ValueError("retrieval checkpoint training selection changed")
    selected_name = selection["selected_mask_name"]
    selected_rows = causal_gate._selected_head_rows(
        masks[selected_name],
        scores[selected_name],
    )
    if training.get("selected_heads") != selected_rows:
        raise ValueError("retrieval checkpoint selected-head ranking changed")
    selected_heads = _validate_selected_heads(selected_rows)
    proxy = training.get("expert_proxy")
    stored_proxy_checks = training.get("expert_proxy_checks")
    if not isinstance(proxy, Mapping):
        raise ValueError("retrieval checkpoint expert proxy is invalid")
    proxy_checks = _proxy_execution_checks(proxy)
    if (
        stored_proxy_checks != proxy_checks
        or not proxy_checks
        or not all(proxy_checks.values())
    ):
        raise ValueError("retrieval checkpoint expert proxy contract failed")
    return selection, selected_heads


def _write_training_checkpoint(
    path: Path,
    *,
    context: Mapping[str, Any],
    training: dict[str, Any],
    post_training_authentication: dict[str, bool],
) -> dict[str, str]:
    target = _new_output(path, "retrieval training checkpoint")
    if not post_training_authentication or not all(
        post_training_authentication.values()
    ):
        raise ValueError("retrieval checkpoint post-training authentication failed")
    _validate_training_payload(training, context=context)
    checkpoint = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _TRAINING_CHECKPOINT_EXPERIMENT,
        "status": _TRAINING_CHECKPOINT_STATUS,
        "protocol": {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "source_sha256": context["protocol"]["source_sha256"],
        "train_record_identity_sha256": sha256_json(
            [row["identity_sha256"] for row in context["train_records"]]
        ),
        "training": training,
        "training_sha256": sha256_json(training),
        "post_training_authentication": post_training_authentication,
        "confirmation_split_opened": False,
    }
    atomic_json(target, checkpoint)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "mode": "created",
    }


def _load_training_checkpoint(
    path: str | Path,
    expected_sha256: str,
    *,
    context: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[tuple[int, int]],
    dict[str, str],
]:
    requested = Path(path).expanduser()
    if requested.is_symlink():
        raise ValueError("retrieval training checkpoint authentication failed")
    source = requested.resolve()
    if (
        not source.is_file()
        or not _is_sha256(expected_sha256)
        or sha256_file(source) != expected_sha256.lower()
    ):
        raise ValueError("retrieval training checkpoint authentication failed")
    checkpoint = _read_json(source, "retrieval training checkpoint")
    expected_keys = {
        "schema_version",
        "experiment",
        "status",
        "protocol",
        "source_sha256",
        "train_record_identity_sha256",
        "training",
        "training_sha256",
        "post_training_authentication",
        "confirmation_split_opened",
    }
    expected_identity = sha256_json(
        [row["identity_sha256"] for row in context["train_records"]]
    )
    if (
        set(checkpoint) != expected_keys
        or checkpoint.get("schema_version") != _SCHEMA_VERSION
        or checkpoint.get("experiment") != _TRAINING_CHECKPOINT_EXPERIMENT
        or checkpoint.get("status") != _TRAINING_CHECKPOINT_STATUS
        or checkpoint.get("protocol")
        != {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        }
        or checkpoint.get("source_sha256") != context["protocol"]["source_sha256"]
        or checkpoint.get("train_record_identity_sha256") != expected_identity
        or checkpoint.get("confirmation_split_opened") is not False
        or checkpoint.get("training_sha256") != sha256_json(checkpoint.get("training"))
    ):
        raise ValueError("retrieval training checkpoint contract changed")
    current_authentication = _fit_post_authentication(context)
    if (
        checkpoint.get("post_training_authentication") != current_authentication
        or not current_authentication
        or not all(current_authentication.values())
    ):
        raise ValueError("retrieval checkpoint post-training authentication changed")
    training = checkpoint.get("training")
    selection, selected_heads = _validate_training_payload(
        training,
        context=context,
    )
    return (
        training,
        selection,
        selected_heads,
        {
            "path": str(source),
            "sha256": expected_sha256.lower(),
            "mode": "resumed",
        },
    )


def _load_frozen_surrogate(context: Mapping[str, Any]) -> Any:
    try:
        import torch

        _prepare_transformers_imports()
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "install engram-lm[conversion] for retrieval selector fitting"
        ) from exc
    torch.set_num_threads(_WORKERS)
    torch.use_deterministic_algorithms(True)
    _progress("loading frozen BF16 surrogate model on CPU")
    loaded = AutoModelForCausalLM.from_pretrained(
        context["model_path"],
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    loaded.to("cpu")
    loaded.requires_grad_(False)
    return loaded


def _fit_training_selector(
    loaded: Any,
    *,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[tuple[int, int]]]:
    gate_state: MutableMapping[str, Any] = {
        "attention_library": context["library_paths"]["attention"],
        "gates": None,
        "diagnostics": [],
    }
    masks: dict[str, np.ndarray] = {"M0": np.zeros((_LAYERS, _HEADS), dtype=np.bool_)}
    scores: dict[str, np.ndarray] = {
        "M0": np.zeros((_LAYERS, _HEADS), dtype=np.float64)
    }
    gradients: dict[str, np.ndarray] = {}
    gradient_rms: dict[str, float] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    originals: list[tuple[Any, Any]] = []
    proxy_stats: Any = None
    started = time.perf_counter()
    try:
        originals = causal_gate._install_causal_gate_attention(
            loaded,
            gate_state,
        )
        with frozen_olmoe_expert_backward_proxy(
            loaded,
            workers=_WORKERS,
        ) as proxy_stats:
            m0_rows, m0_gradient = _training_mask_step(
                loaded,
                gate_state,
                context=context,
                records=context["train_records"],
                mask=masks["M0"],
                mask_name="M0",
            )
            gradients["M0"] = m0_gradient
            m1_scores, m1_mask, m0_rms = causal_gate._projected_gate_step(
                masks["M0"],
                m0_gradient,
            )
            scores["M1"] = m1_scores
            masks["M1"] = m1_mask
            gradient_rms["M0"] = m0_rms
            evaluations["M0"] = {"records": m0_rows}

            m1_rows, m1_gradient = _training_mask_step(
                loaded,
                gate_state,
                context=context,
                records=context["train_records"],
                mask=masks["M1"],
                mask_name="M1",
            )
            gradients["M1"] = m1_gradient
            m2_scores, m2_mask, m1_rms = causal_gate._projected_gate_step(
                masks["M1"],
                m1_gradient,
            )
            scores["M2"] = m2_scores
            masks["M2"] = m2_mask
            gradient_rms["M1"] = m1_rms
            evaluations["M1"] = {"records": m1_rows}
    finally:
        if originals:
            causal_gate._restore_causal_gate_attention(originals)
    if proxy_stats is None:
        raise ValueError("retrieval expert proxy did not initialize")
    proxy_snapshot = proxy_stats.snapshot()
    proxy_checks = _proxy_execution_checks(proxy_snapshot)
    if not all(proxy_checks.values()):
        raise ValueError("retrieval expert proxy execution contract failed")
    if any(parameter.requires_grad for parameter in loaded.parameters()) or any(
        parameter.grad is not None for parameter in loaded.parameters()
    ):
        raise ValueError("retrieval fit modified or accumulated teacher weights")
    evaluations["M2"] = {
        "records": _native_only_training_evaluation(
            context=context,
            records=context["train_records"],
            mask=masks["M2"],
        )
    }
    selection = _select_training_mask(evaluations)
    _progress(
        "training selection complete: "
        f"{selection['selected_mask_name']}, "
        f"eligible={selection['screen_eligible']}"
    )
    selected_name = selection["selected_mask_name"]
    selected_rows = causal_gate._selected_head_rows(
        masks[selected_name],
        scores[selected_name],
    )
    selected_heads = _validate_selected_heads(selected_rows)
    training = {
        "masks": {
            name: {
                "mask": masks[name].tolist(),
                "mask_sha256": sha256_json(masks[name].tolist()),
                "selected_head_count": int(masks[name].sum()),
                "selected_heads": (
                    []
                    if name == "M0"
                    else causal_gate._selected_head_rows(
                        masks[name],
                        scores[name],
                    )
                ),
                "projected_scores": (None if name == "M0" else scores[name].tolist()),
                "average_gradient": (
                    gradients[name].tolist() if name in gradients else None
                ),
                "gradient_rms": gradient_rms.get(name),
                "records": evaluations[name]["records"],
            }
            for name in _MASK_NAMES
        },
        "selection": selection,
        "selected_heads": selected_rows,
        "expert_proxy": proxy_snapshot,
        "expert_proxy_checks": proxy_checks,
        "model_parameters_frozen": True,
        "answer_positions_only": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    validated_selection, validated_heads = _validate_training_payload(
        training,
        context=context,
    )
    if validated_selection != selection or validated_heads != selected_heads:
        raise ValueError("retrieval training payload self-validation failed")
    return training, selection, selected_heads


def fit_and_screen_retrieval_head_selector(
    *,
    protocol: str | Path,
    protocol_sha256: str,
    out: str | Path,
    resume_training_checkpoint: str | Path | None = None,
    resume_training_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit two exact-51 IHT masks, then screen one on development Q7."""

    output = _new_output(out, "retrieval selector development result")
    resume_requested = (
        resume_training_checkpoint is not None
        or resume_training_checkpoint_sha256 is not None
    )
    if resume_requested and (
        resume_training_checkpoint is None or resume_training_checkpoint_sha256 is None
    ):
        raise ValueError(
            "retrieval checkpoint path and SHA-256 must be supplied together"
        )
    if resume_requested:
        checkpoint_path = Path(resume_training_checkpoint).expanduser().resolve()
    else:
        checkpoint_path = _training_checkpoint_path(output)
        _new_output(checkpoint_path, "retrieval training checkpoint")
    _progress("authenticating frozen protocol and train/development inputs")
    context = _authenticate_fit_screen(protocol, protocol_sha256)
    _progress("protocol authenticated; confirmation remains unopened")
    started = time.perf_counter()
    loaded: Any | None = None
    if resume_requested:
        _progress("authenticating completed training checkpoint for resume")
        (
            training,
            selection,
            selected_heads,
            checkpoint,
        ) = _load_training_checkpoint(
            checkpoint_path,
            resume_training_checkpoint_sha256,
            context=context,
        )
        _progress(
            "training checkpoint authenticated; "
            "M0/M1 backward and M2 training evaluation skipped"
        )
    else:
        loaded = _load_frozen_surrogate(context)
        training, selection, selected_heads = _fit_training_selector(
            loaded,
            context=context,
        )
        post_training_authentication = _fit_post_authentication(context)
        checkpoint = _write_training_checkpoint(
            checkpoint_path,
            context=context,
            training=training,
            post_training_authentication=post_training_authentication,
        )
        (
            serialized_training,
            serialized_selection,
            serialized_heads,
            _resume_descriptor,
        ) = _load_training_checkpoint(
            checkpoint["path"],
            checkpoint["sha256"],
            context=context,
        )
        if (
            serialized_training != training
            or serialized_selection != selection
            or serialized_heads != selected_heads
        ):
            raise ValueError("serialized retrieval training checkpoint changed")
        training = serialized_training
        selection = serialized_selection
        selected_heads = serialized_heads
        _progress(
            "training checkpoint written before development: "
            f"{checkpoint['path']} ({checkpoint['sha256']})"
        )
    selected_name = selection["selected_mask_name"]
    teacher_capture: list[dict[str, Any]] | None = None
    teacher_evidence: dict[str, Any] | None = None
    development: dict[str, Any] | None = None
    decision: dict[str, Any]
    if not selection["screen_eligible"]:
        decision = {
            "status": "training_selector_failed",
            "passed": False,
            "confirmation_authorized": False,
            "next_step": (
                "stop this static retrieval selector without opening "
                "confirmation; investigate prefix-conditioned allocation"
            ),
        }
    else:
        if loaded is None:
            loaded = _load_frozen_surrogate(context)
        teacher_capture, teacher_evidence = _capture_dense_teacher(
            loaded,
            context["development_records"],
        )
        if not teacher_evidence["passed"]:
            decision = {
                "status": "teacher_retrieval_evidence_failed",
                "passed": False,
                "confirmation_authorized": False,
                "next_step": (
                    "stop without opening confirmation; redesign the synthetic "
                    "retrieval task or use a teacher that demonstrates retrieval"
                ),
            }
        else:
            del loaded
            loaded = None
            gc.collect()
            full_heads = [
                (layer, head) for layer in range(_LAYERS) for head in range(_HEADS)
            ]
            full_control = _evaluate_native_development(
                context=context,
                records=context["development_records"],
                teacher=teacher_capture,
                selected_heads=full_heads,
                role="full_W128_Q7_control",
            )
            candidate = _evaluate_native_development(
                context=context,
                records=context["development_records"],
                teacher=teacher_capture,
                selected_heads=selected_heads,
                role=f"{selected_name}_exact_51_head_candidate",
            )
            resource_checks = {
                "exactly_51_heads": len(selected_heads) == _RESCUED_HEADS,
                "logical_read_bytes": (
                    candidate["attention_expectations_per_sequence"][
                        "attention_logical_read_bytes"
                    ]
                    == _EXPECTED_ATTENTION_LOGICAL_READ_BYTES
                ),
                "logical_read_fraction": (
                    candidate["attention_expectations_per_sequence"][
                        "attention_logical_read_fraction"
                    ]
                    == _EXPECTED_ATTENTION_READ_FRACTION
                ),
                "state_bytes": (
                    candidate["attention_expectations_per_sequence"][
                        "attention_state_bytes"
                    ]
                    == _EXPECTED_ATTENTION_STATE_BYTES
                ),
                "q7_fraction": (
                    context["protocol"]["budget"]["q7_traffic_contract_per_sequence"][
                        "q7_fraction_of_all_expert_ideal_q4"
                    ]
                    == _EXPECTED_Q7_FRACTION
                ),
                "52_heads_inadmissible": (
                    context["protocol"]["budget"]["next_head_boundary"]["within_budget"]
                    is False
                ),
            }
            development_passed = (
                full_control["quality_passed"]
                and candidate["quality_passed"]
                and all(resource_checks.values())
            )
            development = {
                "selected_mask_name": selected_name,
                "full_control": full_control,
                "candidate": candidate,
                "resource_checks": resource_checks,
                "passed": development_passed,
            }
            decision = {
                "status": (
                    "development_passed"
                    if development_passed
                    else "development_semantic_gate_failed"
                ),
                "passed": development_passed,
                "confirmation_authorized": development_passed,
                "next_step": (
                    "freeze a separate one-shot confirmation command and mask"
                    if development_passed
                    else (
                        "stop this static retrieval selector without opening "
                        "confirmation; investigate prefix-conditioned allocation"
                    )
                ),
            }
    if loaded is not None:
        del loaded
        gc.collect()
    post_authentication = _fit_post_authentication(context)
    if not post_authentication or not all(post_authentication.values()):
        raise ValueError("retrieval selector post-run authentication failed")
    report = {
        "schema_version": _SCHEMA_VERSION,
        "experiment": _RESULT_EXPERIMENT,
        "status": decision["status"],
        "protocol": {
            "path": str(context["protocol_path"]),
            "sha256": context["protocol_sha256"],
        },
        "training_checkpoint": checkpoint,
        "training": training,
        "teacher_retrieval_evidence": teacher_evidence,
        "development": development,
        "decision": decision,
        "post_run_authentication": post_authentication,
        "confirmation_split_opened": False,
        "total_elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output, report)
    _progress(f"result written to {output}")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prospective retrieval-targeted OLMoE head selector",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze",
        help="generate sealed 8/8/8 passkey splits and freeze the protocol",
    )
    freeze.add_argument("--package", required=True)
    freeze.add_argument("--manifest-sha256", required=True)
    freeze.add_argument("--layered-library", required=True)
    freeze.add_argument("--headwise-library", required=True)
    freeze.add_argument("--attention-library", required=True)
    freeze.add_argument("--proxy-qualifier", required=True)
    freeze.add_argument("--out", required=True)
    freeze.add_argument("--train-records", type=int, default=_RECORDS_PER_SPLIT)
    freeze.add_argument(
        "--development-records",
        type=int,
        default=_RECORDS_PER_SPLIT,
    )
    freeze.add_argument(
        "--confirmation-records",
        type=int,
        default=_RECORDS_PER_SPLIT,
    )
    freeze.add_argument("--tokens", type=int, default=_PREDICTION_POSITIONS)
    freeze.add_argument("--answer-tokens", type=int, default=_ANSWER_POSITIONS)
    freeze.add_argument("--seed", type=int, default=_SEED)
    freeze.add_argument("--workers", type=int, default=_WORKERS)
    fit = commands.add_parser(
        "fit-screen",
        help="fit on train and screen packaged Q7 on development",
    )
    fit.add_argument("--protocol", required=True)
    fit.add_argument("--protocol-sha256", required=True)
    fit.add_argument("--out", required=True)
    fit.add_argument("--resume-training-checkpoint")
    fit.add_argument("--resume-training-checkpoint-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        value = freeze_retrieval_head_selector_protocol(
            package=args.package,
            manifest_sha256=args.manifest_sha256,
            layered_library=args.layered_library,
            headwise_library=args.headwise_library,
            attention_library=args.attention_library,
            proxy_qualifier=args.proxy_qualifier,
            out=args.out,
            train_records=args.train_records,
            development_records=args.development_records,
            confirmation_records=args.confirmation_records,
            tokens=args.tokens,
            answer_tokens=args.answer_tokens,
            seed=args.seed,
            workers=args.workers,
        )
    elif args.command == "fit-screen":
        value = fit_and_screen_retrieval_head_selector(
            protocol=args.protocol,
            protocol_sha256=args.protocol_sha256,
            out=args.out,
            resume_training_checkpoint=args.resume_training_checkpoint,
            resume_training_checkpoint_sha256=(args.resume_training_checkpoint_sha256),
        )
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError("unknown retrieval selector command")
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
