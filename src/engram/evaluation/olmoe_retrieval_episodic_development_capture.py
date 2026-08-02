"""Capture independent-development query/key features from the native runtime.

The train artifacts used by the selector were captured before the selector was
chosen.  This module captures only the independent development split, using
the same CPU native shadow route.  It records pre-RoPE normalized inputs,
post-RoPE candidate keys, and candidate Q/K partial bands; no selector is fit
to these records.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from engram.evaluation import olmoe_retrieval_episodic_oracle as oracle
from engram.runtime.olmoe_native import OLMoENativeTokenRuntime
from engram.utils import atomic_json, sha256_file


_ANSWER_START = 96
_READS = 32
_RECORDS = 8
_LAYERS = 16
_HEADS = 16
_CANDIDATES = 8
_HEAD_DIMENSION = 128
_BANDS = 8


def _progress(message: str) -> None:
    print(f"[development-capture] {message}", file=sys.stderr, flush=True)


def _records(path: str | Path, digest: str) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if sha256_file(source) != digest.lower():
        raise ValueError("development capture record digest changed")
    value = oracle.retrieval._read_split(source, split="development")
    if len(value) != _RECORDS:
        raise ValueError("development capture record count changed")
    return value


def _runtime(
    *,
    package: Path,
    library: Path,
    trace_kind: str,
    threads: int,
) -> OLMoENativeTokenRuntime:
    if trace_kind not in {"keys", "qk"}:
        raise ValueError("development capture trace kind is invalid")
    return OLMoENativeTokenRuntime(
        str(package / "model/config.json"),
        str(package / "transformer/non_mlp.safetensors"),
        str(package / "mlp/experts.q7"),
        str(library),
        threads=threads,
        local_window=16,
        older_candidates=8,
        older_top_k=4,
        sink_tokens=2,
        episodic_policy={"slots": 32, "span_size": 8},
        episodic_head_mask=np.ones((_LAYERS, _HEADS), dtype=bool),
        episodic_logit_bias=0.0,
        shadow_attention_policy={
            # The shadow is the full-visible reference.  Episodic source
            # payloads are deliberately far outside the production W=16
            # window, so the shadow ledger must retain the full 128-position
            # sequence while the primary route remains W16/C8/K4.
            "local_window": 128,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
        c28_qk_candidate_key_trace=trace_kind == "keys",
        c28_qk_candidate_trace=trace_kind == "qk",
    )


def _capture_kind(
    records: Sequence[dict[str, Any]],
    *,
    package: Path,
    library: Path,
    anchors: dict[str, tuple[int, ...]],
    trace_kind: str,
    threads: int,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    runtime = _runtime(
        package=package,
        library=library,
        trace_kind=trace_kind,
        threads=threads,
    )
    input_norm = np.empty((_RECORDS, _READS, _LAYERS, _HEADS * _HEAD_DIMENSION), dtype=np.float32)
    trace = (
        np.empty((_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _HEAD_DIMENSION), dtype=np.float32)
        if trace_kind == "keys"
        else np.empty((_RECORDS, _READS, _LAYERS, _HEADS, _CANDIDATES, _BANDS), dtype=np.float32)
    )
    schedules = [oracle._derive_schedule(record["input_ids"], anchors) for record in records]
    ids: list[str] = []
    try:
        if not runtime.shadow_trace_available:
            raise ValueError("development capture shadow trace is unavailable")
        for index, (record, schedule) in enumerate(zip(records, schedules, strict=True)):
            runtime.reset()
            _progress(f"{trace_kind} record {index + 1}/{_RECORDS}")
            ids.append(str(record["record_id"]))
            for position, token_id in enumerate(record["input_ids"][:-1]):
                row = schedule["rows"][position]
                runtime.forward_episodic(
                    [int(token_id)],
                    [int(row["write_slot"])],
                    [int(row["read_span"])],
                )
                if position < _ANSWER_START:
                    continue
                read = position - _ANSWER_START
                input_norm[index, read], _, _ = runtime.last_shadow_trace()
                if trace_kind == "keys":
                    trace[index, read] = runtime.last_c28_qk_candidate_key_trace().candidate_keys
                else:
                    trace[index, read] = runtime.last_c28_qk_candidate_trace().qk_candidates
    finally:
        runtime.close()
    return np.ascontiguousarray(input_norm), np.ascontiguousarray(trace), ids


def _save_tensor(path: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    try:
        from safetensors.numpy import save_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("development capture requires safetensors") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: np.ascontiguousarray(value)}, str(path))
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "tensor_sha256": __import__("hashlib").sha256(value.tobytes(order="C")).hexdigest(),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def capture_development(
    *,
    records_path: str | Path,
    records_sha256: str,
    package_path: str | Path,
    native_library: str | Path,
    out: str | Path,
    threads: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    records_source = Path(records_path).expanduser().resolve()
    package = Path(package_path).expanduser().resolve()
    library = Path(native_library).expanduser().resolve()
    records = _records(records_source, records_sha256)
    anchors = oracle._fact_anchor_ids(package / "tokenizer/tokenizer.json")
    input_norm, keys, ids = _capture_kind(
        records,
        package=package,
        library=library,
        anchors=anchors,
        trace_kind="keys",
        threads=threads,
    )
    qk_input_norm, qk_bands, qk_ids = _capture_kind(
        records,
        package=package,
        library=library,
        anchors=anchors,
        trace_kind="qk",
        threads=threads,
    )
    if ids != qk_ids:
        raise ValueError("development capture record ordering changed")
    input_norm_delta = float(np.max(np.abs(input_norm - qk_input_norm)))
    if input_norm_delta != 0.0:
        raise ValueError("development shadow input trace was not reset deterministic")
    output = Path(out).expanduser().resolve()
    if output.exists():
        raise ValueError("development capture output already exists")
    output.mkdir(parents=True, exist_ok=True)
    tensors = {
        "input_norm": _save_tensor(output / "input_norm.safetensors", "input_norm", input_norm),
        "candidate_keys": _save_tensor(output / "candidate_keys.safetensors", "candidate_keys", keys),
        "candidate_qk_bands": _save_tensor(output / "candidate_qk_bands.safetensors", "candidate_qk_bands", qk_bands),
    }
    report = {
        "schema_version": 1,
        "experiment": "olmoe_q7_retrieval_episodic_development_capture",
        "scope": {"split": "development", "records": _RECORDS, "reads": _READS, "cpu_only": True},
        "records": {"path": str(records_source), "sha256": records_sha256.lower(), "record_ids": ids},
        "package": str(package),
        "native_library": {"path": str(library), "sha256": sha256_file(library)},
        "tensors": tensors,
        "input_norm_reset_max_abs": input_norm_delta,
        "candidate_scores_definition": "sum of eight native pre-top-K QK partial bands",
        "selector_fit_on_development": False,
        "confirmation_split_opened": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "manifest.json", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--native-library", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=12)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = capture_development(
        records_path=args.records,
        records_sha256=args.records_sha256,
        package_path=args.package,
        native_library=args.native_library,
        out=args.out,
        threads=args.threads,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
