"""Full-artifact Python/native parity evidence for the BitNet DIP kernel."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from engram.evaluation.native_bitnet_dip_kernel import (
    NativeBitNetDIPCPUKernel,
    substitute_native_bitnet_dip_kernel_mlps,
)
from engram.models.native_bitnet import load_native_bitnet_artifact
from engram.runtime.native_bitnet import NativeBitNetRuntime
from engram.semantic.native_bitnet_dip import NativeBitNetDIPLayer
from engram.utils import atomic_json, sha256_file


def _first_record(path: Path, *, offset: int) -> dict[str, Any]:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("record offset must be a non-negative integer")
    usable = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid development JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"development JSONL record {line_number} is not an object"
                )
            if usable == offset:
                return value
            usable += 1
    raise ValueError("development JSONL does not contain the requested record")


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _expected_bf16_bits(values: np.ndarray) -> np.ndarray:
    source = np.ascontiguousarray(values, dtype=np.float32)
    return np.ascontiguousarray(
        source.view(np.uint32) >> np.uint32(16),
        dtype=np.uint16,
    )


def evaluate_native_bitnet_dip_full_artifact_parity(
    package: str | Path,
    coordinate_index: str | Path,
    dataset: str | Path,
    *,
    out: str | Path,
    dense_library: str | Path,
    native_library: str | Path,
    record_offset: int = 0,
    input_tokens: int = 33,
    rows_per_layer: int = 6,
    threads: int = 1,
) -> dict[str, Any]:
    """Prove all-layer parity on live BF16 states from a development record.

    One native sparse causal pass captures the exact BF16 input seen by every
    MLP after preceding sparse substitutions.  Each captured row is then run
    through both the frozen NumPy reference and the mmap-backed CPU kernel.
    The protected final holdout is neither accepted nor resolved here.
    """

    if (
        isinstance(input_tokens, bool)
        or not isinstance(input_tokens, int)
        or input_tokens <= 0
    ):
        raise ValueError("input_tokens must be a positive integer")
    if (
        isinstance(rows_per_layer, bool)
        or not isinstance(rows_per_layer, int)
        or not 1 <= rows_per_layer <= input_tokens
    ):
        raise ValueError("rows_per_layer must lie within input_tokens")
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    package_path = Path(package).resolve()
    index_path = Path(coordinate_index).resolve()
    dataset_path = Path(dataset).resolve()
    output_path = Path(out).resolve()
    dense_library_path = Path(dense_library).resolve()
    native_library_path = Path(native_library).resolve()
    protected_name = "milestone2_bitnet_holdout_v1.jsonl"
    if dataset_path.name == protected_name:
        raise ValueError("full-artifact parity must not use the protected holdout")

    record = _first_record(dataset_path, offset=record_offset)
    started = time.perf_counter()
    captured_bits: dict[int, np.ndarray] = {}

    with NativeBitNetRuntime(
        package_path,
        library=dense_library_path,
        threads=threads,
    ) as runtime:
        if "input_ids" in record:
            token_ids = [int(value) for value in record["input_ids"]]
        else:
            token_ids = runtime.encode(str(record.get("text", "")))
        if len(token_ids) < input_tokens:
            raise ValueError(
                f"development record has {len(token_ids)} tokens; "
                f"{input_tokens} are required"
            )
        token_ids = token_ids[:input_tokens]

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("full-artifact parity requires torch") from exc

        with NativeBitNetDIPCPUKernel(
            runtime.artifact_path,
            index_path,
            threads=threads,
            library=native_library_path,
        ) as kernel:
            dense_calls_before = len(runtime.kernel.calls)
            handles = []
            with substitute_native_bitnet_dip_kernel_mlps(
                runtime.model,
                kernel,
                debug_routes=False,
            ) as replacements:
                for layer, replacement in replacements.items():

                    def capture(_module, arguments, *, layer_index=layer):
                        hidden = arguments[0]
                        if hidden.dtype != torch.bfloat16:
                            raise RuntimeError(
                                "live parity boundary is not CPU BF16"
                            )
                        if layer_index not in captured_bits:
                            captured_bits[layer_index] = (
                                hidden.detach()[0]
                                .contiguous()
                                .view(torch.uint16)
                                .cpu()
                                .numpy()
                                .copy()
                            )

                    handles.append(replacement.register_forward_pre_hook(capture))
                try:
                    with torch.inference_mode():
                        runtime.model(
                            input_ids=torch.tensor(
                                [token_ids],
                                dtype=torch.long,
                            ),
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
            if len(runtime.kernel.calls) != dense_calls_before:
                raise RuntimeError(
                    "dense full-record MLP fallback executed during parity pass"
                )
            if set(captured_bits) != set(range(kernel.layer_count)):
                raise RuntimeError("did not capture every live BF16 MLP boundary")

            artifact = load_native_bitnet_artifact(runtime.artifact_path)
            layer_reports: list[dict[str, Any]] = []
            all_equal = {
                "input_coordinate_ids": True,
                "candidate_ids": True,
                "selected_record_ids": True,
                "selected_counts": True,
                "output_bf16_bits": True,
            }
            for layer, policy in enumerate(kernel.policies):
                all_state_bits = captured_bits[layer]
                actual_all = kernel.forward_debug_bf16_bits(
                    layer,
                    all_state_bits,
                )
                if (
                    actual_all.input_coordinate_ids is None
                    or actual_all.candidate_ids is None
                    or actual_all.selected_record_ids is None
                ):
                    raise RuntimeError("native parity omitted route identities")
                counts = np.asarray(
                    actual_all.selected_counts,
                    dtype=np.uint32,
                ).reshape(-1)
                priority = [
                    0,
                    int(np.argmin(counts)),
                    int(np.argmax(counts)),
                    counts.size - 1,
                ]
                priority.extend(
                    np.linspace(
                        0,
                        counts.size - 1,
                        num=rows_per_layer,
                        dtype=np.int64,
                    ).tolist()
                )
                priority.extend(range(counts.size))
                selected_rows: list[int] = []
                for value in priority:
                    row = int(value)
                    if row not in selected_rows:
                        selected_rows.append(row)
                    if len(selected_rows) == rows_per_layer:
                        break
                if len(selected_rows) != rows_per_layer:
                    raise RuntimeError("could not choose parity rows")
                state_bits = np.ascontiguousarray(
                    all_state_bits[selected_rows]
                )
                state = np.ascontiguousarray(
                    (
                        state_bits.astype(np.uint32) << np.uint32(16)
                    ).view(np.float32)
                )
                reference = NativeBitNetDIPLayer(
                    artifact,
                    layer,
                    input_fraction=(
                        policy.input_coordinates / kernel.hidden_size
                    ),
                    candidate_count=policy.candidate_count,
                    top_k=policy.maximum_top_k,
                    rms_audit_count=policy.rms_audit_count,
                    energy_target=policy.energy_target,
                    minimum_top_k=policy.minimum_top_k,
                    maximum_top_k=policy.maximum_top_k,
                    rms_estimator=policy.rms_estimator,
                    rms_audit_strategy=(
                        "top_proxy_raw_square"
                        if policy.rms_audit_strategy
                        == "top_proxy_raw_square"
                        else "hashed_tail"
                    ),
                )
                expected = reference(state)
                expected_selected = expected.selected_indices.copy()
                expected_selected[expected_selected < 0] = np.iinfo(
                    np.uint32
                ).max
                checks = {
                    "input_coordinate_ids": bool(
                        np.array_equal(
                            actual_all.input_coordinate_ids[selected_rows],
                            expected.input_indices.astype(np.uint32),
                        )
                    ),
                    "candidate_ids": bool(
                        np.array_equal(
                            actual_all.candidate_ids[selected_rows],
                            expected.candidate_indices.astype(np.uint32),
                        )
                    ),
                    "selected_record_ids": bool(
                        np.array_equal(
                            actual_all.selected_record_ids[selected_rows],
                            expected_selected.astype(np.uint32),
                        )
                    ),
                    "selected_counts": bool(
                        np.array_equal(
                            actual_all.selected_counts[selected_rows],
                            expected.selected_counts.astype(np.uint32),
                        )
                    ),
                    "output_bf16_bits": bool(
                        np.array_equal(
                            actual_all.output_bf16_bits[selected_rows],
                            _expected_bf16_bits(expected.output),
                        )
                    ),
                }
                for name, passed in checks.items():
                    all_equal[name] = all_equal[name] and passed
                layer_reports.append(
                    {
                        "layer": layer,
                        "rows": rows_per_layer,
                        "row_indices": selected_rows,
                        "selected_counts": [
                            int(actual_all.selected_counts[row])
                            for row in selected_rows
                        ],
                        "includes_observed_minimum_k": bool(
                            int(np.argmin(counts)) in selected_rows
                        ),
                        "includes_observed_maximum_k": bool(
                            int(np.argmax(counts)) in selected_rows
                        ),
                        "equality": checks,
                    }
                )
                del reference, expected, actual_all
                gc.collect()

            report = {
                "experiment": "native_bitnet_dip_full_artifact_parity",
                "scope": "all_30_layers_live_bf16_development",
                "passed": bool(all(all_equal.values())),
                "protected_holdout_used": False,
                "execution": {
                    "device": "cpu",
                    "input_boundary": "live_native_bf16",
                    "python_reference": "native_bitnet_dip_bf16_reference",
                    "native_kernel": "native_cpu",
                    "dense_full_record_fallback_calls": (
                        len(runtime.kernel.calls) - dense_calls_before
                    ),
                },
                "evidence": {
                    "layer_count": kernel.layer_count,
                    "layers_executed": list(range(kernel.layer_count)),
                    "rows_per_layer": rows_per_layer,
                    "total_rows": kernel.layer_count * rows_per_layer,
                    "input_tokens": input_tokens,
                    "row_selection": (
                        "first,last,observed_minimum_k,observed_maximum_k,"
                        "then_evenly_spaced_unique_fill"
                    ),
                },
                "equality": all_equal,
                "artifacts": {
                    "package_manifest": _descriptor(
                        package_path / "manifest.json"
                    ),
                    "base_record_artifact": _descriptor(
                        runtime.artifact_path
                    ),
                    "coordinate_index": _descriptor(index_path),
                    "dip_kernel_library": _descriptor(native_library_path),
                },
                "dataset": {
                    "path": str(dataset_path),
                    "sha256": sha256_file(dataset_path),
                    "record_offset": record_offset,
                },
                "layers": layer_reports,
                "elapsed_seconds": time.perf_counter() - started,
            }

    if not report["passed"]:
        raise RuntimeError("full-artifact Python/native DIP parity failed")
    atomic_json(output_path, report)
    return report


def write_native_bitnet_dip_full_artifact_parity_from_causal_report(
    causal_report: str | Path,
    *,
    out: str | Path,
) -> dict[str, Any]:
    """Extract the separately hash-bound parity proof from a native run.

    The qualifying causal evaluator already performs the expensive all-layer
    Python/native comparison on live development states.  This function
    validates that embedded evidence and publishes the smaller standalone
    schema consumed by the policy-freeze boundary without rerunning the model.
    """

    source_path = Path(causal_report).resolve()
    try:
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid native causal report: {exc}") from exc
    if (
        not isinstance(source, dict)
        or source.get("experiment") != "native_bitnet_dip_native_causal"
        or source.get("dataset_role") != "development"
        or source.get("overall_gate_passed") is not True
    ):
        raise ValueError("causal report is not a passing development run")
    execution = source.get("execution")
    parity = source.get("python_native_parity")
    evidence = source.get("evidence_observed")
    artifacts = source.get("artifacts")
    dataset = source.get("dataset")
    if not all(
        isinstance(value, dict)
        for value in (execution, parity, evidence, artifacts, dataset)
    ):
        raise ValueError("causal report omits parity provenance")
    assert isinstance(execution, dict)
    assert isinstance(parity, dict)
    assert isinstance(evidence, dict)
    assert isinstance(artifacts, dict)
    assert isinstance(dataset, dict)
    if (
        execution.get("input_boundary") != "live_native_bf16"
        or execution.get("kernel") != "native_cpu"
        or execution.get("device") != "cpu"
        or execution.get("dense_fallback") is not False
        or parity.get("evaluated") is not True
        or parity.get("all_layers") is not True
        or parity.get("passed") is not True
    ):
        raise ValueError("causal report does not prove live native parity")
    layer_count = evidence.get("layer_count")
    layers_executed = evidence.get("layers_executed")
    layer_reports = parity.get("layers")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count != 30
        or layers_executed != list(range(layer_count))
        or not isinstance(layer_reports, dict)
        or set(layer_reports) != {str(layer) for layer in range(layer_count)}
    ):
        raise ValueError("causal parity does not cover exactly 30 layers")

    key_mapping = {
        "input_coordinate_ids": "input_coordinate_ids",
        "candidate_ids": "candidate_ids",
        "selected_record_ids": "selected_record_ids",
        "selected_counts": "selected_counts",
        "output_bf16_bits": "output_bf16",
    }
    equality = {}
    for output_name, embedded_name in key_mapping.items():
        equality[output_name] = all(
            isinstance(layer_reports[str(layer)], dict)
            and isinstance(
                layer_reports[str(layer)].get("checks"),
                dict,
            )
            and layer_reports[str(layer)]["checks"].get(embedded_name) is True
            for layer in range(layer_count)
        )
    if not all(equality.values()):
        raise ValueError("causal report contains a failed parity field")
    if Path(str(dataset.get("path", ""))).name == (
        "milestone2_bitnet_holdout_v1.jsonl"
    ):
        raise ValueError("standalone parity cannot use the protected holdout")

    required_artifacts = {
        "package_manifest": "package_manifest",
        "base_record_artifact": "base_record_artifact",
        "coordinate_index": "coordinate_index",
        "dip_kernel_library": "dip_kernel_library",
    }
    standalone_artifacts = {}
    for output_name, source_name in required_artifacts.items():
        descriptor = artifacts.get(source_name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"causal report omits {source_name}")
        path = Path(str(descriptor.get("path", ""))).resolve()
        actual = _descriptor(path)
        if (
            descriptor.get("sha256") != actual["sha256"]
            or descriptor.get("bytes") != actual["bytes"]
        ):
            raise ValueError(f"causal artifact changed: {source_name}")
        standalone_artifacts[output_name] = actual

    rows_per_layer = parity.get("rows_per_layer")
    if (
        isinstance(rows_per_layer, bool)
        or not isinstance(rows_per_layer, int)
        or rows_per_layer < 6
    ):
        raise ValueError(
            "causal parity must cover at least six live rows per layer"
        )
    result = {
        "experiment": "native_bitnet_dip_full_artifact_parity",
        "scope": "all_30_layers_live_bf16_development",
        "passed": True,
        "protected_holdout_used": False,
        "execution": {
            "device": "cpu",
            "input_boundary": "live_native_bf16",
            "python_reference": "native_bitnet_dip_bf16_reference",
            "native_kernel": "native_cpu",
        },
        "evidence": {
            "layer_count": layer_count,
            "layers_executed": list(range(layer_count)),
            "rows_per_layer": rows_per_layer,
            "total_rows": rows_per_layer * layer_count,
        },
        "equality": equality,
        "artifacts": standalone_artifacts,
        "dataset": dict(dataset),
        "source_causal_report": _descriptor(source_path),
        "layers": layer_reports,
    }
    atomic_json(Path(out).resolve(), result)
    return result


__all__ = [
    "evaluate_native_bitnet_dip_full_artifact_parity",
    "write_native_bitnet_dip_full_artifact_parity_from_causal_report",
]
