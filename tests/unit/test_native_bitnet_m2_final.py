from __future__ import annotations

import inspect
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import engram.evaluation.native_bitnet_m2_final as final_confirmation
from engram.evaluation.native_bitnet_m2_final import (
    CANONICAL_TOKEN_HASH_ALGORITHM,
    COMPILER_BUILD_FORMAT,
    FINAL_ADJUDICATION_FORMAT,
    FINAL_CONFIRMATION_FORMAT,
    NATIVE_CAUSAL_EVALUATOR,
    SCORED_PREFIX_TOKEN_HASH_ALGORITHM,
    Milestone2FinalConfirmationError,
    Milestone2FinalRequest,
    adjudicate_native_bitnet_m2_final_confirmation,
    run_native_bitnet_m2_final_confirmation,
    write_native_bitnet_m2_compiler_build_manifest,
    write_native_bitnet_m2_final_authorization_manifest,
)
from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.utils import atomic_json, sha256_file, sha256_json


_REAL_DEFAULT_EVALUATOR = final_confirmation._default_evaluator
_REAL_REPORT_VALIDATOR = final_confirmation._validate_evaluator_report
_REAL_TOKEN_IDENTITY_DERIVER = (
    final_confirmation._derive_final_token_identities
)


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _descriptor(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@dataclass
class _Fixture:
    root: Path
    protocol: Path
    authorization: Path
    dataset: Path
    out: Path
    marker: Path
    pins: dict[str, dict[str, object]]
    layer_count: int


def _fixture(tmp_path: Path) -> _Fixture:
    # This isolated repository is the only dataset a unit test may open.
    root = tmp_path / "fixture-repository"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _write(root / ".gitignore", b"/build/\n")
    _write(
        root / "CMakeLists.txt",
        b"project(engram_fixture LANGUAGES CXX)\n",
    )
    _write(
        root / "native" / "include" / "engram" / "fixture.h",
        b"#pragma once\n",
    )
    _write(
        root / "native" / "src" / "fixture.cpp",
        b'#include "engram/fixture.h"\n',
    )
    _write(
        root / "src" / "engram" / "evaluation" / "native_bitnet_m2_final.py",
        b"# committed fixture runner origin\n",
    )
    _write(
        root / "src" / "engram" / "evaluation" / "native_bitnet_dip_native_causal.py",
        b"# committed fixture evaluator origin\n",
    )

    package = root / "work" / "model.engram-bitnet"
    base = _write(package / "mlp" / "records.bin", b"fixture-base-records")
    tokenizer = _write(
        package / "tokenizer" / "tokenizer.json",
        b'{"fixture":true}',
    )
    index = _write(
        root / "work" / "model.bitnet-dip-index.bin",
        b"fixture-coordinate-index",
    )
    dense_library = _write(
        root / "build" / "libengram_bitnet.so",
        b"fixture-dense-library",
    )
    dip_library = _write(
        root / "build" / "libengram_bitnet_dip.so",
        b"fixture-dip-library",
    )
    build_manifest = _write(
        root / "build" / "compiler-build.json",
        b'{"compiler":"fixture","cpu_only":true}',
    )
    dataset = _write(
        root / "tests" / "fixtures" / "fixture-holdout.jsonl",
        (
            '{"text":"fixture sequence zero with sufficient tokens"}\n'
            '{"text":"fixture sequence one with sufficient tokens"}\n'
            '{"text":"fixture sequence two with sufficient tokens"}\n'
            '{"text":"fixture sequence three with sufficient tokens"}\n'
            '{"text":"fixture sequence four with sufficient tokens"}\n'
            '{"text":"fixture sequence five with sufficient tokens"}\n'
            '{"text":"fixture sequence six with sufficient tokens"}\n'
            '{"text":"fixture sequence seven with sufficient tokens"}\n'
        ).encode(),
    )
    policy = root / "reports" / "fixture-policy.json"
    atomic_json(
        policy,
        {
            "format": "engram-native-bitnet-dip-policy",
            "version": 1,
            "status": "approved",
            "layers": [
                {
                    "layer": layer,
                    "input_fraction": 1 / 320,
                    "input_coordinates": 1,
                    "candidate_count": 10,
                    "minimum_top_k": 2,
                    "maximum_top_k": 5,
                    "energy_target": 1.0,
                    "rms_estimator": "candidate_ratio",
                    "rms_audit_count": 0,
                    "rms_audit_strategy": "none",
                    "rms_variance_scale": 1.0,
                    "rms_variance_bias": 0.0,
                    "output_scale": 1.0,
                }
                for layer in range(2)
            ],
        },
    )
    layer_count = 2
    package_manifest = package / "manifest.json"
    atomic_json(
        package_manifest,
        {
            "format": "engram-native-bitnet",
            "version": 1,
            "source": {
                "repository": "fixture/bitnet",
                "revision": "1" * 40,
                "weight_sha256": "2" * 64,
            },
            "model": {
                "hidden_size": 320,
                "intermediate_size": 320,
                "num_hidden_layers": layer_count,
            },
            "mlp": {
                "path": "mlp/records.bin",
                "sha256": sha256_file(base),
            },
            "tokenizer": {"path": "tokenizer"},
            "files": {
                "mlp/records.bin": {
                    "bytes": base.stat().st_size,
                    "sha256": sha256_file(base),
                },
                "tokenizer/tokenizer.json": {
                    "bytes": tokenizer.stat().st_size,
                    "sha256": sha256_file(tokenizer),
                },
            },
        },
    )
    protocol = root / "reports" / "fixture-protocol.json"
    atomic_json(
        protocol,
        {
            "experiment": (
                "native_bitnet_milestone_2_practical_semantic_memory_confirmation"
            ),
            "protocol_version": 5,
            "test_fixture_only": True,
            "source_model": {
                "repository": "fixture/bitnet",
                "revision": "1" * 40,
                "weight_sha256": "2" * 64,
            },
            "final_confirmation": {
                "dataset": dataset.relative_to(root).as_posix(),
                "dataset_sha256": sha256_file(dataset),
                "record_offset": 0,
                "sequence_count": 8,
                "predictions_per_sequence": 32,
                "prediction_positions": 256,
                "required_tokens_per_sequence": 33,
                "tokenizer_json_sha256": sha256_file(tokenizer),
                "canonical_token_sequence_hashes": [
                    sha256_json(
                        {"input_ids": [sequence, *range(39)]}
                    )
                    for sequence in range(8)
                ],
                "canonical_token_hash_algorithm": (
                    "engram-canonical-token-sequence-sha256-v1"
                ),
                "token_lengths": [40] * 8,
                "reuse_for_configuration_changes_after_opening": False,
            },
            "quality_thresholds": {
                "maximum_teacher_student_kl": 0.05,
                "minimum_top1_agreement": 0.90,
                "maximum_nll_delta": 0.05,
                "maximum_final_hidden_relative_l2": 0.10,
            },
            "practical_router_thresholds": {
                "maximum_mean_active_record_fraction": 0.25,
                "maximum_complete_physical_cold_traffic_fraction_of_dense_q4": (0.45),
                "minimum_held_out_candidate_recall": 0.95,
                "cpu_only_inference_required": True,
                "dense_gate_up_or_down_fallback_allowed": False,
            },
            "candidate_recall_definition": {
                "reference_top_ks": [4, 5],
            },
            "required_system_evidence": {
                "serialized_index_reload": True,
                "python_native_numerical_parity": True,
                "measured_cpu_latency": True,
                "cache_line_honest_traffic_accounting": True,
            },
            "configuration": None,
            "final_result": None,
        },
    )
    tracked_note = _write(root / "implementation.txt", b"fixture implementation")
    assert tracked_note.is_file()
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Engram Test",
        "-c",
        "user.email=engram-test@example.invalid",
        "commit",
        "-m",
        "freeze fixture implementation",
    )
    commit = _git(root, "rev-parse", "HEAD")
    atomic_json(
        build_manifest,
        {
            "format": COMPILER_BUILD_FORMAT,
            "version": 1,
            "status": "frozen",
            "implementation": {
                "git_commit": commit,
                "clean_worktree": True,
            },
            "outputs": {
                "dense_native_library": _descriptor(root, dense_library),
                "native_library": _descriptor(root, dip_library),
            },
            "cpu_inference_only": True,
        },
    )

    pins = {
        "package_manifest": _descriptor(root, package_manifest),
        "base_artifact": _descriptor(root, base),
        "coordinate_index": _descriptor(root, index),
        "policy_manifest": _descriptor(root, policy),
        "tokenizer_json": _descriptor(root, tokenizer),
        "dense_native_library": _descriptor(root, dense_library),
        "native_library": _descriptor(root, dip_library),
        "compiler_build_manifest": _descriptor(root, build_manifest),
        "dataset": _descriptor(root, dataset),
    }
    authorization = root / "build" / "fixture-authorization.json"
    protocol_descriptor = _descriptor(root, protocol)
    execution = {
        "evaluator": NATIVE_CAUSAL_EVALUATOR,
        "dataset_role": "final",
        "debug_recall": True,
        "threads": 2,
    }
    audit = final_confirmation._canonical_audit_record(
        implementation_commit=commit,
        protocol_sha256=str(protocol_descriptor["sha256"]),
        artifact_sha256={
            name: str(descriptor["sha256"]) for name, descriptor in pins.items()
        },
        execution=execution,
    )
    atomic_json(
        authorization,
        {
            "format": FINAL_CONFIRMATION_FORMAT,
            "version": 1,
            "status": "frozen",
            "implementation": {
                "repository_root": str(root),
                "git_commit": commit,
                "require_clean_worktree": True,
            },
            "protocol": protocol_descriptor,
            "artifacts": pins,
            "execution": execution,
            "audit": audit,
        },
    )
    return _Fixture(
        root=root,
        protocol=protocol,
        authorization=authorization,
        dataset=dataset,
        out=root / audit["result"],
        marker=root / audit["opened_marker"],
        pins=pins,
        layer_count=layer_count,
    )


def _passing_report(request: Milestone2FinalRequest) -> dict[str, object]:
    protocol = json.loads(request.protocol_path.read_text())
    full_sequences = [
        [sequence, *range(39)] for sequence in range(request.sequence_count)
    ]
    scored_prefixes = [
        values[: request.predictions_per_sequence + 1]
        for values in full_sequences
    ]
    positions = request.sequence_count * request.predictions_per_sequence
    configuration = {
        str(layer): {
            "input_coordinates": 1,
            "candidate_count": 10,
            "minimum_top_k": 2,
            "maximum_top_k": 5,
            "energy_target": 1.0,
            "rms_audit_count": 0,
            "rms_estimator": "candidate_ratio",
            "rms_audit_strategy": "none",
            "rms_variance_scale": 1.0,
            "rms_variance_bias": 0.0,
            "output_scale": 1.0,
        }
        for layer in range(len(request.reference_top_ks))
    }
    schedules = [[2] * len(request.reference_top_ks) for _ in range(positions)]
    accounting = [
        native_bitnet_dip_physical_accounting(
            320,
            320,
            input_counts=[1] * len(request.reference_top_ks),
            candidate_counts=[10] * len(request.reference_top_ks),
            top_ks=schedule,
        )
        for schedule in schedules
    ]
    token_rows = []
    worst_layer = None
    for token, item in enumerate(accounting):
        traffic = item["traffic"]
        token_rows.append(
            {
                "token": token,
                "sequence": token // request.predictions_per_sequence,
                "prediction_position": (token % request.predictions_per_sequence),
                "scheduled_cache_line_bytes": (traffic["complete_modelled_cold_bytes"]),
                "dense_q4_bytes": traffic["dense_q4_bytes"],
                "fraction_of_dense_q4": traffic["fraction_of_dense_q4"],
            }
        )
        for layer_row in traffic["layers"]:
            candidate = {
                "token": token,
                "layer": layer_row["layer"],
                "fraction_of_dense_q4": layer_row["fraction_of_dense_q4"],
            }
            if (
                worst_layer is None
                or candidate["fraction_of_dense_q4"]
                > worst_layer["fraction_of_dense_q4"]
            ):
                worst_layer = candidate
    assert worst_layer is not None
    worst_token = max(
        token_rows,
        key=lambda row: row["fraction_of_dense_q4"],
    )

    def artifact(path: Path, pin_name: str):
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": request.expected_sha256[pin_name],
        }

    layer_recall = {}
    for layer, reference_top_k in enumerate(request.reference_top_ks):
        secondary = {
            "target_records": positions * 2,
            "candidate_hits": positions * 2,
            "candidate_micro_recall": 1.0,
            "candidate_mean_row_recall": 1.0,
            "candidate_p05_row_recall": 1.0,
            "candidate_minimum_row_recall": 1.0,
            "selected_hits": positions * 2,
            "selected_micro_recall": 1.0,
            "selected_mean_row_recall": 1.0,
            "selected_p05_row_recall": 1.0,
            "selected_minimum_row_recall": 1.0,
        }
        layer_recall[str(layer)] = {
            "layer": layer,
            "rows": positions,
            "reference_top_k": reference_top_k,
            "target_records": positions * reference_top_k,
            "candidate_hits": positions * reference_top_k,
            "candidate_micro_recall": 1.0,
            "candidate_mean_row_recall": 1.0,
            "candidate_p05_row_recall": 1.0,
            "candidate_minimum_row_recall": 1.0,
            "secondary_teacher_positive_utility_recall_clipped_to_"
            "frozen_minimum_and_maximum_k": secondary,
        }
    report: dict[str, object] = {
        "experiment": "native_bitnet_dip_native_causal",
        "dataset_role": "final",
        "artifacts": {
            "package_manifest": artifact(
                request.package_manifest,
                "package_manifest",
            ),
            "base_record_artifact": artifact(
                request.record_artifact,
                "base_artifact",
            ),
            "coordinate_index": artifact(
                request.coordinate_index,
                "coordinate_index",
            ),
            "dense_kernel_library": artifact(
                request.dense_native_library,
                "dense_native_library",
            ),
            "dip_kernel_library": artifact(
                request.native_library,
                "native_library",
            ),
        },
        "dataset": {
            "path": str(request.dataset),
            "sha256": request.expected_sha256["dataset"],
            "record_offset": request.record_offset,
            "sequence_count": request.sequence_count,
            "predictions_per_sequence": request.predictions_per_sequence,
            "required_input_tokens_per_sequence": (
                request.predictions_per_sequence + 1
            ),
            "prediction_positions": positions,
            "input_token_ids_sha256": sha256_json(scored_prefixes),
            "sequence_token_ids_sha256": [
                sha256_json(values) for values in scored_prefixes
            ],
            "canonical_full_sequence_hash_algorithm": (
                CANONICAL_TOKEN_HASH_ALGORITHM
            ),
            "canonical_full_sequence_token_ids_sha256": (
                protocol["final_confirmation"][
                    "canonical_token_sequence_hashes"
                ]
            ),
            "canonical_full_sequence_token_lengths": [40] * 8,
            "scored_prefix_hash_algorithm": (
                SCORED_PREFIX_TOKEN_HASH_ALGORITHM
            ),
            "scored_prefix_token_ids_sha256": [
                sha256_json(values) for values in scored_prefixes
            ],
            "scored_prefix_input_token_ids_sha256": (
                sha256_json(scored_prefixes)
            ),
        },
        "configuration": configuration,
        "reference_top_ks": {
            "values": list(request.reference_top_ks),
            "sha256": sha256_json(list(request.reference_top_ks)),
            "role": "frozen_fixed_per_layer_candidate_recall_denominator",
        },
        "execution": {
            "input_boundary": "live_native_bf16",
            "kernel": "native_cpu",
            "device": "cpu",
            "dense_fallback": False,
            "all_mlp_layers_substituted": True,
            "serialized_index_reloaded": True,
            "python_native_parity_passed": True,
            "timed_sparse_debug_routes": False,
            "debug_pass_outside_timing": True,
            "dense_threads": request.threads,
            "dip_threads": request.threads,
        },
        "quality": {
            "mean_kl_divergence": 0.01,
            "top1_agreement": 0.95,
            "reference_nll": 2.0,
            "candidate_nll": 2.01,
            "nll_delta": 0.01,
            "final_hidden_relative_l2": 0.05,
            "passed": True,
        },
        "quality_passed": True,
        "selected_records": {
            "per_token_layer_k": schedules,
            "global": {
                "sum": positions * len(request.reference_top_ks) * 2,
                "count": positions * len(request.reference_top_ks),
                "minimum": 2,
                "maximum": 2,
                "mean": 2.0,
                "active_fraction": 2 / 320,
            },
        },
        "active_record_budget": {"passes_25_percent": True},
        "physical_cold_traffic": {
            "accounting_version": "native_bitnet_dip_dual_layout_v2",
            "global": {
                "scheduled_cache_line_bytes": sum(
                    row["scheduled_cache_line_bytes"] for row in token_rows
                ),
                "dense_q4_bytes": sum(row["dense_q4_bytes"] for row in token_rows),
            },
            "per_token": token_rows,
            "worst_token": worst_token,
            "worst_layer": worst_layer,
            "passes_45_percent": True,
        },
        "debug_recall": {
            "enabled": True,
            "timed": False,
            "timed_sparse_parity": {"passed": True},
            "global": {
                "target_records": positions * sum(request.reference_top_ks),
                "candidate_hits": positions * sum(request.reference_top_ks),
                "candidate_micro_recall": 1.0,
                "macro_mean_layer_recall": 1.0,
                "candidate_minimum_layer_mean_recall": 1.0,
                "secondary_teacher_positive_utility_recall_clipped_to_"
                "frozen_minimum_and_maximum_k": {
                    "target_records": positions * 2 * len(request.reference_top_ks),
                    "candidate_hits": positions * 2 * len(request.reference_top_ks),
                    "candidate_micro_recall": 1.0,
                    "candidate_macro_mean_layer_recall": 1.0,
                    "selected_hits": positions * 2 * len(request.reference_top_ks),
                    "selected_micro_recall": 1.0,
                    "selected_macro_mean_layer_recall": 1.0,
                },
                "passes_95_percent": True,
            },
            "layers": layer_recall,
        },
        "candidate_recall_passed": True,
        "python_native_parity": {
            "evaluated": True,
            "rows_per_layer": 1,
            "all_layers": True,
            "passed": True,
            "layers": {
                str(layer): {
                    "layer": layer,
                    "rows": 1,
                    "checks": {
                        "output_bf16": True,
                        "input_coordinate_ids": True,
                        "candidate_ids": True,
                        "selected_counts": True,
                        "selected_record_ids": True,
                    },
                    "passed": True,
                    "live_input_bf16_sha256": sha256_json(["live-input", layer]),
                    "native_output_bf16_sha256": sha256_json(["native-output", layer]),
                    "python_output_bf16_sha256": sha256_json(["python-output", layer]),
                }
                for layer in range(len(request.reference_top_ks))
            },
        },
        "evidence_observed": {
            "sequences": request.sequence_count,
            "unique_sequences": request.sequence_count,
            "predictions_per_sequence": request.predictions_per_sequence,
            "prediction_positions": positions,
            "all_mlp_layers": True,
            "layers_executed": list(range(len(request.reference_top_ks))),
        },
        "systems_evidence_passed": True,
        "scoring_protocol_valid": True,
        "evidence_passed": True,
        "protocol_qualifying": True,
        "overall_gate_passed": True,
        "timing": {
            "timed_sparse_seconds": 1.25,
        },
    }
    atomic_json(request.raw_report, report)
    return report


def _seal_consumed_attempt(fixture: _Fixture) -> Path:
    authorization = json.loads(fixture.authorization.read_text())
    audit = authorization["audit"]
    directory = (
        fixture.root
        / audit["directory"]
        / audit["dataset_key"]
    )
    raw = fixture.root / audit["raw_report"]
    archived_authorization = (
        directory / f"{audit['attempt_key']}.authorization.json"
    )
    archived_authorization.parent.mkdir(parents=True, exist_ok=True)
    archived_authorization.write_bytes(fixture.authorization.read_bytes())
    seal = directory / f"{audit['attempt_key']}.evidence-seal.json"
    atomic_json(
        seal,
        {
            "format": (
                "engram-native-bitnet-m2-consumed-attempt-evidence-seal"
            ),
            "version": 1,
            "sealed_at": "2026-01-01T00:00:00Z",
            "implementation_commit": authorization["implementation"][
                "git_commit"
            ],
            "dataset_key": audit["dataset_key"],
            "attempt_key": audit["attempt_key"],
            "dataset_sha256": authorization["artifacts"]["dataset"][
                "sha256"
            ],
            "protocol_sha256": authorization["protocol"]["sha256"],
            "policy_sha256": authorization["artifacts"][
                "policy_manifest"
            ]["sha256"],
            "compiler_build_manifest_sha256": authorization["artifacts"][
                "compiler_build_manifest"
            ]["sha256"],
            "authorization": {
                "original_path": fixture.authorization.relative_to(
                    fixture.root
                ).as_posix(),
                "archived_path": archived_authorization.relative_to(
                    fixture.root
                ).as_posix(),
                "sha256": sha256_file(fixture.authorization),
            },
            "opened_marker": {
                "path": fixture.marker.relative_to(
                    fixture.root
                ).as_posix(),
                "sha256": sha256_file(fixture.marker),
            },
            "result": {
                "path": fixture.out.relative_to(fixture.root).as_posix(),
                "sha256": sha256_file(fixture.out),
            },
            "raw_evaluator_report": {
                "path": raw.relative_to(fixture.root).as_posix(),
                "sha256": sha256_file(raw),
            },
            "trust_limitation": (
                "The original error result did not bind the raw report."
            ),
        },
    )
    _git(fixture.root, "add", directory.relative_to(fixture.root).as_posix())
    _git(
        fixture.root,
        "-c",
        "user.name=Engram Test",
        "-c",
        "user.email=engram-test@example.invalid",
        "commit",
        "-m",
        "seal consumed fixture evidence",
    )
    return directory / f"{audit['attempt_key']}.adjudication.json"


def _consume_and_seal_hash_contract_error(
    fixture: _Fixture,
    monkeypatch,
) -> tuple[Path, Path, dict[str, str]]:
    def historical_hash_contract_bug(*_args, **_kwargs):
        raise Milestone2FinalConfirmationError(
            "native evaluator token sequences differ from frozen canonical hashes"
        )

    monkeypatch.setattr(
        final_confirmation,
        "_validate_evaluator_report",
        historical_hash_contract_bug,
    )
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="token sequences differ",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    authorization = json.loads(fixture.authorization.read_text())
    raw = fixture.root / authorization["audit"]["raw_report"]
    before = {
        "result": sha256_file(fixture.out),
        "marker": sha256_file(fixture.marker),
        "raw": sha256_file(raw),
    }
    adjudication_path = _seal_consumed_attempt(fixture)
    monkeypatch.setattr(
        final_confirmation,
        "_validate_evaluator_report",
        _REAL_REPORT_VALIDATOR,
    )
    return raw, adjudication_path, before


@pytest.fixture(autouse=True)
def _fixture_native_evaluator(monkeypatch):
    def token_identities(request: Milestone2FinalRequest):
        full_sequences = [
            [sequence, *range(39)]
            for sequence in range(request.sequence_count)
        ]
        prefixes = [
            values[: request.predictions_per_sequence + 1]
            for values in full_sequences
        ]
        return final_confirmation._TokenIdentities(
            full_sequence_hashes=tuple(
                sha256_json({"input_ids": values})
                for values in full_sequences
            ),
            full_token_lengths=tuple(
                len(values) for values in full_sequences
            ),
            scored_prefix_hashes=tuple(
                sha256_json(values) for values in prefixes
            ),
            scored_prefix_input_ids_sha256=sha256_json(prefixes),
            scored_prefix_tokens=request.predictions_per_sequence + 1,
            tokenizer_files=(),
        )

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        _passing_report,
    )
    monkeypatch.setattr(
        final_confirmation,
        "_derive_final_token_identities",
        token_identities,
    )


def test_real_token_identity_deriver_separates_full_and_scored_domains(
    tmp_path,
):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer_root = tmp_path / "package" / "tokenizer"
    tokenizer_root.mkdir(parents=True)
    vocabulary = {
        "<unk>": 0,
        "<s>": 1,
        "alpha": 2,
        **{f"seq{index}": index + 3 for index in range(8)},
    }
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            vocabulary,
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="<s> $A",
        special_tokens=[("<s>", 1)],
    )
    tokenizer.save(str(tokenizer_root / "tokenizer.json"))
    atomic_json(
        tokenizer_root / "tokenizer_config.json",
        {
            "bos_token": "<s>",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "<unk>",
        },
    )
    atomic_json(
        tokenizer_root / "special_tokens_map.json",
        {
            "bos_token": "<s>",
            "unk_token": "<unk>",
        },
    )
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    inventory = {
        f"tokenizer/{name}": {
            "bytes": (tokenizer_root / name).stat().st_size,
            "sha256": sha256_file(tokenizer_root / name),
        }
        for name in tokenizer_files
    }
    package_manifest = tmp_path / "package" / "manifest.json"
    atomic_json(
        package_manifest,
        {
            "tokenizer": {
                "path": "tokenizer",
                "files": tokenizer_files,
                "fix_mistral_regex": False,
            },
            "files": inventory,
        },
    )
    dataset = tmp_path / "holdout.jsonl"
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "text": " ".join(
                        [*(["alpha"] * 38), f"seq{sequence}"]
                    )
                }
            )
            + "\n"
            for sequence in range(8)
        )
    )
    unused = tmp_path / "unused"
    request = Milestone2FinalRequest(
        protocol_path=unused,
        authorization_manifest_path=unused,
        package=package_manifest.parent,
        package_manifest=package_manifest,
        record_artifact=unused,
        coordinate_index=unused,
        policy_manifest=unused,
        tokenizer_json=tokenizer_root / "tokenizer.json",
        dense_native_library=unused,
        native_library=unused,
        compiler_build_manifest=unused,
        dataset=dataset,
        raw_report=unused,
        record_offset=0,
        sequence_count=8,
        predictions_per_sequence=32,
        threads=1,
        reference_top_ks=(1,),
        expected_sha256={},
    )

    identities = _REAL_TOKEN_IDENTITY_DERIVER(request)

    assert identities.full_token_lengths == (40,) * 8
    assert identities.scored_prefix_tokens == 33
    assert all(
        full != prefix
        for full, prefix in zip(
            identities.full_sequence_hashes,
            identities.scored_prefix_hashes,
            strict=True,
        )
    )
    assert len(identities.tokenizer_files) == 3


def test_final_runner_opens_fixture_once_and_persists_passing_audit(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    calls = 0

    def evaluator(request: Milestone2FinalRequest):
        nonlocal calls
        calls += 1
        assert fixture.marker.is_file()
        marker = json.loads(fixture.marker.read_text())
        assert marker["status"] == "executing"
        assert marker["dataset"]["actual_sha256"] == sha256_file(fixture.dataset)
        # Only this temporary evaluator simulates tokenizing the fixture.
        assert len(request.dataset.read_text().splitlines()) == 8
        return _passing_report(request)

    monkeypatch.setattr(final_confirmation, "_default_evaluator", evaluator)
    result = run_native_bitnet_m2_final_confirmation(
        fixture.protocol,
        fixture.authorization,
        out=fixture.out,
        opened_marker=fixture.marker,
        confirm_open=True,
    )

    assert calls == 1
    assert result["status"] == "pass"
    assert result["milestone_2_passed"] is True
    persisted = json.loads(fixture.out.read_text())
    marker = json.loads(fixture.marker.read_text())
    assert persisted == result
    assert marker["status"] == "pass"
    assert marker["reuse_allowed"] is False
    assert marker["dataset"]["actual_sha256"] == sha256_file(fixture.dataset)
    protocol = json.loads(fixture.protocol.read_text())
    assert (
        persisted["evaluator_report"]["dataset"][
            "sequence_token_ids_sha256"
        ]
        != protocol["final_confirmation"][
            "canonical_token_sequence_hashes"
        ]
    )
    assert persisted["token_identities"]["canonical_full_sequence"][
        "token_lengths"
    ] == [40] * 8

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="differ from canonical",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=tmp_path / "alternate-result.json",
            opened_marker=tmp_path / "alternate-opened.json",
            confirm_open=True,
        )
    assert calls == 1

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="marker/result already exists",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert calls == 1


def test_dataset_lock_survives_a_different_authorization_attempt(tmp_path):
    fixture = _fixture(tmp_path)
    run_native_bitnet_m2_final_confirmation(
        fixture.protocol,
        fixture.authorization,
        out=fixture.out,
        opened_marker=fixture.marker,
        confirm_open=True,
    )

    second = json.loads(fixture.authorization.read_text())
    second["execution"]["threads"] = 3
    second["audit"] = final_confirmation._canonical_audit_record(
        implementation_commit=second["implementation"]["git_commit"],
        protocol_sha256=second["protocol"]["sha256"],
        artifact_sha256={
            name: descriptor["sha256"]
            for name, descriptor in second["artifacts"].items()
        },
        execution=second["execution"],
    )
    assert (
        second["audit"]["opened_marker"]
        == json.loads(fixture.authorization.read_text())["audit"]["opened_marker"]
    )
    assert (
        second["audit"]["result"]
        != json.loads(fixture.authorization.read_text())["audit"]["result"]
    )
    second_path = tmp_path / "second-authorization.json"
    atomic_json(second_path, second)
    second_result = fixture.root / second["audit"]["result"]

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="marker/result already exists",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            second_path,
            out=second_result,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert not second_result.exists()


def test_final_runner_rejects_preflight_hash_mismatch_without_opening(tmp_path):
    fixture = _fixture(tmp_path)
    authorization = json.loads(fixture.authorization.read_text())
    authorization["artifacts"]["coordinate_index"]["sha256"] = "0" * 64
    atomic_json(fixture.authorization, authorization)

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="coordinate_index SHA-256 mismatch",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )

    assert not fixture.marker.exists()
    error = json.loads(fixture.out.read_text())
    assert error["status"] == "error"
    assert error["opened"] is False
    assert error["decision"] == "final_holdout_not_opened_preflight_error"


def test_fixture_mode_cannot_target_protected_holdout_identity(tmp_path):
    fixture = _fixture(tmp_path)
    authorization = json.loads(fixture.authorization.read_text())
    protocol = json.loads(fixture.protocol.read_text())
    protocol["final_confirmation"]["dataset_sha256"] = (
        final_confirmation.PROTECTED_M2_HOLDOUT_SHA256
    )
    authorization["artifacts"]["dataset"]["sha256"] = (
        final_confirmation.PROTECTED_M2_HOLDOUT_SHA256
    )
    atomic_json(fixture.protocol, protocol)
    authorization["protocol"] = _descriptor(
        fixture.root,
        fixture.protocol,
    )
    authorization["audit"] = final_confirmation._canonical_audit_record(
        implementation_commit=authorization["implementation"]["git_commit"],
        protocol_sha256=authorization["protocol"]["sha256"],
        artifact_sha256={
            name: descriptor["sha256"]
            for name, descriptor in authorization["artifacts"].items()
        },
        execution=authorization["execution"],
    )
    fixture.out = fixture.root / authorization["audit"]["result"]
    fixture.marker = fixture.root / authorization["audit"]["opened_marker"]
    atomic_json(fixture.authorization, authorization)

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="fixture-only mode is forbidden",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert not fixture.marker.exists()


def test_dataset_hash_is_checked_only_after_opened_marker(tmp_path):
    fixture = _fixture(tmp_path)
    authorization = json.loads(fixture.authorization.read_text())
    protocol = json.loads(fixture.protocol.read_text())
    wrong_hash = "0" * 64
    authorization["artifacts"]["dataset"]["sha256"] = wrong_hash
    protocol["final_confirmation"]["dataset_sha256"] = wrong_hash
    atomic_json(fixture.authorization, authorization)
    atomic_json(fixture.protocol, protocol)
    # Updating the committed protocol would normally dirty the implementation.
    # Commit only the fixture protocol so preflight reaches the opened boundary.
    _git(fixture.root, "add", fixture.protocol.relative_to(fixture.root).as_posix())
    _git(
        fixture.root,
        "-c",
        "user.name=Engram Test",
        "-c",
        "user.email=engram-test@example.invalid",
        "commit",
        "-m",
        "freeze mismatched fixture declaration",
    )
    authorization["implementation"]["git_commit"] = _git(
        fixture.root,
        "rev-parse",
        "HEAD",
    )
    build_manifest_path = (
        fixture.root / authorization["artifacts"]["compiler_build_manifest"]["path"]
    )
    build_manifest = json.loads(build_manifest_path.read_text())
    build_manifest["implementation"]["git_commit"] = authorization["implementation"][
        "git_commit"
    ]
    atomic_json(build_manifest_path, build_manifest)
    authorization["artifacts"]["compiler_build_manifest"] = _descriptor(
        fixture.root,
        build_manifest_path,
    )
    authorization["protocol"] = _descriptor(fixture.root, fixture.protocol)
    authorization["audit"] = final_confirmation._canonical_audit_record(
        implementation_commit=authorization["implementation"]["git_commit"],
        protocol_sha256=authorization["protocol"]["sha256"],
        artifact_sha256={
            name: descriptor["sha256"]
            for name, descriptor in authorization["artifacts"].items()
        },
        execution=authorization["execution"],
    )
    fixture.out = fixture.root / authorization["audit"]["result"]
    fixture.marker = fixture.root / authorization["audit"]["opened_marker"]
    atomic_json(fixture.authorization, authorization)

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="dataset SHA-256 mismatch after opening",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )

    marker = json.loads(fixture.marker.read_text())
    result = json.loads(fixture.out.read_text())
    assert marker["status"] == "error"
    assert result["opened"] is True
    assert result["decision"] == "final_holdout_consumed_with_error"


@pytest.mark.parametrize("failure", ["dirty", "commit"])
def test_final_runner_rejects_dirty_or_wrong_implementation(
    tmp_path,
    failure,
):
    fixture = _fixture(tmp_path)
    if failure == "dirty":
        (fixture.root / "implementation.txt").write_text("changed")
        match = "worktree is dirty"
    else:
        authorization = json.loads(fixture.authorization.read_text())
        authorization["implementation"]["git_commit"] = "0" * 40
        atomic_json(fixture.authorization, authorization)
        match = "Git commit mismatch"

    with pytest.raises(Milestone2FinalConfirmationError, match=match):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )

    assert not fixture.marker.exists()
    assert json.loads(fixture.out.read_text())["opened"] is False


def test_final_runner_persists_native_error_after_opening(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)

    def broken(_request):
        raise RuntimeError("fixture native evaluator failed")

    monkeypatch.setattr(final_confirmation, "_default_evaluator", broken)
    with pytest.raises(RuntimeError, match="fixture native evaluator failed"):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )

    result = json.loads(fixture.out.read_text())
    marker = json.loads(fixture.marker.read_text())
    assert result["status"] == "error"
    assert result["opened"] is True
    assert result["error"]["type"] == "RuntimeError"
    assert marker["status"] == "error"
    assert marker["reuse_allowed"] is False


def test_postmortem_adjudicates_only_the_sealed_hash_contract_error(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    raw, adjudication_path, before = (
        _consume_and_seal_hash_contract_error(
            fixture,
            monkeypatch,
        )
    )

    def forbidden_evaluator(_request):
        raise AssertionError("postmortem must not execute the evaluator")

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        forbidden_evaluator,
    )
    adjudication = adjudicate_native_bitnet_m2_final_confirmation(
        fixture.protocol,
        fixture.authorization,
        out=adjudication_path,
        confirm_adjudicate=True,
    )

    assert adjudication["format"] == FINAL_ADJUDICATION_FORMAT
    assert adjudication["status"] == "pass"
    assert adjudication["milestone_2_passed"] is True
    assert adjudication["adjudication"]["model_or_evaluator_executed"] is False
    assert adjudication["original_attempt"]["status"] == "error"
    assert adjudication["checks"]["protocol_token_identities"] == {
        "canonical_hash_algorithm": True,
        "full_sequence_hashes": True,
        "full_token_lengths": True,
        "scored_prefix_length": True,
    }
    assert before == {
        "result": sha256_file(fixture.out),
        "marker": sha256_file(fixture.marker),
        "raw": sha256_file(raw),
    }

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="overwrite an existing final adjudication",
    ):
        adjudicate_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=adjudication_path,
            confirm_adjudicate=True,
        )


def test_postmortem_rejects_evidence_resealed_in_a_later_commit(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    raw, adjudication_path, _before = (
        _consume_and_seal_hash_contract_error(
            fixture,
            monkeypatch,
        )
    )
    authorization = json.loads(fixture.authorization.read_text())
    audit = authorization["audit"]
    seal = (
        fixture.root
        / audit["directory"]
        / audit["dataset_key"]
        / f"{audit['attempt_key']}.evidence-seal.json"
    )
    forged = json.loads(raw.read_text())
    forged["quality"]["mean_kl_divergence"] = 0.02
    atomic_json(raw, forged)
    resealed = json.loads(seal.read_text())
    resealed["raw_evaluator_report"]["sha256"] = sha256_file(raw)
    atomic_json(seal, resealed)
    directory = seal.parent.relative_to(fixture.root).as_posix()
    _git(fixture.root, "add", directory)
    _git(
        fixture.root,
        "-c",
        "user.name=Engram Test",
        "-c",
        "user.email=engram-test@example.invalid",
        "commit",
        "-m",
        "attempt to reseal evidence",
    )

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="modified after its unique origin",
    ):
        adjudicate_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=adjudication_path,
            confirm_adjudicate=True,
        )
    assert not adjudication_path.exists()


def test_postmortem_rejects_head_change_before_publication(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    _raw, adjudication_path, _before = (
        _consume_and_seal_hash_contract_error(
            fixture,
            monkeypatch,
        )
    )

    def advance_head(*args, **kwargs):
        result = _REAL_REPORT_VALIDATOR(*args, **kwargs)
        _write(fixture.root / "post-preflight.txt", b"advance HEAD")
        _git(fixture.root, "add", "post-preflight.txt")
        _git(
            fixture.root,
            "-c",
            "user.name=Engram Test",
            "-c",
            "user.email=engram-test@example.invalid",
            "commit",
            "-m",
            "advance after adjudication preflight",
        )
        return result

    monkeypatch.setattr(
        final_confirmation,
        "_validate_evaluator_report",
        advance_head,
    )
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="Git commit changed before publication",
    ):
        adjudicate_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=adjudication_path,
            confirm_adjudicate=True,
        )
    assert not adjudication_path.exists()


@pytest.mark.parametrize("target", ["protocol", "authorization"])
def test_final_runner_rejects_control_manifest_mutation_during_execution(
    tmp_path,
    target,
    monkeypatch,
):
    fixture = _fixture(tmp_path)

    def mutating(request):
        report = _passing_report(request)
        path = (
            request.protocol_path
            if target == "protocol"
            else request.authorization_manifest_path
        )
        path.write_bytes(path.read_bytes() + b" ")
        return report

    monkeypatch.setattr(final_confirmation, "_default_evaluator", mutating)
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="changed during execution",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )

    result = json.loads(fixture.out.read_text())
    marker = json.loads(fixture.marker.read_text())
    assert result["status"] == "error"
    assert result["opened"] is True
    assert marker["status"] == "error"


def test_final_runner_persists_a_gate_failure_without_retry(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)

    def failing_report(request):
        report = _passing_report(request)
        report["quality"]["candidate_nll"] = 2.051
        report["quality"]["nll_delta"] = 0.051
        atomic_json(request.raw_report, report)
        return report

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        failing_report,
    )
    result = run_native_bitnet_m2_final_confirmation(
        fixture.protocol,
        fixture.authorization,
        out=fixture.out,
        opened_marker=fixture.marker,
        confirm_open=True,
    )

    assert result["status"] == "fail"
    assert result["milestone_2_passed"] is False
    assert result["checks"]["quality"]["nll_delta"] is False
    assert json.loads(fixture.marker.read_text())["status"] == "fail"


@pytest.mark.parametrize("forgery", ["full_hashes_as_prefix", "aggregate"])
def test_final_runner_rejects_scored_prefix_identity_forgery(
    tmp_path,
    monkeypatch,
    forgery,
):
    fixture = _fixture(tmp_path)

    def forged_identity(request):
        report = _passing_report(request)
        if forgery == "full_hashes_as_prefix":
            report["dataset"]["sequence_token_ids_sha256"] = report[
                "dataset"
            ]["canonical_full_sequence_token_ids_sha256"]
        else:
            report["dataset"]["input_token_ids_sha256"] = "0" * 64
        atomic_json(request.raw_report, report)
        return report

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        forged_identity,
    )
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="scored-prefix token identities differ",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert json.loads(fixture.out.read_text())["decision"] == (
        "final_holdout_consumed_with_error"
    )


def test_final_runner_derives_recall_gate_from_integer_counts(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)

    def forged_summary(request):
        report = _passing_report(request)
        first = report["debug_recall"]["layers"]["0"]
        first["candidate_hits"] = 962
        first_recall = first["candidate_hits"] / first["target_records"]
        first["candidate_micro_recall"] = first_recall
        first["candidate_mean_row_recall"] = first_recall
        first["candidate_p05_row_recall"] = 0.90
        first["candidate_minimum_row_recall"] = 0.80
        global_recall = report["debug_recall"]["global"]
        global_recall["candidate_hits"] = (
            first["candidate_hits"]
            + report["debug_recall"]["layers"]["1"]["candidate_hits"]
        )
        global_recall["candidate_micro_recall"] = (
            global_recall["candidate_hits"] / global_recall["target_records"]
        )
        global_recall["macro_mean_layer_recall"] = (first_recall + 1.0) / 2
        global_recall["candidate_minimum_layer_mean_recall"] = first_recall
        # Deliberately retain every evaluator-supplied pass Boolean.
        atomic_json(request.raw_report, report)
        return report

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        forged_summary,
    )
    result = run_native_bitnet_m2_final_confirmation(
        fixture.protocol,
        fixture.authorization,
        out=fixture.out,
        opened_marker=fixture.marker,
        confirm_open=True,
    )

    assert result["status"] == "fail"
    assert (
        result["checks"]["candidate_recall"]["global_micro_membership_recall"] is True
    )
    assert result["checks"]["candidate_recall"]["each_layer_mean_recall"] is False


def test_final_runner_requires_explicit_open_and_production_evaluator(tmp_path):
    fixture = _fixture(tmp_path)
    with pytest.raises(Milestone2FinalConfirmationError, match="remains sealed"):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
        )
    assert not fixture.out.exists()
    assert not fixture.marker.exists()


@pytest.mark.parametrize("existing", ["result", "marker", "raw"])
def test_final_runner_rejects_any_existing_audit_file(tmp_path, existing):
    fixture = _fixture(tmp_path)
    authorization = json.loads(fixture.authorization.read_text())
    raw = fixture.root / authorization["audit"]["raw_report"]
    path = {
        "result": fixture.out,
        "marker": fixture.marker,
        "raw": raw,
    }[existing]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status":"existing"}\n')

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="marker/result already exists",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )


def test_compiler_build_manifest_rebuilds_and_binds_outputs(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    build = fixture.root / "build"
    compiler = _write(build / "fixture-cxx", b"fixture compiler binary")
    _write(
        build / "CMakeCache.txt",
        (
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            f"CMAKE_CXX_COMPILER:FILEPATH={compiler}\n"
            "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        ).encode(),
    )
    _write(build / "build.ninja", b"rule CXX\n  command = fixture\n")
    _write(
        build / "CMakeFiles" / "3.31.4" / "CMakeCXXCompiler.cmake",
        (
            'set(CMAKE_CXX_COMPILER_ID "GNU")\n'
            'set(CMAKE_CXX_COMPILER_VERSION "13.3.0")\n'
        ).encode(),
    )
    dense = build / "libengram_bitnet.so"
    dip = build / "libengram_bitnet_dip.so"
    output = build / "generated-provenance.json"
    calls: list[tuple[Path, Path]] = []

    def fake_rebuild(root: Path, build_directory: Path):
        calls.append((root, build_directory))
        dense.write_bytes(b"fresh dense target")
        dip.write_bytes(b"fresh DIP target")
        return (
            "cmake",
            "--build",
            str(build_directory),
            "--target",
            "engram_bitnet",
            "engram_bitnet_dip",
            "--clean-first",
        )

    monkeypatch.setattr(
        final_confirmation,
        "_execute_native_target_rebuild",
        fake_rebuild,
    )

    actual = write_native_bitnet_m2_compiler_build_manifest(
        fixture.root,
        build,
        dense_native_library=dense,
        native_library=dip,
        out=output,
        required_tracked_paths=(fixture.protocol,),
        sealed_paths=(fixture.dataset,),
    )
    first = actual.read_bytes()
    assert actual == output
    assert calls == [(fixture.root, build)]
    assert (
        write_native_bitnet_m2_compiler_build_manifest(
            fixture.root,
            build,
            dense_native_library=dense,
            native_library=dip,
            out=output,
            required_tracked_paths=(fixture.protocol,),
            sealed_paths=(fixture.dataset,),
        ).read_bytes()
        == first
    )
    assert calls == [(fixture.root, build), (fixture.root, build)]

    report = json.loads(first)
    assert report["format"] == COMPILER_BUILD_FORMAT
    assert report["implementation"]["clean_worktree"] is True
    assert report["implementation"]["sealed_tracked_paths_not_read"] == [
        fixture.dataset.relative_to(fixture.root).as_posix()
    ]
    assert report["implementation"]["post_build_clean_worktree"] is True
    dependency_paths = {
        row["path"] for row in report["implementation"]["tracked_dependencies"]
    }
    assert {
        "CMakeLists.txt",
        "native/include/engram/fixture.h",
        "native/src/fixture.cpp",
        fixture.protocol.relative_to(fixture.root).as_posix(),
    }.issubset(dependency_paths)
    assert report["build"]["invocation"]["command"] == [
        "cmake",
        "--build",
        str(build),
        "--target",
        "engram_bitnet",
        "engram_bitnet_dip",
        "--clean-first",
    ]
    assert report["outputs"]["dense_native_library"]["sha256"] == sha256_file(dense)
    assert report["outputs"]["native_library"]["sha256"] == sha256_file(dip)

    def guarded_sha256(path):
        resolved = Path(path).resolve()
        if resolved == fixture.dataset:
            raise AssertionError("sealed dataset contents were hashed")
        return sha256_file(resolved)

    monkeypatch.setattr(
        final_confirmation,
        "sha256_file",
        guarded_sha256,
    )
    final_confirmation._verify_compiler_build_provenance(
        fixture.root,
        report,
        expected_commit=_git(fixture.root, "rev-parse", "HEAD"),
        output_paths={
            "dense_native_library": dense,
            "native_library": dip,
        },
        output_sha256={
            "dense_native_library": sha256_file(dense),
            "native_library": sha256_file(dip),
        },
        sealed_paths=(fixture.dataset,),
    )
    forged = json.loads(json.dumps(report))
    forged["implementation"]["requested_tracked_paths"].append(
        fixture.dataset.relative_to(fixture.root).as_posix()
    )
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="sealed protocol input is declared",
    ):
        final_confirmation._verify_compiler_build_provenance(
            fixture.root,
            forged,
            expected_commit=_git(fixture.root, "rev-parse", "HEAD"),
            output_paths={
                "dense_native_library": dense,
                "native_library": dip,
            },
            output_sha256={
                "dense_native_library": sha256_file(dense),
                "native_library": sha256_file(dip),
            },
            sealed_paths=(fixture.dataset,),
        )

    dip.write_bytes(b"changed DIP library")
    assert (
        write_native_bitnet_m2_compiler_build_manifest(
            fixture.root,
            build,
            dense_native_library=dense,
            native_library=dip,
            out=output,
            required_tracked_paths=(fixture.protocol,),
            sealed_paths=(fixture.dataset,),
        ).read_bytes()
        == first
    )
    assert dip.read_bytes() == b"fresh DIP target"


def test_authorization_generator_never_hashes_fixture_holdout(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    artifacts = fixture.pins
    bindings = {
        "package_manifest": {
            **artifacts["package_manifest"],
            "path": str(fixture.root / artifacts["package_manifest"]["path"]),
        },
        "base_record_artifact": {
            **artifacts["base_artifact"],
            "path": str(fixture.root / artifacts["base_artifact"]["path"]),
        },
        "coordinate_index": {
            **artifacts["coordinate_index"],
            "path": str(fixture.root / artifacts["coordinate_index"]["path"]),
        },
        "tokenizer_json": {
            **artifacts["tokenizer_json"],
            "path": str(fixture.root / artifacts["tokenizer_json"]["path"]),
        },
        "dense_reference_library": {
            **artifacts["dense_native_library"],
            "path": str(fixture.root / artifacts["dense_native_library"]["path"]),
        },
        "dip_native_library": {
            **artifacts["native_library"],
            "path": str(fixture.root / artifacts["native_library"]["path"]),
        },
    }
    import engram.semantic.native_bitnet_dip_policy_manifest as policy_module

    monkeypatch.setattr(
        policy_module,
        "load_native_bitnet_dip_policy_manifest",
        lambda _path: SimpleNamespace(payload={"bindings": bindings}),
    )
    original_sha256_file = final_confirmation.sha256_file

    def guarded_sha256(path):
        assert Path(path).resolve() != fixture.dataset.resolve()
        return original_sha256_file(Path(path))

    monkeypatch.setattr(final_confirmation, "sha256_file", guarded_sha256)
    package_manifest = fixture.root / artifacts["package_manifest"]["path"]
    authorization = tmp_path / "generated-authorization.json"
    actual = write_native_bitnet_m2_final_authorization_manifest(
        fixture.root,
        protocol=fixture.protocol,
        package_manifest=package_manifest,
        base_artifact=fixture.root / artifacts["base_artifact"]["path"],
        coordinate_index=fixture.root / artifacts["coordinate_index"]["path"],
        policy_manifest=fixture.root / artifacts["policy_manifest"]["path"],
        tokenizer_json=fixture.root / artifacts["tokenizer_json"]["path"],
        dense_native_library=fixture.root / artifacts["dense_native_library"]["path"],
        native_library=fixture.root / artifacts["native_library"]["path"],
        compiler_build_manifest=fixture.root
        / artifacts["compiler_build_manifest"]["path"],
        threads=2,
        out=authorization,
    )

    payload = json.loads(actual.read_text())
    assert payload["trust_chain"] == {
        "acyclic": True,
        "policy_binds_compiler_build_manifest": False,
        "authorization_independently_pins_policy_and_compiler_build": True,
        "compiler_build_binds_same_native_libraries": True,
        "dataset_contents_read": False,
    }
    assert payload["artifacts"]["dataset"] == artifacts["dataset"]
    assert (
        write_native_bitnet_m2_final_authorization_manifest(
            fixture.root,
            protocol=fixture.protocol,
            package_manifest=package_manifest,
            base_artifact=fixture.root / artifacts["base_artifact"]["path"],
            coordinate_index=fixture.root / artifacts["coordinate_index"]["path"],
            policy_manifest=fixture.root / artifacts["policy_manifest"]["path"],
            tokenizer_json=fixture.root / artifacts["tokenizer_json"]["path"],
            dense_native_library=fixture.root
            / artifacts["dense_native_library"]["path"],
            native_library=fixture.root / artifacts["native_library"]["path"],
            compiler_build_manifest=fixture.root
            / artifacts["compiler_build_manifest"]["path"],
            threads=2,
            out=authorization,
        )
        == authorization
    )


def test_default_adapter_forwards_exact_native_final_contract(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    captured = {}
    import engram.evaluation.native_bitnet_dip_native_causal as causal_module

    def fake(package, coordinate_index, dataset, **kwargs):
        captured.update(
            {
                "package": package,
                "coordinate_index": coordinate_index,
                "dataset": dataset,
                **kwargs,
            }
        )
        atomic_json(kwargs["out"], {"fixture": True})
        return {"fixture": True}

    monkeypatch.setattr(
        causal_module,
        "evaluate_native_bitnet_dip_native_causal",
        fake,
    )
    artifacts = fixture.pins
    request = Milestone2FinalRequest(
        protocol_path=fixture.protocol,
        authorization_manifest_path=fixture.authorization,
        package=(fixture.root / artifacts["package_manifest"]["path"]).parent,
        package_manifest=fixture.root / artifacts["package_manifest"]["path"],
        record_artifact=fixture.root / artifacts["base_artifact"]["path"],
        coordinate_index=fixture.root / artifacts["coordinate_index"]["path"],
        policy_manifest=fixture.root / artifacts["policy_manifest"]["path"],
        tokenizer_json=fixture.root / artifacts["tokenizer_json"]["path"],
        dense_native_library=fixture.root / artifacts["dense_native_library"]["path"],
        native_library=fixture.root / artifacts["native_library"]["path"],
        compiler_build_manifest=fixture.root
        / artifacts["compiler_build_manifest"]["path"],
        dataset=fixture.dataset,
        raw_report=tmp_path / "adapter-raw.json",
        record_offset=0,
        sequence_count=8,
        predictions_per_sequence=32,
        threads=2,
        reference_top_ks=(4, 5),
        expected_sha256={
            name: str(descriptor["sha256"]) for name, descriptor in artifacts.items()
        },
    )

    assert _REAL_DEFAULT_EVALUATOR(request) == {"fixture": True}
    assert captured["dataset_role"] == "final"
    assert captured["debug_recall"] is True
    assert captured["reference_top_ks"] == (4, 5)
    assert captured["expected_layer_count"] == 2
    assert captured["dense_library"] == request.dense_native_library
    assert captured["dip_library"] == request.native_library
    parameters = inspect.signature(run_native_bitnet_m2_final_confirmation).parameters
    assert "evaluator" not in parameters
    assert "allow_test_evaluator" not in parameters


def test_runtime_source_origins_must_be_committed_under_pinned_root(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)
    evaluator_module = importlib.import_module(
        "engram.evaluation.native_bitnet_dip_native_causal"
    )

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="does not originate from the pinned repository",
    ):
        final_confirmation._verify_runtime_source_origins(fixture.root)

    runner_source = fixture.root / "src/engram/evaluation/native_bitnet_m2_final.py"
    evaluator_source = (
        fixture.root / "src/engram/evaluation/native_bitnet_dip_native_causal.py"
    )
    monkeypatch.setattr(final_confirmation, "__file__", str(runner_source))
    monkeypatch.setattr(evaluator_module, "__file__", str(evaluator_source))
    assert final_confirmation._verify_runtime_source_origins(fixture.root) == (
        runner_source,
        evaluator_source,
    )


def test_final_runner_rejects_symlinked_audit_directory_before_opening(
    tmp_path,
):
    fixture = _fixture(tmp_path)
    external = tmp_path / "external-audit"
    external.mkdir()
    audit_directory = fixture.root / final_confirmation.FINAL_AUDIT_DIRECTORY
    audit_directory.symlink_to(external, target_is_directory=True)

    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="must not traverse a symbolic link",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert not any(external.iterdir())


def test_final_runner_rejects_forged_python_native_parity_summary(
    tmp_path,
    monkeypatch,
):
    fixture = _fixture(tmp_path)

    def forged_summary(request):
        report = _passing_report(request)
        report["python_native_parity"]["layers"]["0"]["checks"]["output_bf16"] = False
        atomic_json(request.raw_report, report)
        return report

    monkeypatch.setattr(
        final_confirmation,
        "_default_evaluator",
        forged_summary,
    )
    with pytest.raises(
        Milestone2FinalConfirmationError,
        match="Python/native parity failed at layer 0",
    ):
        run_native_bitnet_m2_final_confirmation(
            fixture.protocol,
            fixture.authorization,
            out=fixture.out,
            opened_marker=fixture.marker,
            confirm_open=True,
        )
    assert json.loads(fixture.marker.read_text())["status"] == "error"
