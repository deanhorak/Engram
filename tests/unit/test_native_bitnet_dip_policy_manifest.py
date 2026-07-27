import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import engram.compiler.native_bitnet as native_bitnet_compiler
from engram.compiler.native_bitnet import (
    NATIVE_BITNET_DIP_OPERATOR,
    _validate_semantic_memory_descriptor,
    _verified_package_manifest,
    install_native_bitnet_semantic_memory,
)
from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    NativeBitNetValidationError,
    load_native_bitnet_artifact,
    save_native_bitnet_artifact,
)
from engram.semantic.native_bitnet_dip_index import (
    build_native_bitnet_dip_index,
)
from engram.semantic.native_bitnet_dip_policy_manifest import (
    NATIVE_BITNET_DIP_POLICY_FORMAT,
    NativeBitNetDIPPolicyManifestError,
    build_native_bitnet_dip_policy_manifest,
    load_native_bitnet_dip_policy_manifest,
)
from engram.utils import sha256_file, sha256_json


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _descriptor(path: Path):
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


@pytest.fixture()
def freeze_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        native_bitnet_compiler,
        "_APPROVED_NATIVE_BITNET_M2_ADJUDICATIONS",
        {},
    )
    hidden = 320
    intermediate = 320
    layer_count = 30
    package = tmp_path / "model.engram-bitnet"
    base = package / "mlp" / "model.bitnet-records.bin"
    base.parent.mkdir(parents=True)
    row = np.arange(intermediate, dtype=np.int32)[:, None]
    column = np.arange(hidden, dtype=np.int32)[None, :]
    layers = []
    for layer in range(layer_count):
        gate = ((row + column + layer) % 3 - 1).astype(np.int8)
        up = ((2 * row + column + layer + 1) % 3 - 1).astype(np.int8)
        down = ((column.T + 2 * row.T + layer) % 3 - 1).astype(np.int8)
        layers.append(
            NativeBitNetLayerWeights(
                gate_codes=gate,
                up_codes=up,
                down_codes=down,
                gate_scale=0.5,
                up_scale=0.25,
                down_scale=0.125,
                ffn_sub_norm=np.ones(intermediate, dtype=np.float32),
            )
        )
    save_native_bitnet_artifact(base, layers, rms_norm_eps=1e-5)
    artifact = load_native_bitnet_artifact(base)

    tokenizer = package / "tokenizer" / "tokenizer.json"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_text('{"version":"fixture"}\n', encoding="utf-8")
    manifest = {
        "format": "engram-native-bitnet",
        "version": 1,
        "model": {
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": layer_count,
        },
        "mlp": {
            "path": "mlp/model.bitnet-records.bin",
            "sha256": sha256_file(base),
            "serialized_bytes": base.stat().st_size,
        },
        "runtime": {
            "device": "cpu",
            "dtype": "bfloat16",
            "kernel_threads": 2,
            "attention_mode": "dense_reference",
        },
        "files": {
            "mlp/model.bitnet-records.bin": {
                "sha256": sha256_file(base),
                "bytes": base.stat().st_size,
            },
            "tokenizer/tokenizer.json": {
                "sha256": sha256_file(tokenizer),
                "bytes": tokenizer.stat().st_size,
            },
        },
    }
    package_manifest = package / "manifest.json"
    _write_json(package_manifest, manifest)

    rms = [
        {
            "estimator": (
                "corrected_proxy" if layer == 9 else "candidate_ratio"
            ),
            "audit_count": 8 if layer == 9 else 0,
            "audit_strategy": (
                "top_proxy_raw_square" if layer == 9 else "none"
            ),
        }
        for layer in range(layer_count)
    ]
    fit_hash = "1" * 64
    development_hash = "2" * 64
    final_hash = "3" * 64
    proposal = {
        "experiment": "native_bitnet_dip_joint_candidate_adaptive_k_policy",
        "artifact_sha256": artifact.payload_sha256,
        "decision": "use_joint_policy_for_candidate_only_causal_development",
        "configuration": {
            "input_fraction": 0.75,
            "input_coordinates": 240,
            "minimum_k": 2,
            "energy_target": 1.0,
        },
        "selected_policy": {
            "candidate_counts": [32] * layer_count,
            "maximum_ks": [16] * layer_count,
            "rms_policies": rms,
        },
        "validation_trace": {
            "dataset_hash": fit_hash,
            "causal_or_final_confirmation_corpus_used": False,
        },
        "progression_screen": {"passed": True},
        # This stale local report is deliberately present.  It is retained
        # only as non-approving provenance by the freeze builder.
        "physical_layout": {
            "format": "native_bitnet_dip_dual_layout_v1",
            "layout": {
                "index_header_bytes": 64,
                "index_layer_header_bytes": 64,
            },
        },
    }
    proposal_path = tmp_path / "proposal.json"
    _write_json(proposal_path, proposal)
    index = tmp_path / "model.bitnet-dip-index.bin"
    build_native_bitnet_dip_index(base, proposal_path, index)

    dense_library = tmp_path / "libengram_bitnet.so"
    dip_library = tmp_path / "libengram_bitnet_dip.so"
    dense_library.write_bytes(b"dense-native-fixture")
    dip_library.write_bytes(b"dip-native-fixture")
    protocol = {
        "experiment": (
            "native_bitnet_milestone_2_practical_semantic_memory_confirmation"
        ),
        "protocol_version": 5,
        "configuration_fit": {
            "role": "noncanonical_policy_proposal_only",
            "dataset_sha256": fit_hash,
            "stored_dtype": "float16",
        },
        "causal_development": {
            "full_length_dataset": {
                "sha256": development_hash,
                "allowed_record_offsets": [0, 8],
                "records_per_offset": 8,
                "prediction_positions_per_record": 32,
            }
        },
        "final_confirmation": {"dataset_sha256": final_hash},
        "quality_thresholds": {
            "maximum_teacher_student_kl": 0.05,
            "minimum_top1_agreement": 0.9,
            "maximum_nll_delta": 0.05,
            "maximum_final_hidden_relative_l2": 0.1,
        },
        "practical_router_thresholds": {
            "maximum_mean_active_record_fraction": 0.25,
            "maximum_complete_physical_cold_traffic_fraction_of_dense_q4": 0.45,
            "minimum_held_out_candidate_recall": 0.95,
            "cpu_only_inference_required": True,
            "dense_gate_up_or_down_fallback_allowed": False,
        },
        "candidate_recall_definition": {
            "reference_top_ks": [16] * layer_count,
        },
        "configuration": None,
        "final_result": None,
    }
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)

    configuration = {}
    for layer in range(layer_count):
        configuration[str(layer)] = {
            "input_coordinates": 240,
            "candidate_count": 32,
            "top_k": 16,
            "minimum_top_k": 2,
            "maximum_top_k": 16,
            "energy_target": 1.0,
            "rms_estimator": rms[layer]["estimator"],
            "rms_audit_count": rms[layer]["audit_count"],
            "rms_audit_strategy": rms[layer]["audit_strategy"],
            "rms_variance_scale": 1.0,
            "rms_variance_bias": 0.0,
            "output_scale": 1.0,
        }
    schedules = [[16] * layer_count for _ in range(256)]
    physical = native_bitnet_dip_physical_accounting(
        hidden,
        intermediate,
        input_counts=[240] * layer_count,
        candidate_counts=[32] * layer_count,
        top_ks=schedules[0],
    )
    token_bytes = physical["traffic"]["complete_modelled_cold_bytes"]
    dense_bytes = physical["traffic"]["dense_q4_bytes"]
    token_fraction = token_bytes / dense_bytes
    worst_layer = max(
        physical["traffic"]["layers"],
        key=lambda layer: layer["fraction_of_dense_q4"],
    )
    selected_sum = 256 * layer_count * 16
    selected_count = 256 * layer_count
    artifacts = {
        "package_manifest": _descriptor(package_manifest),
        "base_record_artifact": _descriptor(base),
        "coordinate_index": _descriptor(index),
        "dense_kernel_library": _descriptor(dense_library),
        "dip_kernel_library": _descriptor(dip_library),
    }
    nll_reference = 2.0
    nll_candidate = 2.02
    development = {
        "experiment": "native_bitnet_dip_native_causal",
        "dataset_role": "development",
        "scope": "all_mlp_layers",
        "configuration": configuration,
        "dataset": {
            "path": "development.jsonl",
            "sha256": development_hash,
            "record_offset": 0,
            "sequence_count": 8,
            "predictions_per_sequence": 32,
            "prediction_positions": 256,
        },
        "execution": {
            "input_boundary": "live_native_bf16",
            "kernel": "native_cpu",
            "device": "cpu",
            "dense_fallback": False,
            "all_mlp_layers_substituted": True,
            "serialized_index_reloaded": True,
            "python_native_parity_passed": True,
        },
        "evidence_observed": {
            "sequences": 8,
            "unique_sequences": 8,
            "predictions_per_sequence": 32,
            "prediction_positions": 256,
            "all_mlp_layers": True,
            "layer_count": layer_count,
            "layers_executed": list(range(layer_count)),
        },
        "quality": {
            "mean_kl_divergence": 0.01,
            "top1_agreement": 0.95,
            "reference_nll": nll_reference,
            "candidate_nll": nll_candidate,
            "nll_delta": nll_candidate - nll_reference,
            "final_hidden_relative_l2": 0.05,
            "passed": True,
        },
        "quality_passed": True,
        "reference_top_ks": {
            "values": [16] * layer_count,
            "sha256": sha256_json([16] * layer_count),
            "role": "frozen_fixed_per_layer_candidate_recall_denominator",
        },
        "selected_records": {
            "per_token_layer_k": schedules,
            "global": {
                "sum": selected_sum,
                "count": selected_count,
                "minimum": 16,
                "maximum": 16,
                "mean": 16.0,
                "active_fraction": 16 / intermediate,
            },
        },
        "physical_cold_traffic": {
            "accounting_version": "native_bitnet_dip_dual_layout_v2",
            "global": {
                "scheduled_cache_line_bytes": token_bytes * 256,
                "dense_q4_bytes": dense_bytes * 256,
                "fraction_of_dense_q4": token_fraction,
            },
            "per_token": [
                {
                    "token": token,
                    "scheduled_cache_line_bytes": token_bytes,
                    "dense_q4_bytes": dense_bytes,
                    "fraction_of_dense_q4": token_fraction,
                }
                for token in range(256)
            ],
            "worst_token": {
                "token": 0,
                "fraction_of_dense_q4": token_fraction,
            },
            "worst_layer": {
                "token": 0,
                "layer": worst_layer["layer"],
                "fraction_of_dense_q4": worst_layer[
                    "fraction_of_dense_q4"
                ],
            },
            "passes_45_percent": True,
        },
        "artifacts": artifacts,
        "debug_recall": {
            "global": {
                "rows": 30 * 256,
                "target_records": 30 * 256 * 16,
                "candidate_hits": 30 * 4_000,
                "candidate_micro_recall": 4_000 / (256 * 16),
                "macro_mean_layer_recall": 4_000 / (256 * 16),
                "candidate_minimum_layer_mean_recall": (
                    4_000 / (256 * 16)
                ),
                "global_micro_passes_95_percent": True,
                "every_layer_mean_passes_95_percent": True,
                "passes_95_percent": True,
                "minimum_candidate_recall": 0.95,
            },
            "layers": {
                str(layer): {
                    "layer": layer,
                    "rows": 256,
                    "reference_top_k": 16,
                    "target_records": 256 * 16,
                    "candidate_hits": 4_000,
                    "candidate_micro_recall": 4_000 / (256 * 16),
                    "candidate_mean_row_recall": 4_000 / (256 * 16),
                }
                for layer in range(layer_count)
            },
        },
        "candidate_recall_passed": True,
        "systems_evidence_passed": True,
        "protocol_qualifying": True,
        "overall_gate_passed": True,
        "decision": "freeze_policy_and_run_protected_final_confirmation",
        "milestone_2_status": "development_gate_passed_pending_final",
    }
    development_path = tmp_path / "development.json"
    _write_json(development_path, development)
    parity = {
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
            "rows_per_layer": 6,
            "total_rows": 6 * layer_count,
            "input_tokens": 33,
        },
        "dataset": {"sha256": development_hash},
        "equality": {
            "input_coordinate_ids": True,
            "candidate_ids": True,
            "selected_record_ids": True,
            "selected_counts": True,
            "output_bf16_bits": True,
        },
        "artifacts": {
            key: artifacts[key]
            for key in (
                "package_manifest",
                "base_record_artifact",
                "coordinate_index",
                "dip_kernel_library",
            )
        },
        "layers": [
            {
                "layer": layer,
                "rows": 6,
                "row_indices": [0, 32, 1, 31, 8, 24],
                "selected_counts": [2, 16, 3, 15, 8, 12],
                "includes_observed_minimum_k": True,
                "includes_observed_maximum_k": True,
                "equality": {
                    "input_coordinate_ids": True,
                    "candidate_ids": True,
                    "selected_record_ids": True,
                    "selected_counts": True,
                    "output_bf16_bits": True,
                },
            }
            for layer in range(layer_count)
        ],
    }
    parity_path = tmp_path / "parity.json"
    _write_json(parity_path, parity)
    return {
        "package": package,
        "base": base,
        "index": index,
        "dense_library": dense_library,
        "dip_library": dip_library,
        "proposal": proposal_path,
        "development": development_path,
        "parity": parity_path,
        "protocol": protocol_path,
    }


def _build(fixture, out):
    return build_native_bitnet_dip_policy_manifest(
        proposal_report=fixture["proposal"],
        development_report=fixture["development"],
        native_parity_report=fixture["parity"],
        frozen_protocol=fixture["protocol"],
        package=fixture["package"],
        record_artifact=fixture["base"],
        coordinate_index=fixture["index"],
        dense_library=fixture["dense_library"],
        dip_library=fixture["dip_library"],
        out=out,
    )


def _passing_adjudication(fixture, policy, out):
    checks = {}
    for group, name in (
        native_bitnet_compiler._REQUIRED_M2_ADJUDICATION_CHECKS
    ):
        checks.setdefault(group, {})[name] = True
    payload = {
        "format": "engram-native-bitnet-m2-final-adjudication",
        "version": 1,
        "status": "pass",
        "milestone_2_passed": True,
        "decision": (
            "milestone_2_semantic_gate_passed_by_postmortem_adjudication"
        ),
        "adjudication": {
            "model_or_evaluator_executed": False,
            "original_result_rewritten": False,
            "holdout_reused_for_configuration": False,
        },
        "checks": checks,
        "input_sha256": {
            "package_manifest": sha256_file(
                fixture["package"] / "manifest.json"
            ),
            "base_artifact": sha256_file(fixture["base"]),
            "coordinate_index": sha256_file(fixture["index"]),
            "policy_manifest": sha256_file(policy),
        },
    }
    _write_json(out, payload)
    _authorize_adjudication(fixture, policy, out)
    return out


def _authorize_adjudication(fixture, policy, adjudication):
    native_bitnet_compiler._APPROVED_NATIVE_BITNET_M2_ADJUDICATIONS[
        sha256_file(adjudication)
    ] = {
        "package_manifest": sha256_file(
            fixture["package"] / "manifest.json"
        ),
        "base_artifact": sha256_file(fixture["base"]),
        "coordinate_index": sha256_file(fixture["index"]),
        "policy_manifest": sha256_file(policy),
    }


def test_semantic_memory_installer_derives_fail_closed_dip_package(
    freeze_fixture,
    tmp_path,
):
    policy = _build(freeze_fixture, tmp_path / "frozen-policy.json")
    adjudication = _passing_adjudication(
        freeze_fixture,
        policy,
        tmp_path / "adjudication.json",
    )
    source_manifest_hash = sha256_file(
        freeze_fixture["package"] / "manifest.json"
    )
    out = tmp_path / "derived.engram-bitnet"

    installed = install_native_bitnet_semantic_memory(
        freeze_fixture["package"],
        freeze_fixture["index"],
        policy,
        adjudication,
        out,
        coordinate_index_sha256=sha256_file(freeze_fixture["index"]),
        policy_manifest_sha256=sha256_file(policy),
        adjudication_sha256=sha256_file(adjudication),
    )

    assert installed == out.resolve()
    assert sha256_file(freeze_fixture["package"] / "manifest.json") == (
        source_manifest_hash
    )
    assert not (
        freeze_fixture["package"] / "mlp/model.bitnet-dip-index.bin"
    ).exists()
    manifest = _verified_package_manifest(installed)
    semantic = manifest["semantic_memory"]
    assert semantic["operator"] == NATIVE_BITNET_DIP_OPERATOR
    assert semantic["runtime_scope"] == "native_token_runtime"
    assert semantic["dense_fallback"] is False
    assert semantic["all_mlp_layers_substituted"] is True
    assert semantic["source_package_manifest_sha256"] == source_manifest_hash
    assert manifest["runtime"]["mlp_mode"] == NATIVE_BITNET_DIP_OPERATOR
    assert manifest["runtime"]["attention_mode"] == (
        "native_streaming_w16_c8_k4_sinks2"
    )
    _validate_semantic_memory_descriptor(installed, manifest)
    from engram.runtime.native_bitnet import NativeBitNetRuntime

    with pytest.raises(ValueError, match="require the native token runtime"):
        NativeBitNetRuntime(installed)

    assert (
        install_native_bitnet_semantic_memory(
            freeze_fixture["package"],
            freeze_fixture["index"],
            policy,
            adjudication,
            out,
            coordinate_index_sha256=sha256_file(freeze_fixture["index"]),
            policy_manifest_sha256=sha256_file(policy),
            adjudication_sha256=sha256_file(adjudication),
        )
        == installed
    )

    # An attacker cannot bless a corrupt internal index by merely updating
    # the package-level file hash and semantic descriptor.
    packaged_index = installed / "mlp/model.bitnet-dip-index.bin"
    with packaged_index.open("r+b") as handle:
        handle.seek(-65, 2)
        original = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([original[0] ^ 1]))
    manifest = json.loads((installed / "manifest.json").read_text())
    corrupted_sha256 = sha256_file(packaged_index)
    manifest["semantic_memory"]["sha256"] = corrupted_sha256
    manifest["files"]["mlp/model.bitnet-dip-index.bin"] = {
        "bytes": packaged_index.stat().st_size,
        "sha256": corrupted_sha256,
    }
    _write_json(installed / "manifest.json", manifest)
    outer_verified = _verified_package_manifest(installed)
    with pytest.raises(
        NativeBitNetValidationError,
        match="checksum mismatch",
    ):
        _validate_semantic_memory_descriptor(installed, outer_verified)


def test_semantic_memory_installer_rejects_non_index_target_tampering(
    freeze_fixture,
    tmp_path,
):
    policy = _build(freeze_fixture, tmp_path / "frozen-policy.json")
    adjudication = _passing_adjudication(
        freeze_fixture,
        policy,
        tmp_path / "adjudication.json",
    )
    installed = install_native_bitnet_semantic_memory(
        freeze_fixture["package"],
        freeze_fixture["index"],
        policy,
        adjudication,
        tmp_path / "derived.engram-bitnet",
        coordinate_index_sha256=sha256_file(freeze_fixture["index"]),
        policy_manifest_sha256=sha256_file(policy),
        adjudication_sha256=sha256_file(adjudication),
    )
    tampered = tmp_path / "tampered.engram-bitnet"
    shutil.copytree(installed, tampered)
    tokenizer = tampered / "tokenizer/tokenizer.json"
    tokenizer.write_text('{"version":"attacker"}\n', encoding="utf-8")
    manifest_path = tampered / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["tokenizer/tokenizer.json"] = {
        "bytes": tokenizer.stat().st_size,
        "sha256": sha256_file(tokenizer),
    }
    _write_json(manifest_path, manifest)

    with pytest.raises(
        NativeBitNetValidationError,
        match="not the exact authenticated source derivation",
    ):
        install_native_bitnet_semantic_memory(
            freeze_fixture["package"],
            freeze_fixture["index"],
            policy,
            adjudication,
            tampered,
            coordinate_index_sha256=sha256_file(freeze_fixture["index"]),
            policy_manifest_sha256=sha256_file(policy),
            adjudication_sha256=sha256_file(adjudication),
        )


def test_python_runtime_rejects_incomplete_or_unknown_dip_modes(
    freeze_fixture,
    tmp_path,
):
    from engram.runtime.native_bitnet import NativeBitNetRuntime

    policy = _build(freeze_fixture, tmp_path / "frozen-policy.json")
    adjudication = _passing_adjudication(
        freeze_fixture,
        policy,
        tmp_path / "adjudication.json",
    )
    installed = install_native_bitnet_semantic_memory(
        freeze_fixture["package"],
        freeze_fixture["index"],
        policy,
        adjudication,
        tmp_path / "derived.engram-bitnet",
        coordinate_index_sha256=sha256_file(freeze_fixture["index"]),
        policy_manifest_sha256=sha256_file(policy),
        adjudication_sha256=sha256_file(adjudication),
    )
    manifest_path = installed / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["runtime"]["mlp_mode"]
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        NativeBitNetRuntime(installed)

    manifest["runtime"]["mlp_mode"] = "native_bitnet_dynamic_input_prunin_v2"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="unsupported or incomplete"):
        NativeBitNetRuntime(installed)


def test_semantic_memory_installer_rejects_mutation_and_false_approval(
    freeze_fixture,
    tmp_path,
):
    policy = _build(freeze_fixture, tmp_path / "frozen-policy.json")
    adjudication = _passing_adjudication(
        freeze_fixture,
        policy,
        tmp_path / "adjudication.json",
    )
    arguments = (
        freeze_fixture["package"],
        freeze_fixture["index"],
        policy,
        adjudication,
    )
    keywords = {
        "coordinate_index_sha256": sha256_file(freeze_fixture["index"]),
        "policy_manifest_sha256": sha256_file(policy),
        "adjudication_sha256": sha256_file(adjudication),
    }
    with pytest.raises(NativeBitNetValidationError, match="outside"):
        install_native_bitnet_semantic_memory(
            *arguments,
            freeze_fixture["package"],
            **keywords,
        )
    descendant = freeze_fixture["package"] / "derived.engram-bitnet"
    with pytest.raises(NativeBitNetValidationError, match="outside"):
        install_native_bitnet_semantic_memory(
            *arguments,
            descendant,
            **keywords,
        )
    assert not descendant.exists()

    payload = json.loads(adjudication.read_text())
    payload["checks"]["quality"]["top1_agreement"] = False
    _write_json(adjudication, payload)
    _authorize_adjudication(freeze_fixture, policy, adjudication)
    with pytest.raises(
        NativeBitNetValidationError,
        match="missing a required passing check",
    ):
        install_native_bitnet_semantic_memory(
            *arguments,
            tmp_path / "rejected.engram-bitnet",
            **{
                **keywords,
                "adjudication_sha256": sha256_file(adjudication),
            },
        )


def test_frozen_policy_manifest_roundtrip_reconciles_and_rebuilds_index(
    freeze_fixture,
    tmp_path,
):
    path = _build(freeze_fixture, tmp_path / "frozen-policy.json")
    payload = json.loads(path.read_text())

    assert payload["format"] == NATIVE_BITNET_DIP_POLICY_FORMAT
    assert payload["status"] == "approved"
    assert len(payload["policy"]["layers"]) == 30
    assert payload["policy"]["layers"][9] == {
        "layer": 9,
        "input_fraction": 0.75,
        "input_coordinates": 240,
        "candidate_count": 32,
        "minimum_top_k": 2,
        "maximum_top_k": 16,
        "energy_target": 1.0,
        "rms_estimator": "corrected_proxy",
        "rms_audit_count": 8,
        "rms_audit_strategy": "top_proxy_raw_square",
        "rms_variance_scale": 1.0,
        "rms_variance_bias": 0.0,
        "output_scale": 1.0,
    }
    assert payload["proposal_provenance"]["approval_authority"] is False
    assert payload["proposal_provenance"]["traffic_evidence_accepted"] is False
    assert (
        payload["development_evidence"]["activity"]["active_fraction"]
        == 0.05
    )
    assert (
        payload["development_evidence"]["physical_cold_traffic"][
            "accounting_version"
        ]
        == "native_bitnet_dip_dual_layout_v2"
    )
    assert payload["storage"]["combined_semantic_mlp_bytes"] == (
        freeze_fixture["base"].stat().st_size
        + freeze_fixture["index"].stat().st_size
    )
    loaded = load_native_bitnet_dip_policy_manifest(
        path,
        expected_sha256=sha256_file(path),
    )
    assert len(loaded.layers) == 30
    assert loaded.layers[9].rms_audit_count == 8

    rebuilt = tmp_path / "rebuilt.bitnet-dip-index.bin"
    build_native_bitnet_dip_index(
        freeze_fixture["base"],
        path,
        rebuilt,
    )
    assert sha256_file(rebuilt) == sha256_file(freeze_fixture["index"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda report, protocol: report["execution"].__setitem__(
                "input_boundary",
                "stored_float16",
            ),
            "input_boundary",
        ),
        (
            lambda report, protocol: report["evidence_observed"].__setitem__(
                "sequences",
                2,
            ),
            "sequences",
        ),
        (
            lambda report, protocol: report[
                "physical_cold_traffic"
            ].__setitem__(
                "accounting_version",
                "native_bitnet_dip_dual_layout_v1",
            ),
            "not v2",
        ),
        (
            lambda report, protocol: report["dataset"].__setitem__(
                "sha256",
                protocol["final_confirmation"]["dataset_sha256"],
            ),
            "declared full-length corpus",
        ),
        (
            lambda report, protocol: report.__setitem__(
                "milestone_2_status",
                "passed",
            ),
            "pending sealed final confirmation",
        ),
    ],
)
def test_freeze_rejects_nonqualifying_development_without_writing(
    freeze_fixture,
    tmp_path,
    mutation,
    message,
):
    report = json.loads(freeze_fixture["development"].read_text())
    protocol = json.loads(freeze_fixture["protocol"].read_text())
    mutation(report, protocol)
    _write_json(freeze_fixture["development"], report)
    destination = tmp_path / "must-not-exist.json"

    with pytest.raises(NativeBitNetDIPPolicyManifestError, match=message):
        _build(freeze_fixture, destination)

    assert not destination.exists()


def test_freeze_recomputes_activity_and_loader_rejects_manifest_tampering(
    freeze_fixture,
    tmp_path,
):
    report = json.loads(freeze_fixture["development"].read_text())
    report["selected_records"]["global"]["sum"] += 1
    _write_json(freeze_fixture["development"], report)
    with pytest.raises(
        NativeBitNetDIPPolicyManifestError,
        match="global.sum",
    ):
        _build(freeze_fixture, tmp_path / "false-summary.json")

    # Restore the exact report, freeze it, then prove the loader does not
    # trust an approval flag or editable summary in isolation.
    report["selected_records"]["global"]["sum"] -= 1
    _write_json(freeze_fixture["development"], report)
    path = _build(freeze_fixture, tmp_path / "valid.json")
    payload = json.loads(path.read_text())
    payload["development_evidence"]["activity"]["sum"] += 1
    tampered = tmp_path / "tampered.json"
    _write_json(tampered, payload)
    with pytest.raises(
        NativeBitNetDIPPolicyManifestError,
        match="differs from reconstructed",
    ):
        load_native_bitnet_dip_policy_manifest(tampered)


def test_freeze_requires_bit_exact_full_artifact_parity(
    freeze_fixture,
    tmp_path,
):
    parity = json.loads(freeze_fixture["parity"].read_text())
    parity["equality"]["selected_record_ids"] = False
    _write_json(freeze_fixture["parity"], parity)

    with pytest.raises(
        NativeBitNetDIPPolicyManifestError,
        match="selected_record_ids",
    ):
        _build(freeze_fixture, tmp_path / "no-parity.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["debug_recall"]["global"].__setitem__(
                "candidate_micro_recall",
                0.999,
            ),
            "candidate_micro_recall does not reconcile",
        ),
        (
            lambda report: report["debug_recall"]["layers"]["0"].__setitem__(
                "candidate_mean_row_recall",
                0.999,
            ),
            "layer 0 candidate_mean_row_recall does not reconcile",
        ),
        (
            lambda report: report["debug_recall"]["global"].__setitem__(
                "global_micro_passes_95_percent",
                False,
            ),
            "pass booleans are forged",
        ),
        (
            lambda report: report["debug_recall"]["layers"].pop("29"),
            "exactly one report per layer",
        ),
        (
            lambda report: report["debug_recall"]["layers"]["0"].__setitem__(
                "target_records",
                256 * 16 + 1,
            ),
            "do not reconcile with rows and fixed K",
        ),
        (
            lambda report: report["debug_recall"]["layers"]["0"].__setitem__(
                "candidate_hits",
                256 * 16 + 1,
            ),
            "candidate_hits must be at most",
        ),
        (
            lambda report: report["debug_recall"]["layers"]["0"].__setitem__(
                "rows",
                255,
            ),
            "exactly 256 scored rows",
        ),
    ],
)
def test_freeze_recomputes_fixed_k_recall_from_layer_integer_counts(
    freeze_fixture,
    tmp_path,
    mutate,
    message,
):
    report = json.loads(freeze_fixture["development"].read_text())
    mutate(report)
    _write_json(freeze_fixture["development"], report)

    with pytest.raises(NativeBitNetDIPPolicyManifestError, match=message):
        _build(freeze_fixture, tmp_path / "forged-recall.json")


def test_freeze_requires_six_live_parity_rows_per_layer(
    freeze_fixture,
    tmp_path,
):
    parity = json.loads(freeze_fixture["parity"].read_text())
    parity["evidence"]["rows_per_layer"] = 5
    parity["evidence"]["total_rows"] = 5 * 30
    _write_json(freeze_fixture["parity"], parity)

    with pytest.raises(
        NativeBitNetDIPPolicyManifestError,
        match="rows_per_layer must be at least 6",
    ):
        _build(freeze_fixture, tmp_path / "thin-parity.json")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda parity: parity["layers"].pop(),
            "exactly 30 layer proofs",
        ),
        (
            lambda parity: parity["layers"][0].__setitem__(
                "row_indices",
                [0, 0, 1, 2, 3, 4],
            ),
            "row_indices are invalid",
        ),
        (
            lambda parity: parity["layers"][0].__setitem__(
                "selected_counts",
                [2, 16],
            ),
            "selected_counts are invalid",
        ),
        (
            lambda parity: parity["layers"][0].__setitem__(
                "includes_observed_maximum_k",
                False,
            ),
            "includes_observed_maximum_k must be true",
        ),
        (
            lambda parity: parity["layers"][0]["equality"].__setitem__(
                "candidate_ids",
                False,
            ),
            "layer 0 equality.candidate_ids must be true",
        ),
    ],
)
def test_freeze_rejects_forged_per_layer_parity_proofs(
    freeze_fixture,
    tmp_path,
    mutate,
    message,
):
    parity = json.loads(freeze_fixture["parity"].read_text())
    mutate(parity)
    _write_json(freeze_fixture["parity"], parity)

    with pytest.raises(NativeBitNetDIPPolicyManifestError, match=message):
        _build(freeze_fixture, tmp_path / "forged-parity.json")
