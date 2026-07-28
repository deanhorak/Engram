import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from engram.compiler.olmoe_native import compile_olmoe_native_package
from engram.evaluation.olmoe_native_causal import _write_npz_atomic
from engram.evaluation.olmoe_native_dense_control import (
    _compare_metrics,
    _control_policy,
    _expected_bands,
    _paths_from_manifest,
    _pre_intervention_identity,
    _structural_checks,
    _validate_failed_sustained_result,
    evaluate_native_olmoe_dense_attention_control,
    freeze_native_olmoe_dense_attention_control_protocol,
)
from engram.evaluation.olmoe_native_sustained import (
    _attention_expectations,
    _q7_expectations,
    evaluate_native_olmoe_sustained_context,
    freeze_olmoe_sustained_context_protocol,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


_PRODUCTION_MODEL = {
    "layers": 16,
    "hidden_size": 2_048,
    "intermediate_size": 1_024,
    "experts": 64,
    "vocab_size": 50_304,
    "query_heads": 16,
    "key_value_heads": 16,
    "head_dimension": 128,
    "top_k": 8,
    "q7_group_size": 64,
}


def _fixture_metrics() -> dict[str, float]:
    return {
        "teacher_to_native_kl": 0.20,
        "teacher_top1_agreement": 0.75,
        "target_nll_delta": 0.10,
        "final_hidden_relative_l2": 0.30,
    }


def _fixture_position_rows() -> list[dict[str, float | int]]:
    return [
        {
            "sequence": sequence,
            "position": position,
            "teacher_to_native_kl": (sequence * 128 + position) / 10_000,
        }
        for sequence in range(8)
        for position in range(128)
    ]


def test_production_w128_control_expectations_are_exact():
    assert _control_policy() == {
        "local_window": 128,
        "older_candidates": 8,
        "older_top_k": 4,
        "sink_tokens": 2,
    }

    attention = _attention_expectations(_PRODUCTION_MODEL, _control_policy())
    assert attention == {
        "positions_processed": 128,
        "attention_state_bytes": 35_825_664,
        "attention_scratch_bytes": 18_176,
        "attention_eviction_events": 0,
        "attention_older_candidate_entries_scored": 0,
        "attention_older_selected_entries": 0,
        "attention_sink_insertions": 0,
        "attention_heavy_hitter_updates_minimum": 0,
        "attention_heavy_hitter_updates_maximum": 0,
        "attention_local_kv_bytes": 2_164_260_864,
        "attention_candidate_key_bytes": 0,
        "attention_selected_value_bytes": 0,
        "attention_logical_read_bytes": 2_164_260_864,
        "dense_full_context_logical_kv_bytes": 2_164_260_864,
        "attention_logical_read_fraction": 1.0,
    }

    q7 = _q7_expectations(_PRODUCTION_MODEL)
    assert q7 == {
        "artifact_bytes": 5_842_733_184,
        "router_bytes_per_layer_position": 262_144,
        "selected_expert_bytes_per_layer_position": 45_613_056,
        "scheduled_bytes_per_layer_position": 45_875_200,
        "scheduled_bytes_per_position": 734_003_200,
        "scheduled_bytes_per_sequence": 93_952_409_600,
    }


def test_w128_structural_checks_require_every_exact_counter():
    attention = _attention_expectations(_PRODUCTION_MODEL, _control_policy())
    q7 = _q7_expectations(_PRODUCTION_MODEL)
    metrics = {
        name: int(attention[name])
        for name in (
            "positions_processed",
            "attention_state_bytes",
            "attention_scratch_bytes",
            "attention_eviction_events",
            "attention_older_candidate_entries_scored",
            "attention_older_selected_entries",
            "attention_sink_insertions",
            "attention_logical_read_bytes",
        )
    }
    metrics["attention_heavy_hitter_updates"] = 0
    metrics["q7_scheduled_bytes"] = q7["scheduled_bytes_per_sequence"]

    checks = _structural_checks(
        metrics,
        attention,
        q7,
        position=128,
    )
    assert set(checks) == {
        "positions_processed",
        "attention_state_bytes",
        "attention_scratch_bytes",
        "attention_eviction_events",
        "attention_older_candidate_entries_scored",
        "attention_older_selected_entries",
        "attention_sink_insertions",
        "attention_logical_read_bytes",
        "cache_position",
        "attention_heavy_hitter_updates",
        "q7_scheduled_bytes",
    }
    assert all(checks.values())

    for name in metrics:
        altered = dict(metrics)
        altered[name] += 1
        altered_checks = _structural_checks(
            altered,
            attention,
            q7,
            position=128,
        )
        assert altered_checks[name] is False
        assert sum(not passed for passed in altered_checks.values()) == 1

    wrong_position = _structural_checks(
        metrics,
        attention,
        q7,
        position=127,
    )
    assert wrong_position["cache_position"] is False
    assert sum(not passed for passed in wrong_position.values()) == 1


def test_control_helpers_preserve_frozen_bands_paths_and_metric_direction():
    assert _expected_bands() == [
        {"name": "positions_0_15", "start": 0, "stop": 16},
        {"name": "positions_16_31", "start": 16, "stop": 32},
        {"name": "positions_32_63", "start": 32, "stop": 64},
        {"name": "positions_64_95", "start": 64, "stop": 96},
        {"name": "positions_96_127", "start": 96, "stop": 128},
    ]
    assert _paths_from_manifest(
        Path("/package"),
        {"tokenizer": {"path": "tokenizer"}},
    ) == (
        Path("/package/model/config.json"),
        Path("/package/transformer/non_mlp.safetensors"),
        Path("/package/mlp/experts.q7"),
        Path("/package/tokenizer/tokenizer.json"),
    )

    comparison = _compare_metrics(
        {
            "teacher_to_native_kl": 0.20,
            "teacher_top1_agreement": 0.75,
            "target_nll_delta": 0.10,
            "final_hidden_relative_l2": 0.30,
        },
        {
            "teacher_to_native_kl": 0.05,
            "teacher_top1_agreement": 0.95,
            "target_nll_delta": 0.02,
            "final_hidden_relative_l2": 0.10,
        },
    )
    assert comparison["teacher_to_native_kl"] == pytest.approx(
        {
            "bounded": 0.20,
            "control": 0.05,
            "control_minus_bounded": -0.15,
        }
    )
    assert comparison["teacher_top1_agreement"] == pytest.approx(
        {
            "bounded": 0.75,
            "control": 0.95,
            "control_minus_bounded": 0.20,
        }
    )
    assert comparison["target_nll_delta"]["control_minus_bounded"] == pytest.approx(
        -0.08
    )
    assert comparison["final_hidden_relative_l2"][
        "control_minus_bounded"
    ] == pytest.approx(-0.20)


def test_pre_intervention_identity_requires_exact_first_sixteen_position_rows():
    bounded = _fixture_position_rows()
    control = deepcopy(bounded)

    assert _pre_intervention_identity(bounded, control) == {
        "expected_positions": 128,
        "bounded_positions": 128,
        "control_positions": 128,
        "exact_position_metrics_match": True,
    }

    post_intervention_tamper = deepcopy(control)
    post_intervention_tamper[16]["teacher_to_native_kl"] += 1.0
    assert _pre_intervention_identity(
        bounded,
        post_intervention_tamper,
    )["exact_position_metrics_match"]

    pre_intervention_tamper = deepcopy(control)
    pre_intervention_tamper[0]["teacher_to_native_kl"] += 1.0
    tampered = _pre_intervention_identity(bounded, pre_intervention_tamper)
    assert tampered["bounded_positions"] == 128
    assert tampered["control_positions"] == 128
    assert tampered["exact_position_metrics_match"] is False

    missing_pre_intervention_row = _pre_intervention_identity(bounded, control[1:])
    assert missing_pre_intervention_row["control_positions"] == 127
    assert missing_pre_intervention_row["exact_position_metrics_match"] is False


def test_failed_sustained_prerequisite_rejects_incomplete_or_tampered_evidence():
    protocol_hash = "a" * 64
    protocol = {
        "schema_version": 1,
        "experiment": "olmoe_native_sustained_context_confirmation",
        "status": "frozen_before_candidate_execution",
        "package_manifest_sha256": "package",
        "native_library_sha256": "library",
        "dataset_sha256": "dataset",
        "corpus_manifest_sha256": "corpus",
        "teacher_reference_sha256": "teacher-reference",
        "teacher_arrays_sha256": "teacher-arrays",
    }
    failed_result = {
        "schema_version": 1,
        "experiment": "olmoe_native_sustained_context_confirmation",
        "status": "frozen_confirmation_failed",
        "gate_passed": False,
        "evidence_passed": True,
        "quality_passed": False,
        "decision": "run_matched_q7_dense_attention_control",
        "artifacts": {
            "protocol_sha256": protocol_hash,
            "package_manifest_sha256": "package",
            "native_library_sha256": "library",
            "dataset_sha256": "dataset",
            "corpus_manifest_sha256": "corpus",
            "teacher_reference_sha256": "teacher-reference",
            "teacher_arrays_sha256": "teacher-arrays",
        },
        "post_run_authentication": {
            "package": True,
            "teacher": True,
        },
        "metrics": _fixture_metrics(),
        "position_bands": {
            band["name"]: _fixture_metrics() for band in _expected_bands()
        },
        "position_results": _fixture_position_rows(),
    }

    assert (
        _validate_failed_sustained_result(
            protocol,
            failed_result,
            protocol_hash=protocol_hash,
        )
        is None
    )

    bad_quality_state = deepcopy(failed_result)
    bad_quality_state["quality_passed"] = True
    with pytest.raises(ValueError, match="control prerequisite"):
        _validate_failed_sustained_result(
            protocol,
            bad_quality_state,
            protocol_hash=protocol_hash,
        )

    bad_identity = deepcopy(failed_result)
    bad_identity["artifacts"]["dataset_sha256"] = "tampered"
    with pytest.raises(ValueError, match="control prerequisite"):
        _validate_failed_sustained_result(
            protocol,
            bad_identity,
            protocol_hash=protocol_hash,
        )

    bad_authentication = deepcopy(failed_result)
    bad_authentication["post_run_authentication"]["teacher"] = False
    with pytest.raises(ValueError, match="control prerequisite"):
        _validate_failed_sustained_result(
            protocol,
            bad_authentication,
            protocol_hash=protocol_hash,
        )

    missing_metrics = deepcopy(failed_result)
    del missing_metrics["metrics"]
    with pytest.raises(ValueError, match="control prerequisite"):
        _validate_failed_sustained_result(
            protocol,
            missing_metrics,
            protocol_hash=protocol_hash,
        )

    missing_position = deepcopy(failed_result)
    missing_position["position_results"].pop()
    with pytest.raises(ValueError, match="control prerequisite"):
        _validate_failed_sustained_result(
            protocol,
            missing_position,
            protocol_hash=protocol_hash,
        )


def test_tiny_native_w16_failure_to_w128_control_smoke(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")

    model = create_tiny_olmoe_fixture(
        tmp_path / "model",
        num_experts=64,
        num_experts_per_token=1,
    )
    vocabulary = {"[UNK]": 0, "x": 1}
    vocabulary.update({f"marker{index}": index + 2 for index in range(8)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=64)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=12,
    )
    manifest_hash = compiled["manifest_sha256"]

    dataset = tmp_path / "sustained.jsonl"
    records = [
        {
            "record_id": f"fixture-{index}",
            "source_kind": "engram_authored_holdout",
            "domain": f"fixture-domain-{index}",
            "text": " ".join([f"marker{index}", *(["x"] * 128)]),
        }
        for index in range(8)
    ]
    dataset.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    inputs = [tokenizer.encode(record["text"]).ids[:129] for record in records]
    corpus_manifest = {
        "schema_version": 1,
        "experiment": "olmoe_sustained_context_authored_holdout",
        "status": "authored_and_fixed_before_candidate_execution",
        "created_utc": "2026-07-28",
        "dataset_sha256": sha256_file(dataset),
        "tokenizer_sha256": sha256_file(model / "tokenizer.json"),
        "input_identity": sha256_json(inputs),
        "sequences": 8,
        "tokens_per_sequence": 129,
        "selection": {
            "source_kind": "engram_authored_holdout",
            "rule": "fixed dense-control fixture records",
            "candidate_or_teacher_outputs_inspected_during_selection": False,
            "previous_engram_calibration_or_confirmation_text_reused": False,
        },
        "records": [
            {
                "record_id": record["record_id"],
                "domain": record["domain"],
                "full_token_count": len(tokenizer.encode(record["text"]).ids),
                "window_identity": sha256_json({"input_ids": inputs[index]}),
            }
            for index, record in enumerate(records)
        ],
    }
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    atomic_json(corpus_manifest_path, corpus_manifest)

    logits: list[np.ndarray] = []
    hidden: list[np.ndarray] = []
    targets: list[int] = []
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
        threads=12,
    ) as runtime:
        for sequence in inputs:
            runtime.reset()
            for position, token_id in enumerate(sequence[:-1]):
                runtime.runtime.forward([token_id])
                state, scores = runtime.runtime.last_diagnostics()
                hidden.append(state)
                logits.append(scores)
                targets.append(sequence[position + 1])
    arrays_path = tmp_path / "teacher.npz"
    _write_npz_atomic(
        arrays_path,
        logits=np.asarray(logits, dtype=np.float32),
        hidden=np.asarray(hidden, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
    )
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_causal_reference",
        "source": {
            "model": str(model),
            "revision": None,
            "config_sha256": sha256_file(model / "config.json"),
            "index_sha256": sha256_file(model / "model.safetensors.index.json"),
            "adapter": "olmoe_sparse_expert_v1",
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "sequences": 8,
            "tokens_per_sequence": 129,
            "prediction_positions": 1_024,
            "input_identity": sha256_json(inputs),
            "input_ids": inputs,
        },
        "configuration": {
            "dtype": "bfloat16",
            "device": "cpu",
            "threads": 12,
            "batch_size": 1,
            "expert_workers": 1,
            "sequence_workers": 4,
            "threaded_expert_layers": 0,
            "expert_backend": "transformers_reference",
            "sequence_backend": "thread_pool_shared_model_v1",
            "attention_implementation": "eager",
            "use_cache": False,
            "output_hidden_states": True,
            "weights_modified": False,
        },
        "arrays": {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
        },
    }
    reference_path = tmp_path / "teacher.json"
    atomic_json(reference_path, reference)

    sustained_protocol_path = tmp_path / "sustained-protocol.json"
    freeze_olmoe_sustained_context_protocol(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        out=sustained_protocol_path,
        threads=12,
    )
    sustained_result_path = tmp_path / "sustained-result.json"
    sustained = evaluate_native_olmoe_sustained_context(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        protocol=sustained_protocol_path,
        protocol_sha256=sha256_file(sustained_protocol_path),
        out=sustained_result_path,
        threads=12,
    )
    assert sustained["gate_passed"]
    assert sustained["evidence_passed"]
    assert sustained["quality_passed"]

    failed = deepcopy(sustained)
    failed["status"] = "frozen_confirmation_failed"
    failed["gate_passed"] = False
    failed["quality_passed"] = False
    failed["decision"] = "run_matched_q7_dense_attention_control"
    failed_path = tmp_path / "authenticated-failure.json"
    atomic_json(failed_path, failed)

    tampered_failure = deepcopy(failed)
    tampered_failure["artifacts"]["dataset_sha256"] = "tampered"
    tampered_failure_path = tmp_path / "tampered-failure.json"
    atomic_json(tampered_failure_path, tampered_failure)
    with pytest.raises(ValueError, match="control prerequisite"):
        freeze_native_olmoe_dense_attention_control_protocol(
            package=package,
            manifest_sha256=manifest_hash,
            library=library,
            dataset=dataset,
            corpus_manifest=corpus_manifest_path,
            teacher_reference=reference_path,
            teacher_arrays=arrays_path,
            sustained_protocol=sustained_protocol_path,
            sustained_protocol_sha256=sha256_file(sustained_protocol_path),
            sustained_result=tampered_failure_path,
            sustained_result_sha256=sha256_file(tampered_failure_path),
            out=tmp_path / "tampered-control-protocol.json",
            threads=12,
        )

    control_protocol_path = tmp_path / "control-protocol.json"
    control_protocol = freeze_native_olmoe_dense_attention_control_protocol(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        sustained_protocol=sustained_protocol_path,
        sustained_protocol_sha256=sha256_file(sustained_protocol_path),
        sustained_result=failed_path,
        sustained_result_sha256=sha256_file(failed_path),
        out=control_protocol_path,
        threads=12,
    )
    assert control_protocol["scope"]["only_intervention"] == "local_window_16_to_128"
    assert control_protocol["control_attention_policy"] == _control_policy()

    control_result = evaluate_native_olmoe_dense_attention_control(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        sustained_protocol=sustained_protocol_path,
        sustained_protocol_sha256=sha256_file(sustained_protocol_path),
        sustained_result=failed_path,
        sustained_result_sha256=sha256_file(failed_path),
        control_protocol=control_protocol_path,
        control_protocol_sha256=sha256_file(control_protocol_path),
        out=tmp_path / "control-result.json",
        threads=12,
    )

    assert control_result["evidence_passed"]
    assert all(control_result["evidence_checks"].values())
    assert control_result["pre_intervention_identity"] == {
        "expected_positions": 128,
        "bounded_positions": 128,
        "control_positions": 128,
        "exact_position_metrics_match": True,
    }
    assert (
        control_result["attention_expectations_per_sequence"][
            "attention_eviction_events"
        ]
        == 0
    )
    assert (
        control_result["attention_expectations_per_sequence"][
            "attention_logical_read_fraction"
        ]
        == 1.0
    )
    assert all(
        result["structural_passed"] for result in control_result["sequence_results"]
    )
    assert all(control_result["post_run_authentication"].values())
    assert control_result["traffic"]["q7_scheduled_bytes"] == (
        8
        * control_result["q7_expectations_per_sequence"]["scheduled_bytes_per_sequence"]
    )
