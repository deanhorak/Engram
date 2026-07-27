import json
import sys
import types
from pathlib import Path

import pytest

import engram.evaluation.native_bitnet_dip_token_generation as evaluation
from engram.compiler.native_bitnet import (
    NATIVE_BITNET_DIP_OPERATOR,
    NATIVE_BITNET_M2_ADJUDICATION_SHA256,
    NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256,
    NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256,
    NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256,
    NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256,
)
from engram.evaluation.native_bitnet_dip_token_generation import (
    _parse_native_metrics,
)
from engram.utils import sha256_file


def _metrics(reset=1):
    return (
        f"semantic_backend={NATIVE_BITNET_DIP_OPERATOR} "
        "positions=5 stage_calls=8 semantic_calls=8 semantic_rows=10 "
        "selected_records=10 semantic_kernel_cache_line_bytes=100 "
        "semantic_global_metadata_bytes=20 "
        "semantic_cache_line_bytes=120 "
        "semantic_seconds=1.25 attention_seconds=0.5 "
        "attention_evictions=2 attention_older_candidates_scored=40 "
        "attention_older_entries_selected=20 attention_sink_insertions=8 "
        "attention_heavy_hitter_updates=12 "
        f"reset_verified={reset} reset_counters_zeroed={reset} "
        f"replay_metrics_match={reset}"
    )


def test_native_dip_token_metrics_require_backend_traffic_and_reset_evidence():
    metrics = _parse_native_metrics(_metrics())

    assert metrics == {
        "semantic_backend": NATIVE_BITNET_DIP_OPERATOR,
        "positions": 5,
        "stage_calls": 8,
        "semantic_calls": 8,
        "semantic_rows": 10,
        "selected_records": 10,
        "semantic_kernel_cache_line_bytes": 100,
        "semantic_global_metadata_bytes": 20,
        "semantic_cache_line_bytes": 120,
        "semantic_seconds": 1.25,
        "attention_seconds": 0.5,
        "attention_evictions": 2,
        "attention_older_candidates_scored": 40,
        "attention_older_entries_selected": 20,
        "attention_sink_insertions": 8,
        "attention_heavy_hitter_updates": 12,
        "reset_verified": 1,
        "reset_counters_zeroed": 1,
        "replay_metrics_match": 1,
    }


def test_native_dip_token_metrics_reject_incomplete_or_duplicate_output():
    with pytest.raises(ValueError, match="incomplete"):
        _parse_native_metrics(
            f"semantic_backend={NATIVE_BITNET_DIP_OPERATOR} positions=5"
        )
    with pytest.raises(ValueError, match="duplicate"):
        _parse_native_metrics(_metrics() + " positions=5")


def _reference(prompts):
    results = [
        {
            "prompt": prompt,
            "prompt_tokens": 2,
            "baseline_tokens": [10, 11, 12, 13],
        }
        for prompt in prompts
    ]
    return {
        "schema_version": 1,
        "experiment": "native_bitnet_controller_incremental_generation",
        "status": "frozen_controller_generation_confirmation",
        "gate_passed": True,
        "configuration": {
            "controller_correction_enabled": False,
            "episodic_operator": "native_streaming_w16_c8_k4_sinks2",
            "greedy": True,
            "max_new_tokens": 4,
            "native_embedding_lookup": True,
            "native_packed_attention_projections": True,
            "native_rms_norm": True,
            "native_rope": True,
            "native_vocabulary_argmax": True,
            "semantic_operator": "native_packed_bitnet_phase_stream",
        },
        "scope": {
            "absolute_rope_positions_advanced": True,
            "decoder_layer_forward_used": False,
            "persistent_native_attention_cache_used": True,
            "source_mlp_tensors_loaded": False,
            "torch_embedding_used": False,
            "torch_rms_norm_used": False,
            "torch_rope_used": False,
            "torch_vocabulary_projection_used": False,
        },
        "checks": {
            "cache_positions": True,
            "decoder_layers_bypassed": True,
            "exact_prompt_fraction": True,
            "generated_reference_tokens": True,
            "prompt_count": True,
            "weighted_token_agreement": True,
        },
        "thresholds": dict(evaluation._REFERENCE_THRESHOLDS),
        "prompt_suite": {
            "sha256": "",
            "prompts": len(prompts),
        },
        "results": results,
    }


def _evaluation_fixture(tmp_path: Path, monkeypatch, *, reset=1):
    package = tmp_path / "model.engram-bitnet-dip"
    package.mkdir()
    binary = tmp_path / "engram-bitnet-token-generate"
    binary.write_bytes(b"fixture executable")
    prompts = [f"prompt-{index}" for index in range(8)]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    reference = _reference(prompts)
    reference["prompt_suite"]["sha256"] = sha256_file(prompt_path)
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps(reference, sort_keys=True),
        encoding="utf-8",
    )
    manifest = {
        "model": {
            "num_hidden_layers": 2,
            "hidden_size": 10,
            "intermediate_size": 10,
        },
        "tokenizer": {"path": "tokenizer", "fix_mistral_regex": False},
        "semantic_memory": {
            "operator": NATIVE_BITNET_DIP_OPERATOR,
            "dense_fallback": False,
            "all_mlp_layers_substituted": True,
            "path": "mlp/index.bin",
            "source_package_manifest_sha256": (
                NATIVE_BITNET_M2_PACKAGE_MANIFEST_SHA256
            ),
            "source_artifact_sha256": NATIVE_BITNET_M2_BASE_ARTIFACT_SHA256,
            "sha256": NATIVE_BITNET_M2_COORDINATE_INDEX_SHA256,
            "policy_manifest_sha256": NATIVE_BITNET_M2_POLICY_MANIFEST_SHA256,
            "adjudication_sha256": NATIVE_BITNET_M2_ADJUDICATION_SHA256,
        },
    }
    manifest_path = package / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        evaluation,
        "NATIVE_BITNET_DIP_DERIVED_MANIFEST_SHA256",
        sha256_file(manifest_path),
    )
    monkeypatch.setattr(
        evaluation,
        "_TRUSTED_NATIVE_EXECUTABLE_SHA256",
        sha256_file(binary),
    )
    monkeypatch.setattr(
        evaluation,
        "_CANONICAL_PROMPT_SUITE_SHA256",
        sha256_file(prompt_path),
    )
    monkeypatch.setattr(
        evaluation,
        "_CANONICAL_DENSE_REFERENCE_SHA256",
        sha256_file(reference_path),
    )
    monkeypatch.setattr(
        evaluation,
        "validate_native_bitnet_package",
        lambda _path: {"valid": True},
    )
    monkeypatch.setattr(
        evaluation,
        "_expected_traffic",
        lambda **_kwargs: {
            "kernel_cache_line_bytes": 100,
            "global_metadata_bytes": 20,
            "complete_modelled_cold_bytes": 120,
            "minimum_selected_records": 1,
            "maximum_selected_records": 20,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "_disable_broken_optional_transformers_dependencies",
        lambda: None,
    )

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def encode(self, _prompt, *, add_special_tokens):
            assert add_special_tokens is True
            return [1, 2]

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=Tokenizer),
    )
    monkeypatch.setattr(
        evaluation.subprocess,
        "run",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            stdout="10 11 12 13\n",
            stderr=_metrics(reset),
        ),
    )
    return {
        "package": package,
        "executable": binary,
        "prompts": prompt_path,
        "reference_report": reference_path,
        "out": tmp_path / "report.json",
        "package_manifest_sha256": sha256_file(manifest_path),
        "executable_sha256": sha256_file(binary),
    }


def test_authenticated_evaluator_passes_only_with_reset_and_fixed_thresholds(
    tmp_path,
    monkeypatch,
):
    kwargs = _evaluation_fixture(tmp_path, monkeypatch)
    report = evaluation.evaluate_native_bitnet_dip_token_generation(**kwargs)

    assert report["gate_passed"] is True
    assert report["summary"]["weighted_token_agreement"] == 1.0
    assert report["summary"]["exact_prompts"] == 8
    assert report["summary"]["complete_modelled_cold_bytes"] == 8 * 120
    assert report["artifacts"]["authenticated_before_and_after"] is True
    assert Path(kwargs["out"]).is_file()


def test_evaluator_cannot_pass_when_reset_replay_is_disabled(
    tmp_path,
    monkeypatch,
):
    kwargs = _evaluation_fixture(tmp_path, monkeypatch, reset=0)
    report = evaluation.evaluate_native_bitnet_dip_token_generation(
        **kwargs,
        verify_reset=False,
    )

    assert report["gate_passed"] is False
    assert report["gate_checks"]["all_runtime_invariants"] is False
    assert all(
        not row["runtime_checks"]["reset_requested"]
        for row in report["results"]
    )


def test_dense_reference_cannot_weaken_frozen_thresholds():
    prompts = [f"prompt-{index}" for index in range(8)]
    reference = _reference(prompts)
    reference["prompt_suite"]["sha256"] = "a" * 64
    reference["thresholds"]["minimum_weighted_token_agreement"] = 0.0

    with pytest.raises(ValueError, match="frozen dense baseline"):
        evaluation._validate_dense_reference(
            reference,
            prompts=prompts,
            prompt_sha256="a" * 64,
            max_new_tokens=4,
        )
