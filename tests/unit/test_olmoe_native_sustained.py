import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from engram.compiler.olmoe_native import compile_olmoe_native_package
from engram.evaluation.olmoe_native_causal import _write_npz_atomic
from engram.evaluation.olmoe_native_sustained import (
    _attention_expectations,
    _validate_corpus_manifest,
    evaluate_native_olmoe_sustained_context,
    freeze_olmoe_sustained_context_protocol,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


def test_production_sustained_attention_expectations_are_derived_exactly():
    expectations = _attention_expectations(
        {
            "layers": 16,
            "query_heads": 16,
            "key_value_heads": 16,
            "head_dimension": 128,
        },
        {
            "local_window": 16,
            "older_candidates": 8,
            "older_top_k": 4,
            "sink_tokens": 2,
        },
    )

    assert expectations["attention_state_bytes"] == 6_336_512
    assert expectations["attention_scratch_bytes"] == 3_840
    assert expectations["attention_eviction_events"] == 1_792
    assert expectations["attention_older_candidate_entries_scored"] == 222_208
    assert expectations["attention_older_selected_entries"] == 113_152
    assert expectations["attention_sink_insertions"] == 512
    assert expectations["attention_heavy_hitter_updates_minimum"] == 1_536
    assert expectations["attention_heavy_hitter_updates_maximum"] == 28_160
    assert expectations["attention_logical_read_bytes"] == 677_117_952
    assert expectations["dense_full_context_logical_kv_bytes"] == 2_164_260_864
    assert expectations["attention_logical_read_fraction"] == pytest.approx(
        0.31286337209302323
    )


def test_sustained_context_fixture_passes_all_bands_and_reset(tmp_path):
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
        kernel_threads=2,
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
            "rule": "fixed fixture records",
            "candidate_or_teacher_outputs_inspected_during_selection": False,
            "previous_engram_calibration_or_confirmation_text_reused": False,
        },
        "records": [
            {
                "record_id": record["record_id"],
                "domain": record.get("domain"),
                "full_token_count": len(tokenizer.encode(record["text"]).ids),
                "window_identity": sha256_json({"input_ids": inputs[index]}),
            }
            for index, record in enumerate(records)
        ],
    }
    corpus_manifest_path = tmp_path / "corpus-manifest.json"
    atomic_json(corpus_manifest_path, corpus_manifest)
    logits = []
    hidden = []
    targets = []
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
        threads=2,
    ) as runtime:
        assert runtime.runtime.attention_metrics_available
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
            "prediction_positions": 1024,
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
    protocol_path = tmp_path / "protocol.json"
    protocol = freeze_olmoe_sustained_context_protocol(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        out=protocol_path,
        threads=2,
    )

    report = evaluate_native_olmoe_sustained_context(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        corpus_manifest=corpus_manifest_path,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        protocol=protocol_path,
        protocol_sha256=sha256_file(protocol_path),
        out=tmp_path / "result.json",
        threads=2,
    )

    assert protocol["status"] == "frozen_before_candidate_execution"
    assert report["gate_passed"]
    assert report["metrics"]["teacher_top1_agreement"] == 1.0
    assert report["metrics"]["final_hidden_relative_l2"] == 0.0
    assert set(report["position_bands"]) == {
        "positions_0_15",
        "positions_16_31",
        "positions_32_63",
        "positions_64_95",
        "positions_96_127",
    }
    assert report["attention"]["reset_replay"]["passed"]
    assert all(report["post_run_authentication"].values())
    assert all(result["structural_passed"] for result in report["sequence_results"])
    tampered = json.loads(json.dumps(corpus_manifest))
    tampered["selection"]["candidate_or_teacher_outputs_inspected_during_selection"] = (
        True
    )
    tampered_path = tmp_path / "tampered-corpus-manifest.json"
    atomic_json(tampered_path, tampered)
    with pytest.raises(ValueError, match="corpus manifest contract"):
        _validate_corpus_manifest(
            dataset,
            tampered_path,
            tokenizer,
            tokenizer_sha256=sha256_file(model / "tokenizer.json"),
        )
