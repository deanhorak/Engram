import json

import pytest

from engram.cli import _parser, main


def test_olmoe_causal_capture_forwards_batch_size(tmp_path, monkeypatch):
    captured = {}
    out = tmp_path / "teacher.json"

    def fake_capture(**kwargs):
        captured.update(kwargs)
        out.write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(
        "engram.cli.capture_olmoe_teacher_causal_reference",
        fake_capture,
    )

    main(
        [
            "capture-olmoe-teacher-causal",
            "--model",
            str(tmp_path / "model"),
            "--dataset",
            str(tmp_path / "dataset.jsonl"),
            "--out",
            str(out),
            "--arrays-out",
            str(tmp_path / "teacher.npz"),
            "--batch-size",
            "4",
            "--expert-workers",
            "3",
            "--sequence-workers",
            "1",
        ]
    )

    assert captured["batch_size"] == 4
    assert captured["expert_workers"] == 3
    assert captured["sequence_workers"] == 1


def test_composite_gate_cannot_overwrite_an_input_report(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "mlp_intervention.json"
    second = second_dir / "mlp_intervention.json"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        main(
            [
                "gate-mlp-intervention",
                "--report",
                str(first),
                str(second),
                "--out",
                str(first_dir),
            ]
        )


def test_width_pruned_cli_exposes_strict_q4_deployment_flag(tmp_path):
    arguments = _parser().parse_args(
        [
            "train-width-pruned-student",
            "--model",
            "model",
            "--training-dataset",
            str(tmp_path / "train.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--strict-q4-deployment",
            "--fake-q4-training",
            "--target-widths",
            "384",
            "672",
        ]
    )

    assert arguments.strict_q4_deployment
    assert arguments.fake_q4_training
    assert arguments.target_widths == [384, 672]


def test_hybrid_benchmark_defaults_to_compact_warm_execution(tmp_path):
    arguments = _parser().parse_args(
        [
            "benchmark-hybrid",
            "--endpoint",
            "http://127.0.0.1:11434/api/chat",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
        ]
    )
    assert arguments.context_format == "compact"
    assert arguments.warmup_requests == 1
    assert "directly and concisely" in arguments.system


def test_hybrid_benchmark_scores_frozen_answer_and_retrieval_rubrics(
    tmp_path, monkeypatch
):
    memory = tmp_path / "memory.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    report = tmp_path / "report.json"
    memory.write_text(
        json.dumps(
            {
                "id": "cpu-policy",
                "text": "CPU and GPU deployment policy for inference and training",
                "prompt_text": "Inference is CPU-only; CUDA is for training.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompts.write_text(
        json.dumps(
            {
                "id": "cpu",
                "prompt": "What is the CPU and GPU policy?",
                "required_terms": ["CPU", "CUDA"],
                "expected_memory_ids": ["cpu-policy"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def complete(self, messages, **_kwargs):
            system = messages[0]["content"]
            return (
                "CPU inference; CUDA training."
                if "CPU-only" in system
                else "No project context.",
                {},
            )

    monkeypatch.setattr("engram.cli.OpenAICompatibleClient", _Client)
    main(
        [
            "benchmark-hybrid",
            "--endpoint",
            "http://local/api/chat",
            "--model",
            "fixture",
            "--prompts",
            str(prompts),
            "--memory",
            str(memory),
            "--min-score",
            "0",
            "--max-tokens",
            "8",
            "--out",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["quality_summary"]["baseline_answer_pass_rate"] == 0.0
    assert payload["quality_summary"]["hybrid_answer_pass_rate"] == 1.0
    assert payload["quality_summary"]["hybrid_retrieval_pass_rate"] == 1.0
    assert payload["quality_claim"] == "task_specific_rubric_only"


def test_hybrid_retrieval_benchmark_is_host_independent(tmp_path):
    memory = tmp_path / "memory.jsonl"
    queries = tmp_path / "queries.jsonl"
    report = tmp_path / "retrieval.json"
    memory.write_text(
        json.dumps({"id": "cpu", "text": "CPU inference deployment"}) + "\n",
        encoding="utf-8",
    )
    queries.write_text(
        json.dumps(
            {
                "id": "cpu-query",
                "query": "CPU deployment",
                "expected_memory_ids": ["cpu"],
                "category": "lexical",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    main(
        [
            "benchmark-hybrid-retrieval",
            "--memory",
            str(memory),
            "--queries",
            str(queries),
            "--top-k-values",
            "1",
            "4",
            "--out",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["quality_claim"] == "retrieval_membership_only"
    assert payload["evaluation"]["metrics"]["1"]["all_expected_hit_rate"] == 1.0
    assert len(payload["memory_sha256"]) == 64
    assert len(payload["queries_sha256"]) == 64


def test_beir_hybrid_benchmark_reports_public_metric_gate(tmp_path):
    dataset = tmp_path / "scifact"
    (dataset / "qrels").mkdir(parents=True)
    (dataset / "corpus.jsonl").write_text(
        json.dumps({"_id": "d1", "title": "CPU", "text": "CPU inference"})
        + "\n"
        + json.dumps({"_id": "d2", "title": "Rain", "text": "weather"})
        + "\n",
        encoding="utf-8",
    )
    (dataset / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "CPU inference"}) + "\n",
        encoding="utf-8",
    )
    (dataset / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\n",
        encoding="utf-8",
    )
    report = tmp_path / "beir-report.json"
    main(
        [
            "benchmark-beir-hybrid-retrieval",
            "--dataset",
            str(dataset),
            "--top-k-values",
            "1",
            "10",
            "100",
            "--out",
            str(report),
        ]
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["quality_claim"] == "public_retrieval_metrics_only"
    assert payload["gate"]["passed"]
    assert payload["evaluation"]["metrics"]["10"]["ndcg"] == 1.0
    assert payload["dataset"]["relevance_pairs"] == 1


def test_width_pruned_cli_forwards_layer_schedule(tmp_path, monkeypatch):
    captured = {}

    def fake_train(*_args, **kwargs):
        captured.update(kwargs)
        return {"gate": {"passed": True}}

    monkeypatch.setattr("engram.cli.train_width_pruned_student", fake_train)
    result = main(
        [
            "train-width-pruned-student",
            "--model",
            "model",
            "--training-dataset",
            str(tmp_path / "train.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--target-widths",
            "384",
            "672",
            "--strict-q4-deployment",
            "--fake-q4-training",
        ]
    )

    assert result == 0
    assert captured["target_width"] == 672
    assert captured["target_widths"] == [384, 672]
    assert captured["fake_q4_training"]


def test_budget_native_ternary_cli_forwards_training_protocol(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_train(*_args, **kwargs):
        captured.update(kwargs)
        return {"gate": {"passed": True}}

    monkeypatch.setattr(
        "engram.cli.train_budget_native_ternary_student",
        fake_train,
    )
    result = main(
        [
            "train-budget-native-ternary",
            "--model",
            "model",
            "--training-dataset",
            str(tmp_path / "train.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--group-size",
            "128",
            "--steps",
            "16",
            "--dense-warmup-steps",
            "2",
            "--anneal-steps",
            "6",
            "--transition-mode",
            "global",
            "--final-hidden-weight",
            "4",
            "--final-cka-weight",
            "2",
            "--teacher-top1-weight",
            "0.5",
            "--coadapt-backbone",
            "--backbone-start-step",
            "8",
            "--initial-checkpoint",
            str(tmp_path / "initial.pt"),
            "--training-record-offset",
            "128",
        ]
    )

    assert result == 0
    assert captured["group_size"] == 128
    assert captured["steps"] == 16
    assert captured["dense_warmup_steps"] == 2
    assert captured["anneal_steps"] == 6
    assert captured["transition_mode"] == "global"
    assert captured["final_hidden_weight"] == 4.0
    assert captured["final_cka_weight"] == 2.0
    assert captured["teacher_top1_weight"] == 0.5
    assert captured["coadapt_backbone"]
    assert captured["backbone_start_step"] == 8
    assert captured["initial_checkpoint"] == tmp_path / "initial.pt"
    assert captured["training_record_offset"] == 128


def test_sparse_teacher_cli_exposes_progressive_layer_pilot_controls(tmp_path):
    arguments = _parser().parse_args(
        [
            "train-sparse-student",
            "--model",
            "model",
            "--calibration-dataset",
            str(tmp_path / "calibration.jsonl"),
            "--validation-dataset",
            str(tmp_path / "validation.jsonl"),
            "--calibration-traces",
            str(tmp_path / "traces"),
            "--out",
            str(tmp_path / "out"),
            "--layers",
            "25",
            "26",
            "27",
            "28",
            "29",
            "--exact-dense-start",
            "--dense-warmup-steps",
            "32",
            "--anneal-steps",
            "256",
            "--checkpoint-selection-records",
            "128",
            "--checkpoint-selection-every",
            "64",
        ]
    )

    assert arguments.layers == [25, 26, 27, 28, 29]
    assert arguments.exact_dense_start
    assert arguments.dense_warmup_steps == 32
    assert arguments.anneal_steps == 256
    assert arguments.checkpoint_selection_records == 128
    assert arguments.checkpoint_selection_every == 64


def test_trace_cli_exposes_selected_layers_and_dry_run_plan(tmp_path):
    arguments = _parser().parse_args(
        [
            "trace",
            "--model",
            "model",
            "--dataset",
            str(tmp_path / "corpus.jsonl"),
            "--out",
            str(tmp_path / "traces"),
            "--layers",
            "14",
            "--samples",
            "2048",
            "--mlp-only",
            "--dry-run",
            "--plan-out",
            str(tmp_path / "plan.json"),
        ]
    )

    assert arguments.layers == [14]
    assert arguments.samples == 2048
    assert arguments.mlp_only
    assert arguments.dry_run
    assert arguments.plan_out == tmp_path / "plan.json"


def test_native_bitnet_audit_cli_is_metadata_only(tmp_path, monkeypatch):
    captured = {}

    class Result:
        def to_dict(self):
            return {
                "decision": "proceed_to_exact_weight_repack",
                "combined_gate_status": "not_evaluated_metadata_only",
            }

    def fake_audit(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr("engram.cli.audit_native_bitnet_source", fake_audit)
    output = tmp_path / "audit.json"

    result = main(
        [
            "audit-native-bitnet",
            "--model",
            "microsoft/bitnet-b1.58-2B-4T",
            "--revision",
            "abc123",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(output),
        ]
    )

    assert result == 0
    assert captured["model"] == "microsoft/bitnet-b1.58-2B-4T"
    assert captured["revision"] == "abc123"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert output.is_file()


def test_olmoe_audit_cli_is_metadata_only(tmp_path, monkeypatch):
    captured = {}

    class Result:
        def to_dict(self):
            return {
                "decision": "proceed_to_exact_weight_shape_audit",
                "combined_gate_status": "not_evaluated_source_audit_only",
            }

    def fake_audit(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr("engram.cli.audit_olmoe_source", fake_audit)
    output = tmp_path / "audit.json"

    result = main(
        [
            "audit-olmoe",
            "--model",
            "allenai/OLMoE-1B-7B-0125",
            "--revision",
            "abc123",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(output),
            "--verify-remote-shapes",
        ]
    )

    assert result == 0
    assert captured["model"] == "allenai/OLMoE-1B-7B-0125"
    assert captured["revision"] == "abc123"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["verify_remote_shapes"]
    assert output.is_file()


def test_native_bitnet_repack_cli_forwards_pinned_verification(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_repack(model, out, **kwargs):
        captured["model"] = model
        captured["out"] = out
        captured.update(kwargs)
        return {"combined_gate_status": "not_yet_evaluated"}

    monkeypatch.setattr("engram.cli.repack_native_bitnet_model", fake_repack)
    artifact = tmp_path / "model.bin"

    result = main(
        [
            "repack-native-bitnet",
            "--model",
            "model",
            "--out",
            str(artifact),
            "--report",
            str(tmp_path / "report.json"),
        ]
    )

    assert result == 0
    assert captured["model"] == "model"
    assert captured["out"] == artifact
    assert captured["verify_official_weight_hash"]


def test_native_bitnet_parity_cli_forwards_cpu_smoke_and_status(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_parity(model, artifact, **kwargs):
        captured["model"] = model
        captured["artifact"] = artifact
        captured.update(kwargs)
        return {"smoke_gate": {"passed": True}}

    monkeypatch.setattr("engram.cli.evaluate_native_bitnet_parity", fake_parity)
    artifact = tmp_path / "model.bin"
    report = tmp_path / "parity.json"

    result = main(
        [
            "evaluate-native-bitnet-parity",
            "--model",
            "model",
            "--artifact",
            str(artifact),
            "--artifact-sha256",
            "a" * 64,
            "--out",
            str(report),
            "--revision",
            "abc123",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--local-layers",
            "1",
            "3",
            "--local-states",
            "4",
            "--input-ids",
            "10",
            "11",
            "--no-causal-substitution",
        ]
    )

    assert result == 0
    assert captured["model"] == "model"
    assert captured["artifact"] == artifact
    assert captured["expected_artifact_sha256"] == "a" * 64
    assert captured["out"] == report
    assert captured["revision"] == "abc123"
    assert captured["cache_dir"] == tmp_path / "cache"
    assert captured["local_layers"] == [1, 3]
    assert captured["local_states"] == 4
    assert captured["input_ids"] == [10, 11]
    assert not captured["run_causal_substitution"]


def test_native_bitnet_parity_cli_returns_nonzero_when_smoke_fails(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_parity",
        lambda *_args, **_kwargs: {"smoke_gate": {"passed": False}},
    )

    result = main(
        [
            "evaluate-native-bitnet-parity",
            "--model",
            "model",
            "--artifact",
            str(tmp_path / "model.bin"),
            "--out",
            str(tmp_path / "parity.json"),
        ]
    )

    assert result == 2


def test_native_bitnet_kernel_cli_forwards_frozen_protocol(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(model, artifact, dataset, **kwargs):
        captured.update(
            {"model": model, "artifact": artifact, "dataset": dataset, **kwargs}
        )
        return {"gate_passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_kernel_confirmation",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-kernel",
            "--model",
            "model",
            "--artifact",
            str(tmp_path / "model.bin"),
            "--artifact-sha256",
            "a" * 64,
            "--dataset",
            str(tmp_path / "confirmation.jsonl"),
            "--out",
            str(tmp_path / "report.json"),
            "--threads",
            "6",
            "--sequence-count",
            "8",
            "--prediction-positions",
            "256",
            "--parity-layers",
            "0",
            "29",
            "--parity-states",
            "3",
        ]
    )

    assert result == 0
    assert captured["model"] == "model"
    assert captured["artifact"] == tmp_path / "model.bin"
    assert captured["dataset"] == tmp_path / "confirmation.jsonl"
    assert captured["artifact_sha256"] == "a" * 64
    assert captured["threads"] == 6
    assert captured["sequence_count"] == 8
    assert captured["prediction_positions"] == 256
    assert captured["record_offset"] == 0
    assert captured["parity_layers"] == [0, 29]
    assert captured["parity_states"] == 3


def test_native_bitnet_package_cli_forwards_integrity_inputs(tmp_path, monkeypatch):
    captured = {}

    def fake_compile(model, artifact, out, **kwargs):
        captured.update({"model": model, "artifact": artifact, "out": out, **kwargs})
        return out

    monkeypatch.setattr("engram.cli.compile_native_bitnet_package", fake_compile)
    result = main(
        [
            "compile-native-bitnet",
            "--model",
            "microsoft/model",
            "--artifact",
            str(tmp_path / "mlp.bin"),
            "--artifact-sha256",
            "b" * 64,
            "--out",
            str(tmp_path / "package"),
            "--threads",
            "6",
        ]
    )

    assert result == 0
    assert captured["artifact"] == tmp_path / "mlp.bin"
    assert captured["out"] == tmp_path / "package"
    assert captured["artifact_sha256"] == "b" * 64
    assert captured["kernel_threads"] == 6


def test_native_bitnet_semantic_memory_cli_forwards_approval_inputs(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_install(package, index, policy, adjudication, out, **kwargs):
        captured.update(
            {
                "package": package,
                "index": index,
                "policy": policy,
                "adjudication": adjudication,
                "out": out,
                **kwargs,
            }
        )
        return out

    monkeypatch.setattr(
        "engram.cli.install_native_bitnet_semantic_memory",
        fake_install,
    )
    result = main(
        [
            "install-native-bitnet-semantic-memory",
            "--model",
            str(tmp_path / "source"),
            "--index",
            str(tmp_path / "index.bin"),
            "--policy",
            str(tmp_path / "policy.json"),
            "--adjudication",
            str(tmp_path / "adjudication.json"),
            "--out",
            str(tmp_path / "derived"),
            "--index-sha256",
            "a" * 64,
            "--policy-sha256",
            "b" * 64,
            "--adjudication-sha256",
            "c" * 64,
        ]
    )

    assert result == 0
    assert captured == {
        "package": tmp_path / "source",
        "index": tmp_path / "index.bin",
        "policy": tmp_path / "policy.json",
        "adjudication": tmp_path / "adjudication.json",
        "out": tmp_path / "derived",
        "coordinate_index_sha256": "a" * 64,
        "policy_manifest_sha256": "b" * 64,
        "adjudication_sha256": "c" * 64,
    }


def test_native_bitnet_dip_token_generation_cli_forwards_runtime_inputs(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"gate_passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_dip_token_generation",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-dip-token-generation",
            "--model",
            str(tmp_path / "model"),
            "--executable",
            str(tmp_path / "runtime"),
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--reference",
            str(tmp_path / "reference.json"),
            "--package-manifest-sha256",
            "d" * 64,
            "--executable-sha256",
            "e" * 64,
            "--out",
            str(tmp_path / "result.json"),
            "--max-tokens",
            "6",
            "--threads",
            "4",
            "--no-verify-reset",
            "--timeout",
            "90",
        ]
    )

    assert result == 0
    assert captured == {
        "package": tmp_path / "model",
        "executable": tmp_path / "runtime",
        "prompts": tmp_path / "prompts.jsonl",
        "reference_report": tmp_path / "reference.json",
        "out": tmp_path / "result.json",
        "package_manifest_sha256": "d" * 64,
        "executable_sha256": "e" * 64,
        "max_new_tokens": 6,
        "threads": 4,
        "verify_reset": False,
        "timeout_seconds": 90.0,
    }


def test_native_bitnet_chat_cli_uses_authenticated_dip_runtime(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeRuntime:
        def __init__(self, package, **kwargs):
            captured["package"] = package
            captured.update(kwargs)

        def __enter__(self):
            captured["entered"] = True
            return self

        def __exit__(self, *_args):
            captured["closed"] = True

    def fail_legacy(*_args, **_kwargs):
        pytest.fail("legacy Transformers runtime must not be constructed")

    def fake_chat(runtime, **kwargs):
        captured["chat_runtime"] = runtime
        captured["chat_kwargs"] = kwargs
        return 0

    monkeypatch.setattr(
        "engram.cli.NativeBitNetDIPTokenRuntime",
        FakeRuntime,
    )
    monkeypatch.setattr("engram.cli.NativeBitNetRuntime", fail_legacy)
    monkeypatch.setattr("engram.cli.run_native_bitnet_chat", fake_chat)
    result = main(
        [
            "chat-native-bitnet",
            "--model",
            str(tmp_path / "model.engram-bitnet-dip"),
            "--library",
            str(tmp_path / "libengram_bitnet_token_runtime.so"),
            "--threads",
            "6",
            "--max-tokens",
            "4",
            "--system",
            "fixture system",
        ]
    )

    assert result == 0
    assert captured["package"] == tmp_path / "model.engram-bitnet-dip"
    assert captured["library"] == (tmp_path / "libengram_bitnet_token_runtime.so")
    assert captured["threads"] == 6
    assert captured["entered"] and captured["closed"]
    assert captured["chat_kwargs"] == {
        "max_new_tokens": 4,
        "system_prompt": "fixture system",
    }


def test_native_bitnet_chat_cli_rejects_legacy_attention_overrides(tmp_path):
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "chat-native-bitnet",
                "--model",
                str(tmp_path / "model"),
                "--attention-library",
                str(tmp_path / "attention.so"),
            ]
        )


def test_native_bitnet_attention_cli_forwards_confirmation_split(tmp_path, monkeypatch):
    captured = {}

    def fake_evaluate(package, dataset, **kwargs):
        captured.update({"package": package, "dataset": dataset, **kwargs})
        return {"semantic_confirmation_passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_attention_substitution",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-attention",
            "--model",
            str(tmp_path / "package"),
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--out",
            str(tmp_path / "report.json"),
            "--sequence-count",
            "8",
            "--prediction-positions",
            "256",
            "--record-offset",
            "8",
            "--modes",
            "hybrid",
            "--local-window",
            "16",
            "--retrieval-top-k",
            "4",
            "--retrieval-candidates",
            "10",
            "--lsh-tables",
            "6",
            "--lsh-bits",
            "7",
            "--lsh-radius",
            "2",
        ]
    )

    assert result == 0
    assert captured["package"] == tmp_path / "package"
    assert captured["record_offset"] == 8
    assert captured["modes"] == ["hybrid"]
    assert captured["prediction_positions"] == 256
    assert captured["retrieval_candidates"] == 10
    assert captured["lsh_tables"] == 6
    assert captured["lsh_bits"] == 7
    assert captured["lsh_radius"] == 2


def test_native_bitnet_controller_cli_forwards_compiled_protocol(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_evaluate(package, dataset, controller, **kwargs):
        captured.update(
            {
                "package": package,
                "dataset": dataset,
                "controller": controller,
                **kwargs,
            }
        )
        return {"frozen_gate_passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_controller_substitution",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-controller",
            "--model",
            str(tmp_path / "package"),
            "--dataset",
            str(tmp_path / "records.jsonl"),
            "--controller",
            str(tmp_path / "controller"),
            "--out",
            str(tmp_path / "report.json"),
            "--sequence-count",
            "8",
            "--prediction-positions",
            "256",
            "--record-offset",
            "8",
            "--retrieval-candidates",
            "8",
            "--retrieval-top-k",
            "4",
        ]
    )

    assert result == 0
    assert captured["package"] == tmp_path / "package"
    assert captured["controller"] == tmp_path / "controller"
    assert captured["record_offset"] == 8
    assert captured["prediction_positions"] == 256
    assert captured["native_projections"] is True


def test_native_bitnet_controller_generation_cli_forwards_suite(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"gate_passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_controller_generation",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-controller-generation",
            "--model",
            str(tmp_path / "package"),
            "--controller",
            str(tmp_path / "controller"),
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--out",
            str(tmp_path / "report.json"),
            "--max-tokens",
            "4",
            "--threads",
            "12",
        ]
    )

    assert result == 0
    assert captured["package"] == tmp_path / "package"
    assert captured["controller"] == tmp_path / "controller"
    assert captured["max_new_tokens"] == 4
    assert captured["threads"] == 12


def test_native_bitnet_dip_attention_cli_forwards_boundaries(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(
        "engram.cli.evaluate_native_bitnet_dip_attention_confirmation",
        fake_evaluate,
    )
    result = main(
        [
            "evaluate-native-bitnet-dip-attention",
            "--model",
            str(tmp_path / "package"),
            "--library",
            str(tmp_path / "runtime.so"),
            "--out",
            str(tmp_path / "report.json"),
            "--lengths",
            "16",
            "24",
            "32",
            "--prompt",
            "fixture prompt",
            "--threads",
            "6",
        ]
    )

    assert result == 0
    assert captured == {
        "package": tmp_path / "package",
        "library": tmp_path / "runtime.so",
        "out": tmp_path / "report.json",
        "lengths": [16, 24, 32],
        "prompt": "fixture prompt",
        "threads": 6,
    }
