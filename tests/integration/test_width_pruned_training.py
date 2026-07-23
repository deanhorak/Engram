import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
transformers = pytest.importorskip("transformers")

import transformers.utils as transformers_utils  # noqa: E402
import transformers.utils.import_utils as transformers_imports  # noqa: E402

if transformers_imports.is_sklearn_available():
    try:
        import sklearn  # noqa: F401
    except ImportError:

        def sklearn_unavailable():
            return False

        transformers_imports.is_sklearn_available = sklearn_unavailable
        transformers_utils.is_sklearn_available = sklearn_unavailable

try:
    LlamaForCausalLM = transformers.LlamaForCausalLM
except (ImportError, RuntimeError) as error:
    pytest.skip(
        f"local Transformers Llama stack is unavailable: {error}",
        allow_module_level=True,
    )

from engram.training.width_pruning import train_width_pruned_student  # noqa: E402


def _write_record(path, values):
    path.write_text(json.dumps({"input_ids": values}) + "\n", encoding="utf-8")


def test_width_pruned_training_runs_all_compact_cpu_validation(tmp_path):
    torch.manual_seed(53)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
    )
    model_path = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(model_path, safe_serialization=True)
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_record(training, [1, 5, 7, 2])
    _write_record(validation, [1, 6, 8, 9, 2])

    arguments = {
        "target_widths": [3, 5],
        "batch_size": 1,
        "max_train_records": 1,
        "max_validation_records": 1,
        "device": "cpu",
        "save_artifact": False,
        "checkpoint_every": 1,
        "coadapt_backbone": True,
        "coadapt_embeddings_and_head": True,
        "backbone_learning_rate": 2e-5,
    }
    report = train_width_pruned_student(
        model_path,
        training,
        validation,
        tmp_path / "run",
        steps=1,
        **arguments,
    )

    assert report["training"]["history"][-1]["converted_layers"] == 2
    assert report["validation"]["all_layers_compact"]
    assert report["projected_traffic"]["fraction_of_dense"] == 1 / 3
    assert report["projected_traffic"]["target_intermediate_size"] is None
    assert report["projected_traffic"]["layer_widths"] == [3, 5]
    assert report["configuration"]["target_width"] is None
    assert report["configuration"]["target_widths"] == [3, 5]
    assert report["gate"]["checks"]["all_layers_compact"]
    assert not report["artifact"]["written"]
    assert report["training"]["checkpoint"]["written"]
    assert report["configuration"]["coadapt_backbone"]
    assert report["configuration"]["coadapt_embeddings_and_head"]
    assert report["configuration"]["backbone_trainable_parameters"] > 0
    assert report["configuration"]["incremental_mlp_traffic_from_backbone"] == 0
    checkpoint_manifest = json.loads(
        (tmp_path / "run" / "width_pruned_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["configuration"]["target_width"] is None
    assert checkpoint_manifest["configuration"]["target_widths"] == [3, 5]

    resumed = train_width_pruned_student(
        model_path,
        training,
        validation,
        tmp_path / "run",
        steps=2,
        resume=True,
        **arguments,
    )
    assert resumed["training"]["completed_before_resume"] == 1
    assert len(resumed["training"]["history"]) == 2

    fresh_training = tmp_path / "fresh_training.jsonl"
    _write_record(fresh_training, [1, 10, 11, 2])
    transferred = train_width_pruned_student(
        model_path,
        fresh_training,
        validation,
        tmp_path / "transferred",
        steps=1,
        initial_checkpoint=tmp_path / "run" / "width_pruned_checkpoint.pt",
        **arguments,
    )
    lineage = transferred["training"]["initial_checkpoint"]
    assert transferred["training"]["completed_before_resume"] == 0
    assert len(transferred["training"]["history"]) == 1
    assert lineage["mode"] == "parameters_only"
    assert not lineage["optimizer_restored"]
    assert not lineage["history_restored"]


def test_width_pruned_strict_q4_reload_drives_causal_validation(tmp_path):
    torch.manual_seed(59)
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
    )
    model_path = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(model_path, safe_serialization=True)
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_record(training, [1, 5, 7, 2])
    _write_record(validation, [1, 6, 8, 9, 2])
    output = tmp_path / "strict"

    report = train_width_pruned_student(
        model_path,
        training,
        validation,
        output,
        target_width=4,
        steps=1,
        batch_size=1,
        max_train_records=1,
        max_validation_records=1,
        device="cpu",
        save_artifact=True,
        strict_q4_deployment=True,
        fake_q4_training=True,
        checkpoint_every=1,
        coadapt_backbone=True,
    )

    q4_path = output / "width_pruned_mlp.q4.bin"
    backbone_path = output / "width_pruned_backbone.safetensors"
    assert q4_path.is_file()
    assert backbone_path.is_file()
    assert not (output / "width_pruned_student.safetensors").exists()
    assert report["configuration"]["strict_q4_deployment"]
    assert report["configuration"]["fake_q4_training"]
    assert report["validation"]["mlp_weight_source"] == (
        "serialized_reloaded_q4_decode"
    )
    assert report["validation"]["fake_q4_disabled_after_serialized_reload"]
    assert report["artifact"]["fake_q4_training"]
    assert report["artifact"]["q4"]["reloaded_before_validation"]
    assert report["artifact"]["backbone"]["reloaded_before_validation"]
    assert report["gate"]["checks"]["q4_serialized_reload"]
    assert report["projected_traffic"]["target_intermediate_size"] == 4
    checkpoint_manifest = json.loads(
        (output / "width_pruned_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["configuration"]["fake_q4_training"]
    traffic = report["strict_q4_traffic"]
    assert traffic["total_cold_bytes"] == q4_path.stat().st_size
    assert report["deployment_traffic_fraction"] == traffic["fraction_of_dense_q4"]
