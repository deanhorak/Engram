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

from engram.training.budget_native_ternary_codec import (  # noqa: E402
    load_budget_native_ternary_artifact,
)
from engram.training.budget_native_ternary_training import (  # noqa: E402
    train_budget_native_ternary_student,
)


def _write_record(path, values):
    path.write_text(json.dumps({"input_ids": values}) + "\n", encoding="utf-8")


def test_budget_native_ternary_training_serializes_and_reloads_hard_path(
    tmp_path,
):
    torch.manual_seed(20260723)
    config = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
        eos_token_id=2,
    )
    model_path = tmp_path / "model"
    LlamaForCausalLM(config).save_pretrained(
        model_path,
        safe_serialization=True,
    )
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_record(training, [1, 5, 7, 2])
    _write_record(validation, [1, 6, 8, 9, 2])
    output = tmp_path / "run"

    report = train_budget_native_ternary_student(
        model_path,
        training,
        validation,
        output,
        group_size=128,
        steps=1,
        anneal_steps=1,
        transition_mode="deepest_first",
        batch_size=1,
        max_train_records=1,
        max_validation_records=1,
        device="cpu",
        save_artifact=True,
        checkpoint_every=1,
        coadapt_backbone=True,
        backbone_start_step=0,
    )

    mlp_path = output / "budget_native_ternary_mlp.bin"
    backbone_path = output / "budget_native_ternary_backbone.safetensors"
    loaded = load_budget_native_ternary_artifact(mlp_path)
    assert len(loaded.layers) == 2
    assert mlp_path.stat().st_size == report["physical_traffic"][
        "total_cold_bytes"
    ]
    assert backbone_path.is_file()
    assert report["artifact"]["reloaded_before_validation"]
    assert report["validation"]["mlp_weight_source"] == (
        "serialized_reloaded_grouped_ternary_decode"
    )
    assert report["gate"]["checks"]["hard_validation"]
    assert report["gate"]["checks"]["serialized_reload"]
    assert report["gate"]["checks"]["physical_traffic"]
    assert report["configuration"]["coadapt_backbone"]
    assert report["configuration"]["transition_mode"] == "deepest_first"
    assert report["training"]["history"][0]["quantization_strength"] == 1.0
    assert report["training"]["history"][0]["hard_ternary_layers"] == 2
    assert report["training"]["history"][0]["backbone_active"]
    assert (
        output / "budget_native_ternary_checkpoint.pt"
    ).is_file()

    continued_output = tmp_path / "continued"
    continued = train_budget_native_ternary_student(
        model_path,
        training,
        validation,
        continued_output,
        group_size=128,
        steps=1,
        anneal_steps=1,
        transition_mode="global",
        batch_size=1,
        max_train_records=1,
        max_validation_records=1,
        device="cpu",
        save_artifact=True,
        coadapt_backbone=True,
        coadapt_embeddings_and_head=True,
        backbone_start_step=0,
        hidden_weight=2.0,
        final_hidden_weight=3.0,
        final_cka_weight=0.5,
        teacher_top1_weight=0.25,
        initial_checkpoint=(
            output / "budget_native_ternary_checkpoint.pt"
        ),
    )

    initialization = continued["training"]["initialization"]
    assert initialization["source"] == "initial_checkpoint"
    assert not initialization["optimizer_state_retained"]
    assert initialization["source_model_parameters_retained"]
    assert initialization["checkpoint_sha256"]
    assert (
        continued["configuration"]["loss_weights"]["final_hidden"] == 3.0
    )
    assert continued["configuration"]["loss_weights"]["final_cka"] == 0.5
    assert (
        continued["configuration"]["loss_weights"]["teacher_top1"]
        == 0.25
    )
