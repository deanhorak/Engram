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

from engram.training.structured_experts import train_native_gate_end_to_end  # noqa: E402


def _write_record(path, values):
    path.write_text(
        json.dumps({"input_ids": values, "input_type": "prose"}) + "\n",
        encoding="utf-8",
    )


def test_native_gate_cpu_checkpoint_resumes_to_larger_step_target(tmp_path):
    torch.manual_seed(311)
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
    LlamaForCausalLM(config).save_pretrained(
        model_path, safe_serialization=True
    )
    training = tmp_path / "training.jsonl"
    validation = tmp_path / "validation.jsonl"
    _write_record(training, [1, 5, 7, 2])
    _write_record(validation, [1, 6, 8, 9, 2])
    output = tmp_path / "run"
    arguments = {
        "target_top_k": 4,
        "target_input_fraction": 0.5,
        "warmup_steps": 0,
        "anneal_steps": 2,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "max_train_records": 1,
        "max_validation_records": 1,
        "device": "cpu",
        "save_artifact": False,
        "checkpoint_every": 1,
    }
    first = train_native_gate_end_to_end(
        model_path,
        training,
        validation,
        output,
        steps=1,
        **arguments,
    )
    assert len(first["training"]["history"]) == 1
    assert first["training"]["checkpoint"]["written"]
    assert first["validation"]["hard_sparse_path_only"]

    resumed = train_native_gate_end_to_end(
        model_path,
        training,
        validation,
        output,
        steps=2,
        resume=True,
        **arguments,
    )
    assert resumed["training"]["completed_before_resume"] == 1
    assert len(resumed["training"]["history"]) == 2
    assert resumed["training"]["history"][-1]["top_k"] == 4
    assert resumed["configuration"]["device_neutral_semantics"]
    assert not resumed["artifact"]["written"]
