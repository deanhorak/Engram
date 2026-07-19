import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
try:
    LlamaForCausalLM = transformers.LlamaForCausalLM
except (ImportError, RuntimeError) as error:
    pytest.skip(f"local Transformers Llama stack is unavailable: {error}", allow_module_level=True)

from engram.compiler import compile_model
from engram.evaluation.end_to_end import evaluate_end_to_end


def test_tiny_local_teacher_quality_report_is_measured_not_claimed(tmp_path):
    torch.manual_seed(55)
    config = transformers.LlamaConfig(
        vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    )
    teacher = tmp_path / "teacher"
    LlamaForCausalLM(config).save_pretrained(teacher, safe_serialization=True)
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(json.dumps({"input_ids": [1, 4, 7, 9], "input_type": "code"}) + "\n")
    package = compile_model(teacher, tmp_path / "model.engram", semantic_top_k=4, semantic_candidates=8)
    report = evaluate_end_to_end(package, teacher, dataset)
    assert report["tokens_evaluated"] == 3
    assert report["status"] == "local_teacher_measurement"
    assert report["quality_targets_met"] is None
    assert np_is_finite(report["teacher_student_kl"])


def np_is_finite(value):
    return value == value and abs(value) != float("inf")
