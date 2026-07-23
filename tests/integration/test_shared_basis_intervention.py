import hashlib
import json

import pytest

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from engram.evaluation.shared_basis_intervention import (  # noqa: E402
    _load_transformers,
    evaluate_shared_basis_intervention,
)
from engram.models.inspection import inspect_model  # noqa: E402
from engram.semantic.shared_basis import (  # noqa: E402
    ESIB_MANIFEST_FORMAT,
    decode_shared_basis_artifact,
)
from tests.shared_basis_fixture import write_esib_fixture  # noqa: E402

try:
    torch, _, _ = _load_transformers()
    LlamaForCausalLM = transformers.LlamaForCausalLM
except (ImportError, RuntimeError) as error:
    pytest.skip(
        f"local Transformers Llama stack is unavailable: {error}",
        allow_module_level=True,
    )


def _dataset(path, offset):
    records = [
        {"input_ids": [1, 5 + offset, 9 + offset, 3 + offset, 2]},
        {"input_ids": [1, 7 + offset, 4 + offset, 8 + offset]},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_shared_basis_arm_physically_substitutes_authenticated_all_layer_artifacts(
    tmp_path,
):
    torch.manual_seed(911)
    model_path = tmp_path / "teacher"
    config = transformers.LlamaConfig(
        vocab_size=48,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    LlamaForCausalLM(config).save_pretrained(model_path, safe_serialization=True)
    source_hash = inspect_model(model_path).source_hash

    artifact_paths = []
    entries = []
    for layer in range(2):
        artifact_path = tmp_path / f"layer-{layer}.esib"
        write_esib_fixture(
            artifact_path,
            layer=layer,
            hidden=8,
            width=12,
            rank=4,
            top_k=12,
        )
        artifact = decode_shared_basis_artifact(artifact_path.read_bytes())
        artifact_paths.append(artifact_path)
        entries.append(
            {
                "layer": layer,
                "path": artifact_path.name,
                "sha256": artifact.artifact_sha256,
                "content_checksum": artifact.content_checksum,
            }
        )
    calibration = tmp_path / "calibration.jsonl"
    validation = tmp_path / "validation.jsonl"
    _dataset(calibration, 0)
    _dataset(validation, 2)
    manifest = tmp_path / "artifacts.json"
    manifest.write_text(
        json.dumps(
            {
                "format": ESIB_MANIFEST_FORMAT,
                "version": 1,
                "source_model_hash": source_hash,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 2,
                "artifacts": entries,
                "provenance": {
                    "calibration_dataset_hash": hashlib.sha256(
                        calibration.read_bytes()
                    ).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_shared_basis_intervention(
        model_path,
        validation,
        manifest,
        calibration_dataset=calibration,
        device="cpu",
    )

    arm = report["arms"][0]
    assert arm["variant"] == "shared_basis"
    assert arm["layer_indices"] == [0, 1]
    assert arm["selection_scope"]["mode"] == "full_converted_width"
    assert arm["selection_scope"]["candidate_shortlist"] is False
    assert "candidate_recall" not in arm["local_mlp"]
    assert "candidate_count" not in arm
    assert arm["execution"] == {
        "physical_mlp_substitution": True,
        "post_forward_output_replacement": False,
        "artifact_only_selected_mlp_forward": True,
        "dense_source_mlp_parameters_present_in_student": False,
        "decoded_float32_quality_execution": True,
        "native_compressed_kernel_timing_valid": False,
    }
    assert arm["projected_accounting"]["cold_fraction_of_dense_q4"] > 0
    assert report["calibration"]["record_level_disjoint"] is True
    assert report["artifact_manifest"]["strict_reload_completed"] is True
