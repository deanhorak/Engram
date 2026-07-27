import json
import struct

import numpy as np

from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe import _parse_safetensors_header, audit_olmoe_source


def test_tiny_olmoe_fixture_passes_exact_local_contract(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")

    audit = audit_olmoe_source(model)

    assert audit.decision == "proceed_to_router_trace"
    assert audit.tensor_contract["required_tensor_count"] == 45
    assert audit.tensor_contract["required_names_complete"]
    assert audit.tensor_contract["exact_shapes_validated"]
    assert audit.tensor_contract["shape_errors"] == []
    assert audit.projected_traffic is not None
    assert audit.projected_traffic["active_expert_fraction"] == 0.5
    assert not audit.projected_traffic["passes_45_percent_projection"]


def test_olmoe_audit_rejects_missing_expert_tensor(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    weights = model / "weights.npz"
    with np.load(weights, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "model.layers.1.mlp.experts.3.down_proj.weight"
        }
    np.savez(weights, **arrays)
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    del index["weight_map"]["model.layers.1.mlp.experts.3.down_proj.weight"]
    index_path.write_text(json.dumps(index) + "\n", encoding="utf-8")

    audit = audit_olmoe_source(model)

    assert audit.decision == "reject_incompatible_olmoe_contract"
    assert audit.tensor_contract["missing_tensor_names"] == [
        "model.layers.1.mlp.experts.3.down_proj.weight"
    ]


def test_olmoe_audit_rejects_shape_mismatch(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    weights = model / "weights.npz"
    with np.load(weights, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["model.layers.0.mlp.gate.weight"] = np.zeros(
        (3, 16), dtype=np.float32
    )
    np.savez(weights, **arrays)

    audit = audit_olmoe_source(model)

    assert audit.decision == "reject_incompatible_olmoe_contract"
    assert audit.tensor_contract["shape_errors"] == [
        {
            "name": "model.layers.0.mlp.gate.weight",
            "actual": [3, 16],
            "expected": [4, 16],
        }
    ]


def test_olmoe_metadata_only_index_advances_to_shape_audit(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    (model / "weights.npz").unlink()

    audit = audit_olmoe_source(model)

    assert audit.decision == "proceed_to_exact_weight_shape_audit"
    assert audit.tensor_contract["required_names_complete"]
    assert not audit.tensor_contract["exact_shapes_validated"]
    assert audit.capabilities["metadata_tensor_inventory"]


def test_olmoe_audit_rejects_invalid_top_k(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    config_path = model / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["num_experts_per_tok"] = config["num_experts"] + 1
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    audit = audit_olmoe_source(model)

    assert audit.decision == "reject_incompatible_olmoe_contract"
    assert not audit.checks["valid_expert_selection"]


def test_bounded_safetensors_header_parser_extracts_shapes():
    header = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "model.layers.0.mlp.gate.weight": {
                "dtype": "BF16",
                "shape": [64, 2048],
                "data_offsets": [0, 262144],
            },
        }
    ).encode("utf-8")
    payload = struct.pack("<Q", len(header)) + header

    inventory = _parse_safetensors_header(payload, "model-00001.safetensors")

    assert len(inventory) == 1
    assert inventory[0].name == "model.layers.0.mlp.gate.weight"
    assert inventory[0].shape == (64, 2048)
    assert inventory[0].dtype == "BF16"
