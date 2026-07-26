import json
import struct

import numpy as np
import pytest

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    NativeBitNetValidationError,
    load_native_bitnet_artifact,
    save_native_bitnet_artifact,
    unpack_base3_rows,
)
from engram.semantic.native_bitnet_dip_index import (
    build_native_bitnet_dip_index,
    load_native_bitnet_dip_index,
)
from engram.utils import sha256_file


def _fixture(tmp_path, *, layers=2):
    hidden = 320
    intermediate = 12
    logical_layers = []
    expected = []
    for layer in range(layers):
        row = np.arange(intermediate)[:, None]
        column = np.arange(hidden)[None, :]
        gate = ((row + column + layer) % 3 - 1).astype(np.int8)
        up = ((2 * row + column + layer + 1) % 3 - 1).astype(np.int8)
        down = (
            (np.arange(hidden)[:, None] + 2 * np.arange(intermediate)[None, :] + layer)
            % 3
            - 1
        ).astype(np.int8)
        logical_layers.append(
            NativeBitNetLayerWeights(
                gate_codes=gate,
                up_codes=up,
                down_codes=down,
                gate_scale=0.5,
                up_scale=0.25,
                down_scale=0.125,
                ffn_sub_norm=np.ones(intermediate, dtype=np.float32),
            )
        )
        expected.append((gate, up, down))
    artifact_path = tmp_path / "model.bitnet-records.bin"
    save_native_bitnet_artifact(
        artifact_path,
        logical_layers,
        rms_norm_eps=1e-5,
    )
    artifact = load_native_bitnet_artifact(artifact_path)
    policy = {
        "experiment": "native_bitnet_dip_joint_candidate_adaptive_k_policy",
        "artifact_sha256": artifact.payload_sha256,
        "decision": "use_joint_policy_for_candidate_only_causal_development",
        "configuration": {
            "input_fraction": 0.75,
            "input_coordinates": 240,
            "minimum_k": 2,
            "energy_target": 1.0,
        },
        "selected_policy": {
            "candidate_counts": [8 + layer for layer in range(layers)],
            "maximum_ks": [5 + layer for layer in range(layers)],
            "rms_policies": [
                {
                    "estimator": "candidate_ratio",
                    "audit_count": 0,
                    "audit_strategy": "none",
                }
                for _ in range(layers)
            ],
        },
        "progression_screen": {"passed": True},
    }
    policy_path = tmp_path / "joint-policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    return artifact_path, policy_path, expected


def test_native_bitnet_dip_index_roundtrip_mmaps_exact_coordinate_rows(tmp_path):
    artifact, policy, expected = _fixture(tmp_path)
    path = build_native_bitnet_dip_index(
        artifact,
        policy,
        tmp_path / "model.bitnet-dip-index.bin",
    )

    with load_native_bitnet_dip_index(path) as index:
        assert index.hidden_size == 320
        assert index.intermediate_size == 12
        assert len(index.layers) == 2
        assert not index.layers[0].gate_coordinates.flags.writeable
        for layer, (gate, up, down) in zip(
            index.layers,
            expected,
            strict=True,
        ):
            np.testing.assert_array_equal(
                unpack_base3_rows(
                    layer.gate_coordinates,
                    logical_width=12,
                ),
                gate.T,
            )
            np.testing.assert_array_equal(
                unpack_base3_rows(
                    layer.up_coordinates,
                    logical_width=12,
                ),
                up.T,
            )
            np.testing.assert_array_equal(
                layer.down_norm_squared,
                np.count_nonzero(down, axis=0),
            )
            assert layer.offset % 64 == 0
            assert layer.gate_stream_offset == 128
            assert layer.up_stream_offset % 64 == 0
            assert layer.down_norm_stream_offset % 64 == 0
        assert index.layers[0].policy.input_coordinates == 240
        assert index.layers[0].policy.candidate_count == 8
        assert index.layers[0].policy.minimum_top_k == 2
        assert index.layers[0].policy.maximum_top_k == 5
        assert index.layers[0].policy.energy_target == 1.0
        assert index.layers[0].policy.rms_audit_count == 0
        assert index.layers[0].policy.rms_estimator == "candidate_ratio"
        assert index.layers[0].policy.rms_audit_strategy == "none"
        metadata = index.metadata()
        assert metadata["endianness"] == "little"
        assert metadata["coordinate_dtype"] == "uint8"
        assert metadata["down_norm_dtype"] == "little_endian_uint16"
        assert metadata["checksum"] == "sha256_per_layer_data_and_policy"
        assert metadata["payload_sha256"] == sha256_file(path)
        assert metadata["source_artifact_sha256"] == sha256_file(artifact)

    # A second load proves the first reader did not materialize or mutate the
    # artifact and that normal context-manager shutdown releases the mapping.
    with load_native_bitnet_dip_index(path) as reloaded:
        assert reloaded.layers[1].policy.candidate_count == 9


def test_native_bitnet_dip_index_bytes_equal_physical_accounting(tmp_path):
    artifact, policy, _ = _fixture(tmp_path)
    path = build_native_bitnet_dip_index(
        artifact,
        policy,
        tmp_path / "model.bitnet-dip-index.bin",
    )
    accounting = native_bitnet_dip_physical_accounting(
        320,
        12,
        input_counts=[240, 240],
        candidate_counts=[8, 9],
        top_ks=[5, 6],
    )

    assert path.stat().st_size == accounting["serialization"]["coordinate_index_bytes"]
    assert path.stat().st_size == 82_496


def test_native_bitnet_dip_index_detects_payload_and_policy_corruption(tmp_path):
    artifact, policy, _ = _fixture(tmp_path, layers=1)
    path = build_native_bitnet_dip_index(
        artifact,
        policy,
        tmp_path / "model.bitnet-dip-index.bin",
    )

    # Header (128) + directory (64) + layer header (128) points at gate data.
    with path.open("r+b") as handle:
        handle.seek(128 + 64 + 128)
        byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([byte[0] ^ 0x01]))
    with pytest.raises(
        NativeBitNetValidationError,
        match="checksum mismatch",
    ):
        load_native_bitnet_dip_index(path)

    # Rebuild, then change a still-plausible candidate count in the directory.
    build_native_bitnet_dip_index(artifact, policy, path)
    with path.open("r+b") as handle:
        handle.seek(128 + 8)
        handle.write(struct.pack("<I", 9))
    with pytest.raises(
        NativeBitNetValidationError,
        match="checksum mismatch",
    ):
        load_native_bitnet_dip_index(path)


def test_native_bitnet_dip_index_rejects_metadata_and_policy_mismatch(tmp_path):
    artifact, policy, _ = _fixture(tmp_path, layers=1)
    payload = json.loads(policy.read_text())
    payload["artifact_sha256"] = "0" * 64
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        NativeBitNetValidationError,
        match="source-artifact SHA-256 mismatch",
    ):
        build_native_bitnet_dip_index(
            artifact,
            policy,
            tmp_path / "model.bitnet-dip-index.bin",
        )

    artifact, policy, _ = _fixture(tmp_path / "second", layers=1)
    path = build_native_bitnet_dip_index(
        artifact,
        policy,
        tmp_path / "second" / "model.bitnet-dip-index.bin",
    )
    with path.open("r+b") as handle:
        handle.seek(12)
        handle.write(struct.pack("<I", 0x04030201))
    with pytest.raises(
        NativeBitNetValidationError,
        match="metadata is unsupported",
    ):
        load_native_bitnet_dip_index(path)


def test_native_bitnet_dip_index_rejects_unapproved_or_inconsistent_policy(tmp_path):
    artifact, policy, _ = _fixture(tmp_path, layers=1)
    payload = json.loads(policy.read_text())
    payload["progression_screen"]["passed"] = False
    policy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        NativeBitNetValidationError,
        match="not an approved",
    ):
        build_native_bitnet_dip_index(
            artifact,
            policy,
            tmp_path / "model.bitnet-dip-index.bin",
        )
