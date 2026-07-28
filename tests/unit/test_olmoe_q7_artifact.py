import numpy as np
import pytest

from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.inspection import load_local_named_tensors
from engram.models.olmoe_q7 import (
    LoadedOLMoEQ7Artifact,
    OLMoEQ7ValidationError,
    dequantize_q7_matrix,
    olmoe_q7_layout,
    pack_q7_codes,
    quantize_q7_matrix,
    repack_olmoe_q7_model,
    unpack_q7_codes,
)
from engram.evaluation.olmoe_q7_native import OLMoEQ7NativeKernel
from engram.evaluation.olmoe_q7_systems import (
    evaluate_olmoe_q7_native_systems,
)


def test_q7_pack_round_trip_and_rejects_noncanonical_values():
    rng = np.random.default_rng(91)
    for count in range(1, 33):
        codes = rng.integers(-63, 64, size=count, dtype=np.int8)
        packed = pack_q7_codes(codes)
        np.testing.assert_array_equal(unpack_q7_codes(packed, count), codes)
        assert len(packed) == (count * 7 + 7) // 8

    with pytest.raises(OLMoEQ7ValidationError, match="reserved code"):
        unpack_q7_codes(bytes([127]), 1)
    with pytest.raises(OLMoEQ7ValidationError, match="tail"):
        unpack_q7_codes(bytes([0x80]), 1)
    with pytest.raises(OLMoEQ7ValidationError, match=r"\[-63, 63\]"):
        pack_q7_codes(np.array([64], dtype=np.int16))


def test_q7_quantizer_executes_stored_bf16_scales():
    rng = np.random.default_rng(22)
    source = rng.normal(size=(7, 70)).astype(np.float32)

    codes, scale_bits = quantize_q7_matrix(source, group_size=64)
    decoded = dequantize_q7_matrix(codes, scale_bits, group_size=64)

    assert codes.shape == source.shape
    assert scale_bits.shape == (7, 2)
    assert int(codes.min()) >= -63
    assert int(codes.max()) <= 63
    assert np.linalg.norm(decoded - source) / np.linalg.norm(source) < 0.02


def test_tiny_olmoe_q7_artifact_is_addressable_and_exactly_accounted(tmp_path):
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    artifact_path = repack_olmoe_q7_model(
        model, tmp_path / "model.engram-olmoe-q7", group_size=8
    )
    names = [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
        "model.layers.0.mlp.experts.2.down_proj.weight",
    ]
    source = load_local_named_tensors(model, names)

    with LoadedOLMoEQ7Artifact(artifact_path) as artifact:
        layout = artifact.layout
        assert artifact_path.stat().st_size == layout.file_bytes
        assert layout == olmoe_q7_layout(
            layer_count=2,
            hidden_size=16,
            intermediate_size=8,
            num_experts=4,
            top_k=2,
            group_size=8,
        )
        expected_router_bits = (
            source[names[0]]
            .astype(np.float32)
            .view(np.uint32)
        )
        bias = np.uint32(0x7FFF) + (
            (expected_router_bits >> np.uint32(16)) & np.uint32(1)
        )
        expected_router = (
            ((expected_router_bits + bias) & np.uint32(0xFFFF0000))
            .view(np.float32)
        )
        np.testing.assert_array_equal(artifact.router(0), expected_router)
        decoded = artifact.expert(0, 2)
        for phase, name in zip(("gate", "up", "down"), names[1:], strict=True):
            relative_error = np.linalg.norm(decoded[phase] - source[name]) / np.linalg.norm(
                source[name]
            )
            assert relative_error < 0.02
        assert artifact.metadata()["quantizer"]["scale_dtype"] == (
            "bfloat16_executed"
        )


def test_q7_loader_rejects_tampered_header_and_packed_tail(tmp_path):
    model = create_tiny_olmoe_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=5,
        num_heads=2,
        num_layers=1,
        num_experts=2,
        num_experts_per_token=1,
    )
    path = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=4)
    payload = bytearray(path.read_bytes())
    payload[0] ^= 1
    broken = tmp_path / "broken.q7"
    broken.write_bytes(payload)

    with pytest.raises(OLMoEQ7ValidationError, match="header contract"):
        LoadedOLMoEQ7Artifact(broken)


def test_native_q7_kernel_matches_decoded_sparse_reference(tmp_path):
    library = (
        __import__("pathlib").Path(__file__).parents[2]
        / "build"
        / "libengram_olmoe_q7.so"
    )
    if not library.is_file():
        pytest.skip("native OLMoE Q7 library has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    path = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    rng = np.random.default_rng(81)
    states = rng.normal(size=(5, 16)).astype(np.float32)

    with LoadedOLMoEQ7Artifact(path) as artifact:
        router = artifact.router(1)
        logits = states @ router.T
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        selected = np.argsort(-probabilities, axis=1, kind="stable")[:, :2]
        reference = np.zeros_like(states)
        decoded = {
            expert: artifact.expert(1, expert)
            for expert in np.unique(selected)
        }
        for row, state in enumerate(states):
            for expert in selected[row]:
                weights = decoded[int(expert)]
                gate = state @ weights["gate"].T
                activation = (gate / (1.0 + np.exp(-gate))) * (
                    state @ weights["up"].T
                )
                reference[row] += probabilities[row, expert] * (
                    activation @ weights["down"].T
                )

    with OLMoEQ7NativeKernel(path, library, threads=2) as kernel:
        result = kernel.forward(1, states)

    np.testing.assert_array_equal(result.selected_experts, selected)
    np.testing.assert_allclose(result.output, reference, rtol=2e-5, atol=2e-6)
    assert result.metrics["selected_experts"] == states.shape[0] * 2
    assert result.metrics["router_stream_bytes"] == states.shape[0] * 4 * 16 * 2
    assert result.metrics["scheduled_stream_bytes"] < (
        states.shape[0] * 4 * artifact.layout.expert_payload_bytes
    )

    report = evaluate_olmoe_q7_native_systems(
        path,
        library,
        tmp_path / "systems.json",
        layer=1,
        states=2,
        threads=2,
        seed=4,
        maximum_traffic_fraction=2.0,
    )
    assert report["gate_passed"]
    assert report["parity"]["route_exact"]
    assert report["traffic"]["fraction_of_all_expert_ideal_q4"] < 2.0
