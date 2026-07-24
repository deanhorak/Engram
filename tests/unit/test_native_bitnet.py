import json
import struct
from pathlib import Path

import numpy as np
import pytest

from engram.evaluation.native_bitnet_parity import evaluate_native_bitnet_parity
from engram.evaluation.native_bitnet_kernel import NativeBitNetCPUKernel
from engram.models.native_bitnet import (
    OFFICIAL_NATIVE_BITNET_REPO,
    OFFICIAL_NATIVE_BITNET_REVISION,
    OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256,
    NativeBitNetLayerWeights,
    NativeBitNetValidationError,
    audit_native_bitnet_source,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
    native_bitnet_mlp_forward,
    native_bitnet_repack_traffic,
    pack_base3_rows,
    pack_hf_bitnet_codes,
    repack_native_bitnet_model,
    save_native_bitnet_artifact,
    unpack_base3_rows,
    unpack_hf_bitnet_codes,
)


def _write_config(path, *, hidden=8, intermediate=12, layers=2):
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["BitNetForCausalLM"],
        "hidden_act": "relu2",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "model_type": "bitnet",
        "num_attention_heads": 2,
        "num_hidden_layers": layers,
        "rms_norm_eps": 1e-5,
        "vocab_size": 32,
        "quantization_config": {
            "quant_method": "bitnet",
            "linear_class": "autobitlinear",
            "quantization_mode": "offline",
        },
    }
    (path / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def _layers(*, hidden=8, intermediate=12, count=2):
    rng = np.random.default_rng(20260723)
    result = []
    for _ in range(count):
        result.append(
            NativeBitNetLayerWeights(
                gate_codes=rng.integers(
                    -1, 2, size=(intermediate, hidden), dtype=np.int8
                ),
                up_codes=rng.integers(
                    -1, 2, size=(intermediate, hidden), dtype=np.int8
                ),
                down_codes=rng.integers(
                    -1, 2, size=(hidden, intermediate), dtype=np.int8
                ),
                gate_scale=0.125,
                up_scale=0.25,
                down_scale=0.5,
                ffn_sub_norm=np.ones(intermediate, dtype=np.float32),
            )
        )
    return result


def _write_packed_source(model, layers):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    intermediate, hidden = np.asarray(layers[0].gate_codes).shape
    _write_config(
        model,
        hidden=hidden,
        intermediate=intermediate,
        layers=len(layers),
    )
    tensors = {}
    for index, layer in enumerate(layers):
        prefix = f"model.layers.{index}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = torch.from_numpy(
            pack_hf_bitnet_codes(layer.gate_codes)
        )
        tensors[f"{prefix}.up_proj.weight"] = torch.from_numpy(
            pack_hf_bitnet_codes(layer.up_codes)
        )
        tensors[f"{prefix}.down_proj.weight"] = torch.from_numpy(
            pack_hf_bitnet_codes(layer.down_codes)
        )
        tensors[f"{prefix}.gate_proj.weight_scale"] = torch.tensor(
            [layer.gate_scale],
            dtype=torch.bfloat16,
        )
        tensors[f"{prefix}.up_proj.weight_scale"] = torch.tensor(
            [layer.up_scale],
            dtype=torch.bfloat16,
        )
        tensors[f"{prefix}.down_proj.weight_scale"] = torch.tensor(
            [layer.down_scale],
            dtype=torch.bfloat16,
        )
        tensors[f"{prefix}.ffn_sub_norm.weight"] = torch.tensor(
            layer.ffn_sub_norm,
            dtype=torch.bfloat16,
        )
    safetensors.save_file(tensors, model / "model.safetensors")


def test_metadata_only_audit_is_fail_closed_for_unverified_local_source(
    tmp_path,
):
    model = tmp_path / "model"
    _write_config(model)

    audit = audit_native_bitnet_source(model)

    assert all(audit.checks.values())
    assert audit.capabilities["exact_native_repack"]
    assert not audit.capabilities["exact_swiglu_decomposition"]
    assert not audit.capabilities["existing_semantic_compiler"]
    assert audit.provenance["status"] == "format_only_unverified"
    assert audit.decision == "reject_or_require_explicit_provenance"
    assert audit.combined_gate_status == "not_evaluated_metadata_only"


def test_official_audit_pins_revision_and_attestation(tmp_path, monkeypatch):
    model = tmp_path / "official"
    _write_config(model, hidden=2560, intermediate=6912, layers=30)
    observed = {}

    def fake_download(*, repo_id, filename, revision, cache_dir):
        observed.update(
            {
                "repo_id": repo_id,
                "filename": filename,
                "revision": revision,
                "cache_dir": cache_dir,
            }
        )
        return str(model / "config.json")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    audit = audit_native_bitnet_source(OFFICIAL_NATIVE_BITNET_REPO)

    assert observed["revision"] == OFFICIAL_NATIVE_BITNET_REVISION
    assert audit.resolved_revision == OFFICIAL_NATIVE_BITNET_REVISION
    assert audit.provenance["status"] == "pinned_official_attestation"
    assert audit.decision == "proceed_to_exact_weight_repack"
    assert audit.projected_traffic["serialized_artifact_bytes"] == 318_924_544


def test_audit_rejects_swiglu_metadata_disguised_as_bitnet(tmp_path):
    model = tmp_path / "model"
    config = _write_config(model)
    config["hidden_act"] = "silu"
    (model / "config.json").write_text(
        json.dumps(config) + "\n",
        encoding="utf-8",
    )

    audit = audit_native_bitnet_source(model)

    assert not audit.checks["activation_relu2"]
    assert audit.projected_traffic is None
    assert audit.decision == "reject_or_require_explicit_provenance"


def test_hf_two_bit_round_trip_and_invalid_code_three():
    rng = np.random.default_rng(7)
    codes = rng.integers(-1, 2, size=(12, 8), dtype=np.int8)
    packed = pack_hf_bitnet_codes(codes)

    np.testing.assert_array_equal(
        unpack_hf_bitnet_codes(packed, out_features=12),
        codes,
    )
    corrupt = packed.copy()
    corrupt[0, 0] |= np.uint8(0b11)
    with pytest.raises(NativeBitNetValidationError, match="invalid.*code 3"):
        unpack_hf_bitnet_codes(corrupt, out_features=12)


def test_record_local_base3_round_trip_and_canonical_tail():
    codes = np.asarray(
        [
            [-1, 0, 1, 1, -1, 0, 1],
            [1, -1, 0, 0, 1, -1, -1],
        ],
        dtype=np.int8,
    )
    packed = pack_base3_rows(codes)

    np.testing.assert_array_equal(
        unpack_base3_rows(packed, logical_width=7),
        codes,
    )
    corrupt = packed.copy()
    corrupt[0, -1] += np.uint8(27)
    with pytest.raises(NativeBitNetValidationError, match="tail"):
        unpack_base3_rows(corrupt, logical_width=7)


def test_record_artifact_preserves_codes_scales_norm_and_exact_size(tmp_path):
    layers = _layers()
    path = tmp_path / "model.bitnet-records.bin"

    save_native_bitnet_artifact(path, layers, rms_norm_eps=1e-5)
    loaded = load_native_bitnet_artifact(path)
    decoded = decode_native_bitnet_layer(loaded, 1)
    traffic = native_bitnet_repack_traffic(8, 12, layer_count=2)

    assert path.stat().st_size == traffic["serialized_artifact_bytes"]
    assert loaded.serialized_artifact_bytes == path.stat().st_size
    np.testing.assert_array_equal(
        decoded["gate_codes"],
        np.asarray(layers[1].gate_codes),
    )
    np.testing.assert_array_equal(
        decoded["up_codes"],
        np.asarray(layers[1].up_codes),
    )
    np.testing.assert_array_equal(
        decoded["down_codes"],
        np.asarray(layers[1].down_codes),
    )
    assert float(decoded["gate_scale"]) == layers[1].gate_scale
    assert float(decoded["up_scale"]) == layers[1].up_scale
    assert float(decoded["down_scale"]) == layers[1].down_scale
    np.testing.assert_array_equal(
        decoded["ffn_sub_norm"],
        np.asarray(layers[1].ffn_sub_norm),
    )
    assert all(offset % 64 == 0 for offset in loaded.layer_offsets)


def test_native_reference_forward_is_finite_and_deterministic(tmp_path):
    path = tmp_path / "model.bitnet-records.bin"
    save_native_bitnet_artifact(
        path,
        _layers(),
        rms_norm_eps=1e-5,
    )
    loaded = load_native_bitnet_artifact(path)
    states = np.asarray(
        [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4, -0.1, 0.05]],
        dtype=np.float32,
    )

    first = native_bitnet_mlp_forward(loaded, 0, states)
    second = native_bitnet_mlp_forward(loaded, 0, states)

    assert first.shape == states.shape
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_official_dual_traffic_accounting_passes_without_baseline_swap():
    traffic = native_bitnet_repack_traffic(
        2560,
        6912,
        layer_count=30,
    )

    assert traffic["serialized_artifact_bytes"] == 318_924_544
    assert traffic["hf_native_two_bit_mlp_bytes"] == 398_546_100
    assert traffic["dense_q4_source_mlp_bytes"] == 796_262_400
    assert traffic["fraction_of_dense_q4"] == pytest.approx(0.400526941872428)
    assert traffic["fraction_of_hf_native_two_bit"] == pytest.approx(0.8002199594977846)
    assert traffic["serialized_layout_passes_45_percent_gate"]
    assert traffic["modelled_full_phase_schedule_passes_45_percent_gate"]
    assert traffic["independently_scattered_records_passes_45_percent_gate"]
    assert (
        traffic["modelled_full_phase_schedule_bytes"]
        == traffic["serialized_artifact_bytes"]
    )
    assert traffic["independently_scattered_records_bytes"] == 331_780_864
    assert not traffic["measured_hardware_traffic"]
    assert traffic["headroom_bytes_to_45_percent"] == 39_393_536
    assert len(OFFICIAL_NATIVE_BITNET_WEIGHT_SHA256) == 64


def test_local_safetensors_source_repack_uses_separate_adapter(tmp_path):
    model = tmp_path / "model"
    layers = _layers()
    _write_packed_source(model, layers)
    artifact = tmp_path / "native.bin"

    report = repack_native_bitnet_model(
        model,
        artifact,
        verify_official_weight_hash=False,
    )
    loaded = load_native_bitnet_artifact(artifact)
    decoded = decode_native_bitnet_layer(loaded, 0)

    assert report["source_track"] == "low_bit_native"
    assert report["dense_llama_conversion_status"] == "not_applicable"
    assert report["combined_gate_status"] == "not_yet_evaluated"
    assert report["artifact"]["reloaded"]
    assert report["provenance_verification"] == {
        "official_revision_pinned": False,
        "weight_hash_verification_requested": False,
        "official_weight_hash_verified": False,
        "source_stability_rechecked": False,
    }
    assert report["weight_hashes"] == {}
    assert report["representation_checks"]["logical_reconstruction"]["passed"]
    assert (
        report["representation_checks"]["logical_reconstruction"][
            "logical_source_sha256"
        ]
        == report["representation_checks"]["logical_reconstruction"][
            "logical_artifact_sha256"
        ]
    )
    saved_report = json.loads(Path(report["report_path"]).read_text(encoding="utf-8"))
    assert saved_report["report_path"] == report["report_path"]
    np.testing.assert_array_equal(decoded["gate_codes"], layers[0].gate_codes)


def test_loader_rejects_oversized_header_dimension_before_derived_work(tmp_path):
    path = tmp_path / "native.bin"
    save_native_bitnet_artifact(path, _layers(count=1), rms_norm_eps=1e-5)
    payload = bytearray(path.read_bytes())
    struct.pack_into("<I", payload, 20, (1 << 20) + 4)
    path.write_bytes(payload)

    with pytest.raises(
        NativeBitNetValidationError,
        match="intermediate_size exceeds",
    ):
        load_native_bitnet_artifact(path)


def test_traffic_rejects_layer_payload_that_exceeds_header_width():
    with pytest.raises(
        NativeBitNetValidationError,
        match="not representable",
    ):
        native_bitnet_repack_traffic(
            1 << 20,
            1 << 20,
            layer_count=1,
        )


def test_parity_rejects_artifact_hash_mismatch_before_model_loading(tmp_path):
    artifact = tmp_path / "native.bin"
    save_native_bitnet_artifact(
        artifact,
        _layers(count=1),
        rms_norm_eps=1e-5,
    )

    with pytest.raises(
        NativeBitNetValidationError,
        match="artifact SHA-256 mismatch",
    ):
        evaluate_native_bitnet_parity(
            tmp_path / "missing-model",
            artifact,
            out=tmp_path / "parity.json",
            expected_artifact_sha256="0" * 64,
        )


def test_failed_reconstruction_does_not_publish_over_destination(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "model"
    _write_packed_source(model, _layers())
    artifact = tmp_path / "native.bin"
    artifact.write_bytes(b"previous verified artifact")

    def fail_reconstruction(*_args, **_kwargs):
        raise NativeBitNetValidationError("forced reconstruction failure")

    monkeypatch.setattr(
        "engram.models.native_bitnet._verify_repacked_layers",
        fail_reconstruction,
    )

    with pytest.raises(
        NativeBitNetValidationError,
        match="forced reconstruction failure",
    ):
        repack_native_bitnet_model(
            model,
            artifact,
            verify_official_weight_hash=False,
        )

    assert artifact.read_bytes() == b"previous verified artifact"
    assert not list(tmp_path.glob(".native.bin.verify-*"))


def test_artifact_torch_oracle_matches_transformers_bitnet_mlp(
    tmp_path,
):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as transformers_imports

    if transformers_imports.is_sklearn_available():
        try:
            import sklearn  # noqa: F401
        except Exception:
            transformers_imports.is_sklearn_available = lambda: False
            transformers_utils.is_sklearn_available = lambda: False
    from transformers import BitNetConfig
    from transformers.integrations.bitnet import AutoBitLinear
    from transformers.models.bitnet.modeling_bitnet import BitNetMLP

    from engram.evaluation.native_bitnet_parity import (
        _artifact_mlp_class,
        _disable_bitnet_torch_compile,
    )

    _disable_bitnet_torch_compile()
    layer = _layers(count=1)[0]
    path = tmp_path / "native.bin"
    save_native_bitnet_artifact(path, [layer], rms_norm_eps=1e-5)
    artifact = load_native_bitnet_artifact(path)
    config = BitNetConfig(
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        hidden_act="relu2",
        rms_norm_eps=1e-5,
    )
    reference = BitNetMLP(config).to(torch.bfloat16)
    reference.gate_proj = AutoBitLinear(
        8,
        12,
        bias=False,
        dtype=torch.bfloat16,
        online_quant=False,
    ).to(torch.bfloat16)
    reference.up_proj = AutoBitLinear(
        8,
        12,
        bias=False,
        dtype=torch.bfloat16,
        online_quant=False,
    ).to(torch.bfloat16)
    reference.down_proj = AutoBitLinear(
        12,
        8,
        bias=False,
        dtype=torch.bfloat16,
        online_quant=False,
    ).to(torch.bfloat16)
    for module, codes, scale in (
        (reference.gate_proj, layer.gate_codes, layer.gate_scale),
        (reference.up_proj, layer.up_codes, layer.up_scale),
        (reference.down_proj, layer.down_codes, layer.down_scale),
    ):
        module.weight.data.copy_(torch.as_tensor(codes, dtype=torch.bfloat16))
        module.weight_scale.data.fill_(scale)
    reference.ffn_sub_norm.weight.data.copy_(
        torch.as_tensor(layer.ffn_sub_norm, dtype=torch.bfloat16)
    )
    candidate = _artifact_mlp_class()(artifact, 0)
    states = torch.tensor(
        [
            [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4, -0.1, 0.05]],
            [[-0.2, 0.3, 0.1, -0.4, 0.7, -0.6, 0.5, 0.2]],
        ],
        dtype=torch.bfloat16,
    )

    with torch.inference_mode():
        expected = reference(states)
        actual = candidate(states)

    assert torch.equal(expected, actual)


def test_direct_cpu_phase_stream_kernel_matches_dense_artifact_oracle(tmp_path):
    torch = pytest.importorskip("torch")
    library = Path("build/libengram_bitnet.so").resolve()
    if not library.is_file():
        pytest.skip("native BitNet shared library has not been built")
    from engram.evaluation.native_bitnet_parity import _artifact_mlp_class

    artifact_path = tmp_path / "native.bin"
    save_native_bitnet_artifact(
        artifact_path,
        _layers(count=1),
        rms_norm_eps=1e-5,
    )
    artifact = load_native_bitnet_artifact(artifact_path)
    reference = _artifact_mlp_class()(artifact, 0)
    states = torch.tensor(
        [
            [[0.25, -0.5, 0.75, 0.1, -0.2, 0.4, -0.1, 0.05]],
            [[-0.2, 0.3, 0.1, -0.4, 0.7, -0.6, 0.5, 0.2]],
        ],
        dtype=torch.bfloat16,
    )

    with NativeBitNetCPUKernel(
        artifact_path,
        threads=2,
        library=library,
        expected_sha256=artifact.payload_sha256,
    ) as kernel:
        with torch.inference_mode():
            expected = reference(states)
            actual = kernel.forward(0, states)

        assert torch.equal(expected, actual)
        assert (
            kernel.calls[0]["scheduled_cache_line_bytes"]
            == (artifact.layer_block_bytes[0])
        )
        assert kernel.calls[0]["rows"] == 2
