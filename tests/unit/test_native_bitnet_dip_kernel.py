import json
import shutil

import numpy as np
import pytest

from engram.evaluation.native_bitnet_dip_kernel import (
    NativeBitNetDIPCPUKernel,
    NativeBitNetDIPKernelError,
    build_native_bitnet_dip_kernel_mlp,
    substitute_native_bitnet_dip_kernel_mlps,
)
from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    decode_native_bitnet_layer,
    load_native_bitnet_artifact,
    save_native_bitnet_artifact,
)
from engram.semantic.native_bitnet_dip import NativeBitNetDIPLayer
from engram.semantic.native_bitnet_dip_index import (
    build_native_bitnet_dip_index,
)


def _bf16_round(values):
    source = np.asarray(values, dtype=np.float32)
    bits = source.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + (
        (bits >> np.uint32(16)) & np.uint32(1)
    )
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _fixture(tmp_path, *, perturb=0):
    hidden = 320
    intermediate = 20
    rng = np.random.default_rng(1701 + perturb)
    layers = []
    for layer in range(2):
        layers.append(
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
                gate_scale=0.03125,
                up_scale=0.015625,
                down_scale=0.0078125,
                ffn_sub_norm=_bf16_round(
                    np.linspace(
                        -1.125, 1.375, intermediate, dtype=np.float32
                    )
                ),
            )
        )
    artifact_path = tmp_path / f"model-{perturb}.bitnet-records.bin"
    save_native_bitnet_artifact(
        artifact_path,
        layers,
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
            "candidate_counts": [14, 15],
            "maximum_ks": [10, 10],
            "rms_policies": [
                {
                    "estimator": "candidate_ratio",
                    "audit_count": 0,
                    "audit_strategy": "none",
                },
                {
                    "estimator": "corrected_proxy",
                    "audit_count": 3,
                    "audit_strategy": "top_proxy_raw_square",
                },
            ],
        },
        "progression_screen": {"passed": True},
    }
    policy_path = tmp_path / f"policy-{perturb}.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    index_path = build_native_bitnet_dip_index(
        artifact_path,
        policy_path,
        tmp_path / f"model-{perturb}.bitnet-dip-index.bin",
    )
    return artifact_path, index_path, artifact


def _states():
    values = np.linspace(-1.0, 1.0, 3 * 320, dtype=np.float32).reshape(3, 320)
    # Exercise values immediately around BF16 ties and the Q8 zero floor.
    values[0, :8] = np.asarray(
        [
            1.00390625,
            1.003906369,
            -1.00390625,
            -1.003906369,
            1.0e-7,
            -1.0e-7,
            0.0,
            -0.0,
        ],
        dtype=np.float32,
    )
    return values


def _scalar_teacher_ids(
    artifact,
    layer,
    states,
    top_k,
    *,
    return_positive_counts=False,
):
    """Canonical scalar native-BF16 teacher utility reference."""

    decoded = decode_native_bitnet_layer(artifact, layer)
    gate_codes = decoded["gate_codes"]
    up_codes = decoded["up_codes"]
    down_codes = decoded["down_codes"]
    gain = decoded["ffn_sub_norm"]
    gate_scale = np.float32(decoded["gate_scale"])
    up_scale = np.float32(decoded["up_scale"])
    width = artifact.intermediate_size
    result = []
    positive_counts = []
    for source in _bf16_round(states):
        maximum = np.float32(np.max(np.abs(source)))
        scale = np.float32(127.0) / np.maximum(maximum, np.float32(1e-5))
        state = _bf16_round(
            np.rint(source * scale)
            .clip(np.float32(-128.0), np.float32(127.0))
            .astype(np.float32)
            / scale
        )
        raw = np.empty(width, dtype=np.float32)
        for record in range(width):
            gate_accumulator = np.float32(0.0)
            up_accumulator = np.float32(0.0)
            for coordinate in range(artifact.hidden_size):
                gate_accumulator = np.float32(
                    gate_accumulator
                    + np.float32(gate_codes[record, coordinate]) * state[coordinate]
                )
                up_accumulator = np.float32(
                    up_accumulator
                    + np.float32(up_codes[record, coordinate]) * state[coordinate]
                )
            gate = _bf16_round(
                _bf16_round(gate_accumulator) * gate_scale
            )
            up = _bf16_round(_bf16_round(up_accumulator) * up_scale)
            positive = np.maximum(gate, np.float32(0.0))
            raw[record] = _bf16_round(
                _bf16_round(positive * positive) * up
            )
        square_sum = np.float32(0.0)
        for value in raw:
            square_sum = np.float32(square_sum + np.float32(value * value))
        inverse_rms = np.float32(
            1.0
            / np.sqrt(
                np.float32(square_sum / np.float32(width))
                + np.float32(artifact.rms_norm_eps)
            )
        )
        normalized = _bf16_round(_bf16_round(raw * inverse_rms) * gain)
        normalized_max = np.float32(np.max(np.abs(normalized)))
        coefficient_scale = np.float32(127.0) / np.maximum(
            normalized_max,
            np.float32(1e-5),
        )
        coefficients = _bf16_round(
            np.rint(normalized * coefficient_scale)
            .clip(np.float32(-128.0), np.float32(127.0))
            .astype(np.float32)
            / coefficient_scale
        )
        down_norm = np.count_nonzero(down_codes, axis=0).astype(np.float32)
        utility = coefficients * coefficients * down_norm
        positive_counts.append(int(np.count_nonzero(utility > 0.0)))
        result.append(
            np.argsort(-utility, kind="stable")[:top_k].astype(np.uint32)
        )
    ids = np.stack(result)
    if return_positive_counts:
        return ids, np.asarray(positive_counts, dtype=np.uint32)
    return ids


@pytest.mark.parametrize(
    ("layer", "candidate_count", "audit_count", "estimator", "strategy"),
    [
        (0, 14, 0, "candidate_ratio", "none"),
        (
            1,
            15,
            3,
            "corrected_proxy",
            "top_proxy_raw_square",
        ),
    ],
)
def test_native_dip_kernel_matches_frozen_python_hybrid_bit_exact(
    tmp_path,
    layer,
    candidate_count,
    audit_count,
    estimator,
    strategy,
):
    artifact_path, index_path, artifact = _fixture(tmp_path)
    states = _states()
    python_layer = NativeBitNetDIPLayer(
        artifact,
        layer,
        input_fraction=0.75,
        candidate_count=candidate_count,
        top_k=10,
        rms_audit_count=audit_count,
        energy_target=1.0,
        minimum_top_k=2,
        maximum_top_k=10,
        rms_estimator=estimator,
        rms_audit_strategy=(
            strategy if strategy != "none" else "hashed_tail"
        ),
    )
    expected = python_layer(_bf16_round(states))

    with NativeBitNetDIPCPUKernel(
        artifact_path,
        index_path,
        threads=2,
    ) as kernel:
        actual = kernel.forward_debug(layer, states)
        teacher_ids = kernel.teacher_top_k(layer, states, top_k=10)
        teacher_ids_with_counts, positive_counts = (
            kernel.teacher_top_k_with_positive_counts_bf16_bits(
                layer,
                (
                    _bf16_round(states).view(np.uint32)
                    >> np.uint32(16)
                ).astype(np.uint16),
                top_k=10,
            )
        )
        policy = kernel.policies[layer]
        assert policy.candidate_count == candidate_count
        assert policy.rms_audit_count == audit_count
        assert policy.rms_estimator == estimator
        assert policy.rms_audit_strategy == strategy
        # Canonical teacher ties are broken by ascending record ID.
        zero_state_ids = kernel.teacher_top_k(
            layer,
            np.zeros((1, 320), dtype=np.float32),
            top_k=10,
        )
        zero_route = kernel.forward_debug(
            layer,
            np.zeros((1, 320), dtype=np.float32),
        )

    np.testing.assert_array_equal(
        actual.output_bf16_bits,
        (
            _bf16_round(expected.output).view(np.uint32)
            >> np.uint32(16)
        ).astype(np.uint16),
    )
    np.testing.assert_array_equal(actual.output, _bf16_round(expected.output))
    np.testing.assert_array_equal(
        actual.selected_counts,
        expected.selected_counts.astype(np.uint32),
    )
    assert actual.input_coordinate_ids is not None
    assert actual.candidate_ids is not None
    assert actual.selected_record_ids is not None
    np.testing.assert_array_equal(
        actual.input_coordinate_ids,
        expected.input_indices.astype(np.uint32),
    )
    np.testing.assert_array_equal(
        actual.candidate_ids,
        expected.candidate_indices.astype(np.uint32),
    )
    expected_selected = expected.selected_indices.copy()
    expected_selected[expected_selected < 0] = np.iinfo(np.uint32).max
    np.testing.assert_array_equal(
        actual.selected_record_ids,
        expected_selected.astype(np.uint32),
    )
    assert teacher_ids.shape == (states.shape[0], 10)
    assert np.all(teacher_ids < artifact.intermediate_size)
    np.testing.assert_array_equal(
        teacher_ids,
        _scalar_teacher_ids(artifact, layer, states, 10),
    )
    expected_teacher_ids, expected_positive_counts = _scalar_teacher_ids(
        artifact,
        layer,
        states,
        10,
        return_positive_counts=True,
    )
    np.testing.assert_array_equal(
        teacher_ids_with_counts,
        expected_teacher_ids,
    )
    np.testing.assert_array_equal(
        positive_counts,
        expected_positive_counts,
    )
    np.testing.assert_array_equal(
        zero_state_ids,
        np.arange(10, dtype=np.uint32)[None, :],
    )
    np.testing.assert_array_equal(
        zero_route.input_coordinate_ids,
        np.arange(240, dtype=np.uint32)[None, :],
    )
    np.testing.assert_array_equal(
        zero_route.candidate_ids,
        np.arange(candidate_count, dtype=np.uint32)[None, :],
    )
    assert zero_route.selected_counts.tolist() == [2]
    expected_zero_selected = np.full(
        (1, 10),
        np.iinfo(np.uint32).max,
        dtype=np.uint32,
    )
    expected_zero_selected[0, :2] = [0, 1]
    np.testing.assert_array_equal(
        zero_route.selected_record_ids,
        expected_zero_selected,
    )
    assert actual.metrics["candidate_count"] == candidate_count
    assert actual.metrics["selected_count_total"] == int(
        np.sum(actual.selected_counts)
    )
    assert actual.metrics["selected_count_min"] == int(
        np.min(actual.selected_counts)
    )
    assert actual.metrics["selected_count_max"] == int(
        np.max(actual.selected_counts)
    )
    expected_bytes = (
        2 * 240 * 64 * 3
        + 2 * candidate_count * 64 * 3
        + 64 * 3
        + 64 * 3
        + int(np.sum(actual.selected_counts)) * 64
        + 256 * 3
    )
    assert actual.metrics["scheduled_cache_line_bytes"] == expected_bytes
    assert actual.metrics["elapsed_ns"] > 0


def test_native_dip_kernel_reloads_and_source_binding_fails_closed(tmp_path):
    artifact_path, index_path, _ = _fixture(tmp_path / "first")
    for _ in range(2):
        with NativeBitNetDIPCPUKernel(
            artifact_path,
            index_path,
            threads=1,
        ) as kernel:
            assert kernel.layer_count == 2
            assert kernel.hidden_size == 320
            assert kernel.forward(0, _states()[:1]).output.shape == (1, 320)

    other_artifact, _, _ = _fixture(tmp_path / "second", perturb=1)
    with pytest.raises(
        NativeBitNetDIPKernelError,
        match="source-artifact SHA-256 mismatch",
    ):
        NativeBitNetDIPCPUKernel(other_artifact, index_path, threads=1)

    corrupted = tmp_path / "corrupted-index.bin"
    shutil.copyfile(index_path, corrupted)
    with corrupted.open("r+b") as handle:
        handle.seek(128 + 64 + 128)
        value = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([value[0] ^ 1]))
    with pytest.raises(
        NativeBitNetDIPKernelError,
        match="checksum mismatch",
    ):
        NativeBitNetDIPCPUKernel(artifact_path, corrupted, threads=1)


def test_native_dip_kernel_rejects_nonfinite_input(tmp_path):
    artifact_path, index_path, _ = _fixture(tmp_path)
    values = _states()[:1]
    values[0, 0] = np.nan
    with NativeBitNetDIPCPUKernel(
        artifact_path,
        index_path,
        threads=1,
    ) as kernel:
        with pytest.raises(
            NativeBitNetDIPKernelError,
            match="finite",
        ):
            kernel.forward(0, values)


def test_native_dip_torch_module_is_direct_inference_only_and_restores(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    artifact_path, index_path, _ = _fixture(tmp_path)
    hidden = torch.from_numpy(_states()[:2]).to(torch.bfloat16)

    class DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Identity()

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer(), DecoderLayer()])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Backbone()

    model = Model()
    originals = [layer.mlp for layer in model.model.layers]
    with NativeBitNetDIPCPUKernel(
        artifact_path,
        index_path,
        threads=2,
    ) as kernel:
        module = build_native_bitnet_dip_kernel_mlp(
            kernel,
            0,
            debug_routes=True,
        )
        with torch.inference_mode():
            output = module(hidden)
        reference = kernel.forward(0, hidden.float().numpy())
        np.testing.assert_array_equal(
            output.view(torch.uint16).numpy(),
            reference.output_bf16_bits,
        )
        assert module.last_result is not None
        assert module.last_result.candidate_ids is not None
        assert module.last_result.selected_counts.shape == (2,)

        with substitute_native_bitnet_dip_kernel_mlps(
            model,
            kernel,
            debug_routes=False,
        ) as replacements:
            assert set(replacements) == {0, 1}
            with torch.inference_mode():
                assert model.model.layers[1].mlp(hidden).shape == hidden.shape

    assert [layer.mlp for layer in model.model.layers] == originals
