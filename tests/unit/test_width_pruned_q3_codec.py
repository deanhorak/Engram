from pathlib import Path

import numpy as np
import pytest

from engram.training.width_pruned_q3_codec import (
    PackedQ3Rows,
    WidthPrunedQ3LayerWeights,
    _decode_signed_q3_rows,
    _pack_signed_q3_rows,
    decode_width_pruned_q3_artifact,
    load_width_pruned_q3_artifact,
    save_width_pruned_q3_artifact,
    width_pruned_q3_dynamic_traffic,
    width_pruned_q3_forward,
    width_pruned_q3_traffic,
)


def _fixture(layer_count: int = 3):
    rng = np.random.default_rng(3031)
    hidden = 7
    source_intermediate = 13
    widths = [3 + index % 3 for index in range(layer_count)]
    layers = []
    for index, width in enumerate(widths):
        layers.append(
            WidthPrunedQ3LayerWeights(
                rng.normal(scale=0.3, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.3, size=(width, hidden)).astype(np.float32),
                rng.normal(scale=0.3, size=(hidden, width)).astype(np.float32),
                np.arange(index, index + width, dtype=np.int64) % source_intermediate,
            )
        )
    return layers, hidden, source_intermediate, widths


def _save_fixture(path: Path, layer_count: int = 3):
    layers, hidden, source_intermediate, widths = _fixture(layer_count)
    save_width_pruned_q3_artifact(
        path, layers, source_intermediate_size=source_intermediate
    )
    return (
        load_width_pruned_q3_artifact(path),
        hidden,
        source_intermediate,
        widths,
    )


def test_width_pruned_q3_round_trip_and_exact_file_traffic(tmp_path):
    path = tmp_path / "model.width-q3.bin"
    loaded, hidden, source_intermediate, widths = _save_fixture(path, layer_count=30)
    traffic = width_pruned_q3_traffic(hidden, source_intermediate, widths)
    decoded = decode_width_pruned_q3_artifact(loaded)

    assert path.stat().st_size == traffic["serialized_artifact_bytes"]
    assert traffic["traffic_numerator_bytes"] == path.stat().st_size
    assert traffic["total_cold_bytes"] == path.stat().st_size
    assert (
        traffic["dense_q4_source_mlp_bytes"]
        == (len(widths) * 3 * hidden * source_intermediate + 1) // 2
    )
    assert traffic["bits_per_weight"] == 3
    assert traffic["row_scale_codec"] == "fp16"
    assert loaded.serialized_artifact_bytes == path.stat().st_size
    assert len(decoded) == 30
    assert all(offset % 64 == 0 for offset in loaded.layer_offsets)
    assert all(block % 64 == 0 for block in loaded.layer_block_bytes)
    for layer, width in zip(decoded, widths, strict=True):
        assert layer["gate"].shape == (width, hidden)
        assert layer["up"].shape == (width, hidden)
        assert layer["down"].shape == (hidden, width)


def test_reloaded_width_pruned_q3_forward_matches_reference_decode(tmp_path):
    loaded, _, _, _ = _save_fixture(tmp_path / "model.width-q3.bin")
    decoded = decode_width_pruned_q3_artifact(loaded)
    states = np.asarray(
        [
            [0.25, -0.5, 0.75, 0.1, -0.2, 0.4, 0.3],
            [-0.3, 0.2, 0.4, -0.8, 0.6, 0.1, -0.2],
        ],
        dtype=np.float32,
    )

    actual = width_pruned_q3_forward(loaded, 1, states)
    gate = states @ decoded[1]["gate"].T
    expected = (gate / (1.0 + np.exp(-gate))) * (states @ decoded[1]["up"].T)
    expected = expected @ decoded[1]["down"].T

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_q3_decoder_rejects_forbidden_code_and_nonzero_row_padding():
    # Seven values occupy 21 bits, leaving three high padding bits in byte 3.
    codes = np.zeros(3, dtype=np.uint8)
    scales = np.ones(1, dtype=np.float16)
    forbidden = PackedQ3Rows(codes.copy(), scales, 1, 7)
    forbidden.packed[0] = 4
    with pytest.raises(ValueError, match="forbidden -4"):
        _decode_signed_q3_rows(forbidden)

    padded = PackedQ3Rows(codes.copy(), scales, 1, 7)
    padded.packed[-1] = np.uint8(0x80)
    with pytest.raises(ValueError, match="padding bits"):
        _decode_signed_q3_rows(padded)


def test_q3_known_signed_codes_round_trip_across_byte_boundaries():
    values = np.asarray([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    packed = _pack_signed_q3_rows(values)

    assert packed.bytes_per_row == 3
    np.testing.assert_array_equal(_decode_signed_q3_rows(packed), values)


def test_width_pruned_q3_loader_rejects_corrupt_packed_code(tmp_path):
    path = tmp_path / "model.width-q3.bin"
    loaded, _, _, _ = _save_fixture(path)
    payload = bytearray(path.read_bytes())
    # The first code byte follows the 64-byte local layer header. Set its first
    # three-bit code to the forbidden two's-complement -4 pattern.
    code_offset = loaded.layer_offsets[0] + 64
    payload[code_offset] = (payload[code_offset] & 0xF8) | 4
    corrupt = tmp_path / "corrupt.width-q3.bin"
    corrupt.write_bytes(payload)

    with pytest.raises(ValueError, match="forbidden -4"):
        load_width_pruned_q3_artifact(corrupt)


def test_smollm2_q3_width_768_and_maximum_45_percent_width():
    width_768 = width_pruned_q3_traffic(576, 1536, [768] * 30)
    passing = [
        width
        for width in range(1, 1537)
        if width_pruned_q3_traffic(576, 1536, [width] * 30)[
            "passes_45_percent_traffic_gate"
        ]
    ]
    maximum = max(passing)

    assert width_768["passes_45_percent_traffic_gate"]
    assert maximum > 768
    assert width_pruned_q3_traffic(576, 1536, [maximum] * 30)[
        "passes_45_percent_traffic_gate"
    ]
    assert not width_pruned_q3_traffic(576, 1536, [maximum + 1] * 30)[
        "passes_45_percent_traffic_gate"
    ]


def test_dynamic_q3_768_reports_code_budget_and_caller_router_bytes():
    without_router = width_pruned_q3_dynamic_traffic(576, 1536, 768, layer_count=30)
    with_router = width_pruned_q3_dynamic_traffic(
        576,
        1536,
        768,
        layer_count=30,
        router_bytes_per_token=64,
    )

    assert without_router["code_only_fraction_of_dense_q4"] == 0.375
    assert without_router["fraction_of_dense_q4"] > 0.375
    assert without_router["passes_45_percent_traffic_gate"]
    assert with_router["router_cache_aligned_bytes"] == 64
    assert with_router["total_cold_bytes"] == without_router["total_cold_bytes"] + 64
