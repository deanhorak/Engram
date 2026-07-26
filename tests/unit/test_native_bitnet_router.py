import json

import numpy as np
import pytest

from engram.evaluation.native_bitnet_router import (
    analyze_native_bitnet_dip_layer,
    load_native_bitnet_adaptive_schedule,
    maximum_native_bitnet_dip_candidates,
    native_bitnet_dip_traffic,
    native_bitnet_router_traffic,
)
from engram.models.native_bitnet import (
    NativeBitNetLayerWeights,
    load_native_bitnet_artifact,
    save_native_bitnet_artifact,
)


def test_native_bitnet_router_traffic_counts_complete_candidate_records():
    report = native_bitnet_router_traffic(
        2560,
        6912,
        rank=128,
        candidate_count=2160,
    )

    assert report["router_parameters"] == 1_219_328
    assert report["candidate_record_bytes"] == 2160 * 1538
    assert report["complete_modelled_bytes"] == (
        report["router_bytes"] + report["candidate_record_bytes"]
    )
    assert report["passes_45_percent"]


def test_native_bitnet_router_traffic_rejects_invalid_candidate_count():
    with pytest.raises(ValueError, match="exceeds"):
        native_bitnet_router_traffic(8, 12, rank=2, candidate_count=13)


@pytest.mark.parametrize(
    ("top_k", "expected"),
    [
        (1037, 5838),
        (1383, 5665),
        (1728, 5492),
        (2074, 5319),
        (2420, 5146),
    ],
)
def test_native_bitnet_dip_candidate_cap_stays_within_traffic_limit(
    top_k,
    expected,
):
    maximum = maximum_native_bitnet_dip_candidates(
        2560,
        6912,
        input_fraction=0.75,
        top_k=top_k,
        maximum_traffic_fraction=0.45,
    )
    at_limit = native_bitnet_dip_traffic(
        2560,
        6912,
        input_fraction=0.75,
        candidate_count=maximum,
        top_k=top_k,
    )
    above_limit = native_bitnet_dip_traffic(
        2560,
        6912,
        input_fraction=0.75,
        candidate_count=maximum + 1,
        top_k=top_k,
    )

    assert maximum == expected
    assert at_limit["passes_traffic_limit"]
    assert not above_limit["passes_traffic_limit"]


def _zero_tie_artifact(tmp_path):
    hidden = 320
    width = 320
    layer = NativeBitNetLayerWeights(
        gate_codes=-np.ones((width, hidden), dtype=np.int8),
        up_codes=np.ones((width, hidden), dtype=np.int8),
        down_codes=np.eye(hidden, width, dtype=np.int8),
        gate_scale=0.5,
        up_scale=0.5,
        down_scale=0.5,
        ffn_sub_norm=np.ones(width, dtype=np.float32),
    )
    path = tmp_path / "router.bitnet-records.bin"
    save_native_bitnet_artifact(path, [layer], rms_norm_eps=1e-5)
    return load_native_bitnet_artifact(path)


def test_native_bitnet_dip_layer_uses_stable_zero_tie_membership(tmp_path):
    artifact = _zero_tie_artifact(tmp_path)
    kwargs = {
        "top_k": 2,
        "input_fraction": 0.5,
        "candidate_multipliers": (1.0, 2.0),
        "maximum_traffic_fraction": 1.0,
    }
    first = analyze_native_bitnet_dip_layer(
        artifact,
        0,
        np.ones((3, 320), dtype=np.float32),
        **kwargs,
    )
    second = analyze_native_bitnet_dip_layer(
        artifact,
        0,
        np.ones((3, 320), dtype=np.float32),
        **kwargs,
    )

    assert first == second
    assert first["stable_tie_break"].endswith("ascending_source_index")
    assert first["arms"][0]["candidate_count"] == 2
    assert first["arms"][0]["candidate_recall"]["mean"] == 1.0
    assert first["passes_recall_and_traffic"]


def _schedule_payload(*, sequences=8, positions=256):
    return {
        "experiment": "native_bitnet_oracle_causal_substitution",
        "scope": "all_mlp_layers_exact_membership_ceiling",
        "quality_passed": True,
        "configuration": {
            "requested_fraction": None,
            "layer_top_ks": [1, 1],
            "intermediate_size": 4,
            "sequence_count": sequences,
            "prediction_positions": positions,
        },
    }


def test_native_bitnet_adaptive_schedule_requires_frozen_protocol(tmp_path):
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(_schedule_payload()), encoding="utf-8")

    schedule = load_native_bitnet_adaptive_schedule(
        path,
        layer_count=2,
        intermediate_size=4,
    )

    assert schedule["layer_top_ks"] == [1, 1]
    assert schedule["mean_active_fraction"] == 0.25
    path.write_text(
        json.dumps(_schedule_payload(sequences=2, positions=32)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen"):
        load_native_bitnet_adaptive_schedule(
            path,
            layer_count=2,
            intermediate_size=4,
        )
