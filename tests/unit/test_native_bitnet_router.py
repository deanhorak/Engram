import pytest

from engram.evaluation.native_bitnet_router import native_bitnet_router_traffic


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
