import pytest

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)


def test_official_native_bitnet_dip_layout_is_cache_line_honest():
    report = native_bitnet_dip_physical_accounting(
        2560,
        6912,
        input_counts=[1920],
        candidate_counts=[2592],
        top_ks=[1728],
    )

    assert report["format"] == "native_bitnet_dip_dual_layout_v2"
    layout = report["layout"]
    serialization = report["serialization"]
    traffic = report["traffic"]
    layer = traffic["layers"][0]
    assert layout["base_record_payload_bytes"] == 512
    assert layout["base_record_cache_lines"] == 8
    assert layout["coordinate_payload_bytes"] == 1383
    assert layout["coordinate_stride_bytes"] == 1408
    assert layout["coordinate_cache_lines"] == 22
    assert layout["coordinate_row_padding_bytes"] == 25
    assert layout["index_layer_block_bytes"] == 7_222_912
    assert serialization["coordinate_index_bytes"] == 7_223_104
    assert serialization["combined_serialized_bytes"] == 17_854_016
    assert layer["partial_coordinate_scan_bytes"] == 5_406_720
    assert layer["candidate_completion_record_bytes"] == 2_654_208
    assert layer["selected_down_record_bytes"] == 884_736
    assert layer["gain_scan_bytes"] == 13_824
    assert layer["down_norm_scan_bytes"] == 13_824
    assert layer["complete_modelled_cold_bytes"] == 8_973_568
    assert traffic["global_header_directory_bytes"] == 320
    assert traffic["complete_modelled_cold_bytes"] == 8_973_888
    assert traffic["passes_45_percent"]


def test_native_bitnet_dip_adaptive_schedule_accounts_headers_once():
    report = native_bitnet_dip_physical_accounting(
        2560,
        6912,
        input_counts=[1920, 1920],
        candidate_counts=[1556, 3630],
        top_ks=[1037, 2074],
    )

    traffic = report["traffic"]
    serialization = report["serialization"]
    assert len(traffic["layers"]) == 2
    assert traffic["base_global_header_directory_bytes"] == 128
    assert traffic["index_global_header_directory_bytes"] == 192
    assert serialization["coordinate_index_bytes"] == (192 + 2 * 7_222_912)
    assert traffic["complete_modelled_cold_bytes"] == (
        320 + sum(layer["complete_modelled_cold_bytes"] for layer in traffic["layers"])
    )
    assert traffic["passes_45_percent"]


def test_full_coordinate_scan_does_not_charge_unneeded_completion():
    report = native_bitnet_dip_physical_accounting(
        2560,
        6912,
        input_counts=[2560],
        candidate_counts=[2592],
        top_ks=[1728],
    )

    layer = report["traffic"]["layers"][0]
    assert layer["candidate_completion_record_bytes"] == 0
    assert layer["gate_up_trits_duplicated_between_partial_and_completion"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "input_counts": [1920],
                "candidate_counts": [1000],
                "top_ks": [1001],
            },
            "must not exceed",
        ),
        (
            {
                "input_counts": [1920, 1920],
                "candidate_counts": [1000],
                "top_ks": [900],
            },
            "equal lengths",
        ),
    ],
)
def test_native_bitnet_dip_accounting_rejects_invalid_schedules(kwargs, message):
    with pytest.raises(ValueError, match=message):
        native_bitnet_dip_physical_accounting(2560, 6912, **kwargs)


def test_native_bitnet_dip_accounting_fails_closed_for_unaligned_records():
    with pytest.raises(ValueError, match="record width"):
        native_bitnet_dip_physical_accounting(
            644,
            1024,
            input_counts=[320],
            candidate_counts=[512],
            top_ks=[256],
        )
