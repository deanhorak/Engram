from engram.evaluation.olmoe_native_local_fp16 import (
    _compressed_attention_expectations,
)


_MODEL = {
    "layers": 16,
    "query_heads": 16,
    "key_value_heads": 16,
    "head_dimension": 128,
}


def test_int8_w128_expectations_match_frozen_gate():
    values = _compressed_attention_expectations(_MODEL, "int8", 128)
    assert values["attention_state_bytes"] == 10_921_984
    assert values["attention_logical_read_bytes"] == 541_065_216
    assert values["attention_logical_read_fraction"] == 0.25
    assert values["compression"] == "local_kv_symmetric_int8"


def test_fp16_w56_expectations_remain_distinct_from_int8():
    values = _compressed_attention_expectations(_MODEL, "fp16", 56)
    assert values["attention_state_bytes"] == 9_528_320
    assert values["attention_logical_read_bytes"] == 846_462_976
    assert values["attention_logical_read_fraction"] < 0.40
