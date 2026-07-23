import json

from engram.models.fixture import create_tiny_fixture
from engram.tracing import (
    TraceReader,
    capture_teacher_traces,
    plan_teacher_trace_capture,
)


def test_fixture_trace_captures_only_selected_layer_boundaries(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=3,
        num_heads=2,
    )
    traces = tmp_path / "traces"

    capture_teacher_traces(
        model,
        traces,
        samples=4,
        include_attention=False,
        layers=[1],
    )

    reader = TraceReader(traces)
    shard = next(reader.iter_shards())
    assert reader.manifest["metadata"]["selected_layers"] == [1]
    assert reader.manifest["metadata"]["all_layers_captured"] is False
    assert shard["layer_1_mlp_input"].shape == (4, 8)
    assert shard["layer_1_mlp_output"].shape == (4, 8)
    assert not any(name.startswith("layer_0_") for name in shard)
    assert not any(name.startswith("layer_2_") for name in shard)


def test_trace_dry_run_counts_selected_layer_payload_without_capture(tmp_path):
    model = create_tiny_fixture(
        tmp_path / "model",
        hidden_size=8,
        intermediate_size=12,
        num_layers=3,
        num_heads=2,
    )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        "\n".join(
            (
                json.dumps({"input_ids": [1, 2, 3]}),
                json.dumps({"input_ids": [4, 5]}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    plan = plan_teacher_trace_capture(
        model,
        dataset,
        samples=2,
        include_attention=False,
        layers=[1],
    )

    assert plan["captured_sequences"] == 2
    assert plan["source_token_positions"] == 5
    assert plan["captured_token_positions"] == 5
    assert plan["selected_layers"] == [1]
    assert plan["included_boundaries"] == ["mlp"]
    assert plan["estimated_boundary_tensor_bytes"] == 5 * 2 * 8 * 4
    assert plan["estimated_token_metadata_bytes"] == 5 * (128 + 3 * 8)
    assert plan["estimated_npy_payload_bytes"] == 1080
    assert not (tmp_path / "traces").exists()
