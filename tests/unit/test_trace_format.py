from pathlib import Path

import numpy as np
import pytest

from engram.tracing.format import TraceFormatError, TraceReader, TraceWriter


def test_trace_round_trip_and_corruption_detection(tmp_path):
    trace = tmp_path / "trace"
    with TraceWriter(
        trace,
        model_hash="model",
        dataset_hash="dataset",
        split="calibration",
        seed=3,
    ) as writer:
        writer.append({"hidden": np.arange(12, dtype=np.float32).reshape(3, 4), "kind": np.array([1, 2, 3])})
    reader = TraceReader(trace)
    shard = next(reader.iter_shards(["hidden", "kind"]))
    np.testing.assert_array_equal(shard["hidden"], np.arange(12).reshape(3, 4))
    assert reader.manifest["complete"] is True

    field = reader.manifest["shards"][0]["fields"]["hidden"]["file"]
    path = trace / "shard-00000" / field
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)
    with pytest.raises(TraceFormatError, match="checksum mismatch"):
        TraceReader(trace)
