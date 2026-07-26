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
        writer.append(
            {
                "hidden": np.arange(12, dtype=np.float32).reshape(3, 4),
                "kind": np.array([1, 2, 3]),
            }
        )
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


def test_incomplete_trace_can_resume_without_rewriting_existing_shards(tmp_path):
    trace = tmp_path / "trace"
    writer = TraceWriter(
        trace,
        model_hash="model",
        dataset_hash="dataset",
        split="training",
        seed=7,
        metadata={"contract": "resume-test"},
    )
    writer.append({"sample_id": np.array([0, 0]), "value": np.array([1, 2])})
    first_hash = writer._shards[0]["fields"]["value"]["sha256"]

    with TraceWriter(
        trace,
        model_hash="model",
        dataset_hash="dataset",
        split="training",
        seed=7,
        metadata={"contract": "resume-test"},
        resume=True,
    ) as resumed:
        resumed.append({"sample_id": np.array([1]), "value": np.array([3])})

    reader = TraceReader(trace)
    assert len(reader.manifest["shards"]) == 2
    assert reader.manifest["shards"][0]["fields"]["value"]["sha256"] == first_hash
    values = [np.asarray(shard["value"]) for shard in reader.iter_shards(["value"])]
    np.testing.assert_array_equal(np.concatenate(values), [1, 2, 3])


def test_resume_rejects_contract_change(tmp_path):
    trace = tmp_path / "trace"
    writer = TraceWriter(
        trace,
        model_hash="model",
        dataset_hash="dataset",
        split="training",
        seed=7,
    )
    writer.append({"value": np.array([1])})

    with pytest.raises(TraceFormatError, match="different seed"):
        TraceWriter(
            trace,
            model_hash="model",
            dataset_hash="dataset",
            split="training",
            seed=8,
            resume=True,
        )


def test_fresh_writer_rejects_orphan_shard_directory(tmp_path):
    trace = tmp_path / "trace"
    (trace / "shard-00000").mkdir(parents=True)
    with pytest.raises(TraceFormatError, match="orphan shards"):
        TraceWriter(
            trace,
            model_hash="model",
            dataset_hash="dataset",
            split="training",
            seed=1,
        )
