from __future__ import annotations

import numpy as np
import pytest

from engram.tracing.format import TraceReader, TraceWriter
from engram.training.controller_distillation import merge_controller_traces


def _write_chunk(path, sample_ids, *, offset):
    with TraceWriter(
        path,
        model_hash="model",
        dataset_hash="dataset",
        split="training",
        seed=7,
        metadata={
            "contract": "engram.controller.teacher_trajectory",
            "record_offset": offset,
            "requested_samples": len(sample_ids),
            "hidden_size": 2,
            "num_stages": 1,
        },
    ) as writer:
        ids = np.asarray(sample_ids, dtype=np.int64)
        writer.append(
            {
                "sample_id": ids,
                "token_id": ids,
                "token_position": np.zeros_like(ids),
                "token_embedding": np.ones((len(ids), 2), dtype=np.float16),
                "teacher_states": np.ones((len(ids), 2, 2), dtype=np.float16),
                "semantic_outputs": np.ones((len(ids), 1, 2), dtype=np.float16),
                "episodic_outputs": np.ones((len(ids), 1, 2), dtype=np.float16),
            }
        )


def test_merge_controller_traces_preserves_contract_and_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    merged = tmp_path / "merged"
    _write_chunk(first, [0, 1], offset=0)
    _write_chunk(second, [2, 3], offset=2)

    report = merge_controller_traces([first, second], merged)

    assert report["sample_count"] == 4
    assert report["records"] == 4
    trace = TraceReader(merged)
    ids = np.concatenate(
        [shard["sample_id"] for shard in trace.iter_shards(["sample_id"])]
    )
    assert ids.tolist() == [0, 1, 2, 3]
    assert trace.manifest["metadata"]["merged_trace_chunks"] == 2


def test_merge_controller_traces_rejects_overlapping_ids(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_chunk(first, [0, 1], offset=0)
    _write_chunk(second, [1, 2], offset=1)

    with pytest.raises(ValueError, match="sample IDs overlap"):
        merge_controller_traces([first, second], tmp_path / "merged")
