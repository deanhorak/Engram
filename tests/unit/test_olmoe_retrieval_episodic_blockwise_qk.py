from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_blockwise_qk as qk


def _shard(index: int) -> dict[str, object]:
    return {
        "record_index": index,
        "record_id": f"train-{index:02d}",
        "source_record_sha256": "a" * 64,
        "output_evidence_sha256": "b" * 64,
        "reset_output_evidence_sha256": "b" * 64,
        "schedule_rows_sha256": "c" * 64,
    }


def test_cross_check_requires_matching_record_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(qk.full, "_RECORDS", 1)
    qk_manifest = {"shards": [_shard(0)]}
    value_manifest = {"shards": [_shard(0)]}
    qk._cross_check_shards(qk_manifest, value_manifest)
    value_manifest["shards"][0]["record_id"] = "changed"
    with pytest.raises(ValueError, match="not cross-bound"):
        qk._cross_check_shards(qk_manifest, value_manifest)


def test_audit_report_writer_rejects_existing_and_forbidden_paths(
    tmp_path: Path,
) -> None:
    report = {"confirmation_split_opened": False}
    output = tmp_path / "audit.json"
    result = qk.write_audit_report(report, output)
    assert result["path"] == str(output.resolve())
    with pytest.raises(ValueError, match="already exists"):
        qk.write_audit_report(report, output)
    with pytest.raises(ValueError, match="confirmation scope"):
        qk.write_audit_report(report, tmp_path / "confirmation" / "audit.json")


def test_candidate_trace_summary_has_pre_top_k_contract() -> None:
    shape = (
        len(qk.full._READ_POSITIONS),
        qk.full._LAYERS,
        qk.full._QUERY_HEADS,
        qk.full._C28_QK_CANDIDATE_ENTRIES,
        qk.full._QK_PARTIAL_BANDS,
    )
    values = qk.full._qk_candidate_trace_summary(
        np.zeros(shape, dtype=np.float32),
        qk.full._READ_POSITIONS,
    )
    assert values["pre_top_k"] is True
    assert values["older_candidates"] == 8


def test_candidate_key_trace_summary_has_post_rope_contract() -> None:
    shape = (
        len(qk.full._READ_POSITIONS),
        qk.full._LAYERS,
        qk.full._QUERY_HEADS,
        qk.full._C28_QK_CANDIDATE_ENTRIES,
        qk.full._HEAD_DIMENSION,
    )
    values = qk.full._qk_candidate_key_trace_summary(
        np.zeros(shape, dtype=np.float32),
        qk.full._READ_POSITIONS,
    )
    assert values["post_rope"] is True
    assert values["inactive_slots_zero"] is True


def test_candidate_key_compression_rejects_invalid_rank_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(qk.full, "_HEAD_DIMENSION", 4)
    monkeypatch.setattr(
        qk,
        "_checked_manifest",
        lambda *_args, **_kwargs: (
            tmp_path / "manifest.json",
            {
                "experiment": qk.full._QK_CANDIDATE_KEY_CAPTURE_EXPERIMENT,
                "confirmation_split_opened": False,
                "head_dimension": 4,
            },
        ),
    )
    monkeypatch.setattr(
        qk.full,
        "load_stacked_full_visible_qk_candidate_key_trace",
        lambda *_args, **_kwargs: (np.zeros((1, 1, 1, 1, 4), dtype=np.float32), {}),
    )
    with pytest.raises(ValueError, match="compression ranks"):
        qk.audit_candidate_key_compression(
            candidate_key_manifest=tmp_path / "missing.json",
            candidate_key_manifest_sha256="a" * 64,
            ranks=(0,),
        )
