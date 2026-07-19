import numpy as np

from engram.episodic.evaluate import evaluate_attention_replacement
from engram.episodic.hybrid import HybridEpisodicMemory


def test_hybrid_memory_stays_bounded_and_reports_reads():
    rng = np.random.default_rng(8)
    memory = HybridEpisodicMemory(4, 5, local_window=3, retrieval_capacity=7, retrieval_candidates=4, retrieval_top_k=2)
    last = None
    for _ in range(30):
        last = memory.step(rng.normal(size=4), rng.normal(size=4), rng.normal(size=5))
    assert last is not None
    assert memory.store.recent_count == 3
    assert memory.store.older_count == 7
    assert last.retrievals == 2
    assert last.state_bytes > 0


def test_gate3_synthetic_report_is_honestly_labeled():
    report = evaluate_attention_replacement(length=32, key_width=8, value_width=8, local_window=4)
    assert report["status"] == "synthetic_pipeline_validation"
    assert report["teacher_attention_traces"]["status"] == "not_run"
    assert report["copying_accuracy"] == 1.0
