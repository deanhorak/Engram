import numpy as np

from engram.semantic.calibrated_router import TraceCalibratedRouter


def test_trace_calibrated_router_learns_state_specific_records():
    gate = np.array([[4, 0], [0, 4], [1, 1], [-1, 0]], dtype=np.float64)
    up = np.array([[2, 0], [0, 2], [1, 1], [1, 0]], dtype=np.float64)
    values = np.eye(4, 2, dtype=np.float64)
    states = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float64)
    router = TraceCalibratedRouter.fit(
        gate,
        up,
        values,
        states,
        num_clusters=2,
        records_per_cluster=2,
        iterations=8,
    )

    horizontal = router.search([1, 0], probes=1, candidate_count=1)
    vertical = router.search([0, 1], probes=1, candidate_count=1)

    assert horizontal.indices.tolist() == [0]
    assert vertical.indices.tolist() == [1]
    assert horizontal.probed_record_count == 2
    assert vertical.probed_record_count == 2
