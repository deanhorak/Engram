import numpy as np

from engram.evaluation.native_bitnet_dip_rms_estimator import (
    _fit_log_ratio,
    _fit_nonnegative_ratio,
    stratified_proxy_audit,
    top_proxy_raw_audit_union,
)


def test_top_proxy_audit_replaces_tail_slots_without_extra_records():
    order = np.asarray([[0, 1, 2, 3, 4, 5]], dtype=np.int64)
    proxy_square = np.asarray(
        [[9.0, 8.0, 1.0, 2.0, 20.0, 10.0]],
        dtype=np.float64,
    )

    routed, audits, union = top_proxy_raw_audit_union(
        order,
        proxy_square,
        candidate_count=4,
        audit_count=2,
    )

    np.testing.assert_array_equal(routed, [[0, 1]])
    np.testing.assert_array_equal(audits, [[4, 5]])
    assert union.shape == (1, 4)
    assert len(np.unique(union[0])) == 4


def test_stratified_audit_covers_the_entire_unrouted_tail():
    order = np.asarray([np.arange(10)], dtype=np.int64)
    proxy_square = np.asarray([np.arange(10, 0, -1)], dtype=np.float64)

    routed, strata, audits = stratified_proxy_audit(
        order,
        proxy_square,
        candidate_count=6,
        audit_count=2,
    )

    np.testing.assert_array_equal(routed, [[0, 1, 2, 3]])
    assert sorted(np.concatenate(strata[0]).tolist()) == list(range(4, 10))
    assert audits.shape == (1, 2)
    assert all(audit in stratum for audit, stratum in zip(audits[0], strata[0]))


def test_sequence_split_regressions_are_nonnegative_and_learn_ratio():
    features = np.column_stack(
        [
            np.ones(12),
            np.linspace(0.25, 1.5, 12),
        ]
    )
    target = 0.5 + 2.0 * features[:, 1]
    training = np.arange(12) % 2 == 0
    evaluation = ~training

    log_prediction = _fit_log_ratio(
        np.log(features + 1.0),
        target,
        training,
        evaluation,
        ridge=1e-6,
    )
    nnls_prediction = _fit_nonnegative_ratio(
        features,
        target,
        training,
        evaluation,
    )

    assert np.all(log_prediction >= 0)
    assert np.all(nnls_prediction >= 0)
    np.testing.assert_allclose(nnls_prediction, target[evaluation], atol=1e-5)
