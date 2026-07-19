import numpy as np
import pytest

from engram.episodic.recurrent import (
    NormalizedRecurrentAttention,
    RecurrentAttentionError,
    normalized_recurrent_attention,
    positive_feature_map,
)


def _manual_reference(queries, keys, values, decays, epsilon):
    feature_dimension = queries.shape[1]
    value_dimension = values.shape[1]
    numerator = np.zeros((feature_dimension, value_dimension), dtype=np.float64)
    normalizer = np.zeros(feature_dimension, dtype=np.float64)
    outputs = []
    for query, key, value, decay in zip(queries, keys, values, decays):
        query_features = np.where(query >= 0.0, query + 1.0, np.exp(query))
        key_features = np.where(key >= 0.0, key + 1.0, np.exp(key))
        numerator = decay * numerator + np.outer(key_features, value)
        normalizer = decay * normalizer + key_features
        denominator = max(float(query_features @ normalizer), epsilon)
        outputs.append((query_features @ numerator) / denominator)
    return np.asarray(outputs), numerator, normalizer


def test_positive_feature_map_is_exact_positive_and_overflow_safe():
    values = np.array([-1_000.0, -2.0, 0.0, 3.0, 1e200], dtype=np.float64)
    mapped = positive_feature_map(values)
    assert np.all(mapped > 0.0)
    assert np.all(np.isfinite(mapped))
    np.testing.assert_allclose(mapped[1:4], [np.exp(-2.0), 1.0, 4.0])
    assert mapped[-1] == 1e200


def test_sequence_matches_exact_manual_normalized_outer_product_reference():
    queries = np.array([[0.2, -0.4], [-0.7, 0.1], [1.2, -0.2]], dtype=np.float64)
    keys = np.array([[-0.1, 0.3], [0.4, -0.6], [-0.5, 0.8]], dtype=np.float64)
    values = np.array([[2.0, -1.0, 0.5], [-3.0, 4.0, 1.0], [0.2, 0.7, -2.0]])
    decays = np.array([0.8, 0.6, 0.95])
    epsilon = 1e-12
    expected, expected_numerator, expected_normalizer = _manual_reference(
        queries, keys, values, decays, epsilon
    )

    attention = NormalizedRecurrentAttention(2, 3, decay=0.5, epsilon=epsilon)
    actual = attention.sequence(queries, keys, values, decays=decays)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(attention.state.numerator, expected_numerator)
    np.testing.assert_allclose(attention.state.normalizer, expected_normalizer)
    assert attention.state.steps == 3


def test_incremental_sequence_and_convenience_apis_agree():
    rng = np.random.default_rng(123)
    queries = rng.normal(size=(19, 5))
    keys = rng.normal(size=(19, 5))
    values = rng.normal(size=(19, 7))

    incremental = NormalizedRecurrentAttention(5, 7, decay=0.91)
    incremental_outputs = np.stack(
        [incremental.step(query, key, value) for query, key, value in zip(queries, keys, values)]
    )
    sequence = NormalizedRecurrentAttention(5, 7, decay=0.91)
    sequence_outputs = sequence.sequence(queries, keys, values)
    convenience_outputs = normalized_recurrent_attention(
        queries, keys, values, decay=0.91
    )

    np.testing.assert_array_equal(sequence_outputs, incremental_outputs)
    np.testing.assert_array_equal(convenience_outputs, incremental_outputs)

    continued = NormalizedRecurrentAttention(5, 7, decay=0.91)
    first = continued.sequence(queries[:8], keys[:8], values[:8])
    second = continued.sequence(queries[8:], keys[8:], values[8:], reset=False)
    np.testing.assert_array_equal(np.concatenate([first, second]), incremental_outputs)


def test_decay_zero_reads_only_current_value_and_reset_preserves_allocation():
    attention = NormalizedRecurrentAttention(3, 4, decay=0.0, dtype=np.float32)
    numerator_id = id(attention.state.numerator)
    normalizer_id = id(attention.state.normalizer)
    rng = np.random.default_rng(9)
    queries = rng.normal(size=(6, 3)).astype(np.float32)
    keys = rng.normal(size=(6, 3)).astype(np.float32)
    values = rng.normal(size=(6, 4)).astype(np.float32)
    outputs = attention.sequence(queries, keys, values)
    np.testing.assert_allclose(outputs, values, rtol=2e-6, atol=2e-6)

    metrics = attention.state_metrics
    assert metrics == {
        "steps": 6,
        "elements": 15,
        "bytes": 15 * np.dtype(np.float32).itemsize,
        "key_features": 3,
        "value_width": 4,
    }
    attention.reset()
    assert id(attention.state.numerator) == numerator_id
    assert id(attention.state.normalizer) == normalizer_id
    assert attention.state_metrics["steps"] == 0
    assert not np.any(attention.state.numerator)
    assert not np.any(attention.state.normalizer)


def test_long_sequence_state_is_bounded_and_numerically_stable():
    rng = np.random.default_rng(87)
    length = 4_000
    queries = rng.normal(scale=20.0, size=(length, 4)).astype(np.float32)
    keys = rng.normal(scale=20.0, size=(length, 4)).astype(np.float32)
    values = rng.normal(scale=1e4, size=(length, 6)).astype(np.float32)
    attention = NormalizedRecurrentAttention(4, 6, decay=0.995, dtype=np.float32)
    initial_bytes = attention.state.nbytes
    outputs = attention.sequence(queries, keys, values)

    assert np.all(np.isfinite(outputs))
    assert np.all(np.isfinite(attention.state.numerator))
    assert np.all(np.isfinite(attention.state.normalizer))
    assert attention.state.nbytes == initial_bytes
    assert attention.state_metrics["steps"] == length


def test_configuration_shapes_and_nonfinite_values_are_rejected_before_mutation():
    with pytest.raises(RecurrentAttentionError, match="positive integer"):
        NormalizedRecurrentAttention(0, 2)
    with pytest.raises(RecurrentAttentionError, match="decay"):
        NormalizedRecurrentAttention(2, 2, decay=1.01)
    with pytest.raises(RecurrentAttentionError, match="epsilon"):
        NormalizedRecurrentAttention(2, 2, epsilon=0.0)
    with pytest.raises(RecurrentAttentionError, match="dtype"):
        NormalizedRecurrentAttention(2, 2, dtype=np.int32)

    attention = NormalizedRecurrentAttention(2, 3)
    with pytest.raises(RecurrentAttentionError, match="query"):
        attention.step([1.0], [1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(RecurrentAttentionError, match="finite"):
        attention.step([1.0, np.nan], [1.0, 2.0], [1.0, 2.0, 3.0])
    with pytest.raises(RecurrentAttentionError, match="decays"):
        attention.sequence(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((2, 3)),
            decays=[0.9, 2.0],
        )
    assert attention.state.steps == 0
