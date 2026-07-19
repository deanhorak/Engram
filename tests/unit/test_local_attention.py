import numpy as np

from engram.episodic.local_attention import LocalAttentionCache, causal_local_attention


def _dense_causal_reference(query, key, value, window):
    scale = 1.0 / np.sqrt(query.shape[-1])
    scores = np.einsum("...td,...sd->...ts", query, key) * scale
    length = query.shape[-2]
    positions = np.arange(length)
    visible = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] >= positions[:, None] - window + 1
    )
    scores = np.where(visible, scores, -np.inf)
    scores -= np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("...ts,...sv->...tv", weights, value)


def test_multi_head_local_attention_matches_dense_causal_reference():
    rng = np.random.default_rng(31)
    query = rng.normal(size=(3, 11, 5))
    key = rng.normal(size=(3, 11, 5))
    value = rng.normal(size=(3, 11, 7))

    expected = _dense_causal_reference(query, key, value, window=4)
    actual = causal_local_attention(query, key, value, window=4)

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_single_head_window_one_returns_current_values():
    rng = np.random.default_rng(2)
    query = rng.normal(size=(8, 4))
    key = rng.normal(size=(8, 4))
    value = rng.normal(size=(8, 6))

    np.testing.assert_array_equal(
        causal_local_attention(query, key, value, window=1), value
    )


def test_incremental_cache_matches_batch_and_stays_bounded():
    rng = np.random.default_rng(19)
    query = rng.normal(size=(2, 13, 4))
    key = rng.normal(size=(2, 13, 4))
    value = rng.normal(size=(2, 13, 6))
    expected = causal_local_attention(query, key, value, window=5)
    cache = LocalAttentionCache(window=5)

    outputs = []
    for position in range(query.shape[-2]):
        outputs.append(
            cache.step(
                query[..., position, :],
                key[..., position, :],
                value[..., position, :],
            )
        )
        assert cache.cache_length == min(position + 1, cache.window)

    actual = np.stack(outputs, axis=-2)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert cache.tokens_seen == query.shape[-2]
    assert cache.cache_length == 5

    cache.reset()
    assert cache.cache_length == 0
    assert cache.tokens_seen == 0


def test_softmax_is_stable_for_large_scores():
    query = np.array([[10_000.0, -10_000.0], [10_001.0, -9_999.0]])
    key = np.array([[10_000.0, -10_000.0], [9_999.0, -10_001.0]])
    value = np.array([[1.0, 2.0], [3.0, 4.0]])

    output = causal_local_attention(query, key, value, window=2, scale=1.0)

    assert np.all(np.isfinite(output))
    np.testing.assert_array_equal(output[0], value[0])
