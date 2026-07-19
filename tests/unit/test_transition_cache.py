import json

import numpy as np
import pytest

from engram.transitions.cache import CacheFormatError, TransitionCache


def _put(cache, state, token, *, source="online", next_state=None):
    state = np.asarray(state, dtype=np.float32)
    if next_state is None:
        next_state = state + 1.0
    return cache.put(
        state,
        token,
        next_state,
        [(token + 1, 2.5), (token + 2, 1.0)],
        0.9,
        source=source,
    )


def test_fingerprint_is_deterministic_and_input_token_is_part_of_key():
    cache = TransitionCache(
        4, quantization_step=0.5, subvector_width=2, similarity_radius=0.5
    )
    state = np.array([0.1, -0.1, 1.1, -1.1])
    assert cache.fingerprint(state, 7) == cache.fingerprint(state.copy(), 7)
    assert cache.fingerprint(state, 7) != cache.fingerprint(state, 8)

    _put(cache, state, 7)
    nearby = state + np.array([0.02, 0.0, 0.0, 0.0])
    lookup = cache.lookup(nearby, 7)
    assert lookup.hit
    assert lookup.transition.output_candidates[0] == (8, 2.5)
    assert not cache.lookup(nearby, 8).hit


def test_fingerprint_collision_outside_radius_is_rejected():
    cache = TransitionCache(
        2, quantization_step=10.0, similarity_radius=0.1, subvector_width=1
    )
    _put(cache, [1.0, 0.0], 3)
    rejected = cache.lookup([2.0, 0.0], 3)

    assert not rejected.hit
    assert rejected.reason == "radius_rejection"
    assert rejected.state_distance == pytest.approx(0.5)
    assert cache.metrics.radius_rejections == 1
    assert cache.metrics.collisions == 1


def test_lru_eviction_online_offline_counts_and_error_metrics():
    cache = TransitionCache(2, capacity=2, quantization_step=0.1, similarity_radius=0.01)
    _put(cache, [1.0, 0.0], 1, source="offline")
    _put(cache, [0.0, 1.0], 2, source="online")
    assert cache.lookup([1.0, 0.0], 1).hit  # Refresh token 1 as most recent.
    _put(cache, [-1.0, 0.0], 3, source="online")

    assert not cache.lookup([0.0, 1.0], 2).hit
    measured = cache.lookup([1.0, 0.0], 1, actual_next_state=[2.1, 1.0])
    assert measured.hit
    metrics = cache.metrics
    assert metrics.entries == 2
    assert metrics.evictions == 1
    assert metrics.offline_puts == 1
    assert metrics.online_puts == 2
    assert metrics.approximation_error_samples == 1
    assert metrics.mean_approximation_error > 0.0
    assert metrics.hit_rate == pytest.approx(2.0 / 3.0)


def test_bypass_mode_suppresses_population_and_reuse():
    cache = TransitionCache(2, bypass=True)
    assert not _put(cache, [1.0, 0.0], 4)
    lookup = cache.lookup([1.0, 0.0], 4)
    assert not lookup.hit and lookup.reason == "bypass"
    assert cache.metrics.entries == 0
    assert cache.metrics.bypassed_puts == 1
    assert cache.metrics.bypassed_lookups == 1


def test_persistent_round_trip_preserves_lru_data_and_detects_corruption(tmp_path):
    cache = TransitionCache(
        3,
        capacity=3,
        quantization_step=0.25,
        subvector_width=2,
        similarity_radius=0.1,
    )
    _put(cache, [1.0, 0.0, 0.0], 10, source="offline")
    _put(cache, [0.0, 1.0, 0.0], 11, source="online")
    path = cache.save(tmp_path / "cache.json")

    restored = TransitionCache.load(path)
    result = restored.lookup([1.0, 0.0, 0.0], 10)
    assert result.hit
    np.testing.assert_array_equal(result.transition.next_state, [2.0, 1.0, 1.0])
    assert result.transition.confidence == 0.9
    assert restored.metrics.offline_puts == 0  # Loading does not count as population.

    document = json.loads(path.read_text())
    document["entries"][0]["confidence"] = 0.1
    path.write_text(json.dumps(document))
    with pytest.raises(CacheFormatError, match="checksum mismatch"):
        TransitionCache.load(path)
