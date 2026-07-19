import numpy as np

from engram.semantic.oracle import magnitude_oracle_sample


def test_magnitude_ranking_includes_value_norm():
    activations = np.array([10.0, 2.0])
    values = np.array([[0.01, 0.0], [10.0, 0.0]])
    _, order = magnitude_oracle_sample(activations, values)
    assert order.tolist() == [1, 0]


def test_full_k_is_exact_and_zero_output_needs_zero_neurons():
    rng = np.random.default_rng(9)
    activations = rng.normal(size=7)
    values = rng.normal(size=(7, 3))
    results, order = magnitude_oracle_sample(activations, values, targets=(1.0,))
    assert sorted(order.tolist()) == list(range(7))
    assert results[0].k == 7
    assert results[0].relative_l2 < 1e-12

    zero_results, _ = magnitude_oracle_sample(np.zeros(7), values)
    assert all(result.k == 0 and result.relative_l2 == 0.0 for result in zero_results)
