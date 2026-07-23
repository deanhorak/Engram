import numpy as np

from engram.training.grouped_sparse_boundaries import (
    _fixed_cardinality_labels,
    _learn_pair_permutation,
    _pair_contribution_utility,
)


def test_pair_matching_groups_coactive_records_deterministically():
    membership = np.asarray(
        [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ],
        dtype=bool,
    )
    permutation = _learn_pair_permutation(membership)
    pairs = {frozenset(pair) for pair in permutation.reshape(-1, 2).tolist()}
    assert pairs == {frozenset((0, 2)), frozenset((1, 3))}
    np.testing.assert_array_equal(np.sort(permutation), np.arange(4))


def test_pair_utility_matches_explicit_output_contribution_norm():
    states = np.asarray([[0.5, -1.0], [1.25, 0.75]], dtype=np.float64)
    gate = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 1.0]],
        dtype=np.float64,
    )
    up = np.asarray(
        [[0.25, 1.0], [1.0, -0.5], [0.75, 0.25], [0.5, 0.5]],
        dtype=np.float64,
    )
    down = np.asarray(
        [[1.0, 0.5, -0.25, 0.75], [0.0, 1.0, 0.5, -0.5]],
        dtype=np.float64,
    )
    utility = _pair_contribution_utility(states, gate, up, down)

    gate_values = states @ gate.T
    activations = (gate_values / (1.0 + np.exp(-gate_values))) * (states @ up.T)
    expected = []
    for row in activations:
        expected.append(
            [
                np.linalg.norm(row[start] * down[:, start] + row[start + 1] * down[:, start + 1])
                for start in (0, 2)
            ]
        )
    np.testing.assert_allclose(utility, np.asarray(expected), rtol=1e-12, atol=1e-12)


def test_group_labels_have_exact_cardinality_and_follow_utility():
    utility = np.asarray([[0.5, 3.0, 1.0], [4.0, 2.0, 5.0]])
    labels = _fixed_cardinality_labels(utility, 2)
    assert labels.tolist() == [[False, True, True], [True, False, True]]
    np.testing.assert_array_equal(labels.sum(axis=1), np.asarray([2, 2]))
