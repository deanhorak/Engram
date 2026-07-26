import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.evaluation.controller_substitution import (
    replay_compiled_operator_trajectory,
)


def test_compiled_operator_replay_matches_exact_residual_trajectory():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=43,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    rng = np.random.default_rng(47)
    initial = rng.normal(size=(2, 5, 4)).astype(np.float32)
    semantic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)
    episodic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)

    final, states = replay_compiled_operator_trajectory(
        controller,
        initial,
        semantic,
        episodic,
    )

    expected = initial / np.sqrt(
        np.mean(np.square(initial), axis=-1, keepdims=True) + 1e-6
    )
    expected_states = []
    for stage in range(3):
        expected = expected + semantic[..., stage, :] + episodic[..., stage, :]
        expected = expected / np.sqrt(
            np.mean(np.square(expected), axis=-1, keepdims=True) + 1e-6
        )
        expected_states.append(expected)
    np.testing.assert_allclose(states, np.stack(expected_states, axis=-2))
    np.testing.assert_allclose(final, expected)

