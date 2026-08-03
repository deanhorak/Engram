import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.evaluation.controller_substitution import (
    replay_compiled_operator_trajectory,
)
from engram.runtime.controller_only import ControllerOnlyRuntime


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


def test_controller_only_runtime_matches_compiled_operator_replay():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=53,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    rng = np.random.default_rng(59)
    initial = rng.normal(size=(2, 5, 4)).astype(np.float32)
    semantic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)
    episodic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)

    expected_final, expected_states = replay_compiled_operator_trajectory(
        controller, initial, semantic, episodic
    )
    result = ControllerOnlyRuntime(controller).run(
        initial,
        semantic,
        episodic,
    )
    np.testing.assert_allclose(result.final_state, expected_final)
    np.testing.assert_allclose(result.stage_states, expected_states)


def test_controller_only_runtime_rejects_nonzero_correction_by_default():
    controller = FactorizedRecurrentController.initialize(
        input_dim=6,
        state_dim=2,
        num_stages=1,
        rank=2,
        adapter_rank=0,
        operator_residual=True,
        seed=61,
    )
    with np.testing.assert_raises(ValueError):
        ControllerOnlyRuntime(controller)
    runtime = ControllerOnlyRuntime(controller, allow_correction=True)
    initial = np.ones((1, 2), dtype=np.float32)
    semantic = np.zeros((1, 1, 2), dtype=np.float32)
    episodic = np.zeros((1, 1, 2), dtype=np.float32)
    result = runtime.run(initial, semantic, episodic)
    assert result.stage_states.shape == (1, 1, 2)
