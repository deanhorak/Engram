import numpy as np

from engram.controller import FactorizedRecurrentController
from engram.evaluation.controller_substitution import (
    replay_compiled_operator_trajectory,
)
from engram.runtime.controller_only import ControllerOnlyRuntime
from engram.runtime.operator_stream import (
    PCAOperatorStreamProvider,
    RecurrentContextProvider,
    TraceOperatorStreamProvider,
    TraceSequenceOperatorStreamProvider,
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


def test_controller_only_runtime_provider_matches_stream_replay():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=67,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    rng = np.random.default_rng(71)
    initial = rng.normal(size=(2, 4)).astype(np.float32)
    token = rng.normal(size=(2, 4)).astype(np.float32)
    semantic = rng.normal(scale=0.1, size=(2, 3, 4)).astype(np.float32)
    episodic = rng.normal(scale=0.1, size=(2, 3, 4)).astype(np.float32)
    provider = TraceOperatorStreamProvider(semantic, episodic)
    runtime = ControllerOnlyRuntime(controller)
    expected = runtime.run(initial, semantic, episodic)
    actual = runtime.run_provider(initial, token, provider)
    np.testing.assert_allclose(actual.final_state, expected.final_state)
    np.testing.assert_allclose(actual.stage_states, expected.stage_states)
    assert provider.metadata()["learned"] is False


def test_pca_operator_stream_provider_round_trips_without_model(tmp_path):
    rng = np.random.default_rng(73)
    stages, width, rank = 3, 4, 2
    provider = PCAOperatorStreamProvider(
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        semantic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        metadata_payload={"source_model_hash": "test"},
    )
    path = provider.save(tmp_path / "provider")
    restored = PCAOperatorStreamProvider.load(path)
    state = rng.normal(size=(2, width)).astype(np.float32)
    token = rng.normal(size=(2, width)).astype(np.float32)
    for stage in range(stages):
        expected = provider.step(state, token, stage)
        actual = restored.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
    assert restored.metadata()["provider_kind"] == "pca_state_token"


def test_stateful_provider_advances_once_per_token_and_resets_context():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=79,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    provider = RecurrentContextProvider.initialize(
        state_dim=4, num_stages=3, memory_dim=2, output_rank=2, seed=83
    )
    rng = np.random.default_rng(89)
    initial = rng.normal(size=(2, 5, 4)).astype(np.float32)
    token = rng.normal(size=(2, 5, 4)).astype(np.float32)
    runtime = ControllerOnlyRuntime(controller)
    first = runtime.run_sequence_provider(
        initial, token, provider, initial_is_normalized=False
    )
    second = runtime.run_sequence_provider(
        initial, token, provider, initial_is_normalized=False
    )
    np.testing.assert_array_equal(first.stage_states, second.stage_states)
    assert first.stage_states.shape == (2, 5, 3, 4)
    assert first.final_states.shape == (2, 5, 4)
    assert provider.metadata()["architecture"].startswith("shared_token_memory")


def test_sequence_trace_provider_matches_flat_operator_replay():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=97,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    rng = np.random.default_rng(101)
    initial = rng.normal(size=(2, 5, 4)).astype(np.float32)
    token = rng.normal(size=(2, 5, 4)).astype(np.float32)
    semantic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)
    episodic = rng.normal(scale=0.1, size=(2, 5, 3, 4)).astype(np.float32)
    provider = TraceSequenceOperatorStreamProvider(semantic, episodic)
    runtime = ControllerOnlyRuntime(controller)
    sequence = runtime.run_sequence_provider(
        initial, token, provider, initial_is_normalized=False
    )
    expected = []
    for row in range(initial.shape[0]):
        result = runtime.run(
            initial[row], semantic[row], episodic[row], initial_is_normalized=False
        )
        expected.append(result.stage_states)
    np.testing.assert_allclose(sequence.stage_states, np.stack(expected, axis=0))
