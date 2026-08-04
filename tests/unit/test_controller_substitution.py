import numpy as np
import pytest

from engram.controller import FactorizedRecurrentController
from engram.evaluation.controller_substitution import (
    replay_compiled_operator_trajectory,
)
from engram.runtime.controller_only import ControllerOnlyRuntime
from engram.runtime.operator_stream import (
    CausalAttentionOperatorStreamProvider,
    StageCausalAttentionOperatorStreamProvider,
    PCAOperatorStreamProvider,
    NonlinearResidualOperatorStreamProvider,
    RecurrentContextProvider,
    ResidualStateSpaceOperatorStreamProvider,
    StateSpaceOperatorStreamProvider,
    TraceOperatorStreamProvider,
    TraceSequenceOperatorStreamProvider,
    load_operator_stream_provider,
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


def test_sequence_trace_provider_roundtrip_is_checksum_authenticated(tmp_path):
    semantic = np.arange(2 * 3 * 2 * 4, dtype=np.float32).reshape(2, 3, 2, 4)
    episodic = semantic + 0.5
    provider = TraceSequenceOperatorStreamProvider(
        semantic, episodic, trace_sha256="trace-manifest"
    )
    path = provider.save(tmp_path / "sequence-provider")
    restored = TraceSequenceOperatorStreamProvider.load(path)
    assert restored.metadata()["provider_kind"] == "trace_sequence_replay"
    np.testing.assert_array_equal(restored.semantic_outputs, semantic)
    np.testing.assert_array_equal(restored.episodic_outputs, episodic)

    payload = path / "semantic_outputs.npy"
    payload.write_bytes(payload.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        TraceSequenceOperatorStreamProvider.load(path)


def test_state_space_provider_roundtrip_and_sequence_execution(tmp_path):
    rng = np.random.default_rng(107)
    stages, width, memory, projection, rank = 2, 4, 3, 2, 1
    provider = StateSpaceOperatorStreamProvider(
        memory_input=rng.normal(size=(width, memory)).astype(np.float32),
        decay=np.full(memory, 0.8, dtype=np.float32),
        state_projection=rng.normal(size=(width, projection)).astype(np.float32),
        token_projection=rng.normal(size=(width, projection)).astype(np.float32),
        stage_head=rng.normal(
            size=(stages, 2 * projection + memory + 1, 2 * rank)
        ).astype(np.float32),
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        metadata_payload={"learned": True, "architecture": "diagonal_state_space"},
    )
    path = provider.save(tmp_path / "state-space-provider")
    restored = StateSpaceOperatorStreamProvider.load(path)
    initial = rng.normal(size=(2, 3, width)).astype(np.float32)
    token = rng.normal(size=(2, 3, width)).astype(np.float32)
    controller = FactorizedRecurrentController.initialize(
        input_dim=3 * width,
        state_dim=width,
        num_stages=stages,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=109,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    runtime = ControllerOnlyRuntime(FactorizedRecurrentController(**tensors))
    expected = runtime.run_sequence_provider(initial, token, provider)
    actual = runtime.run_sequence_provider(initial, token, restored)
    np.testing.assert_allclose(actual.stage_states, expected.stage_states)
    assert restored.metadata()["provider_kind"] == "state_space_pca"


def test_residual_state_space_zero_correction_matches_base_provider(tmp_path):
    rng = np.random.default_rng(113)
    stages, width, rank = 2, 4, 1
    base = PCAOperatorStreamProvider(
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        semantic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        metadata_payload={"source_model_hash": "test"},
    )
    provider = ResidualStateSpaceOperatorStreamProvider(
        base_provider=base,
        memory_input=np.zeros((width, 3), dtype=np.float32),
        decay=np.full(3, 0.8, dtype=np.float32),
        correction_head=np.zeros((stages, 4, 2 * rank), dtype=np.float32),
        metadata_payload={"learned": True},
    )
    path = provider.save(tmp_path / "residual-provider")
    restored = ResidualStateSpaceOperatorStreamProvider.load(path)
    state = rng.normal(size=(2, width)).astype(np.float32)
    token = rng.normal(size=(2, width)).astype(np.float32)
    provider.reset((2,))
    restored.reset((2,))
    provider.begin_token(token)
    restored.begin_token(token)
    for stage in range(stages):
        expected = base.step(state, token, stage)
        actual = restored.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
    assert restored.metadata()["provider_kind"] == "state_space_residual_pca"


def test_nonlinear_residual_provider_zero_output_roundtrip(tmp_path):
    rng = np.random.default_rng(127)
    stages, width, rank = 2, 4, 1
    base = PCAOperatorStreamProvider(
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        semantic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        metadata_payload={"source_model_hash": "test"},
    )
    provider = NonlinearResidualOperatorStreamProvider(
        base_provider=base,
        input_down=rng.normal(size=(2 * width + 1, 3)).astype(np.float32),
        input_bias=rng.normal(size=3).astype(np.float32),
        stage_embedding=rng.normal(size=(stages, 2)).astype(np.float32),
        hidden_up=rng.normal(size=(5, 3)).astype(np.float32),
        hidden_bias=rng.normal(size=3).astype(np.float32),
        output_up=np.zeros((3, 2 * rank), dtype=np.float32),
        output_bias=np.zeros(2 * rank, dtype=np.float32),
        metadata_payload={"learned": True},
    )
    path = provider.save(tmp_path / "nonlinear-provider")
    restored = NonlinearResidualOperatorStreamProvider.load(path)
    generic = load_operator_stream_provider(path)
    state = rng.normal(size=(2, width)).astype(np.float32)
    token = rng.normal(size=(2, width)).astype(np.float32)
    for stage in range(stages):
        expected = base.step(state, token, stage)
        actual = restored.step(state, token, stage)
        loaded = generic.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        np.testing.assert_allclose(loaded[0], expected[0])
        np.testing.assert_allclose(loaded[1], expected[1])
    assert restored.metadata()["provider_kind"] == "nonlinear_residual_pca"


def test_causal_attention_provider_zero_output_roundtrip(tmp_path):
    rng = np.random.default_rng(131)
    stages, width, rank = 2, 4, 1
    key_dim, value_dim, query_width = 2, 3, 3
    base = PCAOperatorStreamProvider(
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        semantic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        metadata_payload={"source_model_hash": "test", "source_dataset_hash": "train"},
    )
    provider = CausalAttentionOperatorStreamProvider(
        base_provider=base,
        key_projection=rng.normal(size=(width, key_dim)).astype(np.float32),
        value_projection=rng.normal(size=(width, value_dim)).astype(np.float32),
        state_query_projection=rng.normal(size=(width, query_width)).astype(np.float32),
        token_query_projection=rng.normal(size=(width, query_width)).astype(np.float32),
        query_head=rng.normal(size=(stages, 2 * query_width, key_dim)).astype(np.float32),
        correction_head=np.zeros(
            (stages, 2 * query_width + value_dim + 1, 2 * rank), dtype=np.float32
        ),
        metadata_payload={"learned": True, "source_dataset_hash": "train"},
    )
    path = provider.save(tmp_path / "causal-provider")
    restored = CausalAttentionOperatorStreamProvider.load(path)
    generic = load_operator_stream_provider(path)
    state = rng.normal(size=(2, width)).astype(np.float32)
    token = rng.normal(size=(2, width)).astype(np.float32)
    provider.reset((2,))
    restored.reset((2,))
    generic.reset((2,))
    provider.begin_token(token)
    restored.begin_token(token)
    generic.begin_token(token)
    for stage in range(stages):
        expected = base.step(state, token, stage)
        actual = restored.step(state, token, stage)
        loaded = generic.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        np.testing.assert_allclose(loaded[0], expected[0])
        np.testing.assert_allclose(loaded[1], expected[1])
    assert restored.metadata()["provider_kind"] == "causal_attention_pca"


def test_stage_causal_attention_provider_zero_output_roundtrip(tmp_path):
    rng = np.random.default_rng(137)
    stages, width, rank = 2, 4, 1
    key_dim, value_dim, query_width = 2, 3, 3
    base = PCAOperatorStreamProvider(
        semantic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        semantic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        semantic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        episodic_mean=rng.normal(size=(stages, width)).astype(np.float32),
        episodic_basis=rng.normal(size=(stages, rank, width)).astype(np.float32),
        episodic_projection=rng.normal(size=(stages, 2 * width + 1, rank)).astype(np.float32),
        metadata_payload={"source_model_hash": "test", "source_dataset_hash": "train"},
    )
    provider = StageCausalAttentionOperatorStreamProvider(
        base_provider=base,
        key_projection=rng.normal(size=(stages, width, key_dim)).astype(np.float32),
        value_projection=rng.normal(size=(stages, width, value_dim)).astype(np.float32),
        state_query_projection=rng.normal(size=(width, query_width)).astype(np.float32),
        token_query_projection=rng.normal(size=(width, query_width)).astype(np.float32),
        query_head=rng.normal(size=(stages, 2 * query_width, key_dim)).astype(np.float32),
        correction_head=np.zeros(
            (stages, 2 * query_width + value_dim + 1, 2 * rank), dtype=np.float32
        ),
        metadata_payload={"learned": True, "source_dataset_hash": "train"},
    )
    path = provider.save(tmp_path / "stage-causal-provider")
    restored = StageCausalAttentionOperatorStreamProvider.load(path)
    generic = load_operator_stream_provider(path)
    state = rng.normal(size=(2, width)).astype(np.float32)
    token = rng.normal(size=(2, width)).astype(np.float32)
    for item in (provider, restored, generic):
        item.reset((2,))
        item.begin_token(token)
    for stage in range(stages):
        expected = base.step(state, token, stage)
        actual = restored.step(state, token, stage)
        loaded = generic.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
        np.testing.assert_allclose(loaded[0], expected[0])
        np.testing.assert_allclose(loaded[1], expected[1])
    assert restored.metadata()["provider_kind"] == "stage_causal_attention_pca"

    direct = StageCausalAttentionOperatorStreamProvider(
        base_provider=base,
        key_projection=provider.key_projection,
        value_projection=provider.value_projection,
        state_query_projection=provider.state_query_projection,
        token_query_projection=provider.token_query_projection,
        query_head=provider.query_head,
        correction_head=np.zeros(
            (stages, 2 * query_width + value_dim + 1, 2 * width), dtype=np.float32
        ),
        metadata_payload={"learned": True},
    )
    direct.reset((2,))
    direct.begin_token(token)
    for stage in range(stages):
        actual = direct.step(state, token, stage)
        expected = base.step(state, token, stage)
        np.testing.assert_allclose(actual[0], expected[0])
        np.testing.assert_allclose(actual[1], expected[1])
    assert direct.metadata()["correction_mode"] == "direct_hidden"
