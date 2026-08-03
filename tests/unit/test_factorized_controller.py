import numpy as np
import pytest

from engram.controller import FactorizedRecurrentController
from engram.tracing.format import TraceWriter


def test_factorized_controller_has_identity_biased_shared_core():
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        seed=7,
    )

    assert controller.input_down.shape == (12, 2)
    assert controller.recurrent_down.shape == (4, 2)
    assert controller.gate_up.shape == (2, 8)
    assert controller.stage_embeddings.shape == (3, 4)
    assert controller.parameter_count == sum(
        tensor.size for tensor in controller.tensors().values()
    )
    assert controller.serialized_bytes == 4 * controller.parameter_count

    state = np.linspace(-0.3, 0.3, 4, dtype=np.float32)
    state /= np.sqrt(np.mean(np.square(state)) + 1e-6)
    supplied = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    updated = controller.step(state, supplied, stage=0)
    assert updated.dtype == np.float32
    assert np.max(np.abs(updated - state)) < 0.01
    np.testing.assert_allclose(np.sqrt(np.mean(np.square(updated))), 1.0, atol=2e-6)


def test_factorized_controller_stage_rollout_and_serialized_cpu_reload(tmp_path):
    controller = FactorizedRecurrentController.initialize(
        input_dim=9,
        state_dim=3,
        num_stages=4,
        rank=3,
        adapter_rank=2,
        seed=11,
    )
    tensors = controller.tensors()
    tensors["stage_embeddings"][1] = 0.25
    controller = FactorizedRecurrentController(**tensors)
    initial = np.zeros((2, 3), dtype=np.float32)
    supplied = np.ones((2, 4, 9), dtype=np.float32)

    expected = controller.run_staged(initial, supplied)
    controller.save(tmp_path)
    restored = FactorizedRecurrentController.load(tmp_path)
    actual = restored.run_staged(initial, supplied)

    np.testing.assert_array_equal(actual, expected)
    assert restored.metadata() == controller.metadata()
    assert not np.allclose(
        controller.step(initial, supplied[:, 0], stage=0),
        controller.step(initial, supplied[:, 0], stage=1),
    )


def test_stage_input_adapter_uses_versioned_artifact_and_cpu_reload(tmp_path):
    base = FactorizedRecurrentController.initialize(
        input_dim=9,
        state_dim=3,
        num_stages=4,
        rank=3,
        adapter_rank=2,
        input_adapter_rank=0,
        seed=13,
    )
    controller = FactorizedRecurrentController.initialize(
        input_dim=9,
        state_dim=3,
        num_stages=4,
        rank=3,
        adapter_rank=2,
        input_adapter_rank=2,
        seed=13,
    )
    assert base.metadata()["schema_version"] == 1
    assert "input_adapter_rank" not in base.metadata()
    assert "input_adapter_down" not in base.tensors()
    assert controller.input_adapter_down.shape == (4, 9, 2)
    assert controller.input_adapter_up.shape == (4, 2, 3)
    assert controller.metadata()["schema_version"] == 2
    assert controller.metadata()["input_adapter_rank"] == 2
    assert (
        controller.parameter_count - base.parameter_count
        == 4 * 2 * (9 + 3)
    )

    tensors = controller.tensors()
    tensors["input_adapter_up"][2] = 0.1
    controller = FactorizedRecurrentController(**tensors)
    initial = np.ones((2, 3), dtype=np.float32)
    supplied = np.ones((2, 4, 9), dtype=np.float32)
    expected = controller.run_staged(initial, supplied)
    controller.save(tmp_path)
    restored = FactorizedRecurrentController.load(tmp_path)

    np.testing.assert_array_equal(restored.run_staged(initial, supplied), expected)
    assert restored.metadata() == controller.metadata()


def test_operator_residual_preserves_known_additive_transition(tmp_path):
    controller = FactorizedRecurrentController.initialize(
        input_dim=12,
        state_dim=4,
        num_stages=3,
        rank=2,
        adapter_rank=1,
        operator_residual=True,
        seed=17,
    )
    tensors = controller.tensors()
    tensors["step_scale"][:] = 0.0
    controller = FactorizedRecurrentController(**tensors)
    state = np.linspace(-0.4, 0.4, 8, dtype=np.float32).reshape(2, 4)
    token = np.zeros_like(state)
    semantic = np.full_like(state, 0.2)
    episodic = np.full_like(state, -0.05)
    supplied = np.concatenate((token, semantic, episodic), axis=-1)
    residual = state + semantic + episodic
    expected = residual / np.sqrt(
        np.mean(np.square(residual), axis=-1, keepdims=True) + 1e-6
    )

    np.testing.assert_allclose(
        controller.step(state, supplied, stage=1), expected, rtol=1e-6, atol=1e-6
    )
    assert controller.metadata()["schema_version"] == 3
    assert controller.metadata()["operator_residual_input_order"] == [
        "semantic_output",
        "episodic_output",
    ]
    controller.save(tmp_path)
    restored = FactorizedRecurrentController.load(tmp_path)
    assert restored.has_operator_residual
    assert restored.metadata() == controller.metadata()


def test_factorized_controller_validates_shapes_and_stage():
    controller = FactorizedRecurrentController.initialize(
        input_dim=6,
        state_dim=2,
        num_stages=2,
        rank=2,
        adapter_rank=0,
    )
    with pytest.raises(ValueError, match="trailing dimension"):
        controller.step(np.zeros(3), np.zeros(6), stage=0)
    with pytest.raises(ValueError, match="stage must lie"):
        controller.step(np.zeros(2), np.zeros(6), stage=2)
    with pytest.raises(ValueError, match="leading dimensions"):
        controller.run_staged(np.zeros((2, 2)), np.zeros((3, 2, 6)))


def test_torch_and_numpy_factorized_steps_match():
    torch = pytest.importorskip("torch")
    from engram.training.controller_distillation import _torch_controller_class

    controller = FactorizedRecurrentController.initialize(
        input_dim=15,
        state_dim=5,
        num_stages=3,
        rank=4,
        adapter_rank=2,
        input_adapter_rank=2,
        operator_residual=True,
        seed=19,
    )
    _, TorchController = _torch_controller_class()
    module = TorchController(controller).eval()
    rng = np.random.default_rng(23)
    state = rng.normal(size=(3, 5)).astype(np.float32)
    supplied = rng.normal(size=(3, 15)).astype(np.float32)

    with torch.inference_mode():
        torch_output = module.step(
            torch.from_numpy(state), torch.from_numpy(supplied), 2
        ).numpy()
    numpy_output = controller.step(state, supplied, stage=2)
    np.testing.assert_allclose(numpy_output, torch_output, rtol=2e-6, atol=2e-6)


def _write_normalized_trace(path, *, dataset_hash, seed):
    rng = np.random.default_rng(seed)
    records, stages, width = 8, 3, 4
    embedding = rng.normal(size=(records, width)).astype(np.float32)
    embedding /= np.sqrt(np.mean(np.square(embedding), axis=-1, keepdims=True) + 1e-6)
    semantic = rng.normal(scale=0.2, size=(records, stages, width)).astype(np.float32)
    episodic = rng.normal(scale=0.2, size=(records, stages, width)).astype(np.float32)
    states = np.empty((records, stages + 1, width), dtype=np.float32)
    states[:, 0] = embedding
    for stage in range(stages):
        residual = (
            states[:, stage] + 0.1 * semantic[:, stage] + 0.05 * episodic[:, stage]
        )
        states[:, stage + 1] = residual / np.sqrt(
            np.mean(np.square(residual), axis=-1, keepdims=True) + 1e-6
        )
    with TraceWriter(
        path,
        model_hash="a" * 64,
        dataset_hash=dataset_hash,
        split="training",
        seed=seed,
        metadata={
            "contract": "engram.controller.teacher_trajectory",
            "contract_version": 1,
            "teacher_runtime": "unit_fixture",
            "teacher_device": "cpu",
            "hidden_size": width,
            "num_stages": stages,
            "input_order": [
                "token_embedding",
                "semantic_output",
                "episodic_output",
            ],
            "state_normalization": "per_token_rms",
            "operator_normalization": "divide_by_stage_input_rms",
        },
    ) as writer:
        writer.append(
            {
                "sample_id": np.arange(records, dtype=np.int64),
                "token_embedding": embedding.astype(np.float16),
                "teacher_states": states.astype(np.float16),
                "semantic_outputs": semantic.astype(np.float16),
                "episodic_outputs": episodic.astype(np.float16),
            }
        )


def test_distillation_exports_cpu_only_artifact_with_protected_validation(
    tmp_path,
):
    pytest.importorskip("torch")
    from engram.training.controller_distillation import (
        distill_factorized_controller,
    )

    training = tmp_path / "training"
    validation = tmp_path / "validation"
    _write_normalized_trace(training, dataset_hash="b" * 64, seed=29)
    _write_normalized_trace(validation, dataset_hash="c" * 64, seed=31)

    report = distill_factorized_controller(
        training,
        tmp_path / "artifact",
        validation_trace=validation,
        device="cpu",
        rank=3,
        adapter_rank=1,
        steps=4,
        batch_size=4,
        learning_rate=1e-3,
    )

    assert report["protected_validation"] is True
    assert report["inference_device"] == "cpu"
    assert report["torch_required_for_inference"] is False
    assert report["cpu_reload_parity"]["passed"] is True
    restored = FactorizedRecurrentController.load(tmp_path / "artifact" / "controller")
    assert restored.state_dim == 4

    continuation = distill_factorized_controller(
        training,
        tmp_path / "continuation",
        validation_trace=validation,
        initial_controller=tmp_path / "artifact" / "controller",
        device="cpu",
        rank=3,
        adapter_rank=1,
        steps=1,
        batch_size=4,
        learning_rate=1e-4,
        teacher_forcing_schedule="none",
    )
    assert continuation["initial_controller"] is not None
    assert continuation["training"]["teacher_forcing_schedule"] == "none"
    assert continuation["training"]["history"][0]["teacher_forcing"] == 0.0


def test_batched_trace_flattening_drops_padding_and_preserves_sample_ids():
    torch = pytest.importorskip("torch")
    from engram.training.controller_distillation import (
        _controller_trace_arrays,
    )

    batch, tokens, stages, width = 2, 3, 2, 4
    input_ids = torch.tensor([[11, 12, 13], [21, 22, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    initial = torch.arange(batch * tokens * width, dtype=torch.float32).reshape(
        batch, tokens, width
    )
    layer_inputs = {0: initial, 1: initial + 1.0}
    layer_outputs = {0: initial + 1.0, 1: initial + 2.0}
    attention_outputs = {
        stage: torch.full_like(initial, 0.1 * (stage + 1)) for stage in range(stages)
    }
    mlp_outputs = {
        stage: torch.full_like(initial, 0.2 * (stage + 1)) for stage in range(stages)
    }

    arrays = _controller_trace_arrays(
        torch,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sample_ids=[7, 9],
        layer_inputs=layer_inputs,
        layer_outputs=layer_outputs,
        attention_outputs=attention_outputs,
        mlp_outputs=mlp_outputs,
        layer_count=stages,
        hidden_size=width,
    )

    np.testing.assert_array_equal(arrays["sample_id"], [7, 7, 7, 9, 9])
    np.testing.assert_array_equal(arrays["token_id"], [11, 12, 13, 21, 22])
    np.testing.assert_array_equal(arrays["token_position"], [0, 1, 2, 0, 1])
    assert arrays["teacher_states"].shape == (5, 3, width)
    assert arrays["semantic_outputs"].shape == (5, 2, width)
    rms = np.sqrt(
        np.mean(np.square(arrays["teacher_states"].astype(np.float32)), axis=-1)
    )
    np.testing.assert_allclose(rms, 1.0, atol=5e-4)


def test_batched_trace_can_carry_optional_causal_topk_targets():
    torch = pytest.importorskip("torch")
    from engram.training.controller_distillation import _controller_trace_arrays

    input_ids = torch.tensor([[11, 12, 13], [21, 22, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    initial = torch.ones((2, 3, 4), dtype=torch.float32)
    layer_inputs = {0: initial, 1: initial}
    layer_outputs = {0: initial, 1: initial}
    attention_outputs = {0: initial, 1: initial}
    mlp_outputs = {0: initial, 1: initial}
    top_ids = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.int64,
    )
    top_logits = top_ids.float() / 10.0
    targets = torch.tensor([[12, 13, -1], [22, 0, -1]], dtype=torch.int64)
    arrays = _controller_trace_arrays(
        torch,
        input_ids=input_ids,
        attention_mask=attention_mask,
        sample_ids=[7, 9],
        layer_inputs=layer_inputs,
        layer_outputs=layer_outputs,
        attention_outputs=attention_outputs,
        mlp_outputs=mlp_outputs,
        layer_count=2,
        hidden_size=4,
        causal_top_ids=top_ids,
        causal_top_logits=top_logits,
        causal_target_ids=targets,
    )
    assert arrays["causal_top_ids"].shape == (5, 2)
    assert arrays["causal_top_logits"].dtype == np.float16
    np.testing.assert_array_equal(arrays["causal_target_ids"], [12, 13, -1, 22, 0])


def test_causal_topk_loss_backpropagates_through_final_controller_state():
    torch = pytest.importorskip("torch")
    from engram.training.controller_distillation import (
        _TrajectoryArrays,
        _causal_topk_loss,
    )

    data = _TrajectoryArrays(
        token_embedding=np.zeros((2, 4), dtype=np.float32),
        teacher_states=np.zeros((2, 2, 4), dtype=np.float16),
        semantic_outputs=np.zeros((2, 1, 4), dtype=np.float16),
        episodic_outputs=np.zeros((2, 1, 4), dtype=np.float16),
        sample_id=np.arange(2, dtype=np.int64),
        manifest={},
        causal_top_ids=np.asarray([[0, 1], [2, 3]], dtype=np.int32),
        causal_top_logits=np.asarray([[2.0, 0.0], [1.0, -1.0]], dtype=np.float16),
        causal_target_ids=np.asarray([0, 2], dtype=np.int64),
    )
    state = torch.ones((2, 4), dtype=torch.float32, requires_grad=True)
    lm_head = torch.eye(4, dtype=torch.float32)
    norm = torch.ones(4, dtype=torch.float32)
    loss, metrics = _causal_topk_loss(
        torch,
        state,
        data,
        np.arange(2),
        "cpu",
        lm_head=lm_head,
        norm_weight=norm,
    )
    assert torch.isfinite(loss)
    assert set(metrics) == {"causal_topk_kl", "causal_target_ce"}
    loss.backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
