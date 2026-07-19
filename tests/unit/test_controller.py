import numpy as np
import pytest

from engram.controller import SharedRecurrentController


def test_fixed_cycles_share_base_weights_and_report_deterministic_metrics():
    controller = SharedRecurrentController.initialize(
        input_dim=5, state_dim=7, num_stages=3, adapter_rank=2, seed=23
    )
    initial = np.zeros(7)
    supplied = np.linspace(-0.5, 0.5, 5)

    first = controller.run(initial, supplied, mode="fixed", fixed_cycles=4)
    second = controller.run(initial, supplied, mode="fixed", fixed_cycles=4)

    assert controller.input_kernel.ndim == 2
    assert controller.recurrent_kernel.ndim == 2
    assert first.cycles == 4
    assert first.residual == first.residual_history[-1]
    assert len(first.confidence_history) == 4
    np.testing.assert_array_equal(first.state, second.state)
    assert first.residual_history == second.residual_history


def test_adaptive_early_exit_and_low_confidence_extra_cycle_request():
    state_dim = 4
    controller = SharedRecurrentController(
        input_kernel=np.zeros((3, 3 * state_dim)),
        recurrent_kernel=np.zeros((state_dim, 3 * state_dim)),
        bias=np.zeros(3 * state_dim),
        stage_embeddings=np.zeros((2, state_dim)),
        adapter_down=np.zeros((2, state_dim, 0)),
        adapter_up=np.zeros((2, 0, state_dim)),
    )
    initial = np.zeros(state_dim)
    supplied = np.zeros(3)

    easy = controller.run(
        initial,
        supplied,
        mode="adaptive",
        min_cycles=2,
        max_cycles=6,
        residual_tolerance=0.0,
        confidence_threshold=0.9,
        confidence=1.0,
    )
    difficult = controller.run(
        initial,
        supplied,
        mode="adaptive",
        min_cycles=1,
        max_cycles=3,
        residual_tolerance=0.0,
        confidence_threshold=0.9,
        confidence=0.2,
        requested_extra_cycles=2,
    )

    assert easy.cycles == 2
    assert easy.converged is True
    assert difficult.cycles == 5
    assert difficult.cycle_limit == 5
    assert difficult.requested_extra_cycles == 2
    assert difficult.converged is False


def test_stage_parameters_change_trajectory_and_long_run_stays_finite_and_bounded():
    controller = SharedRecurrentController.initialize(
        input_dim=4, state_dim=6, num_stages=2, adapter_rank=1, seed=8
    )
    state = np.zeros(6)
    supplied = np.ones(4)

    stage_zero = controller.step(state, supplied, stage=0)
    stage_one = controller.step(state, supplied, stage=1)
    assert not np.allclose(stage_zero, stage_one)

    result = controller.run(state, supplied, mode="fixed", fixed_cycles=500)
    assert np.all(np.isfinite(result.state))
    assert np.max(np.abs(result.state)) <= 1.0 + 1e-12


def test_serialized_state_round_trip_is_exact_and_validated():
    controller = SharedRecurrentController.initialize(
        input_dim=3, state_dim=5, num_stages=4, adapter_rank=2, seed=5
    )
    restored = SharedRecurrentController.from_state(controller.metadata(), controller.tensors())
    initial = np.arange(5, dtype=np.float64) / 10.0
    supplied = np.arange(3, dtype=np.float64) / 7.0

    np.testing.assert_array_equal(
        restored.run(initial, supplied, fixed_cycles=7).state,
        controller.run(initial, supplied, fixed_cycles=7).state,
    )
    assert restored.metadata() == controller.metadata()

    bad_metadata = dict(controller.metadata())
    bad_metadata["state_dim"] += 1
    with pytest.raises(ValueError, match="metadata does not match"):
        SharedRecurrentController.from_state(bad_metadata, controller.tensors())
