from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_joint_gamma_oracle as joint


def _tiny_trace() -> dict[str, np.ndarray]:
    regular_mass = np.asarray([[[[0.75, 0.25]]]], dtype=np.float32)
    episodic_mass = np.float32(1.0) - regular_mass
    regular_mean = np.asarray(
        [[[[2.0, -1.0, 1.5, 3.0]]]],
        dtype=np.float32,
    )
    episodic_mean = np.asarray(
        [[[[4.0, 2.0, -2.0, 5.0]]]],
        dtype=np.float32,
    )
    regular = regular_mean.copy()
    episodic = episodic_mean.copy()
    regular[..., :2] *= regular_mass[..., 0, None]
    regular[..., 2:] *= regular_mass[..., 1, None]
    episodic[..., :2] *= episodic_mass[..., 0, None]
    episodic[..., 2:] *= episodic_mass[..., 1, None]
    base = regular + episodic
    return {
        "base_attention_output": base,
        "regular_component": regular,
        "episodic_component": episodic,
        "regular_mass": regular_mass,
        "episodic_mass": episodic_mass,
        "target_residual": np.asarray(
            [[[[0.5, -1.5, 2.0, 0.25]]]],
            dtype=np.float32,
        ),
    }


def test_candidate_coefficients_keep_exact_base_anchor_and_gamma_bounds() -> None:
    regular = np.asarray([[0.75, 0.25]], dtype=np.float32)
    episodic = np.float32(1.0) - regular
    candidates, lower, upper = joint._candidate_coefficients(regular, episodic)
    np.testing.assert_array_equal(candidates[..., 4, :], 0.0)
    np.testing.assert_array_equal(candidates[..., 0, 0], 1.0)
    np.testing.assert_array_equal(candidates[..., 0, 1], 0.0)
    expected_p8 = 8.0 * episodic / (regular + 8.0 * episodic)
    np.testing.assert_allclose(candidates[..., 7, 1], expected_p8, rtol=1.0e-7)
    np.testing.assert_array_equal(lower, 0.0)
    np.testing.assert_array_equal(upper[..., 0], 1.0)
    np.testing.assert_allclose(upper[..., 1], expected_p8, rtol=1.0e-7)


def test_qd_reconstruction_matches_finite_counterfactual_grid() -> None:
    arrays = _tiny_trace()
    for code in range(8):
        codes = np.full((1, 1, 1, 2), code, dtype=np.uint8)
        reconstructed = joint._reconstruct_pre_wo_delta(
            arrays,
            codes,
            query_heads=2,
        )
        direct = joint.mass._counterfactual_pre_wo(
            arrays["base_attention_output"],
            arrays["regular_component"],
            arrays["episodic_component"],
            arrays["regular_mass"],
            arrays["episodic_mass"],
            codes,
            query_heads=2,
        ).astype(np.float64)
        direct -= arrays["base_attention_output"].astype(np.float64)
        np.testing.assert_allclose(reconstructed, direct, rtol=1.0e-6, atol=2.0e-7)
    np.testing.assert_array_equal(
        joint._reconstruct_pre_wo_delta(
            arrays,
            np.full((1, 1, 1, 2), 4, dtype=np.uint8),
            query_heads=2,
        ),
        0.0,
    )


def test_joint_quadratic_matches_direct_post_wo_error() -> None:
    arrays = _tiny_trace()
    weight = np.asarray(
        [
            [1.0, 0.5, -0.25, 2.0],
            [-0.5, 1.5, 0.75, 0.25],
            [2.0, -1.0, 0.5, 0.75],
            [0.25, 0.75, -1.5, 1.0],
        ],
        dtype=np.float32,
    )
    inputs = joint.build_joint_quadratic_inputs(
        arrays,
        weight[None, ...],
        query_heads=2,
        row_batch_size=1,
    )
    assert inputs.batch_shape == (1, 1, 1)
    assert inputs.gram.shape == (1, 2, 2, 2, 2)
    assert inputs.linear.shape == (1, 2, 2)
    assert inputs.candidates.shape == (1, 2, 8, 2)
    assert inputs.base_component_reconstruction_max_abs == 0.0
    assert inputs.gram_factor_construction.endswith("gram=A.T@A")
    assert inputs.minimum_normalized_gram_eigenvalue >= -1.0e-10
    assert inputs.maximum_gram_asymmetry <= 1.0e-12
    assert inputs.float32_per_head_counterfactual_pre_wo_max_abs <= 2.0e-7
    assert inputs.float32_uniform_code_projected_max_abs <= 2.0e-6

    codes = np.asarray([[2, 7]], dtype=np.uint8)
    coefficients = np.take_along_axis(
        inputs.candidates,
        codes[..., None, None],
        axis=2,
    )[:, :, 0, :]
    flat = coefficients.reshape(1, -1)
    quadratic_error = (
        inputs.target_energy
        - 2.0 * np.einsum("ni,ni->n", inputs.linear.reshape(1, -1), flat)
        + np.einsum(
            "ni,nij,nj->n",
            flat,
            inputs.gram.reshape(1, 4, 4),
            flat,
        )
    )
    delta = joint._reconstruct_pre_wo_delta(
        arrays,
        codes.reshape(1, 1, 1, 2),
        query_heads=2,
    ).reshape(1, 4)
    projected = delta @ weight.astype(np.float64).T
    target = arrays["target_residual"].reshape(1, 4).astype(np.float64)
    direct_error = np.sum((target - projected) ** 2, axis=1)
    np.testing.assert_allclose(quadratic_error, direct_error, rtol=2.0e-13)

    base = inputs.candidates[:, :, 4, :].reshape(1, -1)
    base_error = (
        inputs.target_energy
        - 2.0 * np.einsum("ni,ni->n", inputs.linear.reshape(1, -1), base)
        + np.einsum(
            "ni,nij,nj->n",
            base,
            inputs.gram.reshape(1, 4, 4),
            base,
        )
    )
    np.testing.assert_array_equal(base_error, inputs.target_energy)


def test_projection_uses_each_head_input_block_and_native_orientation() -> None:
    basis = np.asarray(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ],
        dtype=np.float64,
    )
    weight = np.arange(1.0, 17.0, dtype=np.float64).reshape(4, 4)
    projected = joint._project_qd_basis(basis, weight)
    for head in range(2):
        block = weight[:, head * 2 : (head + 1) * 2]
        for feature in range(2):
            np.testing.assert_array_equal(
                projected[0, head, feature],
                basis[0, head, feature] @ block.T,
            )


def test_head_mass_failure_authentication_rejects_authorized_result() -> None:
    protocol = Path("/frozen/protocol.json")
    value = {
        "schema_version": joint.mass._SCHEMA_VERSION,
        "experiment": joint.mass._RESULT_EXPERIMENT,
        "status": "train_episodic_head_mass_oracle_gate_failed",
        "protocol": {
            "path": str(protocol),
            "sha256": joint._EXPECTED_HEAD_MASS_PROTOCOL_SHA256,
        },
        "scope": {
            "split": "train",
            "same_state_capacity_evidence_only": True,
            "development_outcomes_used": False,
            "confirmation_split_opened": False,
        },
        "oracle": {
            "passed": False,
            "metrics": {"global": {"recovery": -0.1}},
        },
        "decision": {
            "train_head_mass_capacity_gate_passed": False,
            "native_causal_integration_authorized": False,
            "development_authorized": False,
            "confirmation_authorized": False,
            "failure_scope": "fixed grid",
        },
        "post_run_authentication": {"all": True},
        "trace_manifest": {
            "sha256": joint._EXPECTED_HEAD_MASS_MANIFEST_SHA256,
            "shard_count": joint.mass._RECORDS,
            "shards": [{} for _ in range(joint.mass._RECORDS)],
        },
        "confirmation_split_opened": False,
    }
    authenticated = joint._validate_head_mass_failure(
        value,
        protocol_path=protocol,
    )
    assert authenticated["global_recovery"] == -0.1
    assert authenticated["confirmation_split_opened"] is False
    value["decision"]["development_authorized"] = True
    with pytest.raises(ValueError, match="failure contract"):
        joint._validate_head_mass_failure(value, protocol_path=protocol)
