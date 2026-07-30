from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import engram.evaluation.olmoe_retrieval_episodic_query_features as query
from engram.utils import sha256_file, sha256_json


def _shape() -> query._QueryShape:
    return query._QueryShape(layers=2, query_heads=2, head_dimension=2)


def _weights(
    torch: Any,
    shape: query._QueryShape,
    *,
    query_dtype: Any | None = None,
) -> dict[str, Any]:
    hidden = shape.hidden_size
    dtype = torch.bfloat16
    tensors: dict[str, Any] = {
        "lm_head.weight": torch.zeros((5, hidden), dtype=dtype),
        "model.embed_tokens.weight": torch.zeros((5, hidden), dtype=dtype),
        "model.norm.weight": torch.ones(hidden, dtype=dtype),
    }
    projection_values = (
        torch.eye(hidden, dtype=torch.float32),
        torch.tensor(
            [
                [1.0, 0.5, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.5, 1.0],
                [-1.0, 0.0, 0.0, 0.5],
            ],
            dtype=torch.float32,
        ),
    )
    norm_values = (
        torch.tensor([1.0, 2.0, 0.5, -1.0], dtype=torch.float32),
        torch.tensor([0.5, -1.0, 2.0, 1.0], dtype=torch.float32),
    )
    for layer in range(shape.layers):
        base = f"model.layers.{layer}"
        attention = f"{base}.self_attn"
        tensors[f"{base}.input_layernorm.weight"] = torch.ones(
            hidden, dtype=dtype
        )
        tensors[f"{base}.post_attention_layernorm.weight"] = torch.ones(
            hidden, dtype=dtype
        )
        tensors[f"{attention}.k_norm.weight"] = torch.ones(hidden, dtype=dtype)
        tensors[f"{attention}.q_norm.weight"] = norm_values[layer].to(
            dtype=query_dtype or dtype
        )
        for projection in ("k_proj", "o_proj", "v_proj"):
            tensors[f"{attention}.{projection}.weight"] = torch.zeros(
                (hidden, hidden), dtype=dtype
            )
        tensors[f"{attention}.q_proj.weight"] = projection_values[layer].to(
            dtype=query_dtype or dtype
        )
    return tensors


def _fixture(
    tmp_path: Path,
    *,
    query_dtype: Any | None = None,
    remove: str | None = None,
) -> tuple[Path, np.ndarray, np.ndarray]:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    shape = _shape()
    tensors = _weights(torch, shape, query_dtype=query_dtype)
    if remove is not None:
        tensors.pop(remove)
    path = tmp_path / "non_mlp.safetensors"
    safetensors.save_file(tensors, path)
    input_norm = np.ascontiguousarray(
        np.asarray(
            [
                [
                    [[1.0, 2.0, 3.0, 4.0], [-1.0, 2.0, 0.5, 3.0]],
                    [[2.0, -1.0, 0.5, 1.0], [4.0, 1.0, -2.0, 0.5]],
                ]
            ],
            dtype=np.float32,
        )
    )
    positions = np.ascontiguousarray(np.asarray([96, 97], dtype=np.int64))
    return path, input_norm, positions


def _run(
    path: Path,
    input_norm: np.ndarray,
    positions: np.ndarray,
    *,
    device: str = "cpu",
    expected_weight_tensor_sha256: dict[str, str] | None = None,
) -> query.QueryFeatureResult:
    return query._reconstruct_authenticated_query_features(
        non_mlp_path=path,
        non_mlp_sha256=sha256_file(path),
        input_norm=input_norm,
        input_norm_sha256=query.tensor_sha256(input_norm),
        positions=positions,
        positions_sha256=query.tensor_sha256(positions),
        device=device,
        expected_weight_tensor_sha256=expected_weight_tensor_sha256,
        shape=_shape(),
    )


def _reference(
    path: Path,
    input_norm: np.ndarray,
) -> np.ndarray:
    torch = pytest.importorskip("torch")
    safe_open = pytest.importorskip("safetensors").safe_open
    shape = _shape()
    output = np.empty(
        (
            input_norm.shape[0],
            input_norm.shape[1],
            shape.layers,
            shape.query_heads,
            shape.head_dimension,
        ),
        dtype=np.float32,
    )
    with safe_open(path, framework="pt", device="cpu") as handle:
        for layer in range(shape.layers):
            projection = (
                handle.get_tensor(query._query_projection_name(layer))
                .to(torch.float32)
                .numpy()
            )
            norm = (
                handle.get_tensor(query._query_norm_name(layer))
                .to(torch.float32)
                .numpy()
            )
            projected = input_norm[:, :, layer, :] @ projection.T
            inverse = 1.0 / np.sqrt(
                np.sum(projected * projected, axis=-1, dtype=np.float32)
                / np.float32(shape.hidden_size)
                + np.float32(query._RMS_EPSILON)
            )
            normalized = (projected * inverse[..., None]) * norm
            output[:, :, layer, :, :] = normalized.reshape(
                input_norm.shape[0],
                input_norm.shape[1],
                shape.query_heads,
                shape.head_dimension,
            )
    return output


def test_cpu_reconstruction_is_deterministic_and_contract_bound(
    tmp_path: Path,
) -> None:
    path, input_norm, positions = _fixture(tmp_path)
    weights = query.query_weight_contract(
        path,
        non_mlp_sha256=sha256_file(path),
        layers=2,
        hidden_size=4,
    )
    first = _run(path, input_norm, positions)
    second = _run(
        path,
        input_norm,
        positions,
        expected_weight_tensor_sha256=dict(first.weight_tensor_sha256),
    )

    np.testing.assert_array_equal(first.queries, second.queries)
    np.testing.assert_allclose(
        first.queries,
        _reference(path, input_norm),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert first.queries.shape == (1, 2, 2, 2, 2)
    assert first.queries.dtype == np.float32
    assert first.queries.flags.c_contiguous
    assert first.queries.flags.writeable is False
    assert (
        first.tensor_sha256["post_qnorm_pre_rope_queries"]
        == query.tensor_sha256(np.ascontiguousarray(first.queries))
    )
    assert first.contract["source"]["sha256"] == sha256_file(path)
    assert first.contract["source"]["exact_tensor_count"] == 19
    assert weights["exact_non_mlp_tensor_count"] == 19
    assert weights["query_tensor_count"] == 4
    assert weights["tensor_sha256"] == first.weight_tensor_sha256
    assert (
        weights["query_weight_root_sha256"]
        == first.contract["source"]["query_weight_root_sha256"]
    )
    assert first.contract["derivation"] == {
        "projection": "input_norm @ q_proj.weight.T",
        "projection_weight_layout": "output_input",
        "projection_accumulator_dtype": "float32",
        "query_normalization": "flattened_hidden_rms",
        "query_normalization_width": 4,
        "query_normalization_epsilon": 1.0e-5,
        "query_normalization_weight_dtype": "BF16_decoded_to_float32",
        "rope_applied": False,
    }
    assert first.contract["input"]["positions"]["minimum"] == 96
    assert first.contract["input"]["positions"]["maximum"] == 97
    assert first.contract["output"]["layout"] == (
        "record_read_layer_query_head_head_dimension"
    )
    assert first.contract_sha256 == sha256_json(first.contract)
    assert first.contract_sha256 != second.contract_sha256
    assert second.contract["source"]["caller_supplied_query_weight_hashes"] is True


def test_per_record_positions_are_bound_but_not_applied(tmp_path: Path) -> None:
    path, input_norm, positions = _fixture(tmp_path)
    shared = _run(path, input_norm, positions)
    position_grid = np.ascontiguousarray(positions[None, :])
    per_record = _run(path, input_norm, position_grid)
    np.testing.assert_array_equal(shared.queries, per_record.queries)
    assert (
        shared.tensor_sha256["position_grid"]
        == per_record.tensor_sha256["position_grid"]
    )
    assert shared.tensor_sha256["positions"] == per_record.tensor_sha256["positions"]
    assert shared.contract["input"]["positions"]["supplied_shape"] == [2]
    assert per_record.contract["input"]["positions"]["supplied_shape"] == [1, 2]
    assert shared.contract_sha256 != per_record.contract_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("input_dtype", "input_norm"),
        ("input_noncontiguous", "input_norm"),
        ("input_shape", "input_norm"),
        ("input_nonfinite", "input_norm"),
        ("position_dtype", "positions"),
        ("position_shape", "positions"),
        ("position_order", "strictly increasing"),
        ("input_hash", "input_norm tensor SHA-256"),
        ("position_hash", "positions tensor SHA-256"),
    ],
)
def test_inputs_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path, input_norm, positions = _fixture(tmp_path)
    supplied_input = input_norm
    supplied_positions = positions
    input_hash = query.tensor_sha256(input_norm)
    position_hash = query.tensor_sha256(positions)
    if mutation == "input_dtype":
        supplied_input = input_norm.astype(np.float64)
    elif mutation == "input_noncontiguous":
        supplied_input = input_norm[..., ::-1]
    elif mutation == "input_shape":
        supplied_input = np.ascontiguousarray(input_norm[:, :, :1, :])
    elif mutation == "input_nonfinite":
        supplied_input = input_norm.copy()
        supplied_input[0, 0, 0, 0] = np.nan
    elif mutation == "position_dtype":
        supplied_positions = positions.astype(np.int32)
    elif mutation == "position_shape":
        supplied_positions = np.ascontiguousarray(positions[:, None])
    elif mutation == "position_order":
        supplied_positions = np.ascontiguousarray(positions[::-1])
    elif mutation == "input_hash":
        input_hash = "0" * 64
    else:
        position_hash = "0" * 64
    with pytest.raises(query.OLMoEQueryFeatureError, match=message):
        query._reconstruct_authenticated_query_features(
            non_mlp_path=path,
            non_mlp_sha256=sha256_file(path),
            input_norm=supplied_input,
            input_norm_sha256=input_hash,
            positions=supplied_positions,
            positions_sha256=position_hash,
            device="cpu",
            expected_weight_tensor_sha256=None,
            shape=_shape(),
        )


def test_non_mlp_binding_inventory_and_bf16_contract_fail_closed(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    path, input_norm, positions = _fixture(tmp_path)
    with pytest.raises(query.OLMoEQueryFeatureError, match="SHA-256 changed"):
        query._reconstruct_authenticated_query_features(
            non_mlp_path=path,
            non_mlp_sha256="0" * 64,
            input_norm=input_norm,
            input_norm_sha256=query.tensor_sha256(input_norm),
            positions=positions,
            positions_sha256=query.tensor_sha256(positions),
            device="cpu",
            expected_weight_tensor_sha256=None,
            shape=_shape(),
        )

    missing_dir = tmp_path / "missing"
    missing_dir.mkdir()
    missing_path, missing_input, missing_positions = _fixture(
        missing_dir,
        remove=query._query_norm_name(1),
    )
    with pytest.raises(query.OLMoEQueryFeatureError, match="inventory is not exact"):
        _run(missing_path, missing_input, missing_positions)

    dtype_dir = tmp_path / "dtype"
    dtype_dir.mkdir()
    dtype_path, dtype_input, dtype_positions = _fixture(
        dtype_dir,
        query_dtype=torch.float32,
    )
    with pytest.raises(query.OLMoEQueryFeatureError, match="tensor .* is invalid"):
        _run(dtype_path, dtype_input, dtype_positions)


def test_symlink_and_weight_tensor_hashes_fail_closed(tmp_path: Path) -> None:
    path, input_norm, positions = _fixture(tmp_path)
    linked = tmp_path / "linked.safetensors"
    linked.symlink_to(path)
    with pytest.raises(query.OLMoEQueryFeatureError, match="symlink"):
        _run(linked, input_norm, positions)
    with pytest.raises(query.OLMoEQueryFeatureError, match="query-weight tensor"):
        _run(
            path,
            input_norm,
            positions,
            expected_weight_tensor_sha256={
                query._query_projection_name(0): "0" * 64
            },
        )


def test_deterministic_context_restores_process_settings(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    path, input_norm, positions = _fixture(tmp_path)
    deterministic = torch.are_deterministic_algorithms_enabled()
    precision = torch.get_float32_matmul_precision()
    cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    _run(path, input_norm, positions)
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cuda.matmul.allow_tf32 is cuda_tf32
    assert torch.backends.cudnn.allow_tf32 is cudnn_tf32


def test_cuda_reconstruction_is_repeatable_when_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if query.os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        pytest.skip("deterministic CUBLAS workspace configuration is unavailable")
    path, input_norm, positions = _fixture(tmp_path)
    first = _run(path, input_norm, positions, device="cuda")
    second = _run(path, input_norm, positions, device="cuda")
    np.testing.assert_array_equal(first.queries, second.queries)
    np.testing.assert_allclose(
        first.queries,
        _run(path, input_norm, positions, device="cpu").queries,
        atol=2.0e-6,
        rtol=0.0,
    )
    assert first.contract["execution"]["device"].startswith("cuda:")
    assert first.contract["execution"]["cuda_matmul_tf32"] is False
    assert first.contract["execution"]["cudnn_tf32"] is False
