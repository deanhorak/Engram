import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from engram.compiler.olmoe_native import compile_olmoe_native_package
from engram.evaluation.olmoe_native_causal import (
    _THRESHOLDS,
    _position_metrics,
    _threaded_expert_forward,
    _write_npz_atomic,
    evaluate_native_olmoe_causal,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


def test_position_metrics_are_exact_for_identical_diagnostics():
    logits = np.array([0.25, -1.0, 2.0], dtype=np.float32)
    hidden = np.array([1.0, -2.0], dtype=np.float32)

    metrics = _position_metrics(logits, logits, hidden, hidden, 1)

    assert abs(metrics["kl"]) < 1e-15
    assert metrics["top1_match"]
    assert abs(metrics["target_nll_delta"]) < 1e-15
    assert metrics["hidden_relative_l2"] == 0.0


def test_threaded_expert_forward_preserves_source_reduction_order():
    torch = pytest.importorskip("torch")

    class Experts:
        num_experts = 4

        def __init__(self):
            generator = torch.Generator().manual_seed(73)
            self.gate_up_proj = torch.randn(
                4,
                6,
                5,
                generator=generator,
                dtype=torch.bfloat16,
            )
            self.down_proj = torch.randn(
                4,
                5,
                3,
                generator=generator,
                dtype=torch.bfloat16,
            )
            self.act_fn = torch.nn.functional.silu

    experts = Experts()
    hidden = torch.randn(
        7,
        5,
        generator=torch.Generator().manual_seed(79),
        dtype=torch.bfloat16,
    )
    indices = torch.tensor(
        [[0, 1], [3, 0], [2, 1], [3, 2], [1, 0], [2, 3], [0, 2]],
        dtype=torch.int64,
    )
    weights = torch.softmax(
        torch.randn(
            7,
            2,
            generator=torch.Generator().manual_seed(83),
        ),
        dim=-1,
    )
    expected = torch.zeros_like(hidden)
    mask = torch.nn.functional.one_hot(indices, num_classes=4).permute(2, 1, 0)
    for expert_index in range(4):
        top_k_position, token_index = torch.where(mask[expert_index])
        state = hidden[token_index]
        gate, up = torch.nn.functional.linear(
            state,
            experts.gate_up_proj[expert_index],
        ).chunk(2, dim=-1)
        value = torch.nn.functional.silu(gate) * up
        value = torch.nn.functional.linear(
            value,
            experts.down_proj[expert_index],
        )
        value *= weights[token_index, top_k_position, None]
        expected.index_add_(0, token_index, value.to(expected.dtype))

    with ThreadPoolExecutor(max_workers=3) as executor:
        actual = _threaded_expert_forward(
            experts,
            hidden,
            indices,
            weights,
            executor,
        )

    assert torch.equal(actual, expected)


def test_complete_native_olmoe_causal_confirmation_fixture(tmp_path):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(
        tmp_path / "model",
        num_experts=64,
        num_experts_per_token=1,
    )
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "x": 1}, unk_token="[UNK]"))
    tokenizer.save(str(model / "tokenizer.json"))
    q7 = repack_olmoe_q7_model(model, tmp_path / "model.q7", group_size=8)
    non_mlp = tmp_path / "non_mlp.safetensors"
    repack_olmoe_non_mlp_weights(model, non_mlp)
    package = tmp_path / "package"
    compiled = compile_olmoe_native_package(
        model,
        q7,
        non_mlp,
        package,
        kernel_threads=2,
    )
    manifest_hash = compiled["manifest_sha256"]
    inputs = [
        [1 if position != sequence else 2 + sequence for position in range(33)]
        for sequence in range(8)
    ]
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps({"input_ids": row}) + "\n" for row in inputs),
        encoding="utf-8",
    )
    logits = []
    hidden = []
    targets = []
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
    ) as runtime:
        for sequence in inputs:
            runtime.reset()
            for position, token_id in enumerate(sequence[:-1]):
                runtime.runtime.forward([token_id])
                state, scores = runtime.runtime.last_diagnostics()
                hidden.append(state)
                logits.append(scores)
                targets.append(sequence[position + 1])
    arrays_path = tmp_path / "teacher.npz"
    _write_npz_atomic(
        arrays_path,
        logits=np.asarray(logits, dtype=np.float32),
        hidden=np.asarray(hidden, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
    )
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_causal_reference",
        "source": {
            "model": str(model),
            "revision": None,
            "config_sha256": sha256_file(model / "config.json"),
            "index_sha256": sha256_file(model / "model.safetensors.index.json"),
            "adapter": "olmoe_sparse_expert_v1",
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256_file(dataset),
            "sequences": 8,
            "tokens_per_sequence": 33,
            "prediction_positions": 256,
            "input_identity": sha256_json(inputs),
            "input_ids": inputs,
        },
        "configuration": {"weights_modified": False},
        "arrays": {
            "path": str(arrays_path),
            "sha256": sha256_file(arrays_path),
        },
    }
    reference_path = tmp_path / "reference.json"
    atomic_json(reference_path, reference)
    protocol = {
        "schema_version": 1,
        "experiment": "olmoe_native_package_causal_confirmation",
        "status": "frozen_before_candidate_execution",
        "source_revision": None,
        "source_config_sha256": sha256_file(model / "config.json"),
        "source_index_sha256": sha256_file(model / "model.safetensors.index.json"),
        "source_shard_sha256": {"weights.npz": sha256_file(model / "weights.npz")},
        "package_manifest_sha256": manifest_hash,
        "native_library_sha256": sha256_file(library),
        "dataset_sha256": sha256_file(dataset),
        "input_identity": sha256_json(inputs),
        "teacher_reference_sha256": sha256_file(reference_path),
        "teacher_arrays_sha256": sha256_file(arrays_path),
        "sequences": 8,
        "tokens_per_sequence": 33,
        "model": {
            "layers": 2,
            "hidden_size": 16,
            "intermediate_size": 8,
            "experts": 64,
            "vocab_size": 32,
        },
        "thresholds": _THRESHOLDS,
        "post_window_uses_same_quality_thresholds": True,
        "scope": {"candidate_threads": 2},
        "evaluator_source_sha256": {
            "src/engram/evaluation/olmoe_native_causal.py": sha256_file(
                Path("src/engram/evaluation/olmoe_native_causal.py")
            )
        },
    }
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, protocol)

    report = evaluate_native_olmoe_causal(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        dataset=dataset,
        teacher_reference=reference_path,
        teacher_arrays=arrays_path,
        protocol=protocol_path,
        protocol_sha256=sha256_file(protocol_path),
        out=tmp_path / "result.json",
    )

    assert report["gate_passed"]
    assert report["metrics"]["teacher_top1_agreement"] == 1.0
    assert report["metrics"]["final_hidden_relative_l2"] == 0.0
    assert report["configuration"]["candidate_threads"] == 2
    assert all(report["post_run_authentication"].values())
    assert report["artifacts"]["evaluator_source_sha256"]
    assert (
        report["position_splits"]["bounded_retrieval_positions_16_31"][
            "prediction_positions"
        ]
        == 128
    )
