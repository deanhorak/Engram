from __future__ import annotations

import copy
import json

import pytest

from engram.evaluation.olmoe_expert_proxy_benchmark import (
    benchmark_frozen_olmoe_expert_proxy,
    load_frozen_olmoe_expert_layer,
    validate_frozen_olmoe_expert_proxy_report,
)


torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")


def _write_fixture(tmp_path, *, packed=False):
    model_path = tmp_path / "olmoe"
    model_path.mkdir()
    config = {
        "model_type": "olmoe",
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": False,
    }
    (model_path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(41)
    expert_tensors = {}
    for expert in range(config["num_experts"]):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        expert_tensors[f"{prefix}.gate_proj.weight"] = torch.randn(
            config["intermediate_size"],
            config["hidden_size"],
            generator=generator,
        )
        expert_tensors[f"{prefix}.up_proj.weight"] = torch.randn(
            config["intermediate_size"],
            config["hidden_size"],
            generator=generator,
        )
        expert_tensors[f"{prefix}.down_proj.weight"] = torch.randn(
            config["hidden_size"],
            config["intermediate_size"],
            generator=generator,
        )
    if packed:
        tensors = {
            "model.layers.0.mlp.experts.gate_up_proj": torch.stack(
                [
                    torch.cat(
                        (
                            expert_tensors[
                                f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"
                            ],
                            expert_tensors[
                                f"model.layers.0.mlp.experts.{expert}.up_proj.weight"
                            ],
                        ),
                        dim=0,
                    )
                    for expert in range(config["num_experts"])
                ]
            ),
            "model.layers.0.mlp.experts.down_proj": torch.stack(
                [
                    expert_tensors[
                        f"model.layers.0.mlp.experts.{expert}.down_proj.weight"
                    ]
                    for expert in range(config["num_experts"])
                ]
            ),
        }
    else:
        tensors = expert_tensors
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_torch.save_file(tensors, model_path / shard_name)
    index = {
        "metadata": {
            "total_size": sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
        },
        "weight_map": {name: shard_name for name in tensors},
    }
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(index),
        encoding="utf-8",
    )
    return model_path, tensors


@pytest.fixture(scope="module")
def durable_report(tmp_path_factory):
    root = tmp_path_factory.mktemp("olmoe-expert-benchmark")
    model_path, _ = _write_fixture(root)
    output = root / "result" / "benchmark.json"
    report = benchmark_frozen_olmoe_expert_proxy(
        model_path,
        out=output,
        layer=0,
        tokens=2,
        repeats=2,
        workers=(1, 2),
        seed=53,
    )
    return report, output


def test_stream_loader_materializes_only_selected_frozen_bf16_experts(tmp_path):
    model_path, tensors = _write_fixture(tmp_path)

    loaded = load_frozen_olmoe_expert_layer(model_path, layer=0)

    assert loaded.layer_index == 0
    assert loaded.hidden_size == 8
    assert loaded.intermediate_size == 6
    assert loaded.num_experts == 4
    assert loaded.top_k == 2
    assert loaded.model_info["source_layout"] == "legacy_per_expert"
    assert loaded.model_info["selected_tensor_keys"] == 12
    assert loaded.model_info["streamed_expert_slices"] == 12
    assert len(loaded.model_info["selected_source_state_sha256"]) == 64
    assert loaded.model_info["loaded_parameter_bytes"] == 4 * (2 * 6 * 8 + 8 * 6) * 2
    assert loaded.experts.gate_up_proj.dtype == torch.bfloat16
    assert loaded.experts.down_proj.dtype == torch.bfloat16
    assert not loaded.experts.gate_up_proj.requires_grad
    assert not loaded.experts.down_proj.requires_grad
    assert loaded.experts.gate_up_proj.grad is None
    assert loaded.experts.down_proj.grad is None
    for expert in range(4):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        assert torch.equal(
            loaded.experts.gate_up_proj[expert, :6],
            tensors[f"{prefix}.gate_proj.weight"].to(torch.bfloat16),
        )
        assert torch.equal(
            loaded.experts.gate_up_proj[expert, 6:],
            tensors[f"{prefix}.up_proj.weight"].to(torch.bfloat16),
        )
        assert torch.equal(
            loaded.experts.down_proj[expert],
            tensors[f"{prefix}.down_proj.weight"].to(torch.bfloat16),
        )


def test_stream_loader_accepts_exact_packed_expert_outer_shapes(tmp_path):
    model_path, tensors = _write_fixture(tmp_path, packed=True)

    loaded = load_frozen_olmoe_expert_layer(model_path, layer=0)

    assert loaded.model_info["source_layout"] == "packed_experts"
    assert loaded.model_info["selected_tensor_keys"] == 2
    assert loaded.model_info["streamed_expert_slices"] == 8
    assert torch.equal(
        loaded.experts.gate_up_proj,
        tensors["model.layers.0.mlp.experts.gate_up_proj"].to(torch.bfloat16),
    )
    assert torch.equal(
        loaded.experts.down_proj,
        tensors["model.layers.0.mlp.experts.down_proj"].to(torch.bfloat16),
    )


def test_stream_loader_rejects_packed_tensor_with_extra_expert(tmp_path):
    model_path, tensors = _write_fixture(tmp_path, packed=True)
    shard_path = model_path / "model-00001-of-00001.safetensors"
    tensors["model.layers.0.mlp.experts.gate_up_proj"] = torch.cat(
        (
            tensors["model.layers.0.mlp.experts.gate_up_proj"],
            tensors["model.layers.0.mlp.experts.gate_up_proj"][:1],
        )
    )
    safetensors_torch.save_file(tensors, shard_path)

    with pytest.raises(ValueError, match="packed expert tensor.*has shape"):
        load_frozen_olmoe_expert_layer(model_path, layer=0)


def test_benchmark_proves_exact_repeated_parity_and_atomically_writes_json(
    durable_report,
):
    report, output = durable_report

    assert report["schema_version"] == 2
    assert report["parity_passed"]
    assert report["evidence_passed"]
    assert report["status"] == "exact_parity_passed"
    assert report["authentication"]["passed"]
    assert all(report["authentication"]["checks"].values())
    assert set(report["source_sha256"]) == {
        "src/engram/evaluation/olmoe_expert_proxy.py",
        "src/engram/evaluation/olmoe_expert_proxy_benchmark.py",
    }
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    assert report["environment"]["transformers"]
    assert report["environment"]["safetensors"]
    assert (
        len(
            report["authentication"]["pre_run"]["sources"]["transformers_olmoe"][
                "sha256"
            ]
        )
        == 64
    )
    assert all(
        len(value) == 64 for value in report["model"]["loaded_state_sha256"].values()
    )
    assert all(
        len(shard["sha256"]) == 64 for shard in report["model"]["selected_shards"]
    )
    assert report["workload"]["active_experts"] == 4
    assert report["workload"]["assignments_per_expert"] == [1, 1, 1, 1]
    assert report["execution_contract"]["warmups_per_implementation"] == 1
    assert report["execution_contract"]["experts_implementation"] == "eager"
    assert not report["execution_contract"]["counterbalanced"]
    assert (
        report["execution_contract"]["timing_classification"]
        == "host_specific_non_counterbalanced_microbenchmark"
    )
    assert report["eager"]["passed"]
    assert report["eager"]["warmup"]["first_measured_parity"]["exact"]
    assert len(report["eager"]["timing"]["runs"]) == 2
    for candidate in report["proxy"]:
        assert candidate["passed"]
        assert candidate["frozen_expert_gradients_absent"]
        assert candidate["warmup"]["eager_parity"]["exact"]
        assert candidate["warmup"]["first_measured_parity"]["exact"]
        assert candidate["stats"]["patched_layers"] == 1
        assert candidate["stats"]["serial_forward_calls"] == 3
        assert candidate["stats"]["parallel_backward_calls"] == 3
        assert candidate["stats"]["expert_backward_tasks"] == 12
        assert candidate["stats"]["restored_layers"] == 1
        assert not candidate["stats"]["context_active"]
        assert candidate["stats"]["executor_shutdown"]
        assert all(candidate["stats_checks"].values())
        assert all(row["exact"] for row in candidate["eager_parity"])
        assert all(row["exact"] for row in candidate["repeat_parity"])
        assert candidate["timing"]["mean_total_wall_seconds"] > 0.0
        assert candidate["speedup_vs_eager"]["total_wall"] > 0.0
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert validate_frozen_olmoe_expert_proxy_report(output) == report
    assert not list(output.parent.glob(f".{output.name}.tmp-*"))


def _write_tampered_report(tmp_path, report, name):
    path = tmp_path / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_report_validator_rejects_authentication_tamper(
    tmp_path,
    durable_report,
):
    report, _ = durable_report
    tampered = copy.deepcopy(report)
    tampered["authentication"]["post_run"]["sources"]["benchmark"]["sha256"] = "0" * 64
    path = _write_tampered_report(tmp_path, tampered, "auth.json")

    with pytest.raises(ValueError, match="authentication.checks"):
        validate_frozen_olmoe_expert_proxy_report(path)


def test_report_validator_rejects_boolean_integer_coercion(
    tmp_path,
    durable_report,
):
    report, _ = durable_report
    tampered = copy.deepcopy(report)
    tampered["execution_contract"]["workers"][0] = True
    path = _write_tampered_report(tmp_path, tampered, "type.json")

    with pytest.raises(ValueError, match="must be an integer"):
        validate_frozen_olmoe_expert_proxy_report(path)


def test_report_validator_rejects_timing_and_speedup_tamper(
    tmp_path,
    durable_report,
):
    report, _ = durable_report
    timing_tamper = copy.deepcopy(report)
    timing_tamper["proxy"][0]["timing"]["runs"][0]["total"]["wall_seconds"] += 1.0
    timing_path = _write_tampered_report(tmp_path, timing_tamper, "timing.json")
    with pytest.raises(ValueError, match="total"):
        validate_frozen_olmoe_expert_proxy_report(timing_path)

    speedup_tamper = copy.deepcopy(report)
    speedup_tamper["proxy"][0]["speedup_vs_eager"]["total_wall"] += 1.0
    speedup_path = _write_tampered_report(
        tmp_path,
        speedup_tamper,
        "speedup.json",
    )
    with pytest.raises(ValueError, match="speedup_vs_eager.total_wall"):
        validate_frozen_olmoe_expert_proxy_report(speedup_path)


def test_report_validator_rejects_stats_and_parity_tamper(
    tmp_path,
    durable_report,
):
    report, _ = durable_report
    stats_tamper = copy.deepcopy(report)
    stats_tamper["proxy"][0]["stats"]["serial_forward_calls"] += 1
    stats_path = _write_tampered_report(tmp_path, stats_tamper, "stats.json")
    with pytest.raises(ValueError, match="stats_checks.serial_forward_calls"):
        validate_frozen_olmoe_expert_proxy_report(stats_path)

    parity_tamper = copy.deepcopy(report)
    parity_tamper["proxy"][0]["eager_parity"][0]["exact"] = False
    parity_path = _write_tampered_report(tmp_path, parity_tamper, "parity.json")
    with pytest.raises(ValueError, match="exact is inconsistent"):
        validate_frozen_olmoe_expert_proxy_report(parity_path)


def test_report_validator_rejects_loaded_state_and_decision_tamper(
    tmp_path,
    durable_report,
):
    report, _ = durable_report
    state_tamper = copy.deepcopy(report)
    state_tamper["authentication"]["post_loaded_state_sha256"]["gate_up_proj"] = (
        "0" * 64
    )
    state_path = _write_tampered_report(tmp_path, state_tamper, "state.json")
    with pytest.raises(ValueError, match="loaded_gate_up_unchanged"):
        validate_frozen_olmoe_expert_proxy_report(state_path)

    decision_tamper = copy.deepcopy(report)
    decision_tamper["decision"] = "proxy_exact_for_real_layer_performance_is_measured!"
    decision_path = _write_tampered_report(
        tmp_path,
        decision_tamper,
        "decision.json",
    )
    with pytest.raises(ValueError, match="decision is inconsistent"):
        validate_frozen_olmoe_expert_proxy_report(decision_path)


def test_loader_rejects_unsafe_shard_mapping(tmp_path):
    model_path, _ = _write_fixture(tmp_path)
    index_path = model_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    first = next(iter(index["weight_map"]))
    index["weight_map"][first] = "../outside.safetensors"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="local file name"):
        load_frozen_olmoe_expert_layer(model_path)
