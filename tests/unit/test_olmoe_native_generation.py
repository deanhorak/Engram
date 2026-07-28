import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from engram.compiler.olmoe_native import compile_olmoe_native_package
from engram.evaluation.olmoe_native_generation import (
    evaluate_native_olmoe_generation,
)
from engram.models.fixture import create_tiny_olmoe_fixture
from engram.models.olmoe_native import repack_olmoe_non_mlp_weights
from engram.models.olmoe_q7 import repack_olmoe_q7_model
from engram.runtime.olmoe_native import OLMoENativePackageRuntime
from engram.utils import atomic_json, sha256_file, sha256_json


def test_frozen_native_olmoe_generation_confirmation_and_protocol_authentication(
    tmp_path,
):
    library = Path("build/libengram_olmoe_token_runtime.so")
    if not library.is_file():
        pytest.skip("native OLMoE token runtime has not been built")
    model = create_tiny_olmoe_fixture(tmp_path / "model")
    vocabulary = {"[UNK]": 0, "x": 9}
    vocabulary.update({f"p{index}": index + 1 for index in range(8)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
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
    prompts = [f"p{index} x x x x x x x" for index in range(8)]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )

    reference_results = []
    encoded = []
    with OLMoENativePackageRuntime(
        package,
        manifest_sha256=manifest_hash,
        library=library,
    ) as runtime:
        for prompt in prompts:
            input_ids = runtime.tokenizer.encode(prompt).ids
            encoded.append(input_ids)
            runtime.reset()
            forced = [
                runtime.runtime.forward([token_id]).next_token for token_id in input_ids
            ]
            runtime.reset()
            generated = runtime.generate(prompt, max_new_tokens=4)
            reference_results.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids,
                    "teacher_forced_top1": forced,
                    "generated_token_ids": generated["generated_token_ids"],
                    "generated_text": generated["completion"],
                }
            )
    reference = {
        "schema_version": 1,
        "experiment": "olmoe_untouched_teacher_generation_reference",
        "source": {
            "model": str(model),
            "revision": None,
            "config_sha256": sha256_file(model / "config.json"),
            "index_sha256": sha256_file(model / "model.safetensors.index.json"),
            "adapter": "olmoe_sparse_expert_v1",
        },
        "prompt_suite": {
            "path": str(prompt_path),
            "sha256": sha256_file(prompt_path),
            "prompts": 8,
            "input_identity": sha256_json(encoded),
        },
        "configuration": {
            "max_new_tokens": 4,
            "greedy": True,
            "weights_modified": False,
        },
        "results": reference_results,
    }
    reference_path = tmp_path / "teacher.json"
    atomic_json(reference_path, reference)
    protocol = {
        "schema_version": 1,
        "experiment": "olmoe_native_package_generation_confirmation",
        "status": "frozen_before_candidate_execution",
        "source_revision": None,
        "source_config_sha256": sha256_file(model / "config.json"),
        "source_index_sha256": sha256_file(model / "model.safetensors.index.json"),
        "source_shard_sha256": {"weights.npz": sha256_file(model / "weights.npz")},
        "prompt_suite_sha256": sha256_file(prompt_path),
        "teacher_reference_sha256": sha256_file(reference_path),
        "package_manifest_sha256": manifest_hash,
        "native_library_sha256": sha256_file(library),
        "max_new_tokens": 4,
        "thresholds": {
            "minimum_prompts": 8,
            "minimum_prompt_positions": 60,
            "minimum_generated_reference_tokens": 32,
            "minimum_teacher_forced_top1_agreement": 0.90,
            "minimum_weighted_greedy_token_agreement": 0.90,
            "minimum_exact_prompt_fraction": 0.75,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    atomic_json(protocol_path, protocol)
    protocol_hash = sha256_file(protocol_path)

    report = evaluate_native_olmoe_generation(
        package=package,
        manifest_sha256=manifest_hash,
        library=library,
        prompts=prompt_path,
        teacher_reference=reference_path,
        protocol=protocol_path,
        protocol_sha256=protocol_hash,
        out=tmp_path / "result.json",
    )

    assert report["gate_passed"]
    assert report["summary"]["teacher_forced_top1_agreement"] == 1.0
    assert report["summary"]["weighted_greedy_token_agreement"] == 1.0
    assert report["checks"]["longest_prompt_reset_replay"]
    with pytest.raises(ValueError, match="protocol authentication"):
        evaluate_native_olmoe_generation(
            package=package,
            manifest_sha256=manifest_hash,
            library=library,
            prompts=prompt_path,
            teacher_reference=reference_path,
            protocol=protocol_path,
            protocol_sha256="0" * 64,
            out=tmp_path / "rejected.json",
        )
