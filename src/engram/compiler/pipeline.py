from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from engram import __version__
from engram.controller import SharedRecurrentController
from engram.models import inspect_model, load_named_tensors
from engram.semantic.memory import build_semantic_package
from engram.vocabulary.ivf import VocabularyIVFIndex
from engram.utils import atomic_json, npy_file_metadata, sha256_file


def _save_controller(controller: SharedRecurrentController, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name, array in controller.tensors().items():
        np.save(target / f"{name}.npy", array, allow_pickle=False)
    atomic_json(target / "metadata.json", controller.metadata())


def compile_model(
    model: str | Path,
    out: str | Path,
    *,
    seed: int = 71,
    semantic_top_k: int = 8,
    semantic_candidates: int = 16,
    semantic_ivf_clusters: int = 32,
    semantic_ivf_probes: int = 4,
    semantic_ivf_iterations: int = 20,
    vocabulary_candidates: int = 32,
    vocabulary_ivf_clusters: int = 64,
    vocabulary_ivf_probes: int = 4,
    vocabulary_ivf_iterations: int = 20,
    local_window: int = 16,
    cycles: int = 2,
) -> Path:
    """Compile a runnable research package without retaining transformer layers."""
    if semantic_ivf_clusters <= 0 or semantic_ivf_iterations <= 0:
        raise ValueError("semantic IVF clusters and iterations must be positive")
    if (
        vocabulary_candidates <= 0
        or vocabulary_ivf_clusters <= 0
        or vocabulary_ivf_iterations <= 0
    ):
        raise ValueError("vocabulary candidates, IVF clusters, and iterations must be positive")
    source = inspect_model(model)
    model_path = Path(source.model_path)
    target = Path(out)
    compile_options = {
        "seed": seed,
        "semantic_top_k": semantic_top_k,
        "semantic_candidates": semantic_candidates,
        "semantic_ivf_clusters": semantic_ivf_clusters,
        "semantic_ivf_probes": semantic_ivf_probes,
        "semantic_ivf_iterations": semantic_ivf_iterations,
        "vocabulary_candidates": vocabulary_candidates,
        "vocabulary_ivf_clusters": vocabulary_ivf_clusters,
        "vocabulary_ivf_probes": vocabulary_ivf_probes,
        "vocabulary_ivf_iterations": vocabulary_ivf_iterations,
        "local_window": local_window,
        "cycles": cycles,
    }
    existing_manifest = target / "manifest.json"
    if existing_manifest.is_file():
        existing = json.loads(existing_manifest.read_text())
        if existing.get("source_model_hash") != source.source_hash:
            raise ValueError("output package belongs to a different source model; choose a new --out")
        if existing.get("compile_options") != compile_options:
            raise ValueError("compile options changed; choose a new --out to preserve the existing package")
        damaged = [
            relative
            for relative, descriptor in existing.get("files", {}).items()
            if not (target / relative).is_file()
            or sha256_file(target / relative) != descriptor["sha256"]
        ]
        if damaged:
            raise ValueError(f"existing package is corrupt or incomplete: {damaged[:4]}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    metrics_dir = target / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    stage_manifest_path = metrics_dir / "stage_manifest.json"
    stages: list[dict[str, Any]] = []

    def completed(name: str, detail: dict[str, Any] | None = None) -> None:
        stages.append({"stage": name, "status": "completed", "detail": detail or {}})
        atomic_json(
            stage_manifest_path,
            {"version": 1, "source_model_hash": source.source_hash, "seed": seed, "stages": stages},
        )

    completed("inspect_and_validate", {"model_type": source.model_type})
    names = ["model.embed_tokens.weight", "lm_head.weight"]
    extracted = load_named_tensors(model_path, names)
    embeddings_dir = target / "embeddings"
    vocabulary_dir = target / "vocabulary"
    embeddings_dir.mkdir(exist_ok=True)
    vocabulary_dir.mkdir(exist_ok=True)
    np.save(embeddings_dir / "token_embeddings.npy", extracted[names[0]], allow_pickle=False)
    np.save(vocabulary_dir / "embeddings.npy", extracted[names[1]], allow_pickle=False)
    vocabulary_norms = np.linalg.norm(extracted[names[1]], axis=1, keepdims=True)
    normalized_vocabulary = np.divide(
        extracted[names[1]],
        vocabulary_norms,
        out=np.zeros_like(extracted[names[1]]),
        where=vocabulary_norms > 0,
    )
    np.save(vocabulary_dir / "index.npy", normalized_vocabulary.astype(np.float32), allow_pickle=False)
    actual_vocabulary_clusters = min(vocabulary_ivf_clusters, source.vocab_size)
    vocabulary_ivf = VocabularyIVFIndex.build(
        extracted[names[1]],
        num_clusters=actual_vocabulary_clusters,
        iterations=vocabulary_ivf_iterations,
    )
    vocabulary_ivf.save(vocabulary_dir / "ivf")
    if vocabulary_ivf_probes <= 0 or vocabulary_ivf_probes > actual_vocabulary_clusters:
        raise ValueError("vocabulary_ivf_probes must be within the compiled IVF cluster count")
    tokenizer_dir = target / "tokenizer"
    tokenizer_dir.mkdir(exist_ok=True)
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    copied_tokenizer = []
    for file_name in tokenizer_files:
        source_file = model_path / file_name
        if source_file.is_file():
            shutil.copy2(source_file, tokenizer_dir / file_name)
            copied_tokenizer.append(file_name)
    atomic_json(
        tokenizer_dir / "metadata.json",
        {
            "files": copied_tokenizer,
            "fallback": "fixture_byte_mapping" if not copied_tokenizer else None,
        },
    )
    completed("extract_runtime_tensors", {"tokenizer_files": copied_tokenizer})

    build_semantic_package(
        model_path,
        target,
        include_reference=False,
        ivf_clusters=semantic_ivf_clusters,
        ivf_iterations=semantic_ivf_iterations,
    )
    actual_ivf_clusters = min(semantic_ivf_clusters, source.intermediate_size)
    if semantic_ivf_probes <= 0 or semantic_ivf_probes > actual_ivf_clusters:
        raise ValueError("semantic_ivf_probes must be within the compiled IVF cluster count")
    completed(
        "build_semantic_memory",
        {"quantized_only": True, "router": "joint_key_ivf"},
    )

    width = source.hidden_size
    controller = SharedRecurrentController.initialize(
        input_dim=3 * width,
        state_dim=width,
        num_stages=source.num_hidden_layers,
        adapter_rank=min(2, width),
        seed=seed,
        weight_scale=0.02,
    )
    tensors = controller.tensors()
    tensors["input_kernel"].fill(0.0)
    for block in range(3):
        start = block * width
        tensors["input_kernel"][start : start + width, 2 * width :] = np.eye(width) / 3.0
    tensors["stage_embeddings"].fill(0.0)
    controller = SharedRecurrentController.from_state(controller.metadata(), tensors)
    _save_controller(controller, target / "controller")
    completed("initialize_shared_controller", {"initializer": "shared_residual_average_fixture_baseline"})

    episodic_dir = target / "episodic"
    episodic_dir.mkdir(exist_ok=True)
    atomic_json(
        episodic_dir / "config.json",
        {
            "local_window": local_window,
            "retrieval_capacity": 1024,
            "retrieval_candidates": 16,
            "retrieval_top_k": 4,
            "decay": 0.99,
        },
    )
    (target / "transitions").mkdir(exist_ok=True)
    atomic_json(target / "transitions" / "config.json", {"capacity": 4096, "similarity_radius": 0.02})
    (target / "corrections").mkdir(exist_ok=True)
    atomic_json(
        target / "corrections" / "capsules.json",
        {"version": 1, "capsules": [], "fallback": "extra_cycle_and_expanded_search"},
    )
    completed("build_runtime_policies")

    files = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and path != target / "manifest.json":
            relative = path.relative_to(target).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if path.suffix == ".npy":
                files[relative].update(npy_file_metadata(path))
    manifest = {
        "format": "engram-model",
        "version": 1,
        "engram_version": __version__,
        "source_model_hash": source.source_hash,
        "source_architecture": source.architecture,
        "fixture_only": source.model_type == "llama" and bool(
            json.loads((model_path / "config.json").read_text()).get("engram_fixture")
        ),
        "hidden_size": width,
        "vocab_size": source.vocab_size,
        "num_semantic_layers": source.num_hidden_layers,
        "runtime": {
            "cycles": cycles,
            "semantic_top_k": min(semantic_top_k, source.intermediate_size),
            "semantic_candidates": min(semantic_candidates, source.intermediate_size),
            "semantic_ivf_clusters": actual_ivf_clusters,
            "semantic_ivf_probes": semantic_ivf_probes,
            "vocabulary_candidates": min(vocabulary_candidates, source.vocab_size),
            "vocabulary_ivf_clusters": actual_vocabulary_clusters,
            "vocabulary_ivf_probes": vocabulary_ivf_probes,
        },
        "compile_options": compile_options,
        "files": files,
        "does_not_require_source_transformer": True,
    }
    atomic_json(target / "manifest.json", manifest)
    completed("validate_compiled_package", {"files": len(files)})
    atomic_json(
        metrics_dir / "conversion_report.json",
        {
            "status": "runnable_fixture_baseline" if manifest["fixture_only"] else "undistilled_research_baseline",
            "source_model_hash": source.source_hash,
            "quality_claim": None,
            "limitations": [
                "controller is initialized, not end-to-end distilled",
                "semantic IVF quality requires trained-model evaluation",
                "semantic IVF still scans all coarse centroids before posted records",
            ],
        },
    )
    stage_coverage = [
        (1, "inspect_and_validate_source", "completed"),
        (2, "extract_tokenizer_config_embeddings_norm_weights", "completed"),
        (3, "capture_teacher_traces", "external_not_run"),
        (4, "analyze_ffn_sparsity", "external_not_run"),
        (5, "build_semantic_keys_values", "completed"),
        (6, "fit_background_operators", "fallback_none"),
        (7, "construct_semantic_router", "completed_joint_key_ivf"),
        (8, "analyze_attention_heads", "synthetic_only"),
        (9, "distill_recurrent_attention", "fallback_initialized"),
        (10, "build_episodic_indexes", "completed_runtime_bounded_index"),
        (11, "factor_shared_controller", "fallback_shared_initializer"),
        (12, "distill_controller_states", "not_run"),
        (13, "end_to_end_logit_distillation", "not_run"),
        (14, "build_vocabulary_index", "completed_normalized_ivf"),
        (15, "populate_transition_cache", "fallback_online_population"),
        (16, "fit_correction_capsules", "fallback_empty_escalation_policy"),
        (17, "quantize_and_pack", "completed"),
        (18, "validate_compiled_package", "completed"),
        (19, "benchmark", "external_command"),
    ]
    atomic_json(
        stage_manifest_path,
        {
            "version": 1,
            "source_model_hash": source.source_hash,
            "seed": seed,
            "stages": [
                {"number": number, "stage": name, "status": status}
                for number, name, status in stage_coverage
            ],
        },
    )
    # Stage/report files changed after the first manifest snapshot; seal final checksums now.
    files = {}
    for path in sorted(target.rglob("*")):
        if path.is_file() and path != target / "manifest.json":
            relative = path.relative_to(target).as_posix()
            files[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            if path.suffix == ".npy":
                files[relative].update(npy_file_metadata(path))
    manifest["files"] = files
    atomic_json(target / "manifest.json", manifest)
    return target
