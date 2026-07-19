from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from engram.controller import SharedRecurrentController
from engram.episodic import HybridEpisodicMemory
from engram.semantic.compressed import CompressedSemanticLayer
from engram.transitions import TransitionCache
from engram.vocabulary import VocabularyIndex
from engram.vocabulary.ivf import VocabularyIVFIndex
from engram.utils import sha256_file


@dataclass(frozen=True)
class GenerationToken:
    token_id: int
    cycles: int
    semantic_records: int
    semantic_candidates: int
    semantic_proxy_records: int
    semantic_probed_clusters: int
    episodic_retrievals: int
    vocabulary_candidates: int
    vocabulary_proxy_records: int
    vocabulary_probed_clusters: int
    transition_cache_hit: bool


class EngramRuntime:
    """PyTorch-free reference generator for compiled Engram packages."""

    def __init__(self, package: str | Path, *, verify_checksums: bool = True) -> None:
        self.path = Path(package)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("format") != "engram-model":
            raise ValueError("not an Engram model package")
        self._verify_package_files(verify_checksums)
        controller_dir = self.path / "controller"
        metadata = json.loads((controller_dir / "metadata.json").read_text())
        tensors = {
            name: np.load(controller_dir / f"{name}.npy")
            for name in metadata["tensor_layout"]
        }
        self.controller = SharedRecurrentController.from_state(metadata, tensors)
        self.token_embeddings = np.load(self.path / "embeddings" / "token_embeddings.npy", mmap_mode="r")
        self.vocabulary = VocabularyIndex(
            np.load(self.path / "vocabulary" / "embeddings.npy", mmap_mode="r"),
            normalized_embeddings=np.load(
                self.path / "vocabulary" / "index.npy", mmap_mode="r"
            ),
            ivf_index=VocabularyIVFIndex.load(self.path / "vocabulary" / "ivf"),
        )
        self.semantic = [
            CompressedSemanticLayer.load(
                self.path / "semantic" / f"layer-{layer:04d}" / "quantized"
            )
            for layer in range(self.manifest["num_semantic_layers"])
        ]
        episodic_config = json.loads((self.path / "episodic" / "config.json").read_text())
        self.episodic_config = episodic_config
        transition_config = json.loads((self.path / "transitions" / "config.json").read_text())
        self.cache = TransitionCache(
            state_width=self.manifest["hidden_size"],
            capacity=transition_config["capacity"],
            similarity_radius=transition_config["similarity_radius"],
        )
        self._tokenizer = None
        tokenizer_json = self.path / "tokenizer" / "tokenizer.json"
        if tokenizer_json.is_file():
            try:
                from tokenizers import Tokenizer

                self._tokenizer = Tokenizer.from_file(str(tokenizer_json))
            except ImportError as exc:
                raise RuntimeError("install tokenizers to use this package tokenizer") from exc
        self.reset()

    def _verify_package_files(self, verify_checksums: bool) -> None:
        files = self.manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError("package manifest has no file inventory")
        for relative, descriptor in files.items():
            pure = PurePosixPath(relative)
            if pure.is_absolute() or not pure.parts or ".." in pure.parts:
                raise ValueError(f"unsafe package path: {relative!r}")
            if not isinstance(descriptor, dict):
                raise ValueError(f"invalid package descriptor: {relative}")
            path = self.path.joinpath(*pure.parts)
            try:
                expected_bytes = int(descriptor["bytes"])
                expected_hash = str(descriptor["sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid package descriptor: {relative}") from error
            if not path.is_file() or path.stat().st_size != expected_bytes:
                raise ValueError(f"package file missing or has wrong size: {relative}")
            if verify_checksums and sha256_file(path) != expected_hash:
                raise ValueError(f"package checksum mismatch: {relative}")

    def reset(self) -> None:
        width = self.manifest["hidden_size"]
        self.episodic = HybridEpisodicMemory(width, width, **self.episodic_config)
        self.state = np.zeros(width, dtype=np.float64)

    def tokenize_fixture(self, prompt: str) -> list[int]:
        vocab = self.manifest["vocab_size"]
        tokens = [3 + byte % max(vocab - 3, 1) for byte in prompt.encode("utf-8")]
        return tokens or [1]

    def detokenize_fixture(self, tokens: list[int]) -> str:
        return " ".join(f"<{token}>" for token in tokens)

    def tokenize(self, prompt: str) -> list[int]:
        if self._tokenizer is not None:
            tokens = self._tokenizer.encode(prompt).ids
            return tokens or [1]
        if self.manifest["fixture_only"]:
            return self.tokenize_fixture(prompt)
        raise RuntimeError("compiled package has no tokenizer; use a token-ID input path")

    def detokenize(self, tokens: list[int]) -> str:
        if self._tokenizer is not None:
            return self._tokenizer.decode(tokens)
        return self.detokenize_fixture(tokens)

    def step(self, input_token: int, *, exact_vocab: bool = False) -> GenerationToken:
        embedding = np.asarray(self.token_embeddings[input_token], dtype=np.float64)
        previous_state = self.state.copy()
        lookup = self.cache.lookup(previous_state, input_token)
        if lookup.hit and lookup.transition is not None:
            self.state = np.asarray(lookup.transition.next_state, dtype=np.float64).copy()
            token = lookup.transition.output_candidates[0][0]
            return GenerationToken(token, 0, 0, 0, 0, 0, 0, 0, 0, 0, True)

        top_k = self.manifest["runtime"]["semantic_top_k"]
        candidate_count = self.manifest["runtime"]["semantic_candidates"]
        semantic_results = []
        for layer in self.semantic:
            semantic_results.append(
                layer.read(
                    self.state,
                    top_k=top_k,
                    candidate_count=candidate_count,
                    probes=self.manifest["runtime"].get("semantic_ivf_probes"),
                )
            )
        semantic_read = np.mean([item.output for item in semantic_results], axis=0)
        episodic = self.episodic.step(self.state, self.state, self.state)
        supplied = np.concatenate([embedding, semantic_read, episodic.output])
        result = self.controller.run(
            self.state,
            supplied,
            mode="fixed",
            fixed_cycles=self.manifest["runtime"]["cycles"],
        )
        self.state = result.state
        generation = self.vocabulary.greedy(
            self.state,
            exact=exact_vocab,
            candidate_count=self.manifest["runtime"].get(
                "vocabulary_candidates", min(64, self.manifest["vocab_size"])
            ),
            minimum_probes=self.manifest["runtime"].get(
                "vocabulary_ivf_probes", 1
            ),
        )
        self.cache.put_online(
            previous_state,
            input_token,
            self.state,
            [(generation.token_id, generation.logit)],
            confidence=1.0 / (1.0 + result.residual),
        )
        return GenerationToken(
            generation.token_id,
            result.cycles,
            top_k * len(self.semantic),
            candidate_count * len(self.semantic),
            sum(item.proxy_records for item in semantic_results),
            sum(item.probed_clusters for item in semantic_results),
            episodic.retrievals,
            generation.candidate_count,
            generation.proxy_count,
            generation.probed_clusters,
            False,
        )

    def generate_tokens(self, prompt_tokens: list[int], *, max_tokens: int = 16, exact_vocab: bool = False) -> tuple[list[int], list[GenerationToken]]:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must not be empty")
        self.reset()
        current = prompt_tokens[0]
        for token in prompt_tokens[1:]:
            self.step(current, exact_vocab=exact_vocab)
            current = token
        generated, metrics = [], []
        for _ in range(max_tokens):
            result = self.step(current, exact_vocab=exact_vocab)
            generated.append(result.token_id)
            metrics.append(result)
            current = result.token_id
        return generated, metrics
