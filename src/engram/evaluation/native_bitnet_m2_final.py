"""Fail-closed, one-shot Milestone 2 final confirmation.

This module deliberately separates *authorization* from *execution*.  The
committed protocol fixes the holdout and gates.  A post-build authorization
manifest fixes the exact implementation commit and every byte-bearing input.
Only after those inputs pass preflight does the runner atomically create an
``opened`` marker and read the holdout for the first and only time.

Normal tests must use committed temporary fixture protocols and may monkeypatch
the private evaluator adapter.  The public runner has no evaluator-injection
argument.  Merely importing this module never resolves, hashes, tokenizes, or
executes the real Milestone 2 holdout.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

from engram.evaluation.native_bitnet_dip_traffic import (
    native_bitnet_dip_physical_accounting,
)
from engram.utils import atomic_json, sha256_file, sha256_json


FINAL_CONFIRMATION_FORMAT = "engram-native-bitnet-m2-final-confirmation"
FINAL_CONFIRMATION_VERSION = 1
FINAL_ADJUDICATION_FORMAT = "engram-native-bitnet-m2-final-adjudication"
FINAL_ADJUDICATION_VERSION = 1
COMPILER_BUILD_FORMAT = "engram-native-bitnet-m2-compiler-build"
COMPILER_BUILD_VERSION = 1
FINAL_AUDIT_DIRECTORY = PurePosixPath("reports/native_bitnet_m2_final_audit")
PROTECTED_M2_HOLDOUT_SHA256 = (
    "e5606e8d18241b996b4e46cbc1b9559792b1a87ddd491bad5934eed3185b03a4"
)
NATIVE_CAUSAL_EVALUATOR = (
    "engram.evaluation.native_bitnet_dip_native_causal."
    "evaluate_native_bitnet_dip_native_causal"
)
CANONICAL_TOKEN_HASH_ALGORITHM = (
    "engram-canonical-token-sequence-sha256-v1"
)
SCORED_PREFIX_TOKEN_HASH_ALGORITHM = (
    "engram-scored-prefix-bare-list-sha256-v1"
)

_PROTOCOL_EXPERIMENT = (
    "native_bitnet_milestone_2_practical_semantic_memory_confirmation"
)
_POLICY_FORMAT = "engram-native-bitnet-dip-policy"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_NATIVE_BUILD_TARGETS = ("engram_bitnet", "engram_bitnet_dip")
_RUNTIME_SOURCE_PATHS = {
    "final confirmation runner": ("src/engram/evaluation/native_bitnet_m2_final.py"),
    "native causal evaluator": (
        "src/engram/evaluation/native_bitnet_dip_native_causal.py"
    ),
}
_REQUIRED_PINS = frozenset(
    {
        "package_manifest",
        "base_artifact",
        "coordinate_index",
        "policy_manifest",
        "tokenizer_json",
        "dense_native_library",
        "native_library",
        "compiler_build_manifest",
        "dataset",
    }
)


def _canonical_audit_record(
    *,
    implementation_commit: str,
    protocol_sha256: str,
    artifact_sha256: Mapping[str, str],
    execution: Mapping[str, Any],
) -> dict[str, str]:
    """Return canonical dataset lock and authorization-attempt paths."""

    dataset_hash = _sha256(
        artifact_sha256.get("dataset"),
        "artifacts.dataset.sha256",
    )
    dataset_key = sha256_json(
        {
            "format": "engram-native-bitnet-m2-final-dataset-lock-v1",
            "dataset_sha256": dataset_hash,
        }
    )
    attempt_key = sha256_json(
        {
            "format": "engram-native-bitnet-m2-final-attempt-v1",
            "implementation_commit": implementation_commit,
            "protocol_sha256": protocol_sha256,
            "artifact_sha256": dict(sorted(artifact_sha256.items())),
            "execution": {
                "evaluator": execution.get("evaluator"),
                "dataset_role": execution.get("dataset_role"),
                "debug_recall": execution.get("debug_recall"),
                "threads": execution.get("threads"),
            },
        }
    )
    directory = FINAL_AUDIT_DIRECTORY / dataset_key
    return {
        "dataset_key": dataset_key,
        "attempt_key": attempt_key,
        "directory": FINAL_AUDIT_DIRECTORY.as_posix(),
        "result": (directory / f"{attempt_key}.result.json").as_posix(),
        "opened_marker": (directory / "opened.json").as_posix(),
        "raw_report": (directory / f"{attempt_key}.native-causal.raw.json").as_posix(),
    }


def _pinned_audit_paths(
    root: Path,
    authorization: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    audit = authorization.get("audit")
    if not isinstance(audit, Mapping):
        raise Milestone2FinalConfirmationError(
            "final authorization has no canonical audit paths"
        )
    dataset_key = audit.get("dataset_key")
    attempt_key = audit.get("attempt_key")
    if (
        not isinstance(dataset_key, str)
        or _SHA256.fullmatch(dataset_key) is None
        or not isinstance(attempt_key, str)
        or _SHA256.fullmatch(attempt_key) is None
        or audit.get("directory") != FINAL_AUDIT_DIRECTORY.as_posix()
    ):
        raise Milestone2FinalConfirmationError(
            "final authorization audit identity is invalid"
        )
    expected = {
        "result": (
            FINAL_AUDIT_DIRECTORY / dataset_key / f"{attempt_key}.result.json"
        ).as_posix(),
        "opened_marker": (
            FINAL_AUDIT_DIRECTORY / dataset_key / "opened.json"
        ).as_posix(),
        "raw_report": (
            FINAL_AUDIT_DIRECTORY
            / dataset_key
            / f"{attempt_key}.native-causal.raw.json"
        ).as_posix(),
    }
    if any(audit.get(name) != value for name, value in expected.items()):
        raise Milestone2FinalConfirmationError(
            "final authorization audit paths are not canonical"
        )
    paths: list[Path] = []
    for name in ("result", "opened_marker", "raw_report"):
        lexical = root.joinpath(*PurePosixPath(expected[name]).parts)
        resolved = lexical.resolve()
        if not resolved.is_relative_to(root) or lexical.absolute() != resolved:
            raise Milestone2FinalConfirmationError(
                "final authorization audit path must remain inside the "
                "repository and must not traverse a symbolic link"
            )
        paths.append(resolved)
    return tuple(paths)  # type: ignore[return-value]


def _pinned_adjudication_path(
    root: Path,
    authorization: Mapping[str, Any],
) -> Path:
    audit = authorization.get("audit")
    if not isinstance(audit, Mapping):
        raise Milestone2FinalConfirmationError(
            "final authorization has no canonical audit identity"
        )
    dataset_key = audit.get("dataset_key")
    attempt_key = audit.get("attempt_key")
    if (
        not isinstance(dataset_key, str)
        or _SHA256.fullmatch(dataset_key) is None
        or not isinstance(attempt_key, str)
        or _SHA256.fullmatch(attempt_key) is None
    ):
        raise Milestone2FinalConfirmationError(
            "final authorization audit identity is invalid"
        )
    relative = (
        FINAL_AUDIT_DIRECTORY
        / dataset_key
        / f"{attempt_key}.adjudication.json"
    )
    lexical = root.joinpath(*relative.parts)
    resolved = lexical.resolve()
    if (
        not resolved.is_relative_to(root)
        or lexical.absolute() != resolved
    ):
        raise Milestone2FinalConfirmationError(
            "final adjudication path must remain inside the repository"
        )
    return resolved


def _pinned_evidence_seal_paths(
    root: Path,
    authorization: Mapping[str, Any],
) -> tuple[Path, Path]:
    audit = authorization.get("audit")
    if not isinstance(audit, Mapping):
        raise Milestone2FinalConfirmationError(
            "final authorization has no canonical audit identity"
        )
    dataset_key = audit.get("dataset_key")
    attempt_key = audit.get("attempt_key")
    if (
        not isinstance(dataset_key, str)
        or _SHA256.fullmatch(dataset_key) is None
        or not isinstance(attempt_key, str)
        or _SHA256.fullmatch(attempt_key) is None
    ):
        raise Milestone2FinalConfirmationError(
            "final authorization audit identity is invalid"
        )
    directory = FINAL_AUDIT_DIRECTORY / dataset_key
    relative_paths = (
        directory / f"{attempt_key}.evidence-seal.json",
        directory / f"{attempt_key}.authorization.json",
    )
    resolved_paths: list[Path] = []
    for relative in relative_paths:
        lexical = root.joinpath(*relative.parts)
        resolved = lexical.resolve()
        if (
            not resolved.is_relative_to(root)
            or lexical.absolute() != resolved
        ):
            raise Milestone2FinalConfirmationError(
                "final evidence-seal path must remain inside the repository"
            )
        resolved_paths.append(resolved)
    return resolved_paths[0], resolved_paths[1]


class Milestone2FinalConfirmationError(RuntimeError):
    """Raised when a final attempt cannot be executed safely."""


@dataclass(frozen=True)
class Milestone2FinalRequest:
    """Exact, preflight-validated inputs passed to the native evaluator."""

    protocol_path: Path
    authorization_manifest_path: Path
    package: Path
    package_manifest: Path
    record_artifact: Path
    coordinate_index: Path
    policy_manifest: Path
    tokenizer_json: Path
    dense_native_library: Path
    native_library: Path
    compiler_build_manifest: Path
    dataset: Path
    raw_report: Path
    record_offset: int
    sequence_count: int
    predictions_per_sequence: int
    threads: int
    reference_top_ks: tuple[int, ...]
    expected_sha256: Mapping[str, str]


class Milestone2FinalEvaluator(Protocol):
    """Narrow adapter boundary used by the one-shot runner."""

    def __call__(
        self,
        request: Milestone2FinalRequest,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _Preflight:
    protocol: dict[str, Any]
    authorization: dict[str, Any]
    request: Milestone2FinalRequest
    implementation_commit: str
    authorization_sha256: str
    observed_sha256: Mapping[str, str]
    policy_layers: tuple[Mapping[str, Any], ...]
    adjudicator_commit: str | None = None


@dataclass(frozen=True)
class _TokenIdentities:
    full_sequence_hashes: tuple[str, ...]
    full_token_lengths: tuple[int, ...]
    scored_prefix_hashes: tuple[str, ...]
    scored_prefix_input_ids_sha256: str
    scored_prefix_tokens: int
    tokenizer_files: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_full_sequence": {
                "algorithm": CANONICAL_TOKEN_HASH_ALGORITHM,
                "hashes": list(self.full_sequence_hashes),
                "token_lengths": list(self.full_token_lengths),
            },
            "scored_prefix": {
                "algorithm": SCORED_PREFIX_TOKEN_HASH_ALGORITHM,
                "tokens_per_sequence": self.scored_prefix_tokens,
                "sequence_hashes": list(self.scored_prefix_hashes),
                "aggregate_input_ids_sha256": (
                    self.scored_prefix_input_ids_sha256
                ),
            },
            "tokenizer_files": [
                dict(descriptor) for descriptor in self.tokenizer_files
            ],
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Milestone2FinalConfirmationError(
            f"{label} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise Milestone2FinalConfirmationError(f"{label} must contain a JSON object")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Milestone2FinalConfirmationError(f"{label} must be a lowercase SHA-256")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Milestone2FinalConfirmationError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Milestone2FinalConfirmationError(
            f"{label} must be a non-negative integer"
        )
    return value


def _relative_file(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Milestone2FinalConfirmationError(
            f"{label}.path must be a non-empty repository-relative path"
        )
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise Milestone2FinalConfirmationError(
            f"{label}.path must be a safe repository-relative path"
        )
    lexical = root.joinpath(*pure.parts)
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root) or lexical.absolute() != resolved:
        raise Milestone2FinalConfirmationError(
            f"{label}.path must not traverse a symbolic link"
        )
    if not resolved.is_file():
        raise Milestone2FinalConfirmationError(f"{label} is missing: {resolved}")
    return resolved


def _descriptor(
    root: Path,
    value: Any,
    label: str,
    *,
    hash_contents: bool = True,
) -> tuple[Path, str, str | None]:
    if not isinstance(value, dict):
        raise Milestone2FinalConfirmationError(f"{label} pin must be an object")
    path = _relative_file(root, value.get("path"), label)
    expected = _sha256(value.get("sha256"), f"{label}.sha256")
    expected_bytes = value.get("bytes")
    if (
        isinstance(expected_bytes, bool)
        or not isinstance(expected_bytes, int)
        or expected_bytes < 0
        or path.stat().st_size != expected_bytes
    ):
        raise Milestone2FinalConfirmationError(f"{label} byte length mismatch")
    actual = sha256_file(path) if hash_contents else None
    if actual is not None and actual != expected:
        raise Milestone2FinalConfirmationError(f"{label} SHA-256 mismatch")
    return path, expected, actual


def _git(
    root: Path,
    *arguments: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Milestone2FinalConfirmationError(
            f"cannot inspect implementation repository: {exc}"
        ) from exc


def _repository_relative(root: Path, path: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise Milestone2FinalConfirmationError(
            f"{label} must be inside the implementation repository"
        ) from exc


def _verify_git_state(
    root: Path,
    *,
    expected_commit: str,
    tracked_paths: tuple[Path, ...],
    excluded_paths: tuple[Path, ...] = (),
) -> str:
    top = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != root:
        raise Milestone2FinalConfirmationError(
            "implementation repository root mismatch"
        )
    commit = _git(root, "rev-parse", "HEAD").stdout.strip().lower()
    if _COMMIT.fullmatch(commit) is None or commit != expected_commit:
        raise Milestone2FinalConfirmationError("implementation Git commit mismatch")
    for path in tracked_paths:
        relative = _repository_relative(root, path, "frozen tracked input")
        result = _git(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        )
        if result.returncode != 0:
            raise Milestone2FinalConfirmationError(
                f"frozen input is not committed: {relative}"
            )

    pathspecs = ["."]
    for path in excluded_paths:
        if path.is_relative_to(root):
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                pathspecs.append(f":(exclude,top){relative}/**")
            else:
                pathspecs.append(f":(exclude,top,literal){relative}")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *pathspecs,
    ).stdout
    if status:
        raise Milestone2FinalConfirmationError("implementation worktree is dirty")
    return commit


def _verify_worktree_blobs_at_commit(
    root: Path,
    *,
    commit: str,
    paths: Sequence[Path],
) -> None:
    for path in paths:
        relative = _repository_relative(root, path, "commit-bound file")
        committed_blob = _git(
            root,
            "rev-parse",
            f"{commit}:{relative}",
        ).stdout.strip()
        worktree_blob = _git(
            root,
            "hash-object",
            "--",
            relative,
        ).stdout.strip()
        if not committed_blob or worktree_blob != committed_blob:
            raise Milestone2FinalConfirmationError(
                f"worktree file differs from adjudicator commit: {relative}"
            )


def _verify_historical_adjudication_git_state(
    root: Path,
    *,
    original_commit: str,
    original_tracked_paths: tuple[Path, ...],
    adjudication_evidence_paths: tuple[Path, ...],
    adjudication_output_path: Path,
) -> str:
    """Bind a postmortem audit to both the original and adjudicator commits."""

    top = Path(_git(root, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != root:
        raise Milestone2FinalConfirmationError(
            "implementation repository root mismatch"
        )
    original_exists = _git(
        root,
        "cat-file",
        "-e",
        f"{original_commit}^{{commit}}",
        check=False,
    )
    if original_exists.returncode != 0:
        raise Milestone2FinalConfirmationError(
            "original implementation commit is unavailable"
        )
    current_commit = _git(root, "rev-parse", "HEAD").stdout.strip().lower()
    if _COMMIT.fullmatch(current_commit) is None:
        raise Milestone2FinalConfirmationError(
            "adjudicator repository has no full Git commit"
        )
    ancestor = _git(
        root,
        "merge-base",
        "--is-ancestor",
        original_commit,
        current_commit,
        check=False,
    )
    if ancestor.returncode != 0:
        raise Milestone2FinalConfirmationError(
            "original implementation is not an ancestor of the adjudicator"
        )
    for path in original_tracked_paths:
        relative = _repository_relative(
            root,
            path,
            "historical frozen input",
        )
        committed = _git(
            root,
            "cat-file",
            "-e",
            f"{original_commit}:{relative}",
            check=False,
        )
        if committed.returncode != 0:
            raise Milestone2FinalConfirmationError(
                f"historical frozen input was not committed: {relative}"
            )

    current_sources = tuple(
        root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in _RUNTIME_SOURCE_PATHS.values()
    )
    bound_paths = (
        *original_tracked_paths,
        *adjudication_evidence_paths,
        *current_sources,
    )
    _verify_git_state(
        root,
        expected_commit=current_commit,
        tracked_paths=bound_paths,
        excluded_paths=(adjudication_output_path,),
    )
    _verify_worktree_blobs_at_commit(
        root,
        commit=current_commit,
        paths=bound_paths,
    )
    return current_commit


def _tracked_native_build_dependencies(
    root: Path,
    required_paths: tuple[Path, ...],
) -> tuple[Path, ...]:
    """Resolve the complete committed source closure for native targets."""

    discovered = _git(
        root,
        "ls-files",
        "-z",
        "--",
        "CMakeLists.txt",
        "native/include",
        "native/src",
    ).stdout.split("\0")
    candidate_paths = [
        root.joinpath(*PurePosixPath(value).parts) for value in discovered if value
    ]
    candidate_paths.extend(required_paths)
    resolved: dict[str, Path] = {}
    for lexical in candidate_paths:
        absolute = lexical.absolute()
        path = lexical.resolve()
        if not path.is_relative_to(root) or absolute != path or not path.is_file():
            raise Milestone2FinalConfirmationError(
                "native build dependency must be a regular, non-symlinked "
                "repository file"
            )
        relative = _repository_relative(
            root,
            path,
            "native build dependency",
        )
        result = _git(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        )
        if result.returncode != 0:
            raise Milestone2FinalConfirmationError(
                f"native build dependency is not committed: {relative}"
            )
        resolved[relative] = path
    if "CMakeLists.txt" not in resolved:
        raise Milestone2FinalConfirmationError(
            "native build dependency closure has no committed CMakeLists.txt"
        )
    if not any(name.startswith("native/src/") for name in resolved):
        raise Milestone2FinalConfirmationError(
            "native build dependency closure has no committed sources"
        )
    if not any(name.startswith("native/include/") for name in resolved):
        raise Milestone2FinalConfirmationError(
            "native build dependency closure has no committed headers"
        )
    return tuple(resolved[name] for name in sorted(resolved))


def _execute_native_target_rebuild(
    root: Path,
    build: Path,
) -> tuple[str, ...]:
    """Clean and rebuild both final native targets at the pinned commit."""

    command = (
        "cmake",
        "--build",
        str(build),
        "--target",
        *_NATIVE_BUILD_TARGETS,
        "--clean-first",
    )
    try:
        subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Milestone2FinalConfirmationError(
            f"native target rebuild failed: {exc}"
        ) from exc
    return command


def _verify_runtime_source_origins(root: Path) -> tuple[Path, Path]:
    """Prove the executing runner/evaluator came from the pinned checkout."""

    evaluator_module = importlib.import_module(
        "engram.evaluation.native_bitnet_dip_native_causal"
    )
    source_values = {
        "final confirmation runner": __file__,
        "native causal evaluator": getattr(evaluator_module, "__file__", None),
    }
    verified: list[Path] = []
    for label, relative in _RUNTIME_SOURCE_PATHS.items():
        source_value = source_values[label]
        if not isinstance(source_value, str) or not source_value:
            raise Milestone2FinalConfirmationError(
                f"{label} has no filesystem source origin"
            )
        lexical = Path(source_value).absolute()
        resolved = Path(source_value).resolve()
        expected = root.joinpath(*PurePosixPath(relative).parts)
        if resolved != expected or lexical != resolved or not resolved.is_file():
            raise Milestone2FinalConfirmationError(
                f"{label} does not originate from the pinned repository"
            )
        tracked = _git(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        )
        if tracked.returncode != 0:
            raise Milestone2FinalConfirmationError(
                f"{label} source is not committed at the pinned repository"
            )
        verified.append(resolved)
    return verified[0], verified[1]


def _verify_adjudicator_source_origin(root: Path) -> Path:
    """Verify this module without importing the historical evaluator."""

    relative = _RUNTIME_SOURCE_PATHS["final confirmation runner"]
    lexical = Path(__file__).absolute()
    resolved = Path(__file__).resolve()
    expected = root.joinpath(*PurePosixPath(relative).parts)
    if resolved != expected or lexical != resolved or not resolved.is_file():
        raise Milestone2FinalConfirmationError(
            "final adjudicator does not originate from the repository"
        )
    tracked = _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative,
        check=False,
    )
    if tracked.returncode != 0:
        raise Milestone2FinalConfirmationError(
            "final adjudicator source is not committed in the repository"
        )
    return resolved


def _canonical_full_sequence_hash(input_ids: Sequence[int]) -> str:
    return sha256_json({"input_ids": [int(value) for value in input_ids]})


def _scored_prefix_hash(input_ids: Sequence[int]) -> str:
    return sha256_json([int(value) for value in input_ids])


def _load_identity_records(
    path: Path,
    *,
    offset: int,
    count: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    usable = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise Milestone2FinalConfirmationError(
                        f"invalid final JSONL at line {line_number}"
                    ) from exc
                if not isinstance(record, dict):
                    raise Milestone2FinalConfirmationError(
                        f"final JSONL record {line_number} is not an object"
                    )
                if usable >= offset:
                    selected.append(record)
                    if len(selected) == count:
                        break
                usable += 1
    except OSError as exc:
        raise Milestone2FinalConfirmationError(
            f"cannot read opened final dataset: {exc}"
        ) from exc
    if len(selected) != count:
        raise Milestone2FinalConfirmationError(
            f"final dataset contains only {len(selected)} selected records"
        )
    return selected


def _derive_final_token_identities(
    request: Milestone2FinalRequest,
) -> _TokenIdentities:
    """Tokenize the opened records without constructing or executing a model."""

    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_imports

        if transformers_imports.is_sklearn_available():
            try:
                import sklearn  # noqa: F401
            except Exception:

                def sklearn_unavailable() -> bool:
                    return False

                transformers_imports.is_sklearn_available = sklearn_unavailable
                transformers_utils.is_sklearn_available = sklearn_unavailable
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise Milestone2FinalConfirmationError(
            "token-identity verification requires the local tokenizer runtime"
        ) from exc

    package_manifest = _json_object(
        request.package_manifest,
        "native package manifest",
    )
    tokenizer_record = package_manifest.get("tokenizer")
    if not isinstance(tokenizer_record, Mapping):
        raise Milestone2FinalConfirmationError(
            "native package tokenizer record is missing"
        )
    tokenizer_file_names = tokenizer_record.get("files")
    package_inventory = package_manifest.get("files")
    if (
        not isinstance(tokenizer_file_names, list)
        or not tokenizer_file_names
        or any(
            not isinstance(value, str)
            or not value
            or PurePosixPath(value).name != value
            for value in tokenizer_file_names
        )
        or len(set(tokenizer_file_names)) != len(tokenizer_file_names)
        or not isinstance(package_inventory, Mapping)
    ):
        raise Milestone2FinalConfirmationError(
            "native package tokenizer inventory is incomplete"
        )
    tokenizer_root = request.tokenizer_json.parent
    expected_files = set(tokenizer_file_names)
    actual_files = {
        path.name
        for path in tokenizer_root.iterdir()
        if path.is_file()
    }
    if actual_files != expected_files:
        raise Milestone2FinalConfirmationError(
            "native package tokenizer directory differs from its inventory"
        )
    tokenizer_descriptors: list[dict[str, Any]] = []
    for name in tokenizer_file_names:
        lexical = tokenizer_root / name
        resolved = lexical.resolve()
        relative = f"{tokenizer_root.name}/{name}"
        inventory_record = package_inventory.get(relative)
        if (
            lexical.absolute() != resolved
            or not resolved.is_file()
            or not isinstance(inventory_record, Mapping)
            or inventory_record.get("bytes") != resolved.stat().st_size
            or inventory_record.get("sha256") != sha256_file(resolved)
        ):
            raise Milestone2FinalConfirmationError(
                f"native package tokenizer file differs from inventory: {name}"
            )
        tokenizer_descriptors.append(
            {
                "path": relative,
                "bytes": resolved.stat().st_size,
                "sha256": str(inventory_record["sha256"]),
            }
        )
    records = _load_identity_records(
        request.dataset,
        offset=request.record_offset,
        count=request.sequence_count,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root,
        local_files_only=True,
        trust_remote_code=False,
        fix_mistral_regex=bool(
            tokenizer_record.get("fix_mistral_regex", False)
        ),
    )
    required_tokens = request.predictions_per_sequence + 1
    full_sequences: list[list[int]] = []
    scored_prefixes: list[list[int]] = []
    for sequence, record in enumerate(records):
        raw_input_ids = record.get("input_ids")
        if raw_input_ids is not None:
            if (
                not isinstance(raw_input_ids, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in raw_input_ids
                )
            ):
                raise Milestone2FinalConfirmationError(
                    f"final record {sequence} has invalid input_ids"
                )
            token_ids = [int(value) for value in raw_input_ids]
        else:
            text = record.get("text")
            if not isinstance(text, str):
                raise Milestone2FinalConfirmationError(
                    f"final record {sequence} has neither text nor input_ids"
                )
            token_ids = [
                int(value)
                for value in tokenizer.encode(
                    text,
                    add_special_tokens=True,
                )
            ]
        if len(token_ids) < required_tokens:
            raise Milestone2FinalConfirmationError(
                f"final record {sequence} has only {len(token_ids)} tokens"
            )
        full_sequences.append(token_ids)
        scored_prefixes.append(token_ids[:required_tokens])
    return _TokenIdentities(
        full_sequence_hashes=tuple(
            _canonical_full_sequence_hash(values)
            for values in full_sequences
        ),
        full_token_lengths=tuple(len(values) for values in full_sequences),
        scored_prefix_hashes=tuple(
            _scored_prefix_hash(values)
            for values in scored_prefixes
        ),
        scored_prefix_input_ids_sha256=sha256_json(scored_prefixes),
        scored_prefix_tokens=required_tokens,
        tokenizer_files=tuple(tokenizer_descriptors),
    )


def _validate_protocol_token_identities(
    protocol: Mapping[str, Any],
    identities: _TokenIdentities,
) -> dict[str, bool]:
    final = protocol.get("final_confirmation")
    if not isinstance(final, Mapping):
        raise Milestone2FinalConfirmationError(
            "frozen protocol final-confirmation record is missing"
        )
    checks = {
        "canonical_hash_algorithm": (
            final.get("canonical_token_hash_algorithm")
            == CANONICAL_TOKEN_HASH_ALGORITHM
        ),
        "full_sequence_hashes": (
            final.get("canonical_token_sequence_hashes")
            == list(identities.full_sequence_hashes)
        ),
        "full_token_lengths": (
            final.get("token_lengths")
            == list(identities.full_token_lengths)
        ),
        "scored_prefix_length": (
            final.get("required_tokens_per_sequence")
            == identities.scored_prefix_tokens
        ),
    }
    if not all(checks.values()):
        raise Milestone2FinalConfirmationError(
            "opened dataset/tokenizer identities differ from the frozen protocol"
        )
    return checks


def _cmake_cache_value(cache: str, name: str) -> str:
    prefix = f"{name}:"
    for line in cache.splitlines():
        if line.startswith(prefix) and "=" in line:
            return line.split("=", 1)[1]
    raise Milestone2FinalConfirmationError(f"CMake cache has no {name} value")


def _cmake_set_value(source: str, name: str) -> str:
    match = re.search(
        rf'^set\({re.escape(name)} "([^"]*)"\)$',
        source,
        flags=re.MULTILINE,
    )
    if match is None:
        raise Milestone2FinalConfirmationError(
            f"CMake compiler metadata has no {name} value"
        )
    return match.group(1)


def _provenance_file(root: Path, path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise Milestone2FinalConfirmationError(f"{label} is missing")
    return {
        "path": _repository_relative(root, resolved, label),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _verify_compiler_build_provenance(
    root: Path,
    provenance: Mapping[str, Any],
    *,
    expected_commit: str,
    output_paths: Mapping[str, Path],
    output_sha256: Mapping[str, str],
    sealed_paths: tuple[Path, ...],
) -> None:
    """Reconstruct and verify the complete native-build provenance record."""

    implementation = provenance.get("implementation")
    build_record = provenance.get("build")
    compiler = provenance.get("compiler")
    outputs = provenance.get("outputs")
    if (
        provenance.get("format") != COMPILER_BUILD_FORMAT
        or provenance.get("version") != COMPILER_BUILD_VERSION
        or provenance.get("status") != "frozen"
        or provenance.get("cpu_inference_only") is not True
        or not isinstance(implementation, Mapping)
        or implementation.get("git_commit") != expected_commit
        or implementation.get("clean_worktree") is not True
        or implementation.get("post_build_clean_worktree") is not True
        or not isinstance(build_record, Mapping)
        or not isinstance(compiler, Mapping)
        or not isinstance(outputs, Mapping)
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build provenance is not a complete frozen build"
        )

    requested_names = implementation.get("requested_tracked_paths")
    dependency_names = implementation.get("required_tracked_paths")
    dependency_descriptors = implementation.get("tracked_dependencies")
    sealed_names = implementation.get("sealed_tracked_paths_not_read")
    if (
        not isinstance(requested_names, list)
        or any(not isinstance(value, str) for value in requested_names)
        or not isinstance(dependency_names, list)
        or any(not isinstance(value, str) for value in dependency_names)
        or not isinstance(dependency_descriptors, list)
        or not isinstance(sealed_names, list)
        or any(not isinstance(value, str) for value in sealed_names)
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build provenance has no tracked dependency closure"
        )
    expected_sealed_names = sorted(
        _repository_relative(root, path, "sealed compiler build path")
        for path in sealed_paths
    )
    if sorted(sealed_names) != expected_sealed_names:
        raise Milestone2FinalConfirmationError(
            "compiler build sealed paths differ from the protected protocol inputs"
        )
    sealed_name_set = set(expected_sealed_names)
    if sealed_name_set.intersection(requested_names) or sealed_name_set.intersection(
        dependency_names
    ):
        raise Milestone2FinalConfirmationError(
            "a sealed protocol input is declared as a native build dependency"
        )
    requested_paths = tuple(
        _relative_file(root, name, "requested native build dependency")
        for name in requested_names
    )
    reconstructed = _tracked_native_build_dependencies(
        root,
        requested_paths,
    )
    reconstructed_names = [
        _repository_relative(root, path, "native build dependency")
        for path in reconstructed
    ]
    if dependency_names != reconstructed_names or sealed_name_set.intersection(
        reconstructed_names
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build dependency closure differs from the pinned commit"
        )
    descriptors_by_path: dict[str, Mapping[str, Any]] = {}
    for descriptor in dependency_descriptors:
        if not isinstance(descriptor, Mapping):
            raise Milestone2FinalConfirmationError(
                "compiler build dependency descriptor is invalid"
            )
        name = descriptor.get("path")
        if not isinstance(name, str) or name in descriptors_by_path:
            raise Milestone2FinalConfirmationError(
                "compiler build dependency descriptors are ambiguous"
            )
        descriptors_by_path[name] = descriptor
    if set(descriptors_by_path) != set(reconstructed_names):
        raise Milestone2FinalConfirmationError(
            "compiler build dependency descriptors are incomplete"
        )
    for name in reconstructed_names:
        _descriptor(
            root,
            dict(descriptors_by_path[name]),
            f"compiler build dependency {name}",
        )

    build_directory_value = build_record.get("directory")
    if not isinstance(build_directory_value, str):
        raise Milestone2FinalConfirmationError("compiler build directory is missing")
    build_pure = PurePosixPath(build_directory_value)
    if build_pure.is_absolute() or ".." in build_pure.parts or not build_pure.parts:
        raise Milestone2FinalConfirmationError(
            "compiler build directory is not repository-relative"
        )
    build_lexical = root.joinpath(*build_pure.parts)
    build_directory = build_lexical.resolve()
    if (
        not build_directory.is_relative_to(root)
        or build_lexical.absolute() != build_directory
        or not build_directory.is_dir()
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build directory traverses a symbolic link or is missing"
        )
    invocation = build_record.get("invocation")
    expected_command = [
        "cmake",
        "--build",
        str(build_directory),
        "--target",
        *_NATIVE_BUILD_TARGETS,
        "--clean-first",
    ]
    if (
        not isinstance(invocation, Mapping)
        or invocation.get("command") != expected_command
        or invocation.get("cwd") != str(root)
        or invocation.get("targets") != list(_NATIVE_BUILD_TARGETS)
        or invocation.get("clean_first") is not True
        or invocation.get("return_code") != 0
        or invocation.get("post_build_clean_worktree_verified") is not True
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build invocation is missing or differs from the "
            "required clean target rebuild"
        )
    for field, label in (
        ("cmake_cache", "CMake cache"),
        ("compiler_metadata", "CMake compiler metadata"),
        ("build_graph", "native build graph"),
    ):
        descriptor = build_record.get(field)
        if not isinstance(descriptor, Mapping):
            raise Milestone2FinalConfirmationError(
                f"compiler build has no {label} descriptor"
            )
        path, _, _ = _descriptor(
            root,
            dict(descriptor),
            f"compiler build {label}",
        )
        if not path.is_relative_to(build_directory):
            raise Milestone2FinalConfirmationError(
                f"compiler build {label} is outside the build directory"
            )

    compiler_path_value = compiler.get("path")
    if not isinstance(compiler_path_value, str):
        raise Milestone2FinalConfirmationError("compiler build has no compiler path")
    compiler_lexical = Path(compiler_path_value).absolute()
    compiler_path = Path(compiler_path_value).resolve()
    if (
        compiler_lexical != compiler_path
        or not compiler_path.is_file()
        or compiler.get("bytes") != compiler_path.stat().st_size
        or compiler.get("sha256") != sha256_file(compiler_path)
        or not isinstance(compiler.get("id"), str)
        or not compiler.get("id")
        or not isinstance(compiler.get("version"), str)
        or not compiler.get("version")
    ):
        raise Milestone2FinalConfirmationError(
            "compiler executable provenance does not reconcile"
        )

    for name in ("dense_native_library", "native_library"):
        descriptor = outputs.get(name)
        path = output_paths[name]
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("path")
            != _repository_relative(root, path, f"{name} output")
            or descriptor.get("bytes") != path.stat().st_size
            or descriptor.get("sha256") != output_sha256[name]
        ):
            raise Milestone2FinalConfirmationError(
                f"compiler build provenance differs from frozen {name}"
            )


def write_native_bitnet_m2_compiler_build_manifest(
    repository_root: str | Path,
    build_directory: str | Path,
    *,
    dense_native_library: str | Path,
    native_library: str | Path,
    out: str | Path,
    required_tracked_paths: tuple[str | Path, ...],
    sealed_paths: tuple[str | Path, ...] = (),
) -> Path:
    """Write deterministic provenance for the two final native libraries.

    The build directory and explicitly sealed paths are excluded from the
    clean-worktree query.  Sealed paths are checked for index membership only;
    their contents are never opened or hashed by this function.
    """

    root = Path(repository_root).resolve()
    build_lexical = Path(build_directory).absolute()
    build = Path(build_directory).resolve()
    output_lexical = Path(out).absolute()
    output = Path(out).resolve()
    dense = Path(dense_native_library).resolve()
    dip = Path(native_library).resolve()
    if not build.is_dir() or not build.is_relative_to(root) or build_lexical != build:
        raise Milestone2FinalConfirmationError(
            "compiler build directory must be a non-symlinked repository directory"
        )
    if not output.is_relative_to(build) or output_lexical != output:
        raise Milestone2FinalConfirmationError(
            "compiler build manifest must be a non-symlinked path inside "
            "the build directory"
        )
    requested = tuple(Path(path).resolve() for path in required_tracked_paths)
    sealed = tuple(Path(path).resolve() for path in sealed_paths)
    if not requested:
        raise Milestone2FinalConfirmationError(
            "compiler build provenance requires committed implementation paths"
        )
    tracked = _tracked_native_build_dependencies(root, requested)
    if set(tracked).intersection(sealed):
        raise Milestone2FinalConfirmationError(
            "a native build dependency cannot be declared sealed"
        )
    commit = _git(root, "rev-parse", "HEAD").stdout.strip().lower()
    if _COMMIT.fullmatch(commit) is None:
        raise Milestone2FinalConfirmationError(
            "implementation repository has no full Git commit"
        )
    _verify_git_state(
        root,
        expected_commit=commit,
        tracked_paths=tracked + sealed,
        excluded_paths=(build, output, *sealed),
    )

    build_command = _execute_native_target_rebuild(root, build)
    _verify_git_state(
        root,
        expected_commit=commit,
        tracked_paths=tracked + sealed,
        excluded_paths=(build, output, *sealed),
    )

    cache_path = build / "CMakeCache.txt"
    cache = cache_path.read_text(encoding="utf-8")
    compiler_path = Path(_cmake_cache_value(cache, "CMAKE_CXX_COMPILER")).resolve()
    if not compiler_path.is_file():
        raise Milestone2FinalConfirmationError("configured C++ compiler is missing")
    compiler_metadata_paths = sorted(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if len(compiler_metadata_paths) != 1:
        raise Milestone2FinalConfirmationError(
            "compiler build has ambiguous CMake C++ metadata"
        )
    compiler_metadata_path = compiler_metadata_paths[0].resolve()
    compiler_metadata = compiler_metadata_path.read_text(encoding="utf-8")
    build_graph_candidates = [
        candidate
        for candidate in (build / "build.ninja", build / "Makefile")
        if candidate.is_file()
    ]
    if len(build_graph_candidates) != 1:
        raise Milestone2FinalConfirmationError(
            "compiler build must have exactly one supported build graph"
        )
    for library, label in (
        (dense, "dense native library"),
        (dip, "DIP native library"),
    ):
        if not library.is_relative_to(build) or not library.is_file():
            raise Milestone2FinalConfirmationError(
                f"{label} must be a regular build output"
            )

    manifest = {
        "format": COMPILER_BUILD_FORMAT,
        "version": COMPILER_BUILD_VERSION,
        "status": "frozen",
        "implementation": {
            "git_commit": commit,
            "clean_worktree": True,
            "post_build_clean_worktree": True,
            "clean_scope": (
                "repository_except_build_directory_and_declared_sealed_paths"
            ),
            "requested_tracked_paths": [
                _repository_relative(root, path, "requested tracked path")
                for path in requested
            ],
            "required_tracked_paths": [
                _repository_relative(root, path, "tracked path") for path in tracked
            ],
            "tracked_dependencies": [
                _provenance_file(
                    root,
                    path,
                    "native build dependency",
                )
                for path in tracked
            ],
            "sealed_tracked_paths_not_read": [
                _repository_relative(root, path, "sealed path")
                for path in sorted(sealed)
            ],
        },
        "compiler": {
            "path": str(compiler_path),
            "bytes": compiler_path.stat().st_size,
            "sha256": sha256_file(compiler_path),
            "id": _cmake_set_value(
                compiler_metadata,
                "CMAKE_CXX_COMPILER_ID",
            ),
            "version": _cmake_set_value(
                compiler_metadata,
                "CMAKE_CXX_COMPILER_VERSION",
            ),
        },
        "build": {
            "directory": _repository_relative(
                root,
                build,
                "build directory",
            ),
            "build_type": _cmake_cache_value(cache, "CMAKE_BUILD_TYPE"),
            "generator": _cmake_cache_value(cache, "CMAKE_GENERATOR"),
            "invocation": {
                "command": list(build_command),
                "cwd": str(root),
                "targets": list(_NATIVE_BUILD_TARGETS),
                "clean_first": True,
                "return_code": 0,
                "post_build_clean_worktree_verified": True,
            },
            "cmake_cache": _provenance_file(
                root,
                cache_path,
                "CMake cache",
            ),
            "compiler_metadata": _provenance_file(
                root,
                compiler_metadata_path,
                "CMake compiler metadata",
            ),
            "build_graph": _provenance_file(
                root,
                build_graph_candidates[0],
                "native build graph",
            ),
        },
        "outputs": {
            "dense_native_library": _provenance_file(
                root,
                dense,
                "dense native library",
            ),
            "native_library": _provenance_file(
                root,
                dip,
                "DIP native library",
            ),
        },
        "cpu_inference_only": True,
    }
    if output.exists():
        existing = _json_object(output, "compiler build manifest")
        if sha256_json(existing) != sha256_json(manifest):
            raise Milestone2FinalConfirmationError(
                "refusing to replace a different compiler build manifest"
            )
        return output
    atomic_json(output, manifest)
    return output


def _authorization_descriptor(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": _repository_relative(root, resolved, "authorization input"),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_native_bitnet_m2_final_authorization_manifest(
    repository_root: str | Path,
    *,
    protocol: str | Path,
    package_manifest: str | Path,
    base_artifact: str | Path,
    coordinate_index: str | Path,
    policy_manifest: str | Path,
    tokenizer_json: str | Path,
    dense_native_library: str | Path,
    native_library: str | Path,
    compiler_build_manifest: str | Path,
    threads: int,
    out: str | Path,
) -> Path:
    """Freeze the acyclic authorization for one protected final attempt.

    This generator intentionally does not accept a dataset hash from the
    caller.  It copies the already-committed protocol hash and only stats the
    protocol-declared dataset to record its byte length.  Dataset contents are
    first opened later, after the runner has created its exclusive marker.

    The trust chain is acyclic::

        implementation commit -> approved policy
        implementation commit -> compiler/build provenance -> native libraries
        authorization -> protocol + policy + build provenance + all artifacts

    The authorization is generated after the implementation commit and is not
    itself required to be committed.
    """

    root = Path(repository_root).resolve()
    output = Path(out).resolve()
    paths = {
        "package_manifest": Path(package_manifest).resolve(),
        "base_artifact": Path(base_artifact).resolve(),
        "coordinate_index": Path(coordinate_index).resolve(),
        "policy_manifest": Path(policy_manifest).resolve(),
        "tokenizer_json": Path(tokenizer_json).resolve(),
        "dense_native_library": Path(dense_native_library).resolve(),
        "native_library": Path(native_library).resolve(),
        "compiler_build_manifest": Path(compiler_build_manifest).resolve(),
    }
    protocol_path = Path(protocol).resolve()
    for name, path in {"protocol": protocol_path, **paths}.items():
        if (
            not path.is_file()
            or not path.is_relative_to(root)
            or root.joinpath(path.relative_to(root)).absolute() != path
        ):
            raise Milestone2FinalConfirmationError(
                f"authorization {name} must be a regular repository file"
            )
    threads = _positive_integer(threads, "authorization threads")

    protocol_payload = _json_object(protocol_path, "frozen protocol")
    if (
        protocol_payload.get("experiment") != _PROTOCOL_EXPERIMENT
        or protocol_payload.get("protocol_version") != 5
        or not isinstance(protocol_payload.get("final_confirmation"), dict)
        or protocol_payload.get("configuration") is not None
        or protocol_payload.get("final_result") is not None
    ):
        raise Milestone2FinalConfirmationError(
            "authorization requires the unopened protocol-v5 state"
        )
    test_mode = protocol_payload.get("test_fixture_only") is True
    final = protocol_payload["final_confirmation"]
    dataset_hash = _sha256(
        final.get("dataset_sha256"),
        "protocol final dataset_sha256",
    )
    dataset_path = _relative_file(
        root,
        final.get("dataset"),
        "final dataset",
    )
    if final.get("reuse_for_configuration_changes_after_opening") is not False:
        raise Milestone2FinalConfirmationError(
            "protocol does not prohibit final dataset reuse"
        )

    try:
        from engram.semantic.native_bitnet_dip_policy_manifest import (
            load_native_bitnet_dip_policy_manifest,
        )

        loaded_policy = load_native_bitnet_dip_policy_manifest(paths["policy_manifest"])
    except Exception as exc:
        raise Milestone2FinalConfirmationError(
            f"cannot authorize an unreconstructable DIP policy: {exc}"
        ) from exc
    policy_bindings = loaded_policy.payload.get("bindings", {})
    policy_expected = {
        "package_manifest": "package_manifest",
        "base_record_artifact": "base_artifact",
        "coordinate_index": "coordinate_index",
        "tokenizer_json": "tokenizer_json",
        "dense_reference_library": "dense_native_library",
        "dip_native_library": "native_library",
    }
    descriptors = {
        name: _authorization_descriptor(root, path) for name, path in paths.items()
    }
    for binding_name, pin_name in policy_expected.items():
        bound = policy_bindings.get(binding_name)
        descriptor = descriptors[pin_name]
        if (
            not isinstance(bound, Mapping)
            or bound.get("sha256") != descriptor["sha256"]
            or bound.get("bytes") != descriptor["bytes"]
            or Path(str(bound.get("path"))).resolve() != paths[pin_name]
        ):
            raise Milestone2FinalConfirmationError(
                f"approved policy {binding_name} binding mismatch"
            )

    build = _json_object(
        paths["compiler_build_manifest"],
        "compiler build manifest",
    )
    commit = _git(root, "rev-parse", "HEAD").stdout.strip().lower()
    if _COMMIT.fullmatch(commit) is None:
        raise Milestone2FinalConfirmationError(
            "authorization repository has no full implementation commit"
        )
    build_outputs = build.get("outputs")
    if (
        build.get("format") != COMPILER_BUILD_FORMAT
        or build.get("version") != COMPILER_BUILD_VERSION
        or build.get("status") != "frozen"
        or build.get("cpu_inference_only") is not True
        or build.get("implementation", {}).get("git_commit") != commit
        or build.get("implementation", {}).get("clean_worktree") is not True
        or not isinstance(build_outputs, Mapping)
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build manifest does not bind this implementation"
        )
    for name in ("dense_native_library", "native_library"):
        bound = build_outputs.get(name)
        descriptor = descriptors[name]
        if (
            not isinstance(bound, Mapping)
            or bound.get("path") != descriptor["path"]
            or bound.get("bytes") != descriptor["bytes"]
            or bound.get("sha256") != descriptor["sha256"]
        ):
            raise Milestone2FinalConfirmationError(
                f"compiler build {name} output mismatch"
            )
    if not test_mode:
        _verify_compiler_build_provenance(
            root,
            build,
            expected_commit=commit,
            output_paths={
                "dense_native_library": paths["dense_native_library"],
                "native_library": paths["native_library"],
            },
            output_sha256={
                "dense_native_library": str(
                    descriptors["dense_native_library"]["sha256"]
                ),
                "native_library": str(descriptors["native_library"]["sha256"]),
            },
            sealed_paths=(dataset_path,),
        )

    _verify_git_state(
        root,
        expected_commit=commit,
        tracked_paths=(
            protocol_path,
            paths["policy_manifest"],
            dataset_path,
        ),
        excluded_paths=(dataset_path, output),
    )
    protocol_descriptor = _authorization_descriptor(root, protocol_path)
    artifact_descriptors = {
        **descriptors,
        # Deliberately no sha256_file(dataset_path) call here.
        "dataset": {
            "path": dataset_path.relative_to(root).as_posix(),
            "bytes": dataset_path.stat().st_size,
            "sha256": dataset_hash,
        },
    }
    execution = {
        "evaluator": NATIVE_CAUSAL_EVALUATOR,
        "dataset_role": "final",
        "debug_recall": True,
        "threads": threads,
    }
    audit = _canonical_audit_record(
        implementation_commit=commit,
        protocol_sha256=str(protocol_descriptor["sha256"]),
        artifact_sha256={
            name: str(descriptor["sha256"])
            for name, descriptor in artifact_descriptors.items()
        },
        execution=execution,
    )
    authorization = {
        "format": FINAL_CONFIRMATION_FORMAT,
        "version": FINAL_CONFIRMATION_VERSION,
        "status": "frozen",
        "implementation": {
            "repository_root": str(root),
            "git_commit": commit,
            "require_clean_worktree": True,
        },
        "protocol": protocol_descriptor,
        "artifacts": artifact_descriptors,
        "execution": execution,
        "audit": audit,
        "trust_chain": {
            "acyclic": True,
            "policy_binds_compiler_build_manifest": False,
            "authorization_independently_pins_policy_and_compiler_build": True,
            "compiler_build_binds_same_native_libraries": True,
            "dataset_contents_read": False,
        },
    }
    canonical_result, canonical_marker, canonical_raw = _pinned_audit_paths(
        root,
        authorization,
    )
    if any(
        path.exists() for path in (canonical_result, canonical_marker, canonical_raw)
    ):
        raise Milestone2FinalConfirmationError(
            "cannot authorize a final dataset with an existing audit record"
        )
    if output.exists():
        existing = _json_object(output, "final authorization manifest")
        if sha256_json(existing) != sha256_json(authorization):
            raise Milestone2FinalConfirmationError(
                "refusing to replace a different final authorization manifest"
            )
        return output
    atomic_json(output, authorization)
    return output


def _package_bindings(
    protocol: Mapping[str, Any],
    package_manifest_path: Path,
    package_manifest: Mapping[str, Any],
    pins: Mapping[str, Path],
    observed: Mapping[str, str],
) -> Path:
    package_root = package_manifest_path.parent
    if (
        package_manifest.get("format") != "engram-native-bitnet"
        or package_manifest.get("version") != 1
    ):
        raise Milestone2FinalConfirmationError(
            "package manifest is not a supported native BitNet package"
        )
    source = package_manifest.get("source")
    expected_source = protocol.get("source_model")
    if not isinstance(source, dict) or not isinstance(expected_source, dict):
        raise Milestone2FinalConfirmationError(
            "package/protocol source-model binding is missing"
        )
    for package_key, protocol_key in (
        ("repository", "repository"),
        ("revision", "revision"),
        ("weight_sha256", "weight_sha256"),
    ):
        if source.get(package_key) != expected_source.get(protocol_key):
            raise Milestone2FinalConfirmationError(
                f"package source {package_key} differs from protocol"
            )

    mlp = package_manifest.get("mlp")
    tokenizer = package_manifest.get("tokenizer")
    inventory = package_manifest.get("files")
    if not isinstance(mlp, dict) or not isinstance(tokenizer, dict):
        raise Milestone2FinalConfirmationError(
            "package MLP/tokenizer binding is missing"
        )
    if not isinstance(inventory, dict):
        raise Milestone2FinalConfirmationError("package file inventory is missing")
    base_relative = PurePosixPath(str(mlp.get("path", "")))
    tokenizer_relative = (
        PurePosixPath(str(tokenizer.get("path", ""))) / "tokenizer.json"
    )
    if (
        base_relative.is_absolute()
        or tokenizer_relative.is_absolute()
        or ".." in base_relative.parts
        or ".." in tokenizer_relative.parts
        or package_root.joinpath(*base_relative.parts).resolve()
        != pins["base_artifact"]
        or package_root.joinpath(*tokenizer_relative.parts).resolve()
        != pins["tokenizer_json"]
    ):
        raise Milestone2FinalConfirmationError(
            "package paths differ from frozen base/tokenizer pins"
        )
    for relative, pin_name in (
        (base_relative.as_posix(), "base_artifact"),
        (tokenizer_relative.as_posix(), "tokenizer_json"),
    ):
        descriptor = inventory.get(relative)
        if (
            not isinstance(descriptor, dict)
            or descriptor.get("sha256") != observed[pin_name]
            or descriptor.get("bytes") != pins[pin_name].stat().st_size
        ):
            raise Milestone2FinalConfirmationError(
                f"package inventory differs from frozen {pin_name} pin"
            )
    if mlp.get("sha256") != observed["base_artifact"]:
        raise Milestone2FinalConfirmationError(
            "package MLP SHA-256 differs from frozen base artifact"
        )
    return package_root


def _preflight(
    protocol_path: Path,
    authorization_path: Path,
    *,
    result_path: Path,
    opened_marker_path: Path,
    raw_report_path: Path,
    historical_adjudication: bool = False,
    adjudication_output_path: Path | None = None,
    additional_adjudication_evidence_paths: tuple[Path, ...] = (),
) -> _Preflight:
    authorization = _json_object(
        authorization_path,
        "final authorization manifest",
    )
    if (
        authorization.get("format") != FINAL_CONFIRMATION_FORMAT
        or authorization.get("version") != FINAL_CONFIRMATION_VERSION
        or authorization.get("status") != "frozen"
    ):
        raise Milestone2FinalConfirmationError(
            "final authorization manifest is not frozen schema version 1"
        )
    implementation = authorization.get("implementation")
    if not isinstance(implementation, dict):
        raise Milestone2FinalConfirmationError(
            "final authorization implementation pin is missing"
        )
    root_value = implementation.get("repository_root")
    if not isinstance(root_value, str) or not root_value:
        raise Milestone2FinalConfirmationError(
            "implementation.repository_root is missing"
        )
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise Milestone2FinalConfirmationError(
            "implementation repository root is missing"
        )
    expected_commit = implementation.get("git_commit")
    if (
        not isinstance(expected_commit, str)
        or _COMMIT.fullmatch(expected_commit) is None
        or implementation.get("require_clean_worktree") is not True
    ):
        raise Milestone2FinalConfirmationError(
            "implementation must pin a full commit and require a clean worktree"
        )

    protocol_pin = authorization.get("protocol")
    if not isinstance(protocol_pin, dict):
        raise Milestone2FinalConfirmationError("frozen protocol pin is missing")
    pinned_protocol_path = _relative_file(
        root,
        protocol_pin.get("path"),
        "protocol",
    )
    if pinned_protocol_path != protocol_path.resolve():
        raise Milestone2FinalConfirmationError(
            "invoked protocol path differs from authorization"
        )
    expected_protocol_hash = _sha256(
        protocol_pin.get("sha256"),
        "protocol.sha256",
    )
    if sha256_file(pinned_protocol_path) != expected_protocol_hash:
        raise Milestone2FinalConfirmationError("frozen protocol SHA-256 mismatch")
    protocol = _json_object(pinned_protocol_path, "frozen protocol")
    if (
        protocol.get("experiment") != _PROTOCOL_EXPERIMENT
        or protocol.get("protocol_version") != 5
        or not isinstance(protocol.get("final_confirmation"), dict)
    ):
        raise Milestone2FinalConfirmationError("unsupported Milestone 2 final protocol")
    test_mode = protocol.get("test_fixture_only") is True
    if not test_mode:
        if historical_adjudication:
            _verify_adjudicator_source_origin(root)
        else:
            _verify_runtime_source_origins(root)

    artifact_pins = authorization.get("artifacts")
    if not isinstance(artifact_pins, dict) or not _REQUIRED_PINS.issubset(
        artifact_pins
    ):
        missing = sorted(
            _REQUIRED_PINS
            - (artifact_pins.keys() if isinstance(artifact_pins, dict) else set())
        )
        raise Milestone2FinalConfirmationError(
            f"final authorization is missing artifact pins: {missing}"
        )
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    observed: dict[str, str] = {}
    for name in sorted(_REQUIRED_PINS):
        path, expected_hash, actual_hash = _descriptor(
            root,
            artifact_pins[name],
            f"artifacts.{name}",
            hash_contents=name != "dataset",
        )
        paths[name] = path
        expected[name] = expected_hash
        if actual_hash is not None:
            observed[name] = actual_hash

    final = protocol["final_confirmation"]
    if (
        final.get("dataset") != artifact_pins["dataset"].get("path")
        or final.get("dataset_sha256") != expected["dataset"]
        or final.get("tokenizer_json_sha256") != expected["tokenizer_json"]
    ):
        raise Milestone2FinalConfirmationError(
            "protocol dataset/tokenizer pins differ from authorization"
        )
    if final.get("reuse_for_configuration_changes_after_opening") is not False:
        raise Milestone2FinalConfirmationError(
            "protocol does not forbid holdout reuse after opening"
        )
    if test_mode and (
        expected["dataset"] == PROTECTED_M2_HOLDOUT_SHA256
        or final.get("dataset") == "tests/fixtures/milestone2_bitnet_holdout_v1.jsonl"
    ):
        raise Milestone2FinalConfirmationError(
            "fixture-only mode is forbidden for the protected final holdout"
        )
    canonical_hashes = final.get("canonical_token_sequence_hashes")
    token_lengths = final.get("token_lengths")
    if (
        final.get("canonical_token_hash_algorithm")
        != "engram-canonical-token-sequence-sha256-v1"
        or not isinstance(canonical_hashes, list)
        or len(canonical_hashes) != 8
        or any(_SHA256.fullmatch(str(value)) is None for value in canonical_hashes)
        or not isinstance(token_lengths, list)
        or len(token_lengths) != 8
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 33
            for value in token_lengths
        )
        or protocol.get("configuration") is not None
        or protocol.get("final_result") is not None
    ):
        raise Milestone2FinalConfirmationError(
            "frozen protocol token identities or unopened state are invalid"
        )
    practical_contract = protocol.get("practical_router_thresholds")
    required_system = protocol.get("required_system_evidence")
    if (
        not isinstance(practical_contract, dict)
        or practical_contract.get("cpu_only_inference_required") is not True
        or practical_contract.get("dense_gate_up_or_down_fallback_allowed") is not False
        or not isinstance(required_system, dict)
        or any(
            required_system.get(name) is not True
            for name in (
                "serialized_index_reload",
                "python_native_numerical_parity",
                "measured_cpu_latency",
                "cache_line_honest_traffic_accounting",
            )
        )
    ):
        raise Milestone2FinalConfirmationError(
            "frozen protocol system-evidence contract is invalid"
        )

    # The dataset is intentionally excluded.  Its committed-path membership is
    # checked without reading bytes; its worktree content is authenticated only
    # after the opened marker exists.
    original_tracked_paths = (
        pinned_protocol_path,
        paths["policy_manifest"],
        paths["dataset"],
    )
    adjudicator_commit: str | None = None
    if historical_adjudication:
        if adjudication_output_path is None:
            raise Milestone2FinalConfirmationError(
                "historical adjudication output path is missing"
            )
        adjudicator_commit = _verify_historical_adjudication_git_state(
            root,
            original_commit=expected_commit,
            original_tracked_paths=original_tracked_paths,
            adjudication_evidence_paths=(
                result_path.resolve(),
                opened_marker_path.resolve(),
                raw_report_path.resolve(),
                *(
                    path.resolve()
                    for path in additional_adjudication_evidence_paths
                ),
            ),
            adjudication_output_path=adjudication_output_path.resolve(),
        )
        commit = expected_commit
    else:
        commit = _verify_git_state(
            root,
            expected_commit=expected_commit,
            tracked_paths=original_tracked_paths,
            excluded_paths=(
                paths["dataset"],
                result_path.resolve(),
                opened_marker_path.resolve(),
                raw_report_path.resolve(),
            ),
        )

    policy = _json_object(paths["policy_manifest"], "frozen DIP policy")
    policy_layers: tuple[Mapping[str, Any], ...]
    if test_mode:
        if (
            policy.get("format") != _POLICY_FORMAT
            or policy.get("version") != 1
            or policy.get("status")
            not in {"approved", "approved_for_final_confirmation"}
        ):
            raise Milestone2FinalConfirmationError(
                "fixture DIP policy is not an approved schema-v1 policy"
            )
        raw_layers = policy.get("layers", [])
        policy_layers = tuple(
            layer for layer in raw_layers if isinstance(layer, Mapping)
        )
    else:
        try:
            from engram.semantic.native_bitnet_dip_policy_manifest import (
                NATIVE_BITNET_DIP_POLICY_STATUS,
                load_native_bitnet_dip_policy_manifest,
            )

            if policy.get("status") != NATIVE_BITNET_DIP_POLICY_STATUS:
                raise Milestone2FinalConfirmationError(
                    "DIP policy is not approved for final confirmation"
                )
            loaded_policy = load_native_bitnet_dip_policy_manifest(
                paths["policy_manifest"],
                expected_sha256=expected["policy_manifest"],
            )
        except Milestone2FinalConfirmationError:
            raise
        except Exception as exc:
            raise Milestone2FinalConfirmationError(
                f"cannot reconstruct frozen DIP policy: {exc}"
            ) from exc
        bindings = loaded_policy.payload.get("bindings", {})
        expected_bindings = {
            "package_manifest": "package_manifest",
            "base_record_artifact": "base_artifact",
            "coordinate_index": "coordinate_index",
            "tokenizer_json": "tokenizer_json",
            "dense_reference_library": "dense_native_library",
            "dip_native_library": "native_library",
        }
        for binding_name, pin_name in expected_bindings.items():
            descriptor = bindings.get(binding_name)
            if (
                not isinstance(descriptor, Mapping)
                or descriptor.get("sha256") != observed[pin_name]
                or descriptor.get("bytes") != paths[pin_name].stat().st_size
                or Path(str(descriptor.get("path"))).resolve() != paths[pin_name]
            ):
                raise Milestone2FinalConfirmationError(
                    f"frozen DIP policy {binding_name} binding mismatch"
                )
        policy_layers = tuple(layer.to_dict() for layer in loaded_policy.layers)
    build_provenance = _json_object(
        paths["compiler_build_manifest"],
        "compiler build manifest",
    )
    provenance_implementation = build_provenance.get("implementation")
    provenance_outputs = build_provenance.get("outputs")
    if (
        build_provenance.get("format") != COMPILER_BUILD_FORMAT
        or build_provenance.get("version") != COMPILER_BUILD_VERSION
        or build_provenance.get("status") != "frozen"
        or build_provenance.get("cpu_inference_only") is not True
        or not isinstance(provenance_implementation, dict)
        or provenance_implementation.get("git_commit") != expected_commit
        or provenance_implementation.get("clean_worktree") is not True
        or not isinstance(provenance_outputs, dict)
    ):
        raise Milestone2FinalConfirmationError(
            "compiler build provenance is not frozen for this implementation"
        )
    for name in ("dense_native_library", "native_library"):
        output_descriptor = provenance_outputs.get(name)
        if (
            not isinstance(output_descriptor, dict)
            or output_descriptor.get("sha256") != observed[name]
            or output_descriptor.get("bytes") != paths[name].stat().st_size
            or output_descriptor.get("path") != artifact_pins[name].get("path")
        ):
            raise Milestone2FinalConfirmationError(
                f"compiler build provenance differs from frozen {name}"
            )
    if not test_mode:
        _verify_compiler_build_provenance(
            root,
            build_provenance,
            expected_commit=expected_commit,
            output_paths={
                "dense_native_library": paths["dense_native_library"],
                "native_library": paths["native_library"],
            },
            output_sha256={
                "dense_native_library": observed["dense_native_library"],
                "native_library": observed["native_library"],
            },
            sealed_paths=(paths["dataset"],),
        )
    package_manifest = _json_object(
        paths["package_manifest"],
        "native BitNet package manifest",
    )
    package = _package_bindings(
        protocol,
        paths["package_manifest"],
        package_manifest,
        paths,
        observed,
    )

    execution = authorization.get("execution")
    if not isinstance(execution, dict):
        raise Milestone2FinalConfirmationError(
            "final authorization execution record is missing"
        )
    if (
        execution.get("evaluator") != NATIVE_CAUSAL_EVALUATOR
        or execution.get("dataset_role") != "final"
        or execution.get("debug_recall") is not True
    ):
        raise Milestone2FinalConfirmationError(
            "final execution must use the pinned native causal evaluator "
            "with final-role recall diagnostics"
        )
    threads = _positive_integer(execution.get("threads"), "execution.threads")
    canonical_audit = _canonical_audit_record(
        implementation_commit=expected_commit,
        protocol_sha256=expected_protocol_hash,
        artifact_sha256=expected,
        execution=execution,
    )
    if authorization.get("audit") != canonical_audit:
        raise Milestone2FinalConfirmationError(
            "final authorization audit identity differs from frozen inputs"
        )
    canonical_result, canonical_marker, canonical_raw = _pinned_audit_paths(
        root,
        authorization,
    )
    if (
        result_path.resolve() != canonical_result
        or opened_marker_path.resolve() != canonical_marker
        or raw_report_path.resolve() != canonical_raw
    ):
        raise Milestone2FinalConfirmationError(
            "final confirmation audit paths differ from authorization"
        )
    sequence_count = _positive_integer(
        final.get("sequence_count"),
        "protocol final sequence_count",
    )
    predictions = _positive_integer(
        final.get("predictions_per_sequence"),
        "protocol final predictions_per_sequence",
    )
    offset = final.get("record_offset")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise Milestone2FinalConfirmationError(
            "protocol final record_offset must be non-negative"
        )
    if (
        sequence_count != 8
        or predictions != 32
        or final.get("prediction_positions") != sequence_count * predictions
        or final.get("required_tokens_per_sequence") != predictions + 1
    ):
        raise Milestone2FinalConfirmationError(
            "final protocol does not describe the frozen 8x32 confirmation"
        )
    reference = protocol.get("candidate_recall_definition", {}).get("reference_top_ks")
    if (
        not isinstance(reference, list)
        or not reference
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in reference
        )
    ):
        raise Milestone2FinalConfirmationError(
            "protocol candidate-recall reference schedule is invalid"
        )
    layer_count = package_manifest.get("model", {}).get("num_hidden_layers")
    if layer_count != len(reference):
        raise Milestone2FinalConfirmationError(
            "candidate-recall schedule differs from package layer count"
        )

    request = Milestone2FinalRequest(
        protocol_path=pinned_protocol_path,
        authorization_manifest_path=authorization_path.resolve(),
        package=package,
        package_manifest=paths["package_manifest"],
        record_artifact=paths["base_artifact"],
        coordinate_index=paths["coordinate_index"],
        policy_manifest=paths["policy_manifest"],
        tokenizer_json=paths["tokenizer_json"],
        dense_native_library=paths["dense_native_library"],
        native_library=paths["native_library"],
        compiler_build_manifest=paths["compiler_build_manifest"],
        dataset=paths["dataset"],
        raw_report=raw_report_path.resolve(),
        record_offset=offset,
        sequence_count=sequence_count,
        predictions_per_sequence=predictions,
        threads=threads,
        reference_top_ks=tuple(reference),
        expected_sha256=dict(expected),
    )
    return _Preflight(
        protocol=protocol,
        authorization=authorization,
        request=request,
        implementation_commit=commit,
        authorization_sha256=sha256_file(authorization_path),
        observed_sha256=dict(observed),
        policy_layers=policy_layers,
        adjudicator_commit=adjudicator_commit,
    )


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(
        f".{path.name}.exclusive-{os.getpid()}-{uuid.uuid4().hex}"
    )
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-filesystem hard link publishes the fully written inode
            # atomically and fails rather than replacing an existing audit
            # record.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise Milestone2FinalConfirmationError(
                f"refusing to overwrite existing audit file: {path}"
            ) from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _default_evaluator(
    request: Milestone2FinalRequest,
) -> Mapping[str, Any]:
    try:
        from engram.evaluation.native_bitnet_dip_native_causal import (
            evaluate_native_bitnet_dip_native_causal,
        )
    except ImportError as exc:
        raise Milestone2FinalConfirmationError(
            "native Milestone 2 causal evaluator is unavailable"
        ) from exc
    return evaluate_native_bitnet_dip_native_causal(
        request.package,
        request.coordinate_index,
        request.dataset,
        out=request.raw_report,
        sequence_count=request.sequence_count,
        predictions_per_sequence=request.predictions_per_sequence,
        record_offset=request.record_offset,
        dataset_role="final",
        dense_library=request.dense_native_library,
        dip_library=request.native_library,
        threads=request.threads,
        debug_recall=True,
        reference_top_ks=request.reference_top_ks,
        expected_layer_count=len(request.reference_top_ks),
    )


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Milestone2FinalConfirmationError(
            f"native evaluator {label} is missing or non-finite"
        )
    return float(value)


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Milestone2FinalConfirmationError(
            f"native evaluator {label} must be Boolean"
        )
    return value


def _reconciled_ratio(
    reported: Any,
    numerator: int,
    denominator: int,
    label: str,
) -> float:
    if denominator <= 0 or numerator < 0 or numerator > denominator:
        raise Milestone2FinalConfirmationError(
            f"native evaluator {label} integer counts are invalid"
        )
    derived = numerator / denominator
    observed = _number(reported, label)
    if not 0.0 <= observed <= 1.0 or not math.isclose(
        observed,
        derived,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise Milestone2FinalConfirmationError(
            f"native evaluator {label} does not reconcile with integer counts"
        )
    return derived


def _validate_evaluator_report(
    report: Mapping[str, Any],
    preflight: _Preflight,
    *,
    dataset_sha256: str,
    token_identities: _TokenIdentities,
    historical_hash_schema: bool = False,
) -> tuple[bool, dict[str, Any]]:
    if (
        not isinstance(report, Mapping)
        or report.get("experiment") != "native_bitnet_dip_native_causal"
        or report.get("dataset_role") != "final"
    ):
        raise Milestone2FinalConfirmationError(
            "native causal evaluator returned an unsupported final report"
        )
    artifacts = report.get("artifacts")
    dataset = report.get("dataset")
    execution = report.get("execution")
    quality = report.get("quality")
    selected = report.get("selected_records")
    traffic = report.get("physical_cold_traffic")
    recall = report.get("debug_recall")
    parity = report.get("python_native_parity")
    evidence = report.get("evidence_observed")
    timing = report.get("timing")
    for value, label in (
        (artifacts, "artifacts"),
        (dataset, "dataset"),
        (execution, "execution"),
        (quality, "quality"),
        (selected, "selected_records"),
        (traffic, "physical_cold_traffic"),
        (recall, "debug_recall"),
        (parity, "python_native_parity"),
        (evidence, "evidence_observed"),
        (timing, "timing"),
    ):
        if not isinstance(value, Mapping):
            raise Milestone2FinalConfirmationError(
                f"native evaluator report has no {label} object"
            )
    assert isinstance(artifacts, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(execution, Mapping)
    assert isinstance(quality, Mapping)
    assert isinstance(selected, Mapping)
    assert isinstance(traffic, Mapping)
    assert isinstance(recall, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(parity, Mapping)
    assert isinstance(timing, Mapping)

    expected_artifacts = {
        "package_manifest": "package_manifest",
        "base_record_artifact": "base_artifact",
        "coordinate_index": "coordinate_index",
        "dense_kernel_library": "dense_native_library",
        "dip_kernel_library": "native_library",
    }
    for report_name, pin_name in expected_artifacts.items():
        descriptor = artifacts.get(report_name)
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("sha256") != preflight.observed_sha256[pin_name]
            or descriptor.get("bytes")
            != getattr(
                preflight.request,
                {
                    "package_manifest": "package_manifest",
                    "base_artifact": "record_artifact",
                    "coordinate_index": "coordinate_index",
                    "dense_native_library": "dense_native_library",
                    "native_library": "native_library",
                }[pin_name],
            )
            .stat()
            .st_size
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator {report_name} differs from frozen input"
            )
    request = preflight.request
    if (
        dataset.get("sha256") != dataset_sha256
        or dataset.get("record_offset") != request.record_offset
        or dataset.get("sequence_count") != request.sequence_count
        or dataset.get("predictions_per_sequence") != request.predictions_per_sequence
        or dataset.get("required_input_tokens_per_sequence")
        != request.predictions_per_sequence + 1
        or dataset.get("prediction_positions")
        != request.sequence_count * request.predictions_per_sequence
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator did not execute the exact frozen 8x32 slice"
        )
    input_token_hash = _sha256(
        dataset.get("input_token_ids_sha256"),
        "evaluator dataset.input_token_ids_sha256",
    )
    sequence_hashes = dataset.get("sequence_token_ids_sha256")
    if (
        not isinstance(sequence_hashes, list)
        or len(sequence_hashes) != request.sequence_count
        or any(_SHA256.fullmatch(str(value)) is None for value in sequence_hashes)
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator token-sequence hashes are incomplete"
        )
    expected_prefix_hashes = list(token_identities.scored_prefix_hashes)
    if (
        input_token_hash != token_identities.scored_prefix_input_ids_sha256
        or sequence_hashes != expected_prefix_hashes
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator scored-prefix token identities differ from the "
            "independently derived frozen slice"
        )
    if not historical_hash_schema:
        full_hashes = dataset.get("canonical_full_sequence_token_ids_sha256")
        full_lengths = dataset.get("canonical_full_sequence_token_lengths")
        scored_hashes = dataset.get("scored_prefix_token_ids_sha256")
        if (
            dataset.get("canonical_full_sequence_hash_algorithm")
            != CANONICAL_TOKEN_HASH_ALGORITHM
            or full_hashes != list(token_identities.full_sequence_hashes)
            or full_lengths != list(token_identities.full_token_lengths)
            or dataset.get("scored_prefix_hash_algorithm")
            != SCORED_PREFIX_TOKEN_HASH_ALGORITHM
            or scored_hashes != expected_prefix_hashes
            or dataset.get("scored_prefix_input_token_ids_sha256")
            != token_identities.scored_prefix_input_ids_sha256
        ):
            raise Milestone2FinalConfirmationError(
                "native evaluator explicit full/prefix token identity schema "
                "is incomplete or inconsistent"
            )

    configuration = report.get("configuration")
    reference_schedule = report.get("reference_top_ks")
    if (
        not isinstance(reference_schedule, Mapping)
        or reference_schedule.get("values") != list(request.reference_top_ks)
        or reference_schedule.get("sha256")
        != sha256_json(list(request.reference_top_ks))
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator reference top-K schedule differs from protocol"
        )
    if not isinstance(configuration, Mapping) or set(configuration) != {
        str(layer) for layer in range(len(request.reference_top_ks))
    }:
        raise Milestone2FinalConfirmationError(
            "native evaluator configuration does not cover every layer"
        )
    effective_fields = (
        "input_coordinates",
        "candidate_count",
        "minimum_top_k",
        "maximum_top_k",
        "energy_target",
        "rms_audit_count",
        "rms_estimator",
        "rms_audit_strategy",
        "rms_variance_scale",
        "rms_variance_bias",
        "output_scale",
    )
    if all(
        all(field in row for field in effective_fields)
        for row in preflight.policy_layers
    ):
        for layer, frozen in enumerate(preflight.policy_layers):
            observed = configuration[str(layer)]
            if not isinstance(observed, Mapping) or any(
                observed.get(field) != frozen.get(field) for field in effective_fields
            ):
                raise Milestone2FinalConfirmationError(
                    f"native evaluator policy differs at layer {layer}"
                )

    thresholds = preflight.protocol["quality_thresholds"]
    practical_thresholds = preflight.protocol["practical_router_thresholds"]
    mean_kl = _number(
        quality.get("mean_kl_divergence"),
        "quality.mean_kl_divergence",
    )
    top1 = _number(
        quality.get("top1_agreement"),
        "quality.top1_agreement",
    )
    reference_nll = _number(
        quality.get("reference_nll"),
        "quality.reference_nll",
    )
    candidate_nll = _number(
        quality.get("candidate_nll"),
        "quality.candidate_nll",
    )
    nll_delta = _number(
        quality.get("nll_delta"),
        "quality.nll_delta",
    )
    hidden_relative_l2 = _number(
        quality.get("final_hidden_relative_l2"),
        "quality.final_hidden_relative_l2",
    )
    if (
        mean_kl < 0.0
        or not 0.0 <= top1 <= 1.0
        or reference_nll < 0.0
        or candidate_nll < 0.0
        or hidden_relative_l2 < 0.0
        or not math.isclose(
            candidate_nll - reference_nll,
            nll_delta,
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator quality metrics are invalid or unreconciled"
        )
    quality_checks = {
        "teacher_student_kl": mean_kl
        <= _number(
            thresholds.get("maximum_teacher_student_kl"),
            "protocol maximum_teacher_student_kl",
        ),
        "top1_agreement": top1
        >= _number(
            thresholds.get("minimum_top1_agreement"),
            "protocol minimum_top1_agreement",
        ),
        "nll_delta": nll_delta
        <= _number(
            thresholds.get("maximum_nll_delta"),
            "protocol maximum_nll_delta",
        ),
        "final_hidden_relative_l2": hidden_relative_l2
        <= _number(
            thresholds.get("maximum_final_hidden_relative_l2"),
            "protocol maximum_final_hidden_relative_l2",
        ),
    }

    schedules = selected.get("per_token_layer_k")
    expected_positions = request.sequence_count * request.predictions_per_sequence
    if (
        not isinstance(schedules, list)
        or len(schedules) != expected_positions
        or any(
            not isinstance(row, list)
            or len(row) != len(request.reference_top_ks)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in row
            )
            for row in schedules
        )
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator selected-K schedules are incomplete"
        )
    package_manifest = _json_object(
        request.package_manifest,
        "native package manifest",
    )
    model = package_manifest.get("model", {})
    hidden_size = _positive_integer(model.get("hidden_size"), "hidden_size")
    intermediate_size = _positive_integer(
        model.get("intermediate_size"),
        "intermediate_size",
    )
    for layer in range(len(request.reference_top_ks)):
        layer_configuration = configuration[str(layer)]
        if not isinstance(layer_configuration, Mapping):
            raise Milestone2FinalConfirmationError(
                f"native evaluator configuration[{layer}] is invalid"
            )
        minimum_k = _positive_integer(
            layer_configuration.get("minimum_top_k"),
            f"configuration[{layer}].minimum_top_k",
        )
        maximum_k = _positive_integer(
            layer_configuration.get("maximum_top_k"),
            f"configuration[{layer}].maximum_top_k",
        )
        candidate_count = _positive_integer(
            layer_configuration.get("candidate_count"),
            f"configuration[{layer}].candidate_count",
        )
        if (
            minimum_k > maximum_k
            or maximum_k > candidate_count
            or candidate_count > intermediate_size
            or any(
                not minimum_k <= schedule[layer] <= maximum_k for schedule in schedules
            )
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator selected-K bounds fail at layer {layer}"
            )
    flattened = [value for row in schedules for value in row]
    active_fraction = sum(flattened) / (len(flattened) * intermediate_size)
    selected_global = selected.get("global")
    if (
        not isinstance(selected_global, Mapping)
        or selected_global.get("sum") != sum(flattened)
        or selected_global.get("count") != len(flattened)
        or not math.isclose(
            _number(
                selected_global.get("active_fraction"),
                "selected_records.global.active_fraction",
            ),
            active_fraction,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator active-record accounting does not reconcile"
        )
    active_limit = _number(
        practical_thresholds.get("maximum_mean_active_record_fraction"),
        "protocol maximum mean active fraction",
    )

    input_counts = [
        _positive_integer(
            configuration[str(layer)].get("input_coordinates"),
            f"configuration[{layer}].input_coordinates",
        )
        for layer in range(len(request.reference_top_ks))
    ]
    candidate_counts = [
        _positive_integer(
            configuration[str(layer)].get("candidate_count"),
            f"configuration[{layer}].candidate_count",
        )
        for layer in range(len(request.reference_top_ks))
    ]
    if any(value > hidden_size for value in input_counts) or any(
        value > intermediate_size for value in candidate_counts
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator q/C policy exceeds package dimensions"
        )
    accounting = [
        native_bitnet_dip_physical_accounting(
            hidden_size,
            intermediate_size,
            input_counts=input_counts,
            candidate_counts=candidate_counts,
            top_ks=row,
        )
        for row in schedules
    ]
    recomputed_bytes = [
        int(item["traffic"]["complete_modelled_cold_bytes"]) for item in accounting
    ]
    recomputed_dense = [int(item["traffic"]["dense_q4_bytes"]) for item in accounting]
    fractions = [
        value / dense
        for value, dense in zip(
            recomputed_bytes,
            recomputed_dense,
            strict=True,
        )
    ]
    worst_layer_fraction = max(
        float(layer["fraction_of_dense_q4"])
        for item in accounting
        for layer in item["traffic"]["layers"]
    )
    global_traffic = traffic.get("global")
    worst_token = traffic.get("worst_token")
    worst_layer = traffic.get("worst_layer")
    if (
        traffic.get("accounting_version") != "native_bitnet_dip_dual_layout_v2"
        or not isinstance(global_traffic, Mapping)
        or not isinstance(worst_token, Mapping)
        or not isinstance(worst_layer, Mapping)
        or global_traffic.get("scheduled_cache_line_bytes") != sum(recomputed_bytes)
        or global_traffic.get("dense_q4_bytes") != sum(recomputed_dense)
        or not math.isclose(
            _number(
                worst_token.get("fraction_of_dense_q4"),
                "physical_cold_traffic.worst_token.fraction",
            ),
            max(fractions),
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not math.isclose(
            _number(
                worst_layer.get("fraction_of_dense_q4"),
                "physical_cold_traffic.worst_layer.fraction",
            ),
            worst_layer_fraction,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator v2 physical traffic does not reconcile"
        )
    traffic_limit = _number(
        practical_thresholds.get(
            "maximum_complete_physical_cold_traffic_fraction_of_dense_q4"
        ),
        "protocol maximum physical traffic",
    )
    practical_checks = {
        "mean_active_record_fraction": active_fraction <= active_limit,
        "global_complete_physical_cold_traffic": (
            sum(recomputed_bytes) / sum(recomputed_dense) <= traffic_limit
        ),
        "maximum_token_complete_physical_cold_traffic": (
            max(fractions) <= traffic_limit
        ),
        "worst_layer_complete_physical_cold_traffic": (
            worst_layer_fraction <= traffic_limit
        ),
        "cpu_only": (
            execution.get("device") == "cpu"
            and execution.get("kernel") == "native_cpu"
            and execution.get("input_boundary") == "live_native_bf16"
            and execution.get("dense_threads") == request.threads
            and execution.get("dip_threads") == request.threads
        ),
        "no_dense_mlp_fallback": (
            _bool(execution.get("dense_fallback"), "execution.dense_fallback") is False
            and execution.get("all_mlp_layers_substituted") is True
            and execution.get("timed_sparse_debug_routes") is False
            and execution.get("debug_pass_outside_timing") is True
        ),
    }

    recall_global = recall.get("global")
    layer_rows = recall.get("layers")
    if (
        recall.get("enabled") is not True
        or recall.get("timed") is not False
        or not isinstance(recall_global, Mapping)
        or not isinstance(layer_rows, Mapping)
        or len(layer_rows) != len(request.reference_top_ks)
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator candidate-recall layer evidence is incomplete"
        )
    layer_means: list[float] = []
    layer_hits: list[int] = []
    layer_targets: list[int] = []
    secondary_targets_by_layer: list[int] = []
    secondary_candidate_hits_by_layer: list[int] = []
    secondary_selected_hits_by_layer: list[int] = []
    secondary_candidate_means: list[float] = []
    secondary_selected_means: list[float] = []
    for layer, reference_top_k in enumerate(request.reference_top_ks):
        row = layer_rows.get(str(layer))
        if not isinstance(row, Mapping):
            raise Milestone2FinalConfirmationError(
                "native evaluator candidate-recall row is invalid"
            )
        if (
            row.get("layer") != layer
            or row.get("reference_top_k") != reference_top_k
            or row.get("rows") != expected_positions
        ):
            raise Milestone2FinalConfirmationError(
                "native evaluator candidate-recall layer identity is invalid"
            )
        target_records = _positive_integer(
            row.get("target_records"),
            f"debug_recall.layers[{layer}].target_records",
        )
        candidate_hits = _nonnegative_integer(
            row.get("candidate_hits"),
            f"debug_recall.layers[{layer}].candidate_hits",
        )
        if target_records != expected_positions * reference_top_k:
            raise Milestone2FinalConfirmationError(
                f"native evaluator layer {layer} fixed-K target count "
                "does not reconcile"
            )
        derived_layer_recall = _reconciled_ratio(
            row.get("candidate_micro_recall"),
            candidate_hits,
            target_records,
            f"debug_recall.layers[{layer}].candidate_micro_recall",
        )
        reported_mean = _number(
            row.get("candidate_mean_row_recall"),
            f"debug_recall.layers[{layer}].candidate_mean_row_recall",
        )
        if not 0.0 <= reported_mean <= 1.0 or not math.isclose(
            reported_mean,
            derived_layer_recall,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator layer {layer} mean recall does not "
                "reconcile with fixed-K integer counts"
            )
        p05 = _number(
            row.get("candidate_p05_row_recall"),
            f"debug_recall.layers[{layer}].candidate_p05_row_recall",
        )
        minimum = _number(
            row.get("candidate_minimum_row_recall"),
            f"debug_recall.layers[{layer}].candidate_minimum_row_recall",
        )
        if not 0.0 <= minimum <= p05 <= 1.0 or minimum > reported_mean:
            raise Milestone2FinalConfirmationError(
                f"native evaluator layer {layer} row-recall quantiles "
                "are outside [0, 1]"
            )
        layer_means.append(derived_layer_recall)
        layer_hits.append(candidate_hits)
        layer_targets.append(target_records)
        secondary = row.get(
            "secondary_teacher_positive_utility_recall_clipped_to_"
            "frozen_minimum_and_maximum_k"
        )
        if not isinstance(secondary, Mapping):
            raise Milestone2FinalConfirmationError(
                f"native evaluator layer {layer} secondary recall is missing"
            )
        secondary_targets = _positive_integer(
            secondary.get("target_records"),
            f"debug_recall.layers[{layer}].secondary target_records",
        )
        secondary_candidate_hits = _nonnegative_integer(
            secondary.get("candidate_hits"),
            f"debug_recall.layers[{layer}].secondary candidate_hits",
        )
        secondary_selected_hits = _nonnegative_integer(
            secondary.get("selected_hits"),
            f"debug_recall.layers[{layer}].secondary selected_hits",
        )
        _reconciled_ratio(
            secondary.get("candidate_micro_recall"),
            secondary_candidate_hits,
            secondary_targets,
            f"debug_recall.layers[{layer}].secondary candidate recall",
        )
        _reconciled_ratio(
            secondary.get("selected_micro_recall"),
            secondary_selected_hits,
            secondary_targets,
            f"debug_recall.layers[{layer}].secondary selected recall",
        )
        secondary_candidate_mean = _number(
            secondary.get("candidate_mean_row_recall"),
            f"debug_recall.layers[{layer}].secondary candidate mean",
        )
        secondary_selected_mean = _number(
            secondary.get("selected_mean_row_recall"),
            f"debug_recall.layers[{layer}].secondary selected mean",
        )
        if (
            not 0.0 <= secondary_candidate_mean <= 1.0
            or not 0.0 <= secondary_selected_mean <= 1.0
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator layer {layer} secondary recall mean "
                "is outside [0, 1]"
            )
        secondary_targets_by_layer.append(secondary_targets)
        secondary_candidate_hits_by_layer.append(secondary_candidate_hits)
        secondary_selected_hits_by_layer.append(secondary_selected_hits)
        secondary_candidate_means.append(secondary_candidate_mean)
        secondary_selected_means.append(secondary_selected_mean)
    minimum_recall = _number(
        practical_thresholds.get("minimum_held_out_candidate_recall"),
        "protocol minimum held-out candidate recall",
    )
    global_targets = _positive_integer(
        recall_global.get("target_records"),
        "debug_recall.global.target_records",
    )
    global_hits = _nonnegative_integer(
        recall_global.get("candidate_hits"),
        "debug_recall.global.candidate_hits",
    )
    if global_targets != sum(layer_targets) or global_hits != sum(layer_hits):
        raise Milestone2FinalConfirmationError(
            "native evaluator global recall counts do not equal layer sums"
        )
    global_recall = _reconciled_ratio(
        recall_global.get("candidate_micro_recall"),
        global_hits,
        global_targets,
        "debug_recall.global.candidate_micro_recall",
    )
    recall_checks = {
        "global_micro_membership_recall": (global_recall >= minimum_recall),
        "each_layer_mean_recall": min(layer_means) >= minimum_recall,
    }
    macro_recall = _number(
        recall_global.get("macro_mean_layer_recall"),
        "debug_recall.global.macro_mean_layer_recall",
    )
    if not math.isclose(
        macro_recall,
        sum(layer_means) / len(layer_means),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator macro candidate recall does not reconcile"
        )
    reported_minimum_layer = _number(
        recall_global.get("candidate_minimum_layer_mean_recall"),
        "debug_recall.global.candidate_minimum_layer_mean_recall",
    )
    if not math.isclose(
        reported_minimum_layer,
        min(layer_means),
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator minimum layer mean recall does not reconcile"
        )
    global_secondary = recall_global.get(
        "secondary_teacher_positive_utility_recall_clipped_to_"
        "frozen_minimum_and_maximum_k"
    )
    if not isinstance(global_secondary, Mapping):
        raise Milestone2FinalConfirmationError(
            "native evaluator global secondary recall is missing"
        )
    secondary_global_targets = _positive_integer(
        global_secondary.get("target_records"),
        "debug_recall.global.secondary target_records",
    )
    secondary_global_candidate_hits = _nonnegative_integer(
        global_secondary.get("candidate_hits"),
        "debug_recall.global.secondary candidate_hits",
    )
    secondary_global_selected_hits = _nonnegative_integer(
        global_secondary.get("selected_hits"),
        "debug_recall.global.secondary selected_hits",
    )
    if (
        secondary_global_targets != sum(secondary_targets_by_layer)
        or secondary_global_candidate_hits != sum(secondary_candidate_hits_by_layer)
        or secondary_global_selected_hits != sum(secondary_selected_hits_by_layer)
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator global secondary recall counts do not equal layer sums"
        )
    _reconciled_ratio(
        global_secondary.get("candidate_micro_recall"),
        secondary_global_candidate_hits,
        secondary_global_targets,
        "debug_recall.global.secondary candidate recall",
    )
    _reconciled_ratio(
        global_secondary.get("selected_micro_recall"),
        secondary_global_selected_hits,
        secondary_global_targets,
        "debug_recall.global.secondary selected recall",
    )
    for name, reported, values in (
        (
            "candidate",
            global_secondary.get("candidate_macro_mean_layer_recall"),
            secondary_candidate_means,
        ),
        (
            "selected",
            global_secondary.get("selected_macro_mean_layer_recall"),
            secondary_selected_means,
        ),
    ):
        observed = _number(
            reported,
            f"debug_recall.global.secondary {name} macro mean",
        )
        if not math.isclose(
            observed,
            sum(values) / len(values),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator global secondary {name} macro mean "
                "does not reconcile"
            )

    parity_layers = parity.get("layers")
    required_parity_checks = {
        "output_bf16",
        "input_coordinate_ids",
        "candidate_ids",
        "selected_counts",
        "selected_record_ids",
    }
    if (
        parity.get("evaluated") is not True
        or parity.get("rows_per_layer") != 1
        or parity.get("all_layers") is not True
        or parity.get("passed") is not True
        or not isinstance(parity_layers, Mapping)
        or set(parity_layers)
        != {str(layer) for layer in range(len(request.reference_top_ks))}
    ):
        raise Milestone2FinalConfirmationError(
            "native evaluator Python/native parity evidence is incomplete"
        )
    for layer in range(len(request.reference_top_ks)):
        parity_row = parity_layers[str(layer)]
        if not isinstance(parity_row, Mapping):
            raise Milestone2FinalConfirmationError(
                f"native evaluator Python/native parity row {layer} is invalid"
            )
        checks = parity_row.get("checks")
        if (
            parity_row.get("layer") != layer
            or parity_row.get("rows") != 1
            or parity_row.get("passed") is not True
            or not isinstance(checks, Mapping)
            or set(checks) != required_parity_checks
            or any(checks[name] is not True for name in required_parity_checks)
        ):
            raise Milestone2FinalConfirmationError(
                f"native evaluator Python/native parity failed at layer {layer}"
            )
        for hash_name in (
            "live_input_bf16_sha256",
            "native_output_bf16_sha256",
            "python_output_bf16_sha256",
        ):
            _sha256(
                parity_row.get(hash_name),
                f"python_native_parity.layers[{layer}].{hash_name}",
            )

    evidence_checks = {
        "serialized_index_reload": (execution.get("serialized_index_reloaded") is True),
        "python_native_numerical_parity": (
            execution.get("python_native_parity_passed") is True
            and parity.get("evaluated") is True
            and parity.get("all_layers") is True
            and parity.get("passed") is True
        ),
        "timed_debug_parity": (
            recall.get("timed_sparse_parity", {}).get("passed") is True
        ),
        "measured_cpu_latency": (
            _number(
                timing.get("timed_sparse_seconds"),
                "timing.timed_sparse_seconds",
            )
            > 0
        ),
        "cache_line_honest_traffic_accounting": (
            traffic.get("accounting_version") == "native_bitnet_dip_dual_layout_v2"
        ),
        "exact_evidence_shape": (
            evidence.get("sequences") == request.sequence_count
            and evidence.get("unique_sequences") == request.sequence_count
            and evidence.get("predictions_per_sequence")
            == request.predictions_per_sequence
            and evidence.get("prediction_positions") == expected_positions
            and evidence.get("all_mlp_layers") is True
            and evidence.get("layers_executed")
            == list(range(len(request.reference_top_ks)))
        ),
    }
    evaluator_attestations = {
        "quality_passed": (
            quality.get("passed") is True and report.get("quality_passed") is True
        ),
        "active_budget_passed": (
            report.get("active_record_budget", {}).get("passes_25_percent") is True
        ),
        "traffic_passed": traffic.get("passes_45_percent") is True,
        "candidate_recall_passed": (
            report.get("candidate_recall_passed") is True
            and recall_global.get("passes_95_percent") is True
        ),
        "systems_evidence_passed": (report.get("systems_evidence_passed") is True),
        "protocol_qualifying": (
            report.get("scoring_protocol_valid") is True
            and report.get("evidence_passed") is True
            and report.get("protocol_qualifying") is True
        ),
        "overall_gate_passed": report.get("overall_gate_passed") is True,
    }
    checks = {
        "quality": quality_checks,
        "practical": practical_checks,
        "candidate_recall": recall_checks,
        "system_evidence": evidence_checks,
        "evaluator_attestations": evaluator_attestations,
    }
    passed = all(value for group in checks.values() for value in group.values())
    return passed, checks


def _validate_consumed_hash_contract_error(
    *,
    result_path: Path,
    marker_path: Path,
    raw_report_path: Path,
    authorization_path: Path,
    preflight: _Preflight,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, bool]]:
    result = _json_object(result_path, "original final result")
    marker = _json_object(marker_path, "original opened marker")
    expected_error = {
        "type": "Milestone2FinalConfirmationError",
        "message": (
            "native evaluator token sequences differ from frozen canonical hashes"
        ),
    }
    marker_dataset = marker.get("dataset")
    marker_result = marker.get("result")
    checks = {
        "result_is_consumed_execution_error": (
            result.get("format") == FINAL_CONFIRMATION_FORMAT
            and result.get("version") == FINAL_CONFIRMATION_VERSION
            and result.get("status") == "error"
            and result.get("opened") is True
            and result.get("phase") == "execution"
            and result.get("decision") == "final_holdout_consumed_with_error"
            and result.get("milestone_2_passed") is False
            and result.get("error") == expected_error
        ),
        "marker_is_terminal_nonreusable_error": (
            marker.get("format")
            == f"{FINAL_CONFIRMATION_FORMAT}-opened-marker"
            and marker.get("version") == FINAL_CONFIRMATION_VERSION
            and marker.get("status") == "error"
            and marker.get("reuse_allowed") is False
            and marker.get("error") == expected_error
        ),
        "attempt_identity_matches": (
            isinstance(result.get("attempt_id"), str)
            and result.get("attempt_id") == marker.get("attempt_id")
        ),
        "authorization_matches": (
            marker.get("authorization_manifest_sha256")
            == preflight.authorization_sha256
            and result.get("authorization_manifest")
            == str(authorization_path)
        ),
        "protocol_matches": (
            marker.get("protocol_sha256")
            == preflight.authorization["protocol"]["sha256"]
            and result.get("protocol") == str(preflight.request.protocol_path)
        ),
        "implementation_matches": (
            marker.get("implementation_commit")
            == preflight.implementation_commit
        ),
        "dataset_matches": (
            isinstance(marker_dataset, Mapping)
            and marker_dataset.get("path") == str(preflight.request.dataset)
            and marker_dataset.get("expected_sha256")
            == preflight.request.expected_sha256["dataset"]
            and marker_dataset.get("actual_sha256")
            == preflight.request.expected_sha256["dataset"]
            and result.get("dataset_sha256")
            == preflight.request.expected_sha256["dataset"]
        ),
        "result_is_bound_by_marker": (
            isinstance(marker_result, Mapping)
            and marker_result.get("path") == str(result_path)
            and marker_result.get("sha256") == sha256_file(result_path)
            and marker_result.get("milestone_2_passed") is False
        ),
        "raw_report_is_separate_and_present": (
            raw_report_path.is_file()
            and raw_report_path != result_path
            and raw_report_path != marker_path
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise Milestone2FinalConfirmationError(
            "original consumed-attempt audit is inconsistent: "
            + ", ".join(failed)
        )
    return result, marker, checks


def _unique_evidence_origin_commit(
    root: Path,
    paths: Sequence[Path],
    *,
    original_implementation_commit: str,
) -> str:
    origins: set[str] = set()
    for path in paths:
        relative = _repository_relative(root, path, "sealed evidence")
        history = [
            value
            for value in _git(
                root,
                "log",
                "--format=%H",
                "--",
                relative,
            ).stdout.splitlines()
            if value
        ]
        additions = [
            value
            for value in _git(
                root,
                "log",
                "--diff-filter=A",
                "--format=%H",
                "--",
                relative,
            ).stdout.splitlines()
            if value
        ]
        if (
            len(history) != 1
            or len(additions) != 1
            or history[0] != additions[0]
        ):
            raise Milestone2FinalConfirmationError(
                f"sealed evidence was modified after its unique origin: {relative}"
            )
        origin = history[0].lower()
        origin_blob = _git(
            root,
            "rev-parse",
            f"{origin}:{relative}",
        ).stdout.strip()
        head_blob = _git(
            root,
            "rev-parse",
            f"HEAD:{relative}",
        ).stdout.strip()
        if not origin_blob or origin_blob != head_blob:
            raise Milestone2FinalConfirmationError(
                f"sealed evidence blob differs from its origin: {relative}"
            )
        origins.add(origin)
    if len(origins) != 1:
        raise Milestone2FinalConfirmationError(
            "sealed evidence files do not share one origin commit"
        )
    origin_commit = next(iter(origins))
    for ancestor, descendant in (
        (original_implementation_commit, origin_commit),
        (origin_commit, "HEAD"),
    ):
        relation = _git(
            root,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            check=False,
        )
        if relation.returncode != 0:
            raise Milestone2FinalConfirmationError(
                "evidence-seal Git ancestry is invalid"
            )
    return origin_commit


def _validate_evidence_seal(
    *,
    root: Path,
    seal_path: Path,
    archived_authorization_path: Path,
    authorization_path: Path,
    result_path: Path,
    marker_path: Path,
    raw_report_path: Path,
    preflight: _Preflight,
) -> tuple[dict[str, Any], dict[str, bool], str]:
    seal = _json_object(seal_path, "consumed-attempt evidence seal")
    archived_authorization_hash = sha256_file(archived_authorization_path)
    authorization_hash = sha256_file(authorization_path)
    authorization_record = seal.get("authorization")
    result_record = seal.get("result")
    marker_record = seal.get("opened_marker")
    raw_record = seal.get("raw_evaluator_report")
    audit = preflight.authorization.get("audit")
    checks = {
        "schema": (
            seal.get("format")
            == "engram-native-bitnet-m2-consumed-attempt-evidence-seal"
            and seal.get("version") == 1
        ),
        "attempt_identity": (
            isinstance(audit, Mapping)
            and seal.get("dataset_key") == audit.get("dataset_key")
            and seal.get("attempt_key") == audit.get("attempt_key")
            and seal.get("implementation_commit")
            == preflight.implementation_commit
        ),
        "authorization": (
            isinstance(authorization_record, Mapping)
            and authorization_record.get("original_path")
            == _repository_relative(
                root,
                authorization_path,
                "original authorization",
            )
            and authorization_record.get("archived_path")
            == _repository_relative(
                root,
                archived_authorization_path,
                "archived authorization",
            )
            and authorization_record.get("sha256")
            == preflight.authorization_sha256
            and authorization_hash == preflight.authorization_sha256
            and archived_authorization_hash == preflight.authorization_sha256
        ),
        "protocol_policy_and_dataset": (
            seal.get("protocol_sha256")
            == preflight.authorization["protocol"]["sha256"]
            and seal.get("policy_sha256")
            == preflight.request.expected_sha256["policy_manifest"]
            and seal.get("dataset_sha256")
            == preflight.request.expected_sha256["dataset"]
            and seal.get("compiler_build_manifest_sha256")
            == preflight.request.expected_sha256["compiler_build_manifest"]
        ),
        "original_result": (
            isinstance(result_record, Mapping)
            and result_record.get("path")
            == _repository_relative(root, result_path, "original result")
            and result_record.get("sha256") == sha256_file(result_path)
        ),
        "opened_marker": (
            isinstance(marker_record, Mapping)
            and marker_record.get("path")
            == _repository_relative(root, marker_path, "opened marker")
            and marker_record.get("sha256") == sha256_file(marker_path)
        ),
        "raw_evaluator_report": (
            isinstance(raw_record, Mapping)
            and raw_record.get("path")
            == _repository_relative(
                root,
                raw_report_path,
                "raw evaluator report",
            )
            and raw_record.get("sha256") == sha256_file(raw_report_path)
        ),
    }
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise Milestone2FinalConfirmationError(
            "consumed-attempt evidence seal is inconsistent: "
            + ", ".join(failed)
        )
    seal_commit = _unique_evidence_origin_commit(
        root,
        (
            seal_path,
            archived_authorization_path,
            result_path,
            marker_path,
            raw_report_path,
        ),
        original_implementation_commit=preflight.implementation_commit,
    )
    return seal, checks, seal_commit


def _historical_source_blobs(
    root: Path,
    commit: str,
) -> dict[str, dict[str, str]]:
    descriptors: dict[str, dict[str, str]] = {}
    for label, relative in _RUNTIME_SOURCE_PATHS.items():
        blob = _git(root, "rev-parse", f"{commit}:{relative}").stdout.strip()
        if not blob:
            raise Milestone2FinalConfirmationError(
                f"cannot resolve historical {label} source"
            )
        descriptors[label] = {
            "path": relative,
            "git_blob": blob,
        }
    return descriptors


def adjudicate_native_bitnet_m2_final_confirmation(
    protocol: str | Path,
    authorization_manifest: str | Path,
    *,
    out: str | Path,
    confirm_adjudicate: bool = False,
) -> dict[str, Any]:
    """Adjudicate the preserved hash-contract failure without model execution.

    This function is intentionally narrower than the one-shot runner.  It only
    accepts the canonical consumed attempt, requires its three evidence files
    to be committed and clean, independently reconstructs full-record and
    scored-prefix token identities, and runs the original primitive validator
    against the immutable raw report.  It never invokes the causal evaluator.
    """

    if not confirm_adjudicate:
        raise Milestone2FinalConfirmationError(
            "postmortem adjudication requires confirm_adjudicate=True"
        )
    protocol_path = Path(protocol).resolve()
    authorization_path = Path(authorization_manifest).resolve()
    output_path = Path(out).resolve()
    authorization = _json_object(
        authorization_path,
        "final authorization manifest",
    )
    implementation = authorization.get("implementation")
    if not isinstance(implementation, Mapping):
        raise Milestone2FinalConfirmationError(
            "final authorization implementation pin is missing"
        )
    root = Path(str(implementation.get("repository_root", ""))).resolve()
    result_path, marker_path, raw_report_path = _pinned_audit_paths(
        root,
        authorization,
    )
    seal_path, archived_authorization_path = _pinned_evidence_seal_paths(
        root,
        authorization,
    )
    expected_output = _pinned_adjudication_path(root, authorization)
    if output_path != expected_output:
        raise Milestone2FinalConfirmationError(
            "adjudication output path differs from the canonical attempt path"
        )
    if output_path.exists():
        raise Milestone2FinalConfirmationError(
            "refusing to overwrite an existing final adjudication"
        )
    if not all(
        path.is_file()
        for path in (
            result_path,
            marker_path,
            raw_report_path,
            seal_path,
            archived_authorization_path,
        )
    ):
        raise Milestone2FinalConfirmationError(
            "canonical consumed-attempt evidence is incomplete"
        )

    preflight = _preflight(
        protocol_path,
        authorization_path,
        result_path=result_path,
        opened_marker_path=marker_path,
        raw_report_path=raw_report_path,
        historical_adjudication=True,
        adjudication_output_path=output_path,
        additional_adjudication_evidence_paths=(
            seal_path,
            archived_authorization_path,
        ),
    )
    original_result, original_marker, original_checks = (
        _validate_consumed_hash_contract_error(
            result_path=result_path,
            marker_path=marker_path,
            raw_report_path=raw_report_path,
            authorization_path=authorization_path,
            preflight=preflight,
        )
    )
    del original_marker
    evidence_seal, seal_checks, seal_commit = _validate_evidence_seal(
        root=root,
        seal_path=seal_path,
        archived_authorization_path=archived_authorization_path,
        authorization_path=authorization_path,
        result_path=result_path,
        marker_path=marker_path,
        raw_report_path=raw_report_path,
        preflight=preflight,
    )
    del evidence_seal

    input_paths = {
        "package_manifest": preflight.request.package_manifest,
        "base_artifact": preflight.request.record_artifact,
        "coordinate_index": preflight.request.coordinate_index,
        "policy_manifest": preflight.request.policy_manifest,
        "tokenizer_json": preflight.request.tokenizer_json,
        "dense_native_library": preflight.request.dense_native_library,
        "native_library": preflight.request.native_library,
        "compiler_build_manifest": (
            preflight.request.compiler_build_manifest
        ),
        "dataset": preflight.request.dataset,
    }
    observed_inputs = {
        name: sha256_file(path) for name, path in input_paths.items()
    }
    if observed_inputs != dict(preflight.request.expected_sha256):
        raise Milestone2FinalConfirmationError(
            "adjudication inputs differ from the original authorization"
        )
    if sha256_file(protocol_path) != preflight.authorization["protocol"]["sha256"]:
        raise Milestone2FinalConfirmationError(
            "adjudication protocol differs from the original authorization"
        )

    token_identities = _derive_final_token_identities(preflight.request)
    token_checks = _validate_protocol_token_identities(
        preflight.protocol,
        token_identities,
    )
    raw_report = _json_object(
        raw_report_path,
        "original native causal raw report",
    )
    passed, evaluator_checks = _validate_evaluator_report(
        raw_report,
        preflight,
        dataset_sha256=observed_inputs["dataset"],
        token_identities=token_identities,
        historical_hash_schema=True,
    )
    current_commit = preflight.adjudicator_commit
    if (
        not isinstance(current_commit, str)
        or _COMMIT.fullmatch(current_commit) is None
    ):
        raise Milestone2FinalConfirmationError(
            "historical preflight did not bind the adjudicator commit"
        )
    evidence = {
        "original_result": {
            "path": _repository_relative(
                root,
                result_path,
                "original result",
            ),
            "bytes": result_path.stat().st_size,
            "sha256": sha256_file(result_path),
        },
        "opened_marker": {
            "path": _repository_relative(
                root,
                marker_path,
                "opened marker",
            ),
            "bytes": marker_path.stat().st_size,
            "sha256": sha256_file(marker_path),
        },
        "raw_evaluator_report": {
            "path": _repository_relative(
                root,
                raw_report_path,
                "raw evaluator report",
            ),
            "bytes": raw_report_path.stat().st_size,
            "sha256": sha256_file(raw_report_path),
        },
    }
    publication_paths = {
        **input_paths,
        "authorization_manifest": authorization_path,
        "archived_authorization": archived_authorization_path,
        "protocol": protocol_path,
        "original_result": result_path,
        "opened_marker": marker_path,
        "raw_evaluator_report": raw_report_path,
        "evidence_seal": seal_path,
    }
    publication_hashes = {
        name: sha256_file(path)
        for name, path in publication_paths.items()
    }
    status = "pass" if passed else "fail"
    result = {
        "format": FINAL_ADJUDICATION_FORMAT,
        "version": FINAL_ADJUDICATION_VERSION,
        "status": status,
        "completed_at": _utc_now(),
        "adjudication": {
            "kind": "postmortem_hash_contract_correction",
            "model_or_evaluator_executed": False,
            "original_result_rewritten": False,
            "holdout_reused_for_configuration": False,
            "historical_hash_schema": {
                "protocol": (
                    "full sequence with canonical input_ids object envelope"
                ),
                "raw_evaluator": (
                    "first 33 scored input tokens with bare-list envelope"
                ),
            },
        },
        "implementation": {
            "evaluated_commit": preflight.implementation_commit,
            "evidence_seal_commit": seal_commit,
            "adjudicator_commit": current_commit,
            "historical_runtime_source_blobs": _historical_source_blobs(
                root,
                preflight.implementation_commit,
            ),
        },
        "authorization_manifest": {
            "path": str(authorization_path),
            "bytes": authorization_path.stat().st_size,
            "sha256": preflight.authorization_sha256,
        },
        "protocol": {
            "path": _repository_relative(root, protocol_path, "protocol"),
            "bytes": protocol_path.stat().st_size,
            "sha256": sha256_file(protocol_path),
        },
        "original_attempt": {
            "attempt_id": original_result["attempt_id"],
            "status": original_result["status"],
            "decision": original_result["decision"],
            "error": original_result["error"],
            "evidence": evidence,
            "evidence_seal": {
                "path": _repository_relative(
                    root,
                    seal_path,
                    "evidence seal",
                ),
                "bytes": seal_path.stat().st_size,
                "sha256": sha256_file(seal_path),
            },
        },
        "input_sha256": observed_inputs,
        "publication_input_sha256": publication_hashes,
        "token_identities": token_identities.to_dict(),
        "checks": {
            "evidence_seal": seal_checks,
            "original_attempt": original_checks,
            "protocol_token_identities": token_checks,
            **evaluator_checks,
        },
        "evaluator_report": raw_report,
        "milestone_2_passed": passed,
        "decision": (
            "milestone_2_semantic_gate_passed_by_postmortem_adjudication"
            if passed
            else "milestone_2_semantic_gate_failed_by_postmortem_adjudication"
        ),
        "limitations": {
            "original_one_shot_result_remains_error": True,
            "adjudication_is_post_hoc": True,
            "raw_report_not_contemporaneously_bound": True,
            "raw_report_seal_is_prospective_only": True,
            "host_bound_absolute_paths_and_ignored_artifacts": True,
            "traffic_is_cache_line_model_not_measured_dram": True,
            "latency_was_not_a_frozen_gate": True,
            "confirmation_scale": "8 sequences x 32 prediction positions",
        },
    }
    final_bound_paths = (
        preflight.request.protocol_path,
        preflight.request.policy_manifest,
        preflight.request.dataset,
        result_path,
        marker_path,
        raw_report_path,
        seal_path,
        archived_authorization_path,
        *(
            root.joinpath(*PurePosixPath(relative).parts).resolve()
            for relative in _RUNTIME_SOURCE_PATHS.values()
        ),
    )
    if _git(root, "rev-parse", "HEAD").stdout.strip().lower() != current_commit:
        raise Milestone2FinalConfirmationError(
            "adjudicator Git commit changed before publication"
        )
    _verify_git_state(
        root,
        expected_commit=current_commit,
        tracked_paths=final_bound_paths,
        excluded_paths=(output_path,),
    )
    _verify_worktree_blobs_at_commit(
        root,
        commit=current_commit,
        paths=final_bound_paths,
    )
    if {
        name: sha256_file(path)
        for name, path in publication_paths.items()
    } != publication_hashes:
        raise Milestone2FinalConfirmationError(
            "an adjudication input changed before publication"
        )
    _exclusive_json(output_path, result)
    return result


def run_native_bitnet_m2_final_confirmation(
    protocol: str | Path,
    authorization_manifest: str | Path,
    *,
    out: str | Path,
    opened_marker: str | Path,
    confirm_open: bool = False,
) -> dict[str, Any]:
    """Execute the frozen final holdout once, with durable audit records.

    ``confirm_open=True`` is an intentional two-key guard.  The public
    production API has no evaluator-injection path.
    """

    if not confirm_open:
        raise Milestone2FinalConfirmationError(
            "final holdout remains sealed; pass confirm_open=True explicitly"
        )
    protocol_path = Path(protocol).resolve()
    authorization_path = Path(authorization_manifest).resolve()
    output_path = Path(out).resolve()
    marker_path = Path(opened_marker).resolve()
    authorization_preview = _json_object(
        authorization_path,
        "final authorization manifest",
    )
    preview_implementation = authorization_preview.get("implementation")
    if not isinstance(preview_implementation, Mapping):
        raise Milestone2FinalConfirmationError(
            "final authorization implementation pin is missing"
        )
    preview_root = (
        Path(str(preview_implementation.get("repository_root", "")))
        .expanduser()
        .resolve()
    )
    canonical_result, canonical_marker, raw_report = _pinned_audit_paths(
        preview_root,
        authorization_preview,
    )
    if output_path != canonical_result or marker_path != canonical_marker:
        raise Milestone2FinalConfirmationError(
            "caller result/opened-marker paths differ from canonical "
            "authorization audit paths"
        )
    if output_path.exists() or marker_path.exists() or raw_report.exists():
        raise Milestone2FinalConfirmationError(
            "refusing final confirmation because a marker/result already exists"
        )

    attempt_id = str(uuid.uuid4())
    started_at = _utc_now()
    started_ns = time.time_ns()
    preflight: _Preflight | None = None
    dataset_hash: str | None = None
    marker_created = False
    result_reserved = False
    try:
        preflight = _preflight(
            protocol_path,
            authorization_path,
            result_path=output_path,
            opened_marker_path=marker_path,
            raw_report_path=raw_report,
        )
        initial_result = {
            "format": FINAL_CONFIRMATION_FORMAT,
            "version": FINAL_CONFIRMATION_VERSION,
            "attempt_id": attempt_id,
            "status": "ready_to_open",
            "opened": False,
            "started_at": started_at,
            "protocol_sha256": sha256_file(protocol_path),
            "authorization_manifest_sha256": (preflight.authorization_sha256),
            "implementation_commit": preflight.implementation_commit,
        }
        _exclusive_json(output_path, initial_result)
        result_reserved = True

        # Recheck the implementation after reserving the result.  The result
        # path is excluded; the dataset is still excluded and still unread.
        _verify_git_state(
            Path(preflight.authorization["implementation"]["repository_root"])
            .expanduser()
            .resolve(),
            expected_commit=preflight.implementation_commit,
            tracked_paths=(
                preflight.request.protocol_path,
                preflight.request.policy_manifest,
                preflight.request.dataset,
            ),
            excluded_paths=(
                preflight.request.dataset,
                output_path,
                marker_path,
                raw_report,
            ),
        )
        preopen_protocol_hash = sha256_file(protocol_path)
        preopen_authorization_hash = sha256_file(authorization_path)
        if (
            preopen_protocol_hash != preflight.authorization["protocol"]["sha256"]
            or preopen_authorization_hash != preflight.authorization_sha256
        ):
            raise Milestone2FinalConfirmationError(
                "protocol or authorization changed before holdout opening"
            )
        opened_metadata = {
            "format": f"{FINAL_CONFIRMATION_FORMAT}-opened-marker",
            "version": FINAL_CONFIRMATION_VERSION,
            "attempt_id": attempt_id,
            "status": "opened",
            "created_at": _utc_now(),
            "protocol_sha256": preopen_protocol_hash,
            "authorization_manifest_sha256": (preflight.authorization_sha256),
            "implementation_commit": preflight.implementation_commit,
            "dataset": {
                "path": str(preflight.request.dataset),
                "expected_sha256": preflight.request.expected_sha256["dataset"],
                "actual_sha256": None,
            },
            "reuse_allowed": False,
        }
        _exclusive_json(marker_path, opened_metadata)
        marker_created = True

        # This is the first operation that opens/reads holdout contents.
        dataset_hash = sha256_file(preflight.request.dataset)
        opened_metadata["dataset"]["actual_sha256"] = dataset_hash
        if dataset_hash != preflight.request.expected_sha256["dataset"]:
            raise Milestone2FinalConfirmationError(
                "final holdout dataset SHA-256 mismatch after opening"
            )
        token_identities = _derive_final_token_identities(preflight.request)
        _validate_protocol_token_identities(
            preflight.protocol,
            token_identities,
        )
        opened_metadata["status"] = "executing"
        opened_metadata["dataset_authenticated_at"] = _utc_now()
        atomic_json(marker_path, opened_metadata)

        evaluator_report = _default_evaluator(preflight.request)
        if not preflight.request.raw_report.is_file():
            raise Milestone2FinalConfirmationError(
                "native causal evaluator did not persist its raw report"
            )
        persisted_report = _json_object(
            preflight.request.raw_report,
            "native causal evaluator raw report",
        )
        if sha256_json(persisted_report) != sha256_json(dict(evaluator_report)):
            raise Milestone2FinalConfirmationError(
                "native evaluator return value differs from persisted report"
            )

        # Detect any input mutation during execution before accepting metrics.
        post_artifact_hashes = {
            name: sha256_file(path)
            for name, path in {
                "package_manifest": preflight.request.package_manifest,
                "base_artifact": preflight.request.record_artifact,
                "coordinate_index": preflight.request.coordinate_index,
                "policy_manifest": preflight.request.policy_manifest,
                "tokenizer_json": preflight.request.tokenizer_json,
                "dense_native_library": (preflight.request.dense_native_library),
                "native_library": preflight.request.native_library,
                "compiler_build_manifest": (preflight.request.compiler_build_manifest),
                "dataset": preflight.request.dataset,
            }.items()
        }
        post_control_hashes = {
            "protocol": sha256_file(preflight.request.protocol_path),
            "authorization_manifest": sha256_file(
                preflight.request.authorization_manifest_path
            ),
        }
        if (
            post_artifact_hashes != dict(preflight.request.expected_sha256)
            or post_control_hashes["protocol"]
            != preflight.authorization["protocol"]["sha256"]
            or post_control_hashes["authorization_manifest"]
            != preflight.authorization_sha256
        ):
            raise Milestone2FinalConfirmationError(
                "a frozen final-confirmation input changed during execution"
            )
        passed, checks = _validate_evaluator_report(
            persisted_report,
            preflight,
            dataset_sha256=dataset_hash,
            token_identities=token_identities,
        )
        status = "pass" if passed else "fail"
        completed_at = _utc_now()
        result = {
            **initial_result,
            "status": status,
            "opened": True,
            "completed_at": completed_at,
            "elapsed_seconds": (time.time_ns() - started_ns) / 1e9,
            "input_sha256": {
                **post_artifact_hashes,
                **post_control_hashes,
            },
            "checks": checks,
            "token_identities": token_identities.to_dict(),
            "evaluator_report": persisted_report,
            "raw_evaluator_report": {
                "path": str(preflight.request.raw_report),
                "sha256": sha256_file(preflight.request.raw_report),
            },
            "milestone_2_passed": passed,
            "decision": (
                "milestone_2_semantic_gate_passed"
                if passed
                else "milestone_2_semantic_gate_failed_final_holdout"
            ),
        }
        atomic_json(output_path, result)
        opened_metadata.update(
            {
                "status": status,
                "completed_at": completed_at,
                "result": {
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                    "milestone_2_passed": passed,
                },
            }
        )
        atomic_json(marker_path, opened_metadata)
        return result
    except BaseException as exc:
        error_result = {
            "format": FINAL_CONFIRMATION_FORMAT,
            "version": FINAL_CONFIRMATION_VERSION,
            "attempt_id": attempt_id,
            "status": "error",
            "opened": marker_created,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_seconds": (time.time_ns() - started_ns) / 1e9,
            "phase": (
                "execution"
                if marker_created
                else ("ready_to_open" if result_reserved else "preflight")
            ),
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "protocol": str(protocol_path),
            "authorization_manifest": str(authorization_path),
            "dataset_sha256": dataset_hash,
            "milestone_2_passed": False,
            "decision": (
                "final_holdout_consumed_with_error"
                if marker_created
                else "final_holdout_not_opened_preflight_error"
            ),
        }
        try:
            if result_reserved:
                atomic_json(output_path, error_result)
            else:
                _exclusive_json(output_path, error_result)
        except BaseException:
            # Never mask the original failure.  SIGKILL/power loss aside, the
            # result reservation above makes this branch exceptional.
            pass
        if marker_created:
            try:
                marker = _json_object(
                    marker_path,
                    "final opened marker",
                )
                marker.update(
                    {
                        "status": "error",
                        "completed_at": _utc_now(),
                        "error": error_result["error"],
                        "result": {
                            "path": str(output_path),
                            "sha256": (
                                sha256_file(output_path)
                                if output_path.is_file()
                                else None
                            ),
                            "milestone_2_passed": False,
                        },
                    }
                )
                atomic_json(marker_path, marker)
            except BaseException:
                pass
        raise


__all__ = [
    "CANONICAL_TOKEN_HASH_ALGORITHM",
    "COMPILER_BUILD_FORMAT",
    "COMPILER_BUILD_VERSION",
    "FINAL_ADJUDICATION_FORMAT",
    "FINAL_ADJUDICATION_VERSION",
    "FINAL_CONFIRMATION_FORMAT",
    "FINAL_CONFIRMATION_VERSION",
    "Milestone2FinalConfirmationError",
    "Milestone2FinalEvaluator",
    "Milestone2FinalRequest",
    "NATIVE_CAUSAL_EVALUATOR",
    "SCORED_PREFIX_TOKEN_HASH_ALGORITHM",
    "adjudicate_native_bitnet_m2_final_confirmation",
    "run_native_bitnet_m2_final_confirmation",
    "write_native_bitnet_m2_compiler_build_manifest",
    "write_native_bitnet_m2_final_authorization_manifest",
]
