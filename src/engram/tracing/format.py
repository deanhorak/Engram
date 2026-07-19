from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from engram import __version__
from engram.utils import atomic_json, sha256_file


TRACE_SCHEMA_VERSION = 1


class TraceFormatError(ValueError):
    pass


class TraceWriter:
    """Writes independently checksummed, mmap-friendly NPY shards."""

    def __init__(
        self,
        out: str | Path,
        *,
        model_hash: str,
        dataset_hash: str,
        split: str,
        seed: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(out)
        self.path.mkdir(parents=True, exist_ok=True)
        self._shards: list[dict[str, Any]] = []
        self._closed = False
        self._base = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "engram_version": __version__,
            "model_hash": model_hash,
            "dataset_hash": dataset_hash,
            "split": split,
            "seed": seed,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "metadata": dict(metadata or {}),
        }

    def append(self, arrays: Mapping[str, np.ndarray]) -> None:
        if self._closed:
            raise RuntimeError("trace writer is closed")
        if not arrays:
            raise ValueError("cannot append an empty trace shard")
        normalized = {name: np.asarray(value) for name, value in arrays.items()}
        leading = {value.shape[0] for value in normalized.values() if value.ndim > 0}
        if any(value.ndim == 0 for value in normalized.values()) or len(leading) != 1:
            raise ValueError("all trace arrays must have rank >= 1 and the same leading dimension")
        shard_name = f"shard-{len(self._shards):05d}"
        final_dir = self.path / shard_name
        temporary = self.path / f".{shard_name}.tmp"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        fields: dict[str, Any] = {}
        for name in sorted(normalized):
            if "/" in name or ".." in name:
                raise ValueError(f"invalid trace field name {name!r}")
            file_name = f"{name}.npy"
            np.save(temporary / file_name, normalized[name], allow_pickle=False)
            fields[name] = {
                "file": file_name,
                "dtype": str(normalized[name].dtype),
                "shape": list(normalized[name].shape),
                "sha256": sha256_file(temporary / file_name),
            }
        temporary.replace(final_dir)
        self._shards.append(
            {
                "name": shard_name,
                "records": next(iter(leading)),
                "fields": fields,
            }
        )
        self._write_manifest(complete=False)

    def _write_manifest(self, *, complete: bool) -> None:
        manifest = {**self._base, "complete": complete, "shards": self._shards}
        atomic_json(self.path / "manifest.json", manifest)

    def close(self) -> None:
        if not self._closed:
            self._write_manifest(complete=True)
            self._closed = True

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.close()


class TraceReader:
    def __init__(self, path: str | Path, *, verify: bool = True) -> None:
        self.path = Path(path)
        manifest_path = self.path / "manifest.json"
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceFormatError(f"cannot read trace manifest: {exc}") from exc
        if self.manifest.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise TraceFormatError(f"unsupported trace schema {self.manifest.get('schema_version')}")
        if not self.manifest.get("complete"):
            raise TraceFormatError("trace is incomplete")
        if verify:
            self.verify()

    def verify(self) -> None:
        for shard in self.manifest["shards"]:
            for field in shard["fields"].values():
                path = self.path / shard["name"] / field["file"]
                if not path.is_file() or sha256_file(path) != field["sha256"]:
                    raise TraceFormatError(f"trace checksum mismatch: {path}")

    def iter_shards(self, fields: list[str] | None = None) -> Iterator[dict[str, np.ndarray]]:
        for shard in self.manifest["shards"]:
            available = shard["fields"]
            selected = list(available) if fields is None else fields
            missing = set(selected) - set(available)
            if missing:
                raise TraceFormatError(f"missing trace fields: {sorted(missing)}")
            yield {
                name: np.load(self.path / shard["name"] / available[name]["file"], mmap_mode="r")
                for name in selected
            }
