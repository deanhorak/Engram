from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def percentile(values: Iterable[float], q: float) -> float:
    import numpy as np

    array = np.asarray(list(values), dtype=np.float64)
    return float(np.percentile(array, q)) if array.size else float("nan")


def npy_file_metadata(path: Path) -> dict[str, Any]:
    """Return explicit portable metadata for a NumPy binary array."""
    import numpy as np

    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY version {version}")
        offset = handle.tell()
    byte_order = dtype.byteorder
    if byte_order == "=":
        byte_order = "little" if sys.byteorder == "little" else "big"
    elif byte_order == "|":
        byte_order = "not_applicable"
    else:
        byte_order = "little" if byte_order == "<" else "big"
    alignment = 1
    while alignment < 64 and offset % (alignment * 2) == 0:
        alignment *= 2
    return {
        "binary_format": "npy",
        "format_version": list(version),
        "dtype": str(dtype),
        "shape": list(shape),
        "byte_order": byte_order,
        "fortran_order": bool(fortran),
        "data_offset": offset,
        "alignment": alignment,
    }
