import hashlib
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from engram.semantic.shared_basis import (  # noqa: E402
    ESIB_MANIFEST_FORMAT,
    SharedBasisMLP,
    decode_shared_basis_artifact,
    load_shared_basis_artifact_set,
    shared_basis_traffic,
)
from tests.shared_basis_fixture import write_esib_fixture  # noqa: E402


def _write_manifest(path, artifacts, *, source_hash):
    entries = []
    for artifact_path in artifacts:
        artifact = decode_shared_basis_artifact(artifact_path.read_bytes())
        entries.append(
            {
                "layer": artifact.layer,
                "path": artifact_path.name,
                "sha256": artifact.artifact_sha256,
                "content_checksum": artifact.content_checksum,
            }
        )
    path.write_text(
        json.dumps(
            {
                "format": ESIB_MANIFEST_FORMAT,
                "version": 1,
                "source_model_hash": source_hash,
                "hidden_size": 4,
                "intermediate_size": 6,
                "num_hidden_layers": 2,
                "artifacts": entries,
                "provenance": {"calibration_dataset_hash": "b" * 64},
            }
        ),
        encoding="utf-8",
    )


def test_strict_shared_basis_reload_rejects_corruption_and_binds_manifest(tmp_path):
    paths = []
    for layer in range(2):
        path = tmp_path / f"layer-{layer}.esib"
        write_esib_fixture(path, layer=layer, hidden=4, width=6, rank=3, top_k=4)
        paths.append(path)
    manifest = tmp_path / "artifacts.json"
    _write_manifest(manifest, paths, source_hash="a" * 64)

    loaded = load_shared_basis_artifact_set(
        manifest,
        expected_source_model_hash="a" * 64,
        expected_hidden_size=4,
        expected_intermediate_size=6,
        expected_num_hidden_layers=2,
    )

    assert list(loaded.artifacts) == [0, 1]
    assert len(loaded.artifact_set_sha256) == 64
    assert loaded.manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert all(
        shared_basis_traffic(artifact)["adversarial_total_cold_bytes"] > 0
        for artifact in loaded.artifacts.values()
    )

    corrupted = bytearray(paths[0].read_bytes())
    corrupted[-1] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode_shared_basis_artifact(corrupted)
    with pytest.raises(ValueError, match="source-model hash mismatch"):
        load_shared_basis_artifact_set(manifest, expected_source_model_hash="c" * 64)


def test_shared_basis_module_executes_only_decoded_buffers(tmp_path):
    path = tmp_path / "full-width.esib"
    payload = write_esib_fixture(path, layer=0, hidden=4, width=6, rank=3, top_k=6)
    artifact = decode_shared_basis_artifact(payload)
    module = SharedBasisMLP(artifact, capture=True, execution_chunk_size=2)
    hidden = torch.tensor(
        [[0.1, -0.3, 0.2, 0.4], [-0.5, 0.25, 0.75, -0.1]],
        dtype=torch.float32,
    )

    output = module(hidden)
    latent = torch.nn.functional.linear(hidden, module.basis)
    activation = torch.nn.functional.silu(
        torch.nn.functional.linear(latent, module.gate_coeff)
    ) * torch.nn.functional.linear(latent, module.up_coeff)
    expected = activation @ module.down
    captured, selected = module.pop_capture()

    assert list(module.parameters()) == []
    assert torch.allclose(output, expected, atol=1e-6)
    assert torch.equal(captured, output)
    assert selected.shape == (2, 6)
    assert all(len(np.unique(row)) == 6 for row in selected.numpy())
