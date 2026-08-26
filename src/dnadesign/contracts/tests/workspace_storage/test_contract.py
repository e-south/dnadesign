"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/workspace_storage/test_contract.py

Tests strict workspace-storage parsing and filesystem closure.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dnadesign.contracts.workspace_storage import (
    MANIFEST_NAME,
    WorkspaceStorageError,
    verify_workspace_storage,
)


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _manifest(*, input_digest: str, artifact_digest: str, demo: bool = False) -> dict[str, object]:
    return {
        "schema": "dnadesign.workspace-storage/v1",
        "workspace_id": "latentdna-storage-pilot",
        "owner_repository": "dnadesign",
        "owner_tool": "latentdna",
        "workspace_schema": "latentdna.workspace",
        "workspace_schema_version": "1",
        "producer_revision": "test-revision-1",
        "storage_class": "reproducible",
        "retention_policy": "rebuildable",
        "demo": demo,
        "inputs": [{"path": "inputs/payload.txt", "digest": input_digest}],
        "artifacts": [{"path": "outputs/result.json", "digest": artifact_digest}],
        "original_execution_path": "/private/original/latentdna/workspace",
    }


def _write_workspace(workspace_root: Path, *, demo: bool = False) -> dict[str, object]:
    input_content = b"payload\n"
    artifact_content = b'{"status":"ok"}\n'
    (workspace_root / "inputs").mkdir(parents=True)
    (workspace_root / "outputs").mkdir()
    (workspace_root / "inputs" / "payload.txt").write_bytes(input_content)
    (workspace_root / "outputs" / "result.json").write_bytes(artifact_content)
    manifest = _manifest(
        input_digest=_digest(input_content),
        artifact_digest=_digest(artifact_content),
        demo=demo,
    )
    (workspace_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_verify_workspace_storage_closes_paths_and_digests(tmp_path: Path) -> None:
    workspace_root = tmp_path / "latentdna-storage-pilot"
    _write_workspace(workspace_root)

    verified = verify_workspace_storage(workspace_root)

    assert verified.root == workspace_root.resolve()
    assert verified.manifest.workspace_id == "latentdna-storage-pilot"
    assert verified.manifest.original_execution_path == "/private/original/latentdna/workspace"
    assert verified.inputs[0].path == (workspace_root / "inputs" / "payload.txt").resolve()
    assert verified.artifacts[0].path == (workspace_root / "outputs" / "result.json").resolve()
    assert verified.inputs[0].size_bytes == 8


def test_verify_workspace_storage_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    workspace_root = tmp_path / "duplicate-key"
    workspace_root.mkdir()
    (workspace_root / MANIFEST_NAME).write_text(
        '{"schema":"dnadesign.workspace-storage/v1","schema":"wrong"}',
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceStorageError, match="duplicate key 'schema'"):
        verify_workspace_storage(workspace_root)


def test_verify_workspace_storage_rejects_unknown_manifest_fields(tmp_path: Path) -> None:
    workspace_root = tmp_path / "unknown-field"
    manifest = _write_workspace(workspace_root)
    manifest["implicit_fallback_root"] = "/tmp/fallback"
    (workspace_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WorkspaceStorageError, match="unsupported fields: implicit_fallback_root"):
        verify_workspace_storage(workspace_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["inputs"][0].update(path="../escape.txt"), "must be a confined relative path"),
        (lambda manifest: manifest["artifacts"][0].update(digest=f"sha256:{'0' * 64}"), "digest mismatch"),
        (lambda manifest: manifest.update(retention_policy="cold"), "is incompatible with storage_class"),
    ],
)
def test_verify_workspace_storage_fails_fast_on_invalid_state(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    workspace_root = tmp_path / "invalid-state"
    manifest = _write_workspace(workspace_root)
    mutation(manifest)  # type: ignore[operator]
    (workspace_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WorkspaceStorageError, match=message):
        verify_workspace_storage(workspace_root)


def test_verify_workspace_storage_rejects_duplicate_resource_paths(tmp_path: Path) -> None:
    workspace_root = tmp_path / "duplicate-resource"
    manifest = _write_workspace(workspace_root)
    manifest["artifacts"] = [manifest["inputs"][0]]
    (workspace_root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WorkspaceStorageError, match="declared more than once"):
        verify_workspace_storage(workspace_root)


def test_non_demo_workspace_inside_git_checkout_is_rejected(tmp_path: Path) -> None:
    checkout_root = tmp_path / "checkout"
    (checkout_root / ".git").mkdir(parents=True)
    workspace_root = checkout_root / "ignored" / "workspace"
    _write_workspace(workspace_root)

    with pytest.raises(WorkspaceStorageError, match="non-demo workspace cannot live inside a Git checkout"):
        verify_workspace_storage(workspace_root)


def test_explicit_demo_workspace_inside_git_checkout_is_allowed(tmp_path: Path) -> None:
    checkout_root = tmp_path / "checkout"
    (checkout_root / ".git").mkdir(parents=True)
    workspace_root = checkout_root / "examples" / "workspace"
    _write_workspace(workspace_root, demo=True)

    verified = verify_workspace_storage(workspace_root)

    assert verified.manifest.demo is True
