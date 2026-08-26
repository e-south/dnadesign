"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/storage_objects/test_contract.py

Tests exact storage-object and storage-root verification.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from dnadesign.contracts.storage_objects import (
    MANIFEST_NAME,
    StorageObjectError,
    verify_storage_object,
    verify_storage_root,
)


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _write_object(
    root: Path,
    *,
    storage_id: str = "pilot",
    owner_tool: str = "cruncher",
    object_kind: str = "workspace",
    demo: bool = False,
) -> None:
    payload = b"payload\n"
    result = b'{"status":"ok"}\n'
    storage_class = "cache" if object_kind == "tool-cache" else "reproducible"
    retention_policy = "rebuildable" if object_kind == "tool-cache" else "review-before-delete"
    (root / "inputs").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "inputs" / "payload.txt").write_bytes(payload)
    (root / "outputs" / "result.json").write_bytes(result)
    manifest = {
        "schema": "dnadesign.storage-object/v1",
        "storage_id": storage_id,
        "owner_repository": "dnadesign",
        "owner_tool": owner_tool,
        "object_kind": object_kind,
        "content_schema": f"{owner_tool}.{object_kind}",
        "content_schema_version": "1",
        "producer_revision": "test-revision-1",
        "storage_class": storage_class,
        "retention_policy": retention_policy,
        "demo": demo,
        "resources": [
            {"path": "inputs/payload.txt", "digest": _digest(payload), "role": "input"},
            {"path": "outputs/result.json", "digest": _digest(result), "role": "artifact"},
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_storage_object_requires_exact_file_closure(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)

    verified = verify_storage_object(root)

    assert verified.manifest.storage_id == "pilot"
    assert verified.manifest.object_kind.value == "workspace"
    assert [resource.role.value for resource in verified.resources] == ["input", "artifact"]
    assert verified.summary()["resource_count"] == 2

    (root / "undeclared.txt").write_text("not in manifest", encoding="utf-8")
    with pytest.raises(StorageObjectError, match="undeclared files: undeclared.txt"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    (root / "linked.txt").symlink_to(root / "inputs" / "payload.txt")

    with pytest.raises(StorageObjectError, match="symlink is not allowed: linked.txt"):
        verify_storage_object(root)

    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(StorageObjectError, match="storage object root must not be a symlink"):
        verify_storage_object(linked_root)


def test_git_resident_demo_must_be_small_and_tracked(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    _write_object(root, demo=True)

    with pytest.raises(StorageObjectError, match="demo file is not tracked"):
        verify_storage_object(root)

    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)
    assert verify_storage_object(root).manifest.demo is True

    oversized = root / "outputs" / "large.bin"
    oversized.write_bytes(b"x" * 2_000_001)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["resources"].append(
        {"path": "outputs/large.bin", "digest": _digest(oversized.read_bytes()), "role": "artifact"}
    )
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)

    with pytest.raises(StorageObjectError, match="demo exceeds 2000000 bytes"):
        verify_storage_object(root)


def test_external_tool_cache_may_be_a_git_checkout(tmp_path: Path) -> None:
    root = tmp_path / "proteinmpnn"
    _write_object(
        root,
        storage_id="proteinmpnn",
        owner_tool="proteinmpnn",
        object_kind="tool-cache",
    )
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    git_files = sorted(path for path in (root / ".git").rglob("*") if path.is_file())
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].extend(
        {
            "path": path.relative_to(root).as_posix(),
            "digest": _digest(path.read_bytes()),
            "role": "cache",
        }
        for path in git_files
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_storage_object(root).manifest.object_kind.value == "tool-cache"

    manifest["object_kind"] = "workspace"
    manifest["storage_class"] = "reproducible"
    manifest["retention_policy"] = "review-before-delete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageObjectError, match="non-demo storage object cannot live inside"):
        verify_storage_object(root)


def test_verify_storage_root_enforces_shelf_owner_and_identity(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    workspace = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(workspace)
    cache = storage_root / "tool-cache" / "proteinmpnn" / "revision-1"
    _write_object(
        cache,
        storage_id="revision-1",
        owner_tool="proteinmpnn",
        object_kind="tool-cache",
    )
    (storage_root / "stores").mkdir()
    (storage_root / "AGENTS.md").write_text("# Router\n", encoding="utf-8")

    verified = verify_storage_root(storage_root)

    assert verified.summary()["object_count"] == 2
    assert verified.summary()["objects_by_kind"] == {"tool-cache": 1, "workspace": 1}

    manifest = json.loads((workspace / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest["storage_id"] = "wrong-id"
    (workspace / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(StorageObjectError, match="storage_id 'wrong-id' does not match directory 'pilot'"):
        verify_storage_root(storage_root)


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        ("README.md", "unexpected path in storage root"),
        ("misc/pilot.txt", "unexpected path in storage root"),
        ("workspaces/stray.txt", "unexpected path in storage shelf"),
        ("workspaces/cruncher/stray.txt", "unexpected path in storage owner directory"),
    ],
)
def test_verify_storage_root_rejects_unrouted_content(
    tmp_path: Path,
    relative_path: str,
    message: str,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    stray = storage_root / relative_path
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text("stray\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match=message):
        verify_storage_root(storage_root)


@pytest.mark.parametrize("relative_path", ["workspaces", "workspaces/cruncher/pilot"])
def test_verify_storage_root_rejects_routing_symlinks(
    tmp_path: Path,
    relative_path: str,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    target = storage_root / relative_path
    if target == storage_root / "workspaces":
        target.rmdir()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(external, target_is_directory=True)

    with pytest.raises(StorageObjectError, match="symlink"):
        verify_storage_root(storage_root)
