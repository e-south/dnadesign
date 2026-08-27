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
import os
import subprocess
from pathlib import Path

import pytest

import dnadesign.contracts.storage_objects.validation as storage_validation
from dnadesign.contracts.storage_objects import (
    MANIFEST_NAME,
    StorageObjectError,
    verify_storage_object,
    verify_storage_root,
)
from dnadesign.contracts.storage_objects.models import LOCK_NAME
from dnadesign.contracts.storage_objects.validation import storage_file_paths


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
    resource_roles = ("cache", "cache") if object_kind == "tool-cache" else ("input", "artifact")
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
            {"path": "inputs/payload.txt", "digest": _digest(payload), "role": resource_roles[0]},
            {"path": "outputs/result.json", "digest": _digest(result), "role": resource_roles[1]},
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    lock_path = root / LOCK_NAME
    lock_path.touch()
    lock_path.chmod(0o664)


def test_verify_storage_object_requires_exact_file_closure(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)

    verified = verify_storage_object(root)

    assert verified.manifest.storage_id == "pilot"
    assert verified.manifest.object_kind.value == "workspace"
    assert [resource.role.value for resource in verified.resources] == ["input", "artifact"]
    assert verified.summary()["resource_count"] == 2
    assert verified.summary()["manifest_digest"] == _digest((root / MANIFEST_NAME).read_bytes())

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


def test_verify_storage_object_rejects_nul_in_resource_path(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0]["path"] = "inputs/payload\x00.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must not contain NUL bytes"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_non_regular_entries(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    fifo = root / "runtime.fifo"
    os.mkfifo(fifo)

    with pytest.raises(StorageObjectError, match="non-regular storage entry is not allowed: runtime.fifo"):
        verify_storage_object(root)


def test_storage_object_lock_is_shared_state_not_content(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    (root / LOCK_NAME).touch()

    verified = verify_storage_object(root)

    assert verified.summary()["resource_count"] == 2


def test_verify_storage_object_requires_coordination_lock(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    (root / LOCK_NAME).unlink()

    with pytest.raises(StorageObjectError, match="storage object lock is missing"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_nonempty_shared_lock(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    lock_path = root / LOCK_NAME
    lock_path.write_text("unexpected bytes\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must be an empty coordination file"):
        verify_storage_object(root)

    assert lock_path.read_text(encoding="utf-8") == "unexpected bytes\n"


def test_verify_storage_object_rejects_invalid_shared_lock_posture(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    lock_path = root / LOCK_NAME
    lock_path.chmod(0o644)

    with pytest.raises(StorageObjectError, match="lock must be group-writable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_group_unreadable_shared_lock(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    lock_path = root / LOCK_NAME
    lock_path.touch()
    lock_path.chmod(0o620)

    with pytest.raises(StorageObjectError, match="lock must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_other_writable_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o707)

    with pytest.raises(StorageObjectError, match="must not be other-writable"):
        verify_storage_object(root)


@pytest.mark.parametrize("mode", [0o200, 0o400])
def test_verify_storage_object_rejects_owner_inaccessible_lock(tmp_path: Path, mode: int) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    lock_path = root / LOCK_NAME
    lock_path.touch()
    lock_path.chmod(mode)

    with pytest.raises(StorageObjectError, match="lock must be owner-readable and owner-writable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_group_writable_root_without_setgid(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o770)

    with pytest.raises(StorageObjectError, match="must set the setgid bit"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_sticky_group_shared_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o3770)

    with pytest.raises(StorageObjectError, match="must not set the sticky bit"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_shared_root_without_group_traversal(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2720)

    with pytest.raises(StorageObjectError, match="must be group-traversable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_shared_root_without_group_read(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2730)

    with pytest.raises(StorageObjectError, match="must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_inaccessible_shared_content(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    (root / "inputs").chmod(0o700)

    with pytest.raises(StorageObjectError, match="directory must be group-readable and traversable"):
        verify_storage_object(root)

    (root / "inputs").chmod(0o750)
    (root / "inputs" / "payload.txt").chmod(0o600)
    with pytest.raises(StorageObjectError, match="shared resource must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_inaccessible_empty_shared_directory(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    empty = root / "empty"
    empty.mkdir(mode=0o700)

    with pytest.raises(StorageObjectError, match="directory must be group-readable and traversable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_private_manifest_in_shared_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    manifest_path = root / MANIFEST_NAME
    manifest_path.chmod(0o600)

    with pytest.raises(StorageObjectError, match="manifest must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_wraps_resource_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    payload_path = root / "inputs" / "payload.txt"
    original_open = Path.open

    def _open(path: Path, *args: object, **kwargs: object):
        if path == payload_path:
            raise PermissionError(13, "Permission denied", str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open)

    with pytest.raises(StorageObjectError, match="cannot read storage resource"):
        storage_validation.verify_storage_object(root)


def test_verify_storage_object_rejects_bytes_changed_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    original_verify = storage_validation._verify_resource
    calls = 0

    def _verify(root_path, resource):
        nonlocal calls
        verified = original_verify(root_path, resource)
        calls += 1
        if calls == 2:
            (root / "inputs" / "payload.txt").write_text("changed\n", encoding="utf-8")
        return verified

    monkeypatch.setattr(storage_validation, "_verify_resource", _verify)

    with pytest.raises(StorageObjectError, match="declared resource digest mismatch"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_manifest_changed_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    manifest_path = root / MANIFEST_NAME
    initial_bytes = manifest_path.read_bytes()
    changed = json.loads(initial_bytes)
    changed["producer_revision"] = "test-revision-2"
    changed_bytes = json.dumps(changed).encode("utf-8")
    reads = 0
    original_read_bytes = Path.read_bytes

    def _read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == manifest_path:
            reads += 1
            return initial_bytes if reads == 1 else changed_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", _read_bytes)

    with pytest.raises(StorageObjectError, match="storage object changed during validation"):
        verify_storage_object(root)


def test_verify_storage_object_rechecks_shared_resource_access_on_second_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    payload_path = root / "inputs" / "payload.txt"
    original_tree_paths = storage_validation._storage_tree_paths
    calls = 0

    def _tree_paths(root_path: Path):
        nonlocal calls
        paths = original_tree_paths(root_path)
        calls += 1
        if calls == 2:
            payload_path.chmod(0o600)
        return paths

    monkeypatch.setattr(storage_validation, "_storage_tree_paths", _tree_paths)

    with pytest.raises(StorageObjectError, match="shared resource must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_rechecks_coordination_posture_on_second_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(0o2770)
    manifest_path = root / MANIFEST_NAME
    original_tree_paths = storage_validation._storage_tree_paths
    calls = 0

    def _tree_paths(root_path: Path):
        nonlocal calls
        paths = original_tree_paths(root_path)
        calls += 1
        if calls == 2:
            manifest_path.chmod(0o600)
        return paths

    monkeypatch.setattr(storage_validation, "_storage_tree_paths", _tree_paths)

    with pytest.raises(StorageObjectError, match="manifest must be group-readable"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_lock_replaced_between_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    lock_path = root / LOCK_NAME
    initial_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_tree_paths = storage_validation._storage_tree_paths
    calls = 0

    def _tree_paths(root_path: Path):
        nonlocal calls
        paths = original_tree_paths(root_path)
        calls += 1
        if calls == 2:
            lock_path.unlink()
            lock_path.touch()
            lock_path.chmod(0o664)
        return paths

    monkeypatch.setattr(storage_validation, "_storage_tree_paths", _tree_paths)

    # Keep the stale inode alive, as it would be for a non-cooperating process
    # holding the unlinked lock, so Linux cannot recycle its inode immediately.
    with lock_path.open("rb"):
        with pytest.raises(StorageObjectError, match="storage object changed during validation"):
            verify_storage_object(root)

    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) != initial_identity


def test_verify_storage_object_rejects_root_made_other_writable_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    original_tree_paths = storage_validation._storage_tree_paths
    calls = 0

    def _tree_paths(root_path: Path):
        nonlocal calls
        paths = original_tree_paths(root_path)
        calls += 1
        if calls == 2:
            root.chmod(0o707)
        return paths

    monkeypatch.setattr(storage_validation, "_storage_tree_paths", _tree_paths)

    with pytest.raises(StorageObjectError, match="must not be other-writable"):
        verify_storage_object(root)


def test_storage_file_closure_propagates_unreadable_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = PermissionError(13, "Permission denied", str(tmp_path / "blocked"))

    def _walk_with_error(
        _root: Path,
        *,
        followlinks: bool,
        onerror,
    ):
        assert followlinks is False
        onerror(error)
        return iter(())

    monkeypatch.setattr(os, "walk", _walk_with_error)

    with pytest.raises(StorageObjectError, match="cannot traverse storage object"):
        storage_file_paths(tmp_path)


def test_declared_resource_below_symlink_loop_is_contract_error(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0]["path"] = "a/payload.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (root / "a").symlink_to("b", target_is_directory=True)
    (root / "b").symlink_to("a", target_is_directory=True)

    with pytest.raises(StorageObjectError, match="declared resource a/payload.txt does not resolve"):
        verify_storage_object(root)


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


def test_git_resident_demo_treats_resource_paths_literally(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    _write_object(root, demo=True)
    tracked = root / "foo1.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    untracked = root / "foo[1].txt"
    untracked.write_text("untracked\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].extend(
        [
            {"path": "foo1.txt", "digest": _digest(tracked.read_bytes()), "role": "artifact"},
            {"path": "foo[1].txt", "digest": _digest(untracked.read_bytes()), "role": "artifact"},
        ]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "add",
            "examples/pilot/inputs",
            "examples/pilot/outputs",
            "examples/pilot/storage.object.json",
            "examples/pilot/.storage-object.lock",
            "examples/pilot/foo1.txt",
        ],
        check=True,
    )

    with pytest.raises(StorageObjectError, match=r"demo file is not tracked: .*foo\[1\]\.txt"):
        verify_storage_object(root)


def test_git_resident_demo_normalizes_missing_git_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    _write_object(root, demo=True)

    def _missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(storage_validation.subprocess, "run", _missing_git)

    with pytest.raises(StorageObjectError, match="cannot verify demo Git tracking"):
        verify_storage_object(root)


def test_demo_outside_git_checkout_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root, demo=True)

    with pytest.raises(StorageObjectError, match="must live inside a Git checkout"):
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


def test_tool_cache_rejects_noncache_resource_roles(tmp_path: Path) -> None:
    root = tmp_path / "proteinmpnn"
    _write_object(
        root,
        storage_id="proteinmpnn",
        owner_tool="proteinmpnn",
        object_kind="tool-cache",
    )
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0]["role"] = "artifact"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(StorageObjectError, match="every resource role to be 'cache'"):
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


def test_verify_storage_root_rejects_symlinked_agents_router(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    external = tmp_path / "external-agents.md"
    external.write_text("# External\n", encoding="utf-8")
    (storage_root / "AGENTS.md").symlink_to(external)

    with pytest.raises(StorageObjectError, match="routing file must not be a symlink"):
        verify_storage_root(storage_root)


def test_verify_storage_root_wraps_directory_enumeration_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    original_iterdir = Path.iterdir

    def _iterdir(path: Path):
        if path == storage_root:
            raise PermissionError(13, "Permission denied", str(path))
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _iterdir)

    with pytest.raises(StorageObjectError, match="cannot enumerate storage root"):
        verify_storage_root(storage_root)


def test_verify_storage_root_rejects_object_created_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    original_routes = storage_validation._routed_object_directories
    calls = 0

    def _routes(root: Path):
        nonlocal calls
        routes = original_routes(root)
        calls += 1
        if calls == 1:
            _write_object(storage_root / "workspaces" / "cruncher" / "late")
        return routes

    monkeypatch.setattr(storage_validation, "_routed_object_directories", _routes)

    with pytest.raises(StorageObjectError, match="storage root changed during validation"):
        verify_storage_root(storage_root)


def test_verify_storage_root_rejects_earlier_manifest_refreshed_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    earlier = storage_root / "workspaces" / "cruncher" / "earlier"
    later = storage_root / "workspaces" / "cruncher" / "later"
    _write_object(earlier, storage_id="earlier")
    _write_object(later, storage_id="later")
    original_verify = storage_validation.verify_storage_object
    refreshed = False

    def _verify(object_root: Path):
        nonlocal refreshed
        verified = original_verify(object_root)
        if object_root == later and not refreshed:
            manifest_path = earlier / MANIFEST_NAME
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["producer_revision"] = "test-revision-2"
            staged_manifest = manifest_path.with_name(f".{MANIFEST_NAME}.refresh")
            staged_manifest.write_text(json.dumps(manifest), encoding="utf-8")
            os.replace(staged_manifest, manifest_path)
            refreshed = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage object manifest changed during root validation"):
        verify_storage_root(storage_root)


def test_verify_storage_root_revalidates_earlier_resource_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    earlier = storage_root / "workspaces" / "cruncher" / "earlier"
    later = storage_root / "workspaces" / "cruncher" / "later"
    _write_object(earlier, storage_id="earlier")
    _write_object(later, storage_id="later")
    original_verify = storage_validation.verify_storage_object
    changed = False

    def _verify(object_root: Path):
        nonlocal changed
        verified = original_verify(object_root)
        if object_root == later and not changed:
            (earlier / "outputs" / "result.json").write_text('{"status":"changed"}\n', encoding="utf-8")
            changed = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="digest mismatch"):
        verify_storage_root(storage_root)
