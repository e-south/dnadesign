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
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

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


@pytest.mark.parametrize(
    "reserved_name",
    [
        f".{MANIFEST_NAME}.tmp-user-declared",
        f".{MANIFEST_NAME}.restore-user-declared",
        f".{MANIFEST_NAME}.rollback-user-declared",
        f".{MANIFEST_NAME}.cleanup-user-declared",
        f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}",
    ],
)
def test_verify_storage_object_rejects_declared_reserved_recovery_file(
    tmp_path: Path,
    reserved_name: str,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    content = b"user-declared recovery bytes\n"
    reserved = root / reserved_name
    reserved.write_bytes(content)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].append({"path": reserved_name, "digest": _digest(content), "role": "artifact"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        StorageObjectError,
        match=r"ambiguous manifest staging state.*(?:user-declared|cleanup-owner)",
    ):
        verify_storage_object(root)


def test_verify_storage_object_allows_empty_top_level_cleanup_owner_directory(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    cleanup_directory = root / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    cleanup_directory.mkdir()

    verified = verify_storage_object(root)

    assert verified.summary()["status"] == "verified"
    assert not tuple(cleanup_directory.iterdir())


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


@pytest.mark.parametrize(("root_mode", "lock_mode"), [(0o700, 0o606), (0o2770, 0o666)])
def test_verify_storage_object_rejects_other_writable_lock(
    tmp_path: Path,
    root_mode: int,
    lock_mode: int,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(root_mode)
    lock_path = root / LOCK_NAME
    lock_path.chmod(lock_mode)

    with pytest.raises(StorageObjectError, match="lock must not be other-writable"):
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


@pytest.mark.parametrize(("root_mode", "manifest_mode"), [(0o700, 0o606), (0o2770, 0o666)])
def test_verify_storage_object_rejects_other_writable_manifest(
    tmp_path: Path,
    root_mode: int,
    manifest_mode: int,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    root.chmod(root_mode)
    manifest_path = root / MANIFEST_NAME
    manifest_path.chmod(manifest_mode)

    with pytest.raises(StorageObjectError, match="manifest must not be other-writable"):
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

    with pytest.raises(StorageObjectError, match="cannot read storage resource") as exc_info:
        storage_validation.verify_storage_object(root)
    assert not isinstance(exc_info.value, storage_validation._StorageSnapshotInconsistent)


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

    with pytest.raises(StorageObjectError, match="declared resource digest mismatch") as exc_info:
        verify_storage_object(root)
    assert not isinstance(exc_info.value, storage_validation._StorageSnapshotInconsistent)


def test_verify_storage_object_binds_digest_to_post_read_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    payload_path = root / "inputs" / "payload.txt"
    original_sha256 = storage_validation._sha256
    payload_reads = 0

    def _sha256(path: Path) -> str:
        nonlocal payload_reads
        digest = original_sha256(path)
        if path == payload_path:
            payload_reads += 1
            if payload_reads == 2:
                path.write_bytes(b"changed\n")
        return digest

    monkeypatch.setattr(storage_validation, "_sha256", _sha256)

    with pytest.raises(StorageObjectError, match="declared resource changed during validation"):
        verify_storage_object(root)

    assert payload_reads == 2
    assert payload_path.read_bytes() == b"changed\n"


def test_verify_storage_object_rechecks_tree_after_final_resource_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    payload_path = root / "inputs" / "payload.txt"
    original_verify_resource = storage_validation._verify_resource
    payload_reads = 0

    def _verify_resource(root_path, resource):
        nonlocal payload_reads
        verified = original_verify_resource(root_path, resource)
        if resource.relative_path == "inputs/payload.txt":
            payload_reads += 1
            if payload_reads == 2:
                payload_path.write_bytes(b"changed\n")
        return verified

    monkeypatch.setattr(storage_validation, "_verify_resource", _verify_resource)

    with pytest.raises(StorageObjectError, match="storage object changed during validation"):
        verify_storage_object(root)

    assert payload_reads == 2
    assert payload_path.read_bytes() == b"changed\n"


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
    original_read = storage_validation._read_stable_regular_bytes

    def _read(
        path: Path,
        *,
        label: str,
        change_message: str,
        expected_identity: tuple[int, int],
    ) -> bytes:
        nonlocal reads
        if path == manifest_path:
            reads += 1
            return initial_bytes if reads == 1 else changed_bytes
        return original_read(
            path,
            label=label,
            change_message=change_message,
            expected_identity=expected_identity,
        )

    monkeypatch.setattr(storage_validation, "_read_stable_regular_bytes", _read)

    with pytest.raises(StorageObjectError, match="storage object changed during validation"):
        verify_storage_object(root)


def test_verify_storage_object_binds_manifest_reads_to_coordination_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    manifest_path = root / MANIFEST_NAME
    lock_path = root / LOCK_NAME
    coordination_state = storage_validation._verify_coordination_posture(root, manifest_path, lock_path)
    competing_manifest = tmp_path / "competing-manifest.json"
    competing_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    competing_payload["producer_revision"] = "competing-revision"
    competing_manifest.write_text(json.dumps(competing_payload), encoding="utf-8")
    competing_manifest.chmod(stat.S_IMODE(manifest_path.stat(follow_symlinks=False).st_mode))
    competing_manifest.replace(manifest_path)

    monkeypatch.setattr(
        storage_validation,
        "_verify_coordination_posture",
        lambda *_args, **_kwargs: coordination_state,
    )

    with pytest.raises(StorageObjectError, match="manifest changed during validation"):
        verify_storage_object(root)


def test_verify_storage_object_rejects_same_inode_manifest_rewrite_during_final_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    _write_object(root)
    manifest_path = root / MANIFEST_NAME
    original_read = storage_validation._read_stable_regular_bytes
    initial_bytes = manifest_path.read_bytes()
    replacement_bytes = b"!" + initial_bytes[1:]
    reads = 0

    def _read_then_rewrite_same_inode(
        path: Path,
        *,
        label: str,
        change_message: str,
        expected_identity: tuple[int, int],
    ) -> bytes:
        nonlocal reads
        content = original_read(
            path,
            label=label,
            change_message=change_message,
            expected_identity=expected_identity,
        )
        if path == manifest_path:
            reads += 1
            if reads == 2:
                before = path.stat(follow_symlinks=False)
                with path.open("r+b") as handle:
                    handle.write(replacement_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.utime(
                    path,
                    ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
                )
        return content

    monkeypatch.setattr(storage_validation, "_read_stable_regular_bytes", _read_then_rewrite_same_inode)

    with pytest.raises(StorageObjectError, match="changed during validation"):
        verify_storage_object(root)

    assert reads == 2
    assert manifest_path.read_bytes() == replacement_bytes


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


def test_verify_storage_root_holds_all_object_locks_against_concurrent_writer(
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
    expected_digest = _digest((earlier / MANIFEST_NAME).read_bytes())
    original_routes = storage_validation._routed_object_directories
    locks_held = Event()
    continue_validation = Event()
    route_calls = 0

    def _routes(root: Path):
        nonlocal route_calls
        routes = original_routes(root)
        route_calls += 1
        if route_calls == 2:
            locks_held.set()
            assert continue_validation.wait(timeout=10)
        return routes

    monkeypatch.setattr(storage_validation, "_routed_object_directories", _routes)

    writer_script = """
import json
import sys
from pathlib import Path
from dnadesign.contracts.storage_objects import refresh_storage_object

summary = refresh_storage_object(
    Path(sys.argv[1]),
    expected_manifest_digest=sys.argv[2],
    producer_revision="test-revision-2",
)
print(json.dumps(summary))
"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        validation_future = executor.submit(verify_storage_root, storage_root)
        assert locks_held.wait(timeout=10)
        writer = subprocess.Popen(
            [sys.executable, "-c", writer_script, str(earlier), expected_digest],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                writer.wait(timeout=0.5)
            continue_validation.set()
            assert validation_future.result(timeout=10).summary()["status"] == "verified"
            stdout, stderr = writer.communicate(timeout=10)
        finally:
            continue_validation.set()
            if writer.poll() is None:
                writer.kill()
                writer.communicate()
        assert writer.returncode == 0, stderr
        assert json.loads(stdout)["status"] == "verified"

    assert json.loads((earlier / MANIFEST_NAME).read_text(encoding="utf-8"))["producer_revision"] == "test-revision-2"
    assert stat.S_IMODE((earlier / LOCK_NAME).stat().st_mode) == 0o664


def test_verify_storage_root_acquires_object_locks_in_path_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    for storage_id in ("zebra", "alpha"):
        _write_object(storage_root / "workspaces" / "cruncher" / storage_id, storage_id=storage_id)
    original_acquire = storage_validation._acquire_existing_validation_lock
    acquired: list[Path] = []

    def _acquire(lock_path: Path, *args: object, **kwargs: object):
        acquired.append(lock_path)
        return original_acquire(lock_path, *args, **kwargs)

    monkeypatch.setattr(storage_validation, "_acquire_existing_validation_lock", _acquire)

    assert verify_storage_root(storage_root).summary()["status"] == "verified"
    assert acquired == sorted(acquired, key=lambda path: path.as_posix())
    assert [path.parent.name for path in acquired] == ["alpha", "zebra"]


def test_verify_storage_root_rejects_missing_object_lock_before_validation(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    (object_root / LOCK_NAME).unlink()

    with pytest.raises(StorageObjectError, match="storage object lock is missing") as exc_info:
        verify_storage_root(storage_root)

    assert not isinstance(exc_info.value, storage_validation._StorageSnapshotInconsistent)


def test_verify_storage_root_rejects_lock_replaced_during_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    lock_path = object_root / LOCK_NAME
    original_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_acquire = storage_validation._acquire_existing_validation_lock
    replacement_identity: tuple[int, int] | None = None

    def _acquire(lock: Path, *args: object, **kwargs: object):
        nonlocal replacement_identity
        replacement = lock_path.with_name(f".{LOCK_NAME}.replacement")
        replacement.touch(mode=0o664)
        replacement.replace(lock_path)
        replacement_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
        return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(storage_validation, "_acquire_existing_validation_lock", _acquire)

    with pytest.raises(StorageObjectError, match="lock changed before acquisition completed"):
        verify_storage_root(storage_root)

    assert replacement_identity is not None
    assert replacement_identity != original_identity


def test_verify_storage_root_does_not_recreate_lock_unlinked_during_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    lock_path = object_root / LOCK_NAME
    original_acquire = storage_validation._acquire_existing_validation_lock

    def _acquire(lock: Path, *args: object, **kwargs: object):
        lock_path.unlink()
        return original_acquire(lock, *args, **kwargs)

    monkeypatch.setattr(storage_validation, "_acquire_existing_validation_lock", _acquire)

    with pytest.raises(StorageObjectError, match="cannot open existing storage object lock"):
        verify_storage_root(storage_root)

    assert not lock_path.exists()


def test_verify_storage_root_rejects_lock_replaced_before_acquisition_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    lock_path = object_root / LOCK_NAME
    original_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_lock = storage_validation._validation_manifest_lock
    replaced = False

    @contextmanager
    def _lock(root: Path):
        nonlocal replaced
        if not replaced:
            replacement = lock_path.with_name(f".{LOCK_NAME}.replacement")
            replacement.touch(mode=0o664)
            replacement.replace(lock_path)
            replaced = True
        with original_lock(root):
            yield

    monkeypatch.setattr(storage_validation, "_validation_manifest_lock", _lock)

    with pytest.raises(StorageObjectError, match="changed during validation while acquiring object locks"):
        verify_storage_root(storage_root)

    assert replaced
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) != original_identity


def test_verify_storage_root_rejects_lock_replaced_before_validation_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    lock_path = object_root / LOCK_NAME
    original_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_verify = storage_validation._verify_locked_storage_root

    def _verify(root: Path, routes: object):
        verified = original_verify(root, routes)
        replacement = lock_path.with_name(f".{LOCK_NAME}.replacement")
        replacement.touch(mode=0o664)
        replacement.replace(lock_path)
        return verified

    monkeypatch.setattr(storage_validation, "_verify_locked_storage_root", _verify)

    with pytest.raises(StorageObjectError, match="lock changed before validation completion"):
        verify_storage_root(storage_root)

    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) != original_identity


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


def test_verify_storage_root_rechecks_routes_after_final_object_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    pilot = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(pilot)
    original_verify = storage_validation.verify_storage_object
    calls = 0
    created = False

    def _verify(object_root: Path):
        nonlocal calls, created
        verified = original_verify(object_root)
        calls += 1
        if calls == 3:
            _write_object(storage_root / "workspaces" / "cruncher" / "late", storage_id="late")
            created = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage root changed during validation"):
        verify_storage_root(storage_root)

    assert created
    assert calls == 3


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


def test_verify_storage_root_rechecks_earlier_resource_after_final_sequential_pass(
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
    later_calls = 0
    changed = False

    def _verify(object_root: Path):
        nonlocal changed, later_calls
        verified = original_verify(object_root)
        if object_root == later:
            later_calls += 1
            if later_calls == 2:
                (earlier / "outputs" / "result.json").write_text(
                    '{"status":"changed-after-earlier-final-pass"}\n',
                    encoding="utf-8",
                )
                changed = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage root changed"):
        verify_storage_root(storage_root)

    assert changed
    assert later_calls == 2


def test_verify_storage_root_rechecks_earlier_shared_access_after_final_sequential_pass(
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
    for object_root in (earlier, later):
        object_root.chmod(0o2770)
        for directory in (object_root / "inputs", object_root / "outputs"):
            directory.chmod(0o750)
        for path in (
            object_root / MANIFEST_NAME,
            object_root / LOCK_NAME,
            object_root / "inputs" / "payload.txt",
            object_root / "outputs" / "result.json",
        ):
            path.chmod(0o664 if path.name in {MANIFEST_NAME, LOCK_NAME} else 0o640)
    original_verify = storage_validation.verify_storage_object
    later_calls = 0
    changed = False

    def _verify(object_root: Path):
        nonlocal changed, later_calls
        verified = original_verify(object_root)
        if object_root == later:
            later_calls += 1
            if later_calls == 2:
                (earlier / "outputs" / "result.json").chmod(0o600)
                changed = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage root changed"):
        verify_storage_root(storage_root)

    assert changed
    assert later_calls == 2


@pytest.mark.parametrize("drift_kind", ["bytes", "mode"])
def test_verify_storage_root_rechecks_earlier_object_after_later_final_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    earlier = storage_root / "workspaces" / "cruncher" / "earlier"
    later = storage_root / "workspaces" / "cruncher" / "later"
    _write_object(earlier, storage_id="earlier")
    _write_object(later, storage_id="later")
    if drift_kind == "mode":
        for object_root in (earlier, later):
            object_root.chmod(0o2770)
            for directory in (object_root / "inputs", object_root / "outputs"):
                directory.chmod(0o750)
            for path in (
                object_root / MANIFEST_NAME,
                object_root / LOCK_NAME,
                object_root / "inputs" / "payload.txt",
                object_root / "outputs" / "result.json",
            ):
                path.chmod(0o664 if path.name in {MANIFEST_NAME, LOCK_NAME} else 0o640)
    original_verify = storage_validation.verify_storage_object
    later_calls = 0
    changed = False

    def _verify(object_root: Path):
        nonlocal changed, later_calls
        verified = original_verify(object_root)
        if object_root == later:
            later_calls += 1
            if later_calls == 3:
                resource = earlier / "outputs" / "result.json"
                if drift_kind == "bytes":
                    resource.write_text('{"status":"changed-during-final-rebind"}\n', encoding="utf-8")
                else:
                    resource.chmod(0o600)
                changed = True
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage root changed during validation"):
        verify_storage_root(storage_root)

    assert changed
    assert later_calls == 3


def test_verify_storage_root_rejects_earlier_demo_index_drift_during_later_rebind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    storage_root = checkout / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    earlier = storage_root / "workspaces" / "cruncher" / "earlier"
    later = storage_root / "workspaces" / "cruncher" / "later"
    _write_object(earlier, storage_id="earlier", demo=True)
    _write_object(later, storage_id="later", demo=True)
    subprocess.run(["git", "-C", str(checkout), "add", "storage"], check=True)
    original_rebind = storage_validation._rebind_verified_storage_object
    removed_relative = (earlier / "outputs" / "result.json").relative_to(checkout).as_posix()
    changed = False

    def _rebind(verified):
        nonlocal changed
        if verified.root == later and not changed:
            subprocess.run(
                ["git", "-C", str(checkout), "update-index", "--force-remove", "--", removed_relative],
                check=True,
            )
            changed = True
        return original_rebind(verified)

    monkeypatch.setattr(storage_validation, "_rebind_verified_storage_object", _rebind)

    with pytest.raises(StorageObjectError, match="demo Git index changed during root validation"):
        verify_storage_root(storage_root)

    assert changed
    assert (checkout / removed_relative).is_file()
    assert (
        subprocess.run(
            ["git", "-C", str(checkout), "ls-files", "--error-unmatch", "--", removed_relative],
            check=False,
            capture_output=True,
        ).returncode
        != 0
    )


def test_root_demo_authority_rejects_distinct_checkouts_before_sequential_rechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    earlier = checkout / "earlier"
    later = checkout / "later"
    _write_object(earlier, storage_id="earlier", demo=True)
    _write_object(later, storage_id="later", demo=True)
    subprocess.run(["git", "-C", str(checkout), "add", "earlier", "later"], check=True)
    verified = [verify_storage_object(earlier), verify_storage_object(later)]
    distinct_checkouts = {
        earlier.resolve(): tmp_path / "checkout-a",
        later.resolve(): tmp_path / "checkout-b",
    }

    monkeypatch.setattr(
        storage_validation,
        "_git_checkout_ancestor",
        lambda root, *, include_root: distinct_checkouts[root],
    )
    monkeypatch.setattr(storage_validation, "_git_authority_environment", lambda: {})
    monkeypatch.setattr(
        storage_validation,
        "_read_demo_git_index_snapshot",
        lambda checkout, *, git_environment: (checkout.as_posix().encode(), {}),
    )

    with pytest.raises(StorageObjectError, match="root demos must share one Git checkout"):
        storage_validation._capture_demo_git_authorities(verified)


def test_verify_storage_root_rejects_object_directory_replaced_after_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)
    object_root = storage_root / "workspaces" / "cruncher" / "pilot"
    _write_object(object_root)
    replacement = tmp_path / "replacement"
    _write_object(replacement)
    displaced = tmp_path / "displaced"
    original_identity = (object_root.stat().st_dev, object_root.stat().st_ino)
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    original_verify = storage_validation.verify_storage_object
    calls = 0

    def _verify(root: Path):
        nonlocal calls
        verified = original_verify(root)
        calls += 1
        if calls == 2:
            object_root.rename(displaced)
            replacement.rename(object_root)
        return verified

    monkeypatch.setattr(storage_validation, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="storage root changed during validation"):
        verify_storage_root(storage_root)

    assert original_identity != replacement_identity
    assert (object_root.stat().st_dev, object_root.stat().st_ino) == replacement_identity
