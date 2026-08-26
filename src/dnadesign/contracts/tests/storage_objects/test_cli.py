"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/storage_objects/test_cli.py

Tests deterministic storage inventory and validation commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
import hashlib
import json
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import dnadesign.contracts.storage_objects.inventory as storage_inventory
from dnadesign.contracts.storage_objects import (
    MANIFEST_NAME,
    StorageObjectError,
    StorageObjectPublicationUncertain,
    StorageObjectPublicationUnsupported,
    inventory_storage_object,
    refresh_storage_object,
)
from dnadesign.contracts.storage_objects.models import LOCK_NAME


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_inventory_creates_a_closed_manifest_then_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    (root / "inputs").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "inputs" / "payload.txt").write_text("payload\n", encoding="utf-8")
    (root / "outputs" / "result.json").write_text('{"status":"ok"}\n', encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "dnadesign.contracts.storage_objects",
        "inventory",
        str(root),
        "--storage-id",
        "pilot",
        "--owner-repository",
        "dnadesign",
        "--owner-tool",
        "cruncher",
        "--object-kind",
        "workspace",
        "--content-schema",
        "cruncher.workspace",
        "--content-schema-version",
        "1",
        "--producer-revision",
        "test-revision-1",
        "--storage-class",
        "reproducible",
        "--retention-policy",
        "review-before-delete",
        "--input",
        "inputs/payload.txt",
        "--metadata",
        "outputs/result.json",
        "--json",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    manifest_text = (root / MANIFEST_NAME).read_text(encoding="utf-8")

    assert manifest_text.endswith("\n")
    summary = json.loads(completed.stdout)
    assert summary["status"] == "verified"
    assert summary["manifest_digest"] == _digest(root / MANIFEST_NAME)
    assert [row["path"] for row in json.loads(manifest_text)["resources"]] == [
        "inputs/payload.txt",
        "outputs/result.json",
    ]
    assert [row["role"] for row in json.loads(manifest_text)["resources"]] == [
        "input",
        "metadata",
    ]
    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode == 2
    assert "already exists" in repeated.stderr


def test_inventory_bootstraps_demo_then_requires_manifest_to_be_tracked(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout with spaces"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)
    command = [
        sys.executable,
        "-m",
        "dnadesign.contracts.storage_objects",
        "inventory",
        str(root),
        "--storage-id",
        "pilot",
        "--owner-repository",
        "dnadesign",
        "--owner-tool",
        "cruncher",
        "--object-kind",
        "workspace",
        "--content-schema",
        "cruncher.workspace",
        "--content-schema-version",
        "1",
        "--producer-revision",
        "test-revision-1",
        "--storage-class",
        "reproducible",
        "--retention-policy",
        "review-before-delete",
        "--demo",
        "--json",
    ]

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    before_add = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "validate",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    next_step = subprocess.run(
        summary["next_step"],
        shell=True,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    after_add = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "validate",
            str(root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert summary["status"] == "created-pending-git-add"
    assert (root / MANIFEST_NAME).is_file()
    assert before_add.returncode == 2
    assert "demo file is not tracked" in before_add.stderr
    assert json.loads(next_step.stdout)["status"] == "verified"
    assert json.loads(after_add.stdout)["status"] == "verified"


def test_validate_root_emits_inventory_summary(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    for shelf in ("workspaces", "stores", "tool-cache"):
        (storage_root / shelf).mkdir(parents=True)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "validate-root",
            str(storage_root),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "object_count": 0,
        "objects_by_kind": {},
        "owner_count": 0,
        "schema": "dnadesign.storage-root/v1",
        "status": "verified",
    }


def test_refresh_requires_prior_receipt_and_preserves_protected_roles(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    (root / "inputs").mkdir(parents=True)
    (root / "outputs").mkdir()
    (root / "inputs" / "payload.txt").write_text("payload\n", encoding="utf-8")
    (root / "outputs" / "result.json").write_text("first\n", encoding="utf-8")
    inventory = [
        sys.executable,
        "-m",
        "dnadesign.contracts.storage_objects",
        "inventory",
        str(root),
        "--storage-id",
        "pilot",
        "--owner-repository",
        "dnadesign",
        "--owner-tool",
        "cruncher",
        "--object-kind",
        "workspace",
        "--content-schema",
        "cruncher.workspace",
        "--content-schema-version",
        "1",
        "--producer-revision",
        "test-revision-1",
        "--storage-class",
        "reproducible",
        "--retention-policy",
        "review-before-delete",
        "--input",
        "inputs/payload.txt",
    ]
    subprocess.run(inventory, check=True, capture_output=True, text=True)
    manifest_path = root / MANIFEST_NAME
    expected_digest = _digest(manifest_path)
    (root / "outputs" / "result.json").write_text("second\n", encoding="utf-8")
    (root / "outputs" / "new.txt").write_text("new\n", encoding="utf-8")
    refresh = [
        sys.executable,
        "-m",
        "dnadesign.contracts.storage_objects",
        "refresh",
        str(root),
        "--expected-manifest-digest",
        expected_digest,
        "--producer-revision",
        "test-revision-2",
        "--json",
    ]

    stale_refresh = [*refresh]
    stale_refresh[6] = "sha256:" + "0" * 64
    stale = subprocess.run(
        stale_refresh,
        check=False,
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 2
    assert "manifest changed before refresh" in stale.stderr

    completed = subprocess.run(refresh, check=True, capture_output=True, text=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roles = {row["path"]: row["role"] for row in manifest["resources"]}
    assert json.loads(completed.stdout)["resource_count"] == 3
    assert manifest["producer_revision"] == "test-revision-2"
    assert roles == {
        "inputs/payload.txt": "input",
        "outputs/new.txt": "artifact",
        "outputs/result.json": "artifact",
    }


def test_refresh_preserves_existing_manifest_permissions(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    manifest_path.chmod(0o640)
    expected_digest = _digest(manifest_path)
    (root / "result.txt").write_text("result\n", encoding="utf-8")

    refresh_storage_object(
        root,
        expected_manifest_digest=expected_digest,
        producer_revision="test-revision-2",
    )

    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o640


def test_inventory_creates_group_writable_lock_for_shared_object(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )

    assert stat.S_IMODE((root / LOCK_NAME).stat().st_mode) == 0o664
    assert stat.S_IMODE((root / MANIFEST_NAME).stat().st_mode) == 0o664
    assert (root / LOCK_NAME).stat().st_gid == root.stat().st_gid
    assert (root / MANIFEST_NAME).stat().st_gid == root.stat().st_gid


def test_inventory_rejects_non_group_writable_lock_in_shared_object(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    lock_path = root / LOCK_NAME
    lock_path.touch(mode=0o644)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="lock must be group-writable"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rejects_group_unreadable_lock_in_shared_object(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    lock_path = root / LOCK_NAME
    lock_path.touch()
    lock_path.chmod(0o620)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="lock must be group-readable"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o620
    assert not (root / MANIFEST_NAME).exists()


@pytest.mark.parametrize("mode", [0o200, 0o400])
def test_inventory_rejects_owner_inaccessible_lock(tmp_path: Path, mode: int) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    lock_path = root / LOCK_NAME
    lock_path.touch()
    lock_path.chmod(mode)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="lock must be owner-readable and owner-writable"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert stat.S_IMODE(lock_path.stat().st_mode) == mode
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rejects_group_writable_root_without_setgid(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must set the setgid bit"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / LOCK_NAME).exists()
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rejects_shared_root_without_group_traversal(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2720)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must be group-traversable"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / LOCK_NAME).exists()
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rejects_shared_root_without_group_read(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2730)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must be group-readable"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / LOCK_NAME).exists()
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_stages_manifest_on_object_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    observed_directories: list[Path] = []
    real_mkstemp = storage_inventory.tempfile.mkstemp

    def _capture_mkstemp(*, dir: Path, prefix: str) -> tuple[int, str]:
        observed_directories.append(Path(dir))
        return real_mkstemp(dir=dir, prefix=prefix)

    monkeypatch.setattr(storage_inventory.tempfile, "mkstemp", _capture_mkstemp)

    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )

    assert observed_directories == [root]


def test_inventory_fails_closed_on_user_file_that_resembles_manifest_staging(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    collision = root / f".{MANIFEST_NAME}.tmp-user-data"
    collision.write_text("user-owned\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="ambiguous manifest staging state"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert collision.read_text(encoding="utf-8") == "user-owned\n"
    assert not (root / MANIFEST_NAME).exists()


def test_refresh_rejects_missing_root_without_traceback(tmp_path: Path) -> None:
    root = tmp_path / "missing"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "refresh",
            str(root),
            "--expected-manifest-digest",
            "sha256:" + "0" * 64,
            "--producer-revision",
            "test-revision-2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "storage object root is not a directory" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (root / LOCK_NAME).exists()


def test_validate_rejects_symlink_loop_root_without_traceback(tmp_path: Path) -> None:
    (tmp_path / "a").symlink_to("b", target_is_directory=True)
    (tmp_path / "b").symlink_to("a", target_is_directory=True)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "validate",
            str(tmp_path / "a" / "object"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "storage object root does not resolve" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_inventory_and_refresh_reject_symlinked_object_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    linked_root = tmp_path / "linked-pilot"
    linked_root.symlink_to(root, target_is_directory=True)

    inventory = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "inventory",
            str(linked_root),
            "--storage-id",
            "pilot",
            "--owner-repository",
            "dnadesign",
            "--owner-tool",
            "cruncher",
            "--object-kind",
            "workspace",
            "--content-schema",
            "cruncher.workspace",
            "--content-schema-version",
            "1",
            "--producer-revision",
            "test-revision-1",
            "--storage-class",
            "reproducible",
            "--retention-policy",
            "review-before-delete",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert inventory.returncode == 2
    assert "root must not be a symlink" in inventory.stderr


def test_inventory_rejects_symlinked_shared_lock(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    external = tmp_path / "external.lock"
    external.write_text("do not touch\n", encoding="utf-8")
    (root / LOCK_NAME).symlink_to(external)

    with pytest.raises(StorageObjectError, match="storage object lock must be a regular file"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert external.read_text(encoding="utf-8") == "do not touch\n"


def test_inventory_rejects_nonempty_shared_lock_without_changing_it(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    lock_path = root / LOCK_NAME
    lock_path.write_text("user bytes\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must be an empty coordination file"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert lock_path.read_text(encoding="utf-8") == "user bytes\n"


def test_inventory_wraps_resource_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload_path = root / "payload.txt"
    payload_path.write_text("payload\n", encoding="utf-8")
    original_open = Path.open

    def _open(path: Path, *args: object, **kwargs: object):
        if path == payload_path:
            raise PermissionError(13, "Permission denied", str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _open)

    with pytest.raises(StorageObjectError, match="cannot read storage resource"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )


def test_inventory_wraps_lock_acquisition_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()

    def _deny_lock(*_args: object, **_kwargs: object) -> object:
        raise PermissionError(13, "Permission denied", str(root / LOCK_NAME))

    monkeypatch.setattr(storage_inventory.FileLock, "acquire", _deny_lock)

    with pytest.raises(StorageObjectError, match="cannot acquire storage object manifest lock"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )


def test_inventory_wraps_unsupported_filesystem_locking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()

    def _unsupported_lock(*_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("flock unsupported")

    monkeypatch.setattr(storage_inventory.FileLock, "acquire", _unsupported_lock)

    with pytest.raises(StorageObjectError, match="cannot acquire storage object manifest lock"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )


def test_manifest_lock_releases_after_post_acquisition_inspection_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    acquired = False
    released = False

    class _FakeLock:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def acquire(self) -> None:
            nonlocal acquired
            acquired = True

        def release(self) -> None:
            nonlocal released
            released = True

    real_stat = Path.stat

    def _stat(path: Path, *args: object, **kwargs: object):
        if acquired and path == root / LOCK_NAME:
            raise PermissionError("simulated post-acquisition inspection denial")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(storage_inventory, "FileLock", _FakeLock)
    monkeypatch.setattr(Path, "stat", _stat)

    with pytest.raises(StorageObjectError, match="cannot inspect storage object lock after acquisition"):
        with storage_inventory._manifest_lock(root):
            raise AssertionError("lock body must not run")

    assert acquired is True
    assert released is True


def test_refresh_wraps_lock_acquisition_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    expected_digest = _digest(root / MANIFEST_NAME)

    def _deny_lock(*_args: object, **_kwargs: object) -> object:
        raise PermissionError(13, "Permission denied", str(root / LOCK_NAME))

    monkeypatch.setattr(storage_inventory.FileLock, "acquire", _deny_lock)

    with pytest.raises(StorageObjectError, match="cannot acquire storage object manifest lock"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )


def test_inventory_fails_closed_on_preexisting_manifest_temp(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    temporary = root / f".{MANIFEST_NAME}.tmp"
    temporary.write_text("pre-existing bytes\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="ambiguous manifest staging state"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert temporary.read_text(encoding="utf-8") == "pre-existing bytes\n"
    assert not (root / MANIFEST_NAME).exists()


def test_concurrent_refresh_allows_exactly_one_compare_and_swap(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    (root / "inputs").mkdir(parents=True)
    (root / "inputs" / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory = [
        sys.executable,
        "-m",
        "dnadesign.contracts.storage_objects",
        "inventory",
        str(root),
        "--storage-id",
        "pilot",
        "--owner-repository",
        "dnadesign",
        "--owner-tool",
        "cruncher",
        "--object-kind",
        "workspace",
        "--content-schema",
        "cruncher.workspace",
        "--content-schema-version",
        "1",
        "--producer-revision",
        "test-revision-1",
        "--storage-class",
        "reproducible",
        "--retention-policy",
        "review-before-delete",
        "--input",
        "inputs/payload.txt",
    ]
    subprocess.run(inventory, check=True, capture_output=True, text=True)
    assert (root / LOCK_NAME).is_file()
    expected_digest = _digest(root / MANIFEST_NAME)
    (root / "result.txt").write_text("result\n", encoding="utf-8")

    def _refresh() -> str:
        try:
            refresh_storage_object(
                root,
                expected_manifest_digest=expected_digest,
                producer_revision="test-revision-2",
            )
        except StorageObjectError as exc:
            return str(exc)
        return "verified"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: _refresh(), range(2)))

    assert outcomes.count("verified") == 1
    assert sum("manifest changed before refresh" in outcome for outcome in outcomes) == 1


def test_refresh_advances_authoritative_store_receipt(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="store",
        owner_repository="dnadesign",
        owner_tool="usr",
        object_kind="store",
        content_schema="usr.dataset",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="authoritative",
        retention_policy="retain",
    )
    manifest_path = root / MANIFEST_NAME
    expected_digest = _digest(manifest_path)
    (root / "payload.txt").write_text("changed\n", encoding="utf-8")

    summary = refresh_storage_object(
        root,
        expected_manifest_digest=expected_digest,
        producer_revision="test-revision-2",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary["status"] == "verified"
    assert summary["manifest_digest"] == _digest(manifest_path)
    assert manifest["producer_revision"] == "test-revision-2"
    assert manifest["resources"][0]["digest"] == _digest(root / "payload.txt")


def test_refresh_rejects_tool_cache_receipts(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="cache",
        owner_repository="dnadesign",
        owner_tool="proteinmpnn",
        object_kind="tool-cache",
        content_schema="proteinmpnn.checkout",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="cache",
        retention_policy="rebuildable",
    )

    with pytest.raises(StorageObjectError, match="limited to active workspaces and stores"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(root / MANIFEST_NAME),
            producer_revision="test-revision-2",
        )


def test_refresh_explicitly_reclassifies_mutable_metadata_as_artifact(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    event_log = root / ".events.log"
    event_log.write_text('{"event":"created"}\n', encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="store",
        owner_repository="dnadesign",
        owner_tool="usr",
        object_kind="store",
        content_schema="usr.dataset",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="authoritative",
        retention_policy="retain",
        metadata_paths=(".events.log",),
    )
    manifest_path = root / MANIFEST_NAME
    prior_digest = _digest(manifest_path)
    with event_log.open("a", encoding="utf-8") as handle:
        handle.write('{"event":"updated"}\n')

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "dnadesign.contracts.storage_objects",
            "refresh",
            str(root),
            "--expected-manifest-digest",
            prior_digest,
            "--producer-revision",
            "test-revision-2",
            "--artifact",
            ".events.log",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(completed.stdout)
    assert manifest["resources"][0]["role"] == "artifact"
    assert manifest["resources"][0]["digest"] == _digest(event_log)
    assert summary["manifest_digest"] == _digest(manifest_path)


def test_refresh_never_reclassifies_inputs_as_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("original\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="store",
        owner_repository="dnadesign",
        owner_tool="usr",
        object_kind="store",
        content_schema="usr.dataset",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="authoritative",
        retention_policy="retain",
        input_paths=("payload.txt",),
    )
    manifest_path = root / MANIFEST_NAME
    prior_bytes = manifest_path.read_bytes()

    with pytest.raises(StorageObjectError, match="input and cache roles remain protected"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
            artifact_paths=("payload.txt",),
        )

    assert manifest_path.read_bytes() == prior_bytes


def test_refresh_never_reclassifies_deleted_metadata_as_artifact(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    metadata = root / "metadata.txt"
    metadata.write_text("original\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="store",
        owner_repository="dnadesign",
        owner_tool="usr",
        object_kind="store",
        content_schema="usr.dataset",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="authoritative",
        retention_policy="retain",
        metadata_paths=("metadata.txt",),
    )
    manifest_path = root / MANIFEST_NAME
    prior_bytes = manifest_path.read_bytes()
    metadata.unlink()

    with pytest.raises(StorageObjectError, match="reclassification target is missing"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
            artifact_paths=("metadata.txt",),
        )

    assert manifest_path.read_bytes() == prior_bytes


@pytest.mark.parametrize("protected_role", ["input", "metadata"])
def test_refresh_rejects_changed_protected_bytes(tmp_path: Path, protected_role: str) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    protected = root / "protected.txt"
    protected.write_text("original\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
        input_paths=("protected.txt",) if protected_role == "input" else (),
        metadata_paths=("protected.txt",) if protected_role == "metadata" else (),
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    protected.write_text("changed\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="cannot refresh after changing input or metadata files"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes


@pytest.mark.parametrize("protected_role", ["input", "metadata"])
def test_refresh_rejects_protected_bytes_changed_between_digest_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_role: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    protected = root / "protected.txt"
    protected.write_text("original\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
        input_paths=("protected.txt",) if protected_role == "input" else (),
        metadata_paths=("protected.txt",) if protected_role == "metadata" else (),
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    original_sha256 = storage_inventory._sha256
    protected_hashes = 0

    def _mutate_after_first_protected_hash(path: Path) -> str:
        nonlocal protected_hashes
        digest = original_sha256(path)
        if path == protected:
            protected_hashes += 1
            if protected_hashes == 1:
                protected.write_text("changed\n", encoding="utf-8")
        return digest

    monkeypatch.setattr(storage_inventory, "_sha256", _mutate_after_first_protected_hash)

    with pytest.raises(StorageObjectError, match="protected resource changed during refresh"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert protected_hashes == 2
    assert manifest_path.read_bytes() == previous_bytes


def test_inventory_and_refresh_preserve_explicit_cache_roles(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "cache.bin").write_bytes(b"cache-v1")
    (root / "artifact-cache.bin").write_bytes(b"artifact-cache-v1")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
        cache_paths=("cache.bin",),
    )
    manifest_path = root / MANIFEST_NAME
    (root / "new-cache.bin").write_bytes(b"cache-v2")

    refresh_storage_object(
        root,
        expected_manifest_digest=_digest(manifest_path),
        producer_revision="test-revision-2",
        cache_paths=("artifact-cache.bin", "new-cache.bin"),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert {item["path"]: item["role"] for item in manifest["resources"]} == {
        "artifact-cache.bin": "cache",
        "cache.bin": "cache",
        "new-cache.bin": "cache",
    }


@pytest.mark.parametrize("protected_role", ["input", "metadata"])
def test_refresh_never_reclassifies_protected_resources_as_cache(
    tmp_path: Path,
    protected_role: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    protected = root / "protected.txt"
    protected.write_text("protected\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
        input_paths=("protected.txt",) if protected_role == "input" else (),
        metadata_paths=("protected.txt",) if protected_role == "metadata" else (),
    )
    manifest_path = root / MANIFEST_NAME
    prior_bytes = manifest_path.read_bytes()

    with pytest.raises(StorageObjectError, match="input and metadata roles remain protected"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
            cache_paths=("protected.txt",),
        )

    assert manifest_path.read_bytes() == prior_bytes


def test_failed_refresh_atomically_restores_readonly_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    manifest_path.chmod(0o444)
    expected_digest = _digest(manifest_path)
    (root / "result.txt").write_text("result\n", encoding="utf-8")

    def _reject_manifest(*_args: object, **_kwargs: object) -> object:
        raise StorageObjectError("forced post-write validation failure")

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _reject_manifest)

    with pytest.raises(StorageObjectError, match="forced post-write validation failure"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444
    assert not tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))


def test_inventory_refuses_manifest_created_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    external_bytes = b'{"external":"receipt"}\n'
    original_link = storage_inventory.os.link
    injected = False

    def _link_after_competing_receipt(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal injected
        if Path(destination) == manifest_path and not injected:
            injected = True
            manifest_path.write_bytes(external_bytes)
        original_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(storage_inventory.os, "link", _link_after_competing_receipt)

    with pytest.raises(StorageObjectError, match="manifest appeared before publication"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert injected
    assert manifest_path.read_bytes() == external_bytes
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_refresh_hashes_and_parses_one_snapshot_then_rejects_manifest_replaced_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    initial_bytes = manifest_path.read_bytes()
    replacement = json.loads(initial_bytes)
    replacement["owner_tool"] = "replacement-owner"
    replacement_bytes = (json.dumps(replacement, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_digest = _digest(manifest_path)
    original_exchange = storage_inventory._atomic_exchange
    original_load = storage_inventory.load_storage_object_manifest_bytes
    parsed_snapshots: list[bytes] = []
    injected = False

    def _exchange_after_competing_receipt(source: Path, destination: Path) -> None:
        nonlocal injected
        if destination == manifest_path and not injected:
            injected = True
            manifest_path.write_bytes(replacement_bytes)
        original_exchange(source, destination)

    def _record_parsed_snapshot(content: bytes, *, source_label: str) -> object:
        parsed_snapshots.append(content)
        return original_load(content, source_label=source_label)

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _exchange_after_competing_receipt)
    monkeypatch.setattr(storage_inventory, "load_storage_object_manifest_bytes", _record_parsed_snapshot)

    with pytest.raises(StorageObjectError, match="manifest changed before publication"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert parsed_snapshots == [initial_bytes]
    assert injected
    assert manifest_path.read_bytes() == replacement_bytes
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_refresh_reports_typed_unsupported_platform_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    monkeypatch.setattr(storage_inventory.sys, "platform", "unsupported")

    with pytest.raises(
        StorageObjectPublicationUnsupported,
        match="does not support atomic storage manifest exchange",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_inventory_reports_typed_unsupported_hard_link_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    def _unsupported_link(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "injected unsupported hard link")

    monkeypatch.setattr(storage_inventory.os, "link", _unsupported_link)

    with pytest.raises(
        StorageObjectPublicationUnsupported,
        match="does not support atomic create-only manifest publication",
    ):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / MANIFEST_NAME).exists()
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_refresh_preserves_both_receipts_when_atomic_swap_back_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    replacement = json.loads(manifest_path.read_bytes())
    replacement["owner_tool"] = "replacement-owner"
    replacement_bytes = (json.dumps(replacement, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_digest = _digest(manifest_path)
    original_exchange = storage_inventory._atomic_exchange
    exchange_calls = 0

    def _fail_swap_back(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            destination.write_bytes(replacement_bytes)
            original_exchange(source, destination)
            return
        raise OSError("injected atomic swap-back failure")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _fail_swap_back)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="rollback failed.*outcome is uncertain",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert exchange_calls == 2
    published = json.loads(manifest_path.read_bytes())
    assert published["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == replacement_bytes


def test_refresh_restores_manifest_when_verification_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()

    def _interrupt_verification(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _interrupt_verification)

    with pytest.raises(KeyboardInterrupt):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes


def test_inventory_removes_manifest_when_verification_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    def _interrupt_verification(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _interrupt_verification)

    with pytest.raises(KeyboardInterrupt):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rolls_back_when_atomic_link_publication_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    original_link = storage_inventory.os.link

    def _interrupt_after_link(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        original_link(source, destination, follow_symlinks=follow_symlinks)
        raise KeyboardInterrupt

    monkeypatch.setattr(storage_inventory.os, "link", _interrupt_after_link)

    with pytest.raises(KeyboardInterrupt):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rollback_preserves_receipt_replaced_at_commit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    competitor_root = tmp_path / "competitor"
    competitor_root.mkdir()
    (competitor_root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        competitor_root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="competing-revision",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    competitor_bytes = (competitor_root / MANIFEST_NAME).read_bytes()

    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_move = storage_inventory._atomic_move_no_replace
    original_replace = storage_inventory.os.replace
    original_unlink = storage_inventory.os.unlink
    injected = False

    def _inject_competitor() -> None:
        nonlocal injected
        staged = manifest_path.with_name(f".{MANIFEST_NAME}.competitor")
        staged.write_bytes(competitor_bytes)
        original_replace(staged, manifest_path)
        injected = True

    def _move_after_competing_receipt(source: Path, destination: Path) -> None:
        if source == manifest_path and not injected:
            _inject_competitor()
        original_move(source, destination)

    def _unlink_after_competing_receipt(path: str | bytes, *args: object, **kwargs: object) -> None:
        if Path(path) == manifest_path and not injected:
            _inject_competitor()
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_atomic_move_no_replace", _move_after_competing_receipt)
    monkeypatch.setattr(storage_inventory.os, "unlink", _unlink_after_competing_receipt)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectError):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-1",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert injected
    assert manifest_path.read_bytes() == competitor_bytes
    assert not tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))


def test_refresh_swaps_back_when_atomic_exchange_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    original_exchange = storage_inventory._atomic_exchange
    exchange_calls = 0

    def _interrupt_after_exchange(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        original_exchange(source, destination)
        if exchange_calls == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _interrupt_after_exchange)

    with pytest.raises(KeyboardInterrupt):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes
    assert exchange_calls == 2


def test_refresh_rollback_preserves_receipt_replaced_at_commit_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    competitor = json.loads(previous_bytes)
    competitor["producer_revision"] = "competing-revision"
    competitor_bytes = (json.dumps(competitor, indent=2, sort_keys=True) + "\n").encode("utf-8")
    original_exchange = storage_inventory._atomic_exchange
    original_replace = storage_inventory.os.replace
    exchange_calls = 0
    injected = False

    def _exchange_after_competing_receipt(source: Path, destination: Path) -> None:
        nonlocal exchange_calls, injected
        exchange_calls += 1
        if exchange_calls == 2:
            staged = destination.with_name(f".{MANIFEST_NAME}.competitor")
            staged.write_bytes(competitor_bytes)
            original_replace(staged, destination)
            injected = True
        original_exchange(source, destination)

    def _replace_after_competing_receipt(source: Path, destination: Path) -> None:
        nonlocal injected
        if source.name.startswith(f".{MANIFEST_NAME}.restore-"):
            destination.write_bytes(competitor_bytes)
            injected = True
        original_replace(source, destination)

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _exchange_after_competing_receipt)
    monkeypatch.setattr(storage_inventory.os, "replace", _replace_after_competing_receipt)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectError):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert manifest_path.read_bytes() == competitor_bytes
    assert not tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))


def test_refresh_rollback_retains_both_receipts_when_swap_back_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    previous_bytes = b"previous receipt\n"
    published_bytes = b"published receipt\n"
    competitor_bytes = b"competing receipt\n"
    manifest_path.write_bytes(published_bytes)
    original_exchange = storage_inventory._atomic_exchange
    original_replace = storage_inventory.os.replace
    exchange_calls = 0

    def _fail_swap_back(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            staged = destination.with_name(f".{MANIFEST_NAME}.competitor")
            staged.write_bytes(competitor_bytes)
            original_replace(staged, destination)
            original_exchange(source, destination)
            return
        raise OSError("injected rollback swap-back failure")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _fail_swap_back)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="rollback failed.*outcome is uncertain",
    ):
        storage_inventory._rollback_manifest(
            manifest_path,
            published_bytes=published_bytes,
            previous_bytes=previous_bytes,
            previous_mode=0o644,
            operation_error=StorageObjectError("injected validation failure"),
        )

    assert exchange_calls == 2
    assert manifest_path.read_bytes() == previous_bytes
    recovery_paths = tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == competitor_bytes


def test_refresh_rollback_fails_typed_unsupported_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    published_bytes = b"published receipt\n"
    manifest_path.write_bytes(published_bytes)

    def _unsupported_exchange(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectPublicationUnsupported("injected unsupported exchange")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _unsupported_exchange)

    with pytest.raises(StorageObjectPublicationUnsupported, match="injected unsupported exchange"):
        storage_inventory._rollback_manifest(
            manifest_path,
            published_bytes=published_bytes,
            previous_bytes=b"previous receipt\n",
            previous_mode=0o644,
            operation_error=StorageObjectError("injected validation failure"),
        )

    assert manifest_path.read_bytes() == published_bytes
    assert not tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))


def test_refresh_rejects_duplicate_resource_paths_before_collapsing_roles(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload_path = root / "payload.txt"
    payload_path.write_text("payload\n", encoding="utf-8")
    inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="cruncher",
        object_kind="workspace",
        content_schema="cruncher.workspace",
        content_schema_version="1",
        producer_revision="test-revision-1",
        storage_class="reproducible",
        retention_policy="review-before-delete",
        input_paths=("payload.txt",),
    )
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"].append({**manifest["resources"][0], "role": "artifact"})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    expected_digest = _digest(manifest_path)

    with pytest.raises(StorageObjectError, match="resource path is declared more than once"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )
