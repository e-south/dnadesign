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
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from filelock import FileLock

import dnadesign.contracts.storage_objects.inventory as storage_inventory
import dnadesign.contracts.storage_objects.locking as storage_locking
import dnadesign.contracts.storage_objects.validation as storage_validation
from dnadesign.contracts.storage_objects import (
    MANIFEST_NAME,
    StorageObjectError,
    StorageObjectPublicationUncertain,
    StorageObjectPublicationUnsupported,
    inventory_storage_object,
    refresh_storage_object,
    verify_storage_object,
)
from dnadesign.contracts.storage_objects.models import LOCK_NAME


def _digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def test_inventory_retries_one_transient_post_publication_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    original_verify = storage_inventory.verify_storage_object
    calls = 0
    settle_delays: list[float] = []

    def _verify(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise StorageObjectError(storage_inventory._TRANSIENT_POST_PUBLICATION_VALIDATION_ERROR)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _verify)
    monkeypatch.setattr(storage_inventory.time, "sleep", settle_delays.append)

    summary = inventory_storage_object(
        root,
        storage_id="pilot",
        owner_repository="dnadesign",
        owner_tool="usr",
        object_kind="store",
        content_schema="usr.dataset-root",
        content_schema_version="v1",
        producer_revision="test-revision",
        storage_class="cold",
        retention_policy="cold",
    )

    assert summary["status"] == "verified"
    assert calls == 2
    assert settle_delays == [storage_inventory._POST_PUBLICATION_SETTLE_SECONDS]


def test_inventory_rejects_receipt_replaced_before_transient_validation_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_verify = storage_inventory.verify_storage_object
    calls = 0
    settle_delays: list[float] = []
    replacement_identity: tuple[int, int] | None = None

    def _verify(*args: object, **kwargs: object):
        nonlocal calls, replacement_identity
        calls += 1
        if calls == 1:
            replacement = manifest_path.with_name(f".{MANIFEST_NAME}.competitor")
            replacement.write_bytes(manifest_path.read_bytes())
            replacement.chmod(manifest_path.stat(follow_symlinks=False).st_mode & 0o777)
            replacement.replace(manifest_path)
            replacement_stat = manifest_path.stat(follow_symlinks=False)
            replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
            raise StorageObjectError(storage_inventory._TRANSIENT_POST_PUBLICATION_VALIDATION_ERROR)
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _verify)
    monkeypatch.setattr(storage_inventory.time, "sleep", settle_delays.append)

    with pytest.raises(StorageObjectPublicationUncertain, match="cannot identify the receipt moved"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="usr",
            object_kind="store",
            content_schema="usr.dataset-root",
            content_schema_version="v1",
            producer_revision="test-revision",
            storage_class="cold",
            retention_policy="cold",
        )

    assert calls == 1
    assert settle_delays == []
    assert replacement_identity is not None
    assert not manifest_path.exists()
    recovery_paths = tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))
    assert len(recovery_paths) == 1
    recovery_stat = recovery_paths[0].stat(follow_symlinks=False)
    assert (recovery_stat.st_dev, recovery_stat.st_ino) == replacement_identity


@pytest.mark.parametrize("competitor_changes_bytes", [False, True])
def test_inventory_rejects_aba_receipt_replacement_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    competitor_changes_bytes: bool,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_verify = storage_inventory.verify_storage_object
    calls = 0
    published_identity: tuple[int, int] | None = None

    def _verify(*args: object, **kwargs: object):
        nonlocal calls, published_identity
        calls += 1
        published_stat = manifest_path.stat(follow_symlinks=False)
        published_identity = (published_stat.st_dev, published_stat.st_ino)
        published_backup = tmp_path / "published-manifest-backup.json"
        os.link(manifest_path, published_backup)
        competitor = tmp_path / "competing-manifest.json"
        if competitor_changes_bytes:
            competitor_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            competitor_payload["producer_revision"] = "competing-revision"
            competitor.write_text(
                json.dumps(competitor_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            competitor.write_bytes(manifest_path.read_bytes())
        competitor.chmod(stat.S_IMODE(published_stat.st_mode))
        competitor.replace(manifest_path)
        try:
            verified = original_verify(*args, **kwargs)
        finally:
            published_backup.replace(manifest_path)
        restored_stat = manifest_path.stat(follow_symlinks=False)
        assert (restored_stat.st_dev, restored_stat.st_ino) == published_identity
        return verified

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _verify)

    with pytest.raises(StorageObjectError, match="verification result does not match the published receipt"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="usr",
            object_kind="store",
            content_schema="usr.dataset-root",
            content_schema_version="v1",
            producer_revision="test-revision",
            storage_class="cold",
            retention_policy="cold",
        )

    assert calls == 1
    assert published_identity is not None
    assert not manifest_path.exists()


def test_inventory_does_not_retry_semantic_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    calls = 0
    settle_delays: list[float] = []

    def _verify(*_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        raise StorageObjectError("declared resource digest mismatch")

    monkeypatch.setattr(storage_inventory, "verify_storage_object", _verify)
    monkeypatch.setattr(storage_inventory.time, "sleep", settle_delays.append)

    with pytest.raises(StorageObjectError, match="declared resource digest mismatch"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="usr",
            object_kind="store",
            content_schema="usr.dataset-root",
            content_schema_version="v1",
            producer_revision="test-revision",
            storage_class="cold",
            retention_policy="cold",
        )

    assert calls == 1
    assert settle_delays == []
    assert not (root / MANIFEST_NAME).exists()


def _set_git_index_mode(checkout: Path, path: Path, mode: str) -> None:
    relative = path.relative_to(checkout).as_posix()
    if mode == "160000":
        object_id = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        object_id = (
            subprocess.run(
                ["git", "-C", str(checkout), "hash-object", "-w", "--stdin"],
                input=path.read_bytes(),
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )
    subprocess.run(
        ["git", "-C", str(checkout), "update-index", "--add", "--cacheinfo", f"{mode},{object_id},{relative}"],
        check=True,
    )


def _seed_tracked_demo(tmp_path: Path) -> tuple[Path, Path]:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)
    return root, root / MANIFEST_NAME


def _fail_late_demo_manifest_stat(
    monkeypatch: pytest.MonkeyPatch,
    manifest_path: Path,
) -> None:
    original_verify_demo = storage_validation._verify_demo
    original_stat = Path.stat
    armed = False

    def _verify_demo_with_stat_failure(*args: object, **kwargs: object) -> None:
        nonlocal armed
        armed = True
        try:
            original_verify_demo(*args, **kwargs)
        finally:
            armed = False

    def _stat(path: Path, *args: object, **kwargs: object):
        if armed and path == manifest_path:
            raise OSError("injected late demo manifest stat failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(storage_validation, "_verify_demo", _verify_demo_with_stat_failure)
    monkeypatch.setattr(Path, "stat", _stat)


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
    assert stat.S_IMODE((root / LOCK_NAME).stat().st_mode) == 0o600
    indexed_lock = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "--stage", "--", f"examples/pilot/{LOCK_NAME}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert indexed_lock.startswith("100644 ")


def test_demo_validation_requires_empty_lock_to_match_git_index(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    lock_path = root / LOCK_NAME
    subprocess.run(
        ["git", "-C", str(checkout), "add", "examples/pilot/storage.object.json"],
        check=True,
    )

    with pytest.raises(StorageObjectError, match=r"demo file is not tracked: .*\.storage-object\.lock"):
        verify_storage_object(root)

    lock_path.write_text("not empty\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(checkout), "add", "examples/pilot/.storage-object.lock"],
        check=True,
    )
    lock_path.write_bytes(b"")
    with pytest.raises(StorageObjectError, match=r"demo file differs from Git index: .*\.storage-object\.lock"):
        verify_storage_object(root)

    subprocess.run(
        ["git", "-C", str(checkout), "add", "examples/pilot/.storage-object.lock"],
        check=True,
    )
    assert verify_storage_object(root).summary()["status"] == "verified"


def test_demo_validation_reads_original_index_blob_despite_git_replace_ref(tmp_path: Path) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    payload_path = root / "payload.txt"
    relative_payload = payload_path.relative_to(checkout).as_posix()
    indexed_blob = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "--stage", "--", relative_payload],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()[1]
    replacement_bytes = b"replacement payload\n"
    replacement_blob = (
        subprocess.run(
            ["git", "-C", str(checkout), "hash-object", "-w", "--stdin"],
            input=replacement_bytes,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    subprocess.run(
        ["git", "-C", str(checkout), "replace", indexed_blob, replacement_blob],
        check=True,
    )
    payload_path.write_bytes(replacement_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0]["digest"] = f"sha256:{hashlib.sha256(replacement_bytes).hexdigest()}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(checkout), "add", manifest_path.relative_to(checkout).as_posix()],
        check=True,
    )

    replaced_blob = subprocess.run(
        ["git", "-C", str(checkout), "cat-file", "blob", indexed_blob],
        check=True,
        capture_output=True,
    ).stdout
    assert replaced_blob == replacement_bytes

    with pytest.raises(StorageObjectError, match=f"demo file differs from Git index: {relative_payload}"):
        verify_storage_object(root)


@pytest.mark.parametrize("override_kind", ["index", "repository"])
def test_demo_validation_clears_repository_local_git_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_kind: str,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    replacement_bytes = b"alternate authority payload\n"
    (root / "payload.txt").write_bytes(replacement_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resources"][0]["digest"] = f"sha256:{hashlib.sha256(replacement_bytes).hexdigest()}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    relative_root = root.relative_to(checkout).as_posix()

    if override_kind == "index":
        alternate_index = tmp_path / "alternate.index"
        alternate_index.write_bytes((checkout / ".git" / "index").read_bytes())
        alternate_env = {**os.environ, "GIT_INDEX_FILE": str(alternate_index)}
        subprocess.run(
            ["git", "-C", str(checkout), "add", "--", relative_root],
            env=alternate_env,
            check=True,
        )
        monkeypatch.setenv("GIT_INDEX_FILE", str(alternate_index))
    else:
        alternate_git_dir = tmp_path / "alternate.git"
        subprocess.run(["git", "init", "-q", "--bare", str(alternate_git_dir)], check=True)
        subprocess.run(
            [
                "git",
                f"--git-dir={alternate_git_dir}",
                f"--work-tree={checkout}",
                "add",
                "--",
                relative_root,
            ],
            check=True,
        )
        monkeypatch.setenv("GIT_DIR", str(alternate_git_dir))
        monkeypatch.setenv("GIT_WORK_TREE", str(checkout))

    with pytest.raises(StorageObjectError, match="demo file differs from Git index"):
        verify_storage_object(root)


@pytest.mark.parametrize(
    ("target_kind", "error"),
    [
        ("manifest", "demo manifest changed during Git index validation"),
        ("resource", "digest mismatch"),
        ("manifest_identity", "demo storage object changed during Git index validation"),
        ("resource_identity", "demo storage object changed during Git index validation"),
    ],
)
def test_demo_validation_rechecks_filesystem_bytes_after_git_index_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
    error: str,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    payload_path = root / "payload.txt"
    target_path = manifest_path if target_kind.startswith("manifest") else payload_path
    relative_target = target_path.relative_to(checkout).as_posix()
    target_blob = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "--stage", "--", relative_target],
        check=True,
        capture_output=True,
    ).stdout.split()[1]
    original_run = storage_validation.subprocess.run
    changed = False

    def _run(*args: object, **kwargs: object):
        nonlocal changed
        completed = original_run(*args, **kwargs)
        command = args[0]
        if isinstance(command, list) and "cat-file" in command and command[-1] == target_blob and not changed:
            if target_kind.endswith("_identity"):
                replacement = target_path.with_name(f".{target_path.name}.same-bytes")
                replacement.write_bytes(target_path.read_bytes())
                replacement.chmod(stat.S_IMODE(target_path.stat().st_mode))
                os.replace(replacement, target_path)
            elif target_kind == "manifest":
                manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
            else:
                payload_path.write_text("changed after Git validation\n", encoding="utf-8")
            changed = True
        return completed

    monkeypatch.setattr(storage_validation.subprocess, "run", _run)

    with pytest.raises(StorageObjectError, match=error):
        verify_storage_object(root)

    assert changed


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_demo_validation_rechecks_exact_closure_after_git_index_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_kind: str,
) -> None:
    root, _manifest_path = _seed_tracked_demo(tmp_path)
    original_run = storage_validation.subprocess.run
    changed = False

    def _run(*args: object, **kwargs: object):
        nonlocal changed
        completed = original_run(*args, **kwargs)
        command = args[0]
        if isinstance(command, list) and "cat-file" in command and not changed:
            undeclared = root / "undeclared"
            if entry_kind == "file":
                undeclared.write_text("created during Git validation\n", encoding="utf-8")
            else:
                undeclared.mkdir()
            changed = True
        return completed

    monkeypatch.setattr(storage_validation.subprocess, "run", _run)

    with pytest.raises(StorageObjectError, match="changed during Git index validation"):
        verify_storage_object(root)

    assert changed


@pytest.mark.parametrize("replacement_kind", ["root", "nested_directory"])
def test_demo_validation_rejects_same_path_directory_replacement_after_git_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    if replacement_kind == "nested_directory":
        nested = root / "nested"
        nested.mkdir()
        (root / "payload.txt").replace(nested / "payload.txt")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["resources"][0]["path"] = "nested/payload.txt"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(checkout), "add", "-A", "--", root.relative_to(checkout).as_posix()],
            check=True,
        )
        assert verify_storage_object(root).summary()["status"] == "verified"
    original_run = storage_validation.subprocess.run
    changed = False

    def _run(*args: object, **kwargs: object):
        nonlocal changed
        completed = original_run(*args, **kwargs)
        command = args[0]
        if isinstance(command, list) and "cat-file" in command and not changed:
            target = root if replacement_kind == "root" else root / "nested"
            displaced = target.with_name(f".{target.name}.displaced")
            target_mode = stat.S_IMODE(target.stat(follow_symlinks=False).st_mode)
            os.replace(target, displaced)
            target.mkdir()
            target.chmod(target_mode)
            for entry in tuple(displaced.iterdir()):
                os.replace(entry, target / entry.name)
            displaced.rmdir()
            changed = True
        return completed

    monkeypatch.setattr(storage_validation.subprocess, "run", _run)

    with pytest.raises(StorageObjectError, match="changed during Git index validation"):
        verify_storage_object(root)

    assert changed


@pytest.mark.parametrize("target_kind", ["resource", "directory"])
def test_shared_demo_rejects_access_posture_drift_after_git_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    payload = root / "payload.txt"
    if target_kind == "directory":
        nested = root / "nested"
        nested.mkdir()
        payload.replace(nested / payload.name)
        payload = nested / payload.name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["resources"][0]["path"] = "nested/payload.txt"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run(
            ["git", "-C", str(checkout), "add", "-A", "--", root.relative_to(checkout).as_posix()],
            check=True,
        )
        nested.chmod(0o750)
    root.chmod(0o2770)
    manifest_path.chmod(0o664)
    (root / LOCK_NAME).chmod(0o664)
    payload.chmod(0o640)
    (root / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}").chmod(0o750)
    assert verify_storage_object(root).summary()["status"] == "verified"
    original_run = storage_validation.subprocess.run
    changed = False

    def _run(*args: object, **kwargs: object):
        nonlocal changed
        completed = original_run(*args, **kwargs)
        command = args[0]
        if isinstance(command, list) and "cat-file" in command and not changed:
            target = payload if target_kind == "resource" else payload.parent
            target.chmod(0o600 if target_kind == "resource" else 0o740)
            changed = True
        return completed

    monkeypatch.setattr(storage_validation.subprocess, "run", _run)

    with pytest.raises(StorageObjectError, match="changed during Git index validation"):
        verify_storage_object(root)

    assert changed


def test_demo_validation_rejects_index_change_after_checked_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    relative_manifest = manifest_path.relative_to(checkout).as_posix()
    manifest_blob = subprocess.run(
        ["git", "-C", str(checkout), "ls-files", "--stage", "--", relative_manifest],
        check=True,
        capture_output=True,
    ).stdout.split()[1]
    original_run = storage_validation.subprocess.run
    changed = False

    def _run(*args: object, **kwargs: object):
        nonlocal changed
        completed = original_run(*args, **kwargs)
        command = args[0]
        if isinstance(command, list) and "cat-file" in command and command[-1] == manifest_blob and not changed:
            original_run(
                ["git", "-C", str(checkout), "rm", "--cached", "--", relative_manifest],
                check=True,
                capture_output=True,
            )
            changed = True
        return completed

    monkeypatch.setattr(storage_validation.subprocess, "run", _run)

    with pytest.raises(StorageObjectError, match="demo Git index changed during validation"):
        verify_storage_object(root)

    assert changed


def test_demo_validation_rejects_index_change_during_final_filesystem_recheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    relative_manifest = manifest_path.relative_to(checkout).as_posix()
    original_storage_tree_paths = storage_validation._storage_tree_paths
    scan_count = 0

    def _storage_tree_paths(path: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
        nonlocal scan_count
        snapshot = original_storage_tree_paths(path)
        scan_count += 1
        if scan_count == 3:
            subprocess.run(
                ["git", "-C", str(checkout), "rm", "--cached", "--", relative_manifest],
                check=True,
                capture_output=True,
            )
        return snapshot

    monkeypatch.setattr(storage_validation, "_storage_tree_paths", _storage_tree_paths)

    with pytest.raises(StorageObjectError, match="demo Git index changed during validation"):
        verify_storage_object(root)

    assert scan_count == 3


def test_demo_validation_rejects_extra_index_entry_inside_object_root(tmp_path: Path) -> None:
    root, _manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    relative_extra = (root / "checkout-only.txt").relative_to(checkout).as_posix()
    extra_blob = (
        subprocess.run(
            ["git", "-C", str(checkout), "hash-object", "-w", "--stdin"],
            input=b"present only in the index\n",
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    subprocess.run(
        ["git", "-C", str(checkout), "update-index", "--add", "--cacheinfo", f"100644,{extra_blob},{relative_extra}"],
        check=True,
    )

    assert not (root / "checkout-only.txt").exists()
    with pytest.raises(StorageObjectError, match=f"demo Git index has undeclared entries: {relative_extra}"):
        verify_storage_object(root)


def test_demo_validation_rejects_unmerged_declared_index_entry(tmp_path: Path) -> None:
    root, _manifest_path = _seed_tracked_demo(tmp_path)
    checkout = root.parents[1]
    payload = root / "payload.txt"
    relative_payload = payload.relative_to(checkout).as_posix()
    payload_blob = subprocess.run(
        ["git", "-C", str(checkout), "hash-object", "-w", "--stdin"],
        input=payload.read_bytes(),
        check=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(checkout), "update-index", "--index-info"],
        input=b"0 "
        + (b"0" * 40)
        + b"\t"
        + relative_payload.encode()
        + b"\n"
        + b"100644 "
        + payload_blob
        + b" 1\t"
        + relative_payload.encode()
        + b"\n"
        + b"100644 "
        + payload_blob
        + b" 2\t"
        + relative_payload.encode()
        + b"\n",
        check=True,
    )

    with pytest.raises(
        StorageObjectError,
        match=f"demo Git index entry must have exactly one stage-0 record: {relative_payload}",
    ):
        verify_storage_object(root)


def test_demo_validation_normalizes_late_manifest_stat_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    _fail_late_demo_manifest_stat(monkeypatch, manifest_path)

    with pytest.raises(StorageObjectError, match="cannot inspect demo manifest after validation"):
        verify_storage_object(root)

    assert manifest_path.exists()


def test_inventory_demo_normalizes_late_manifest_stat_failure_and_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    payload = root / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
    manifest_path = root / MANIFEST_NAME
    _fail_late_demo_manifest_stat(monkeypatch, manifest_path)

    with pytest.raises(StorageObjectError, match="cannot inspect demo manifest after validation"):
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
            demo=True,
        )

    assert not manifest_path.exists()
    assert payload.read_text(encoding="utf-8") == "payload\n"


def test_refresh_demo_normalizes_late_manifest_stat_failure_and_restores_prior_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest_path = _seed_tracked_demo(tmp_path)
    previous_bytes = manifest_path.read_bytes()
    previous_digest = _digest(manifest_path)
    _fail_late_demo_manifest_stat(monkeypatch, manifest_path)

    with pytest.raises(StorageObjectError, match="cannot inspect demo manifest after validation"):
        refresh_storage_object(
            root,
            expected_manifest_digest=previous_digest,
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes
    assert json.loads(previous_bytes)["producer_revision"] == "test-revision-1"


@pytest.mark.parametrize(
    ("target_name", "index_mode"),
    [
        (MANIFEST_NAME, "120000"),
        (LOCK_NAME, "120000"),
        ("payload.txt", "120000"),
        ("payload.txt", "160000"),
    ],
)
def test_demo_validation_rejects_nonregular_git_index_modes(
    tmp_path: Path,
    target_name: str,
    index_mode: str,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Storage Test",
            "-c",
            "user.email=storage-test@example.invalid",
            "commit",
            "-qm",
            "seed demo",
        ],
        check=True,
    )
    target = root / target_name
    _set_git_index_mode(checkout, target, index_mode)

    assert target.is_file() and not target.is_symlink()
    with pytest.raises(
        StorageObjectError,
        match=rf"demo Git index entry must be a regular file.*mode {index_mode}",
    ):
        verify_storage_object(root)


def test_demo_validation_accepts_regular_executable_git_index_mode(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    payload = root / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot"], check=True)
    _set_git_index_mode(checkout, payload, "100755")

    assert verify_storage_object(root).summary()["status"] == "verified"


def test_inventory_demo_requires_resource_bytes_to_match_git_index(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    payload = root / "payload.txt"
    payload.write_text("indexed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
    payload.write_text("unstaged\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="demo file differs from Git index"):
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
            demo=True,
        )

    assert not (root / MANIFEST_NAME).exists()
    assert payload.read_text(encoding="utf-8") == "unstaged\n"

    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
    assert (
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
            demo=True,
        )["status"]
        == "created-pending-git-add"
    )


def test_refresh_demo_allows_only_manifest_to_enter_pending_git_state(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    payload = root / "payload.txt"
    payload.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "add",
            "examples/pilot/storage.object.json",
            "examples/pilot/.storage-object.lock",
        ],
        check=True,
    )
    manifest_path = root / MANIFEST_NAME
    previous_bytes = manifest_path.read_bytes()
    payload.write_text("second\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="demo file differs from Git index"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes

    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
    summary = refresh_storage_object(
        root,
        expected_manifest_digest=_digest(manifest_path),
        producer_revision="test-revision-2",
    )

    assert summary["status"] == "refreshed-pending-git-add"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["producer_revision"] == "test-revision-2"
    with pytest.raises(StorageObjectError, match="demo file differs from Git index"):
        verify_storage_object(root)
    subprocess.run(summary["next_step"], shell=True, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert verify_storage_object(root).summary()["status"] == "verified"

    revision_only = refresh_storage_object(
        root,
        expected_manifest_digest=_digest(manifest_path),
        producer_revision="test-revision-3",
    )
    assert revision_only["status"] == "refreshed-pending-git-add"
    subprocess.run(revision_only["next_step"], shell=True, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert verify_storage_object(root).manifest.producer_revision == "test-revision-3"


@pytest.mark.parametrize(
    ("prior_state", "error"),
    [
        ("dirty", "demo file differs from Git index"),
        ("dirty-demo-flag", "demo file differs from Git index"),
        ("untracked", "demo file is not tracked"),
        ("symlink-index", "demo Git index entry must be a regular file.*mode 120000"),
    ],
)
def test_refresh_demo_requires_prior_manifest_to_match_git_index(
    tmp_path: Path,
    prior_state: str,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkout = tmp_path / "checkout"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    root = checkout / "examples" / "pilot"
    root.mkdir(parents=True)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "examples/pilot/payload.txt"], check=True)
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
        demo=True,
    )
    manifest_path = root / MANIFEST_NAME
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "add",
            "examples/pilot/storage.object.json",
            "examples/pilot/.storage-object.lock",
        ],
        check=True,
    )
    if prior_state in {"dirty", "dirty-demo-flag"}:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_state == "dirty":
            manifest["retention_policy"] = "retain"
        else:
            manifest["demo"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif prior_state == "untracked":
        subprocess.run(
            ["git", "-C", str(checkout), "rm", "--cached", "--", "examples/pilot/storage.object.json"],
            check=True,
            capture_output=True,
        )
    else:
        _set_git_index_mode(checkout, manifest_path, "120000")
    prior_bytes = manifest_path.read_bytes()
    publication_calls = 0
    original_publish = storage_inventory._publish_refresh_manifest

    def _record_publication(*args: object, **kwargs: object) -> None:
        nonlocal publication_calls
        publication_calls += 1
        original_publish(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_publish_refresh_manifest", _record_publication)

    with pytest.raises(StorageObjectError, match=error):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == prior_bytes
    assert publication_calls == 0
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


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

    assert stat.S_IMODE((root / LOCK_NAME).stat().st_mode) == 0o660
    assert stat.S_IMODE((root / MANIFEST_NAME).stat().st_mode) == 0o664
    assert (root / LOCK_NAME).stat().st_gid == root.stat().st_gid
    assert (root / MANIFEST_NAME).stat().st_gid == root.stat().st_gid


def test_inventory_creates_owner_only_lock_for_private_object(tmp_path: Path) -> None:
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

    assert stat.S_IMODE((root / LOCK_NAME).stat().st_mode) == 0o600


def test_shared_lock_bootstrap_opens_owner_only_before_descriptor_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    original_open = storage_locking.os.open
    creation_modes: list[int] = []

    def _open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == LOCK_NAME and flags & os.O_CREAT:
            creation_modes.append(mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(storage_locking.os, "open", _open)

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

    assert creation_modes == [0o600]
    assert stat.S_IMODE((root / LOCK_NAME).stat().st_mode) == 0o660


def test_inventory_normalizes_shared_cleanup_boundary_under_restrictive_umask(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    previous_umask = os.umask(0o077)
    try:
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
    finally:
        os.umask(previous_umask)

    cleanup_directory = root / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    assert stat.S_IMODE(cleanup_directory.stat().st_mode) & 0o777 == 0o750
    assert cleanup_directory.stat().st_gid == root.stat().st_gid
    assert not tuple(cleanup_directory.iterdir())


def test_inventory_repairs_safe_existing_shared_cleanup_boundary(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    cleanup_directory = root / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    cleanup_directory.mkdir(mode=0o700)
    cleanup_directory.chmod(0o700)

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

    assert stat.S_IMODE(cleanup_directory.stat().st_mode) & 0o777 == 0o750
    assert (root / MANIFEST_NAME).exists()


def test_inventory_rejects_unsafe_shared_cleanup_boundary_before_publication(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o2770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    cleanup_directory = root / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    cleanup_directory.mkdir(mode=0o770)
    cleanup_directory.chmod(0o770)

    with pytest.raises(StorageObjectPublicationUncertain, match="not an owner-write-private.*boundary"):
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


def test_inventory_rejects_other_writable_object_root_before_locking(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o707)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must not be other-writable"):
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


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
@pytest.mark.parametrize(("root_mode", "lock_mode"), [(0o700, 0o606), (0o2770, 0o666)])
def test_writers_reject_other_writable_coordination_lock(
    tmp_path: Path,
    operation: str,
    root_mode: int,
    lock_mode: int,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(root_mode)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    if operation == "refresh":
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
        prior_manifest = manifest_path.read_bytes()
    else:
        (root / LOCK_NAME).touch(mode=0o600)
        prior_manifest = None
    lock_path = root / LOCK_NAME
    lock_path.chmod(lock_mode)

    with pytest.raises(StorageObjectError, match="lock must not be other-writable"):
        if operation == "inventory":
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
        else:
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    assert stat.S_IMODE(lock_path.stat().st_mode) == lock_mode
    if prior_manifest is None:
        assert not manifest_path.exists()
    else:
        assert manifest_path.read_bytes() == prior_manifest


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
@pytest.mark.parametrize(("root_mode", "manifest_mode"), [(0o700, 0o606), (0o2770, 0o666)])
def test_writers_reject_other_writable_manifest_without_mutation(
    tmp_path: Path,
    operation: str,
    root_mode: int,
    manifest_mode: int,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(root_mode)
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
    )
    manifest_path = root / MANIFEST_NAME
    manifest_path.chmod(manifest_mode)
    prior_manifest = manifest_path.read_bytes()
    prior_payload = payload_path.read_bytes()
    prior_lock = (root / LOCK_NAME).read_bytes()

    with pytest.raises(StorageObjectError, match="manifest must not be other-writable"):
        if operation == "inventory":
            inventory_storage_object(
                root,
                storage_id="pilot",
                owner_repository="dnadesign",
                owner_tool="cruncher",
                object_kind="workspace",
                content_schema="cruncher.workspace",
                content_schema_version="1",
                producer_revision="test-revision-2",
                storage_class="reproducible",
                retention_policy="review-before-delete",
            )
        else:
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    assert manifest_path.read_bytes() == prior_manifest
    assert stat.S_IMODE(manifest_path.stat().st_mode) == manifest_mode
    assert payload_path.read_bytes() == prior_payload
    assert (root / LOCK_NAME).read_bytes() == prior_lock
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert not tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))
    assert not tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
def test_writers_report_manifest_posture_drift_at_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    expected_digest: str | None = None
    if operation == "refresh":
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
        expected_digest = _digest(manifest_path)
    original_binding_check = storage_inventory._assert_held_manifest_lock_binding

    def _change_manifest_after_lock_binding(*args: object, **kwargs: object) -> None:
        original_binding_check(*args, **kwargs)
        manifest_path.chmod(0o666)

    monkeypatch.setattr(storage_inventory, "_assert_held_manifest_lock_binding", _change_manifest_after_lock_binding)

    with pytest.raises(StorageObjectPublicationUncertain, match="coordination.*changed.*revalidate"):
        if operation == "inventory":
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
        else:
            assert expected_digest is not None
            refresh_storage_object(
                root,
                expected_manifest_digest=expected_digest,
                producer_revision="test-revision-2",
            )

    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o666
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


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


def test_inventory_rejects_sticky_group_shared_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o3770)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="must not set the sticky bit"):
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


def test_inventory_allows_sticky_private_root(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    root.chmod(0o1700)
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")

    summary = inventory_storage_object(
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

    assert summary["status"] == "verified"


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

    assert observed_directories == [root, root]


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


def test_inventory_rejects_reserved_staging_created_during_resource_enumeration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    recovery = root / f".{MANIFEST_NAME}.rollback-raced"
    original_storage_file_paths = storage_inventory.storage_file_paths
    injected = False

    def _storage_file_paths(storage_root: Path) -> tuple[Path, ...]:
        nonlocal injected
        if not injected:
            recovery.write_text("recoverable receipt\n", encoding="utf-8")
            injected = True
        return original_storage_file_paths(storage_root)

    monkeypatch.setattr(storage_inventory, "storage_file_paths", _storage_file_paths)

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*rollback-raced"):
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
    assert recovery.read_text(encoding="utf-8") == "recoverable receipt\n"
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_rejects_reserved_staging_snapshot_despite_guard_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    recovery = root / f".{MANIFEST_NAME}.rollback-aba"
    original_storage_file_paths = storage_inventory.storage_file_paths
    original_staging_guard = storage_inventory._assert_no_ambiguous_manifest_staging
    guard_calls = 0

    def _storage_file_paths(storage_root: Path) -> tuple[Path, ...]:
        files = original_storage_file_paths(storage_root)
        recovery.write_text("recoverable receipt\n", encoding="utf-8")
        return (*files, recovery)

    def _staging_guard(storage_root: Path) -> None:
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls != 2:
            original_staging_guard(storage_root)
            return
        recovery_bytes = recovery.read_bytes()
        recovery.unlink()
        original_staging_guard(storage_root)
        recovery.write_bytes(recovery_bytes)

    monkeypatch.setattr(storage_inventory, "storage_file_paths", _storage_file_paths)
    monkeypatch.setattr(storage_inventory, "_assert_no_ambiguous_manifest_staging", _staging_guard)

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*rollback-aba"):
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

    assert guard_calls == 2
    assert recovery.read_text(encoding="utf-8") == "recoverable receipt\n"
    assert not (root / MANIFEST_NAME).exists()


def test_inventory_fails_closed_on_create_rollback_recovery_state(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    recovery = root / f".{MANIFEST_NAME}.rollback-user-data"
    recovery.write_text("recoverable receipt\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*rollback-user-data"):
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

    assert recovery.read_text(encoding="utf-8") == "recoverable receipt\n"
    assert not (root / MANIFEST_NAME).exists()


def test_refresh_fails_closed_on_create_rollback_recovery_state(tmp_path: Path) -> None:
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
    original_manifest = manifest_path.read_bytes()
    recovery = root / f".{MANIFEST_NAME}.rollback-recovery"
    recovery.write_text("recoverable receipt\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*rollback-recovery"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == original_manifest
    assert recovery.read_text(encoding="utf-8") == "recoverable receipt\n"


def test_refresh_rejects_reserved_staging_created_during_resource_enumeration(
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
    original_manifest = manifest_path.read_bytes()
    recovery = root / f".{MANIFEST_NAME}.restore-raced"
    original_storage_file_paths = storage_inventory.storage_file_paths
    injected = False

    def _storage_file_paths(storage_root: Path) -> tuple[Path, ...]:
        nonlocal injected
        if not injected:
            recovery.write_text("recoverable receipt\n", encoding="utf-8")
            injected = True
        return original_storage_file_paths(storage_root)

    monkeypatch.setattr(storage_inventory, "storage_file_paths", _storage_file_paths)

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*restore-raced"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert manifest_path.read_bytes() == original_manifest
    assert recovery.read_text(encoding="utf-8") == "recoverable receipt\n"


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
@pytest.mark.parametrize("namespace", ["tmp", "restore", "rollback"])
def test_writers_reject_empty_reserved_directory_created_after_staging_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    namespace: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    if operation == "refresh":
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
        prior_manifest = manifest_path.read_bytes()
    else:
        prior_manifest = None
    recovery = root / f".{MANIFEST_NAME}.{namespace}-raced-directory"
    original_resource_guard = storage_inventory._assert_no_ambiguous_manifest_resources
    injected = False

    def _resource_guard(storage_root: Path, files: tuple[Path, ...]) -> None:
        nonlocal injected
        if not injected:
            recovery.mkdir()
            injected = True
        original_resource_guard(storage_root, files)

    monkeypatch.setattr(storage_inventory, "_assert_no_ambiguous_manifest_resources", _resource_guard)

    with pytest.raises(StorageObjectError, match=r"ambiguous manifest staging state.*raced-directory"):
        if operation == "inventory":
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
        else:
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    assert injected
    assert recovery.is_dir()
    assert not tuple(recovery.iterdir())
    if prior_manifest is None:
        assert not manifest_path.exists()
    else:
        assert manifest_path.read_bytes() == prior_manifest


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

    monkeypatch.setattr(storage_inventory, "_acquire_new_manifest_lock", _deny_lock)

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


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_writer_lock_race_never_opens_or_truncates_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    replacement_kind: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
    lock_path = root / LOCK_NAME
    expected_digest: str | None = None
    if operation == "inventory":
        lock_path.touch(mode=0o644)
    else:
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
    prior_manifest = (root / MANIFEST_NAME).read_bytes() if expected_digest is not None else None
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"do-not-truncate")
    original_acquire = storage_inventory._acquire_existing_manifest_lock
    raced = False

    def _acquire_after_replacement(*args: object, **kwargs: object) -> int:
        nonlocal raced
        if not raced:
            lock_path.unlink()
            if replacement_kind == "symlink":
                lock_path.symlink_to(victim)
            else:
                lock_path.write_bytes(b"competitor-lock-bytes")
                lock_path.chmod(0o644)
            raced = True
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_acquire_existing_manifest_lock", _acquire_after_replacement)

    with pytest.raises(StorageObjectError, match=r"lock(?: posture)? changed|cannot open existing"):
        if operation == "inventory":
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
        else:
            assert expected_digest is not None
            refresh_storage_object(
                root,
                expected_manifest_digest=expected_digest,
                producer_revision="test-revision-2",
            )

    assert raced
    if replacement_kind == "symlink":
        assert victim.read_bytes() == b"do-not-truncate"
    else:
        assert lock_path.read_bytes() == b"competitor-lock-bytes"
    if prior_manifest is not None:
        assert (root / MANIFEST_NAME).read_bytes() == prior_manifest


@pytest.mark.parametrize("replacement_kind", ["regular", "symlink"])
def test_inventory_lock_bootstrap_exclusive_create_preserves_competitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement_kind: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    lock_path = root / LOCK_NAME
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"do-not-truncate")
    original_acquire = storage_inventory._acquire_new_manifest_lock

    def _acquire_after_competitor(*args: object, **kwargs: object) -> tuple[int, tuple[int, int]]:
        if replacement_kind == "symlink":
            lock_path.symlink_to(victim)
        else:
            lock_path.write_bytes(b"competitor-lock-bytes")
            lock_path.chmod(0o644)
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_acquire_new_manifest_lock", _acquire_after_competitor)

    with pytest.raises(StorageObjectError, match="cannot exclusively create storage object lock"):
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
    if replacement_kind == "symlink":
        assert victim.read_bytes() == b"do-not-truncate"
    else:
        assert lock_path.read_bytes() == b"competitor-lock-bytes"


def test_inventory_lock_bootstrap_preserves_uncertainty_when_cleanup_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    lock_path = root / LOCK_NAME
    original_close = storage_locking.os.close
    failed_descriptor: int | None = None

    def _fail_file_fsync(descriptor: int) -> None:
        nonlocal failed_descriptor
        failed_descriptor = descriptor
        raise OSError("injected lock fsync failure")

    def _close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor == failed_descriptor:
            raise OSError("injected cleanup close failure")

    monkeypatch.setattr(storage_locking.os, "fsync", _fail_file_fsync)
    monkeypatch.setattr(storage_locking.os, "close", _close_then_fail)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="bootstrap did not complete.*descriptor cleanup also failed.*cleanup close failure",
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

    assert failed_descriptor is not None
    assert lock_path.is_file() and lock_path.read_bytes() == b""
    assert not (root / MANIFEST_NAME).exists()


def test_refresh_lock_acquisition_preserves_primary_error_when_cleanup_close_fails(
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
    prior_manifest = manifest_path.read_bytes()
    original_close = storage_locking.os.close
    failed_descriptor: int | None = None

    def _fail_acquisition(descriptor: int, *_args: object, **_kwargs: object) -> None:
        nonlocal failed_descriptor
        failed_descriptor = descriptor
        raise StorageObjectError("injected primary acquisition failure")

    def _close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        if descriptor == failed_descriptor:
            raise OSError("injected existing-lock close failure")

    monkeypatch.setattr(storage_locking, "_acquire_flock", _fail_acquisition)
    monkeypatch.setattr(storage_locking.os, "close", _close_then_fail)

    with pytest.raises(
        StorageObjectError,
        match="primary acquisition failure.*descriptor cleanup also failed.*existing-lock close failure",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert failed_descriptor is not None
    assert manifest_path.read_bytes() == prior_manifest
    assert (root / LOCK_NAME).read_bytes() == b""


def test_private_root_lock_bootstrap_binds_actual_file_gid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    original_acquire = storage_inventory._acquire_new_manifest_lock
    observed_expected_gid = object()

    def _acquire(*args: object, **kwargs: object) -> tuple[int, tuple[int, int]]:
        nonlocal observed_expected_gid
        observed_expected_gid = kwargs["expected_gid"]
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_acquire_new_manifest_lock", _acquire)

    summary = inventory_storage_object(
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

    assert observed_expected_gid is None
    assert summary["status"] == "verified"


def test_inventory_wraps_unsupported_filesystem_locking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()

    def _unsupported_lock(*_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("flock unsupported")

    monkeypatch.setattr(storage_inventory, "_acquire_new_manifest_lock", _unsupported_lock)

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

    original_acquire = storage_inventory._acquire_new_manifest_lock
    original_release = storage_inventory._release_manifest_lock

    def _acquire(*args: object, **kwargs: object) -> tuple[int, tuple[int, int]]:
        nonlocal acquired
        result = original_acquire(*args, **kwargs)
        acquired = True
        return result

    def _release(descriptor: int) -> None:
        nonlocal released
        released = True
        original_release(descriptor)

    real_stat = Path.stat

    def _stat(path: Path, *args: object, **kwargs: object):
        if acquired and path == root / LOCK_NAME:
            raise PermissionError("simulated post-acquisition inspection denial")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_acquire_new_manifest_lock", _acquire)
    monkeypatch.setattr(storage_inventory, "_release_manifest_lock", _release)
    monkeypatch.setattr(Path, "stat", _stat)

    with pytest.raises(StorageObjectError, match="cannot inspect storage object lock after acquisition"):
        with storage_inventory._manifest_lock(root, allow_missing=True):
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

    monkeypatch.setattr(storage_inventory, "_acquire_existing_manifest_lock", _deny_lock)

    with pytest.raises(StorageObjectError, match="cannot acquire storage object manifest lock"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )


def test_refresh_rejects_missing_lock_while_an_unlinked_inode_is_held(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
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
    prior_manifest = manifest_path.read_bytes()
    lock_path = root / LOCK_NAME
    held_lock = FileLock(lock_path)
    held_lock.acquire()
    lock_path.unlink()
    try:
        with pytest.raises(StorageObjectError, match="storage object lock is missing"):
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )
    finally:
        held_lock.release()

    assert not lock_path.exists()
    assert manifest_path.read_bytes() == prior_manifest
    assert payload.read_text(encoding="utf-8") == "payload\n"


def test_inventory_does_not_bootstrap_missing_lock_beside_existing_manifest(tmp_path: Path) -> None:
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
    prior_manifest = manifest_path.read_bytes()
    lock_path = root / LOCK_NAME
    lock_path.unlink()

    with pytest.raises(StorageObjectError, match="cannot bootstrap a missing storage object lock"):
        inventory_storage_object(
            root,
            storage_id="pilot",
            owner_repository="dnadesign",
            owner_tool="cruncher",
            object_kind="workspace",
            content_schema="cruncher.workspace",
            content_schema_version="1",
            producer_revision="test-revision-2",
            storage_class="reproducible",
            retention_policy="review-before-delete",
        )

    assert not lock_path.exists()
    assert manifest_path.read_bytes() == prior_manifest


def test_refresh_rejects_lock_replaced_between_inspection_and_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("payload\n", encoding="utf-8")
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
    prior_manifest = manifest_path.read_bytes()
    lock_path = root / LOCK_NAME
    initial_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)
    original_acquire = storage_inventory._acquire_existing_manifest_lock
    replaced = False

    def _acquire_after_replacement(*args: object, **kwargs: object):
        nonlocal replaced
        if not replaced:
            replacement = root / ".replacement-lock"
            replacement.touch(mode=0o644)
            replacement.replace(lock_path)
            replaced = True
        return original_acquire(*args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_acquire_existing_manifest_lock", _acquire_after_replacement)

    # Keep the stale inode alive, as it would be for a non-cooperating process
    # holding the unlinked lock, so Linux cannot recycle its inode immediately.
    with lock_path.open("rb"):
        with pytest.raises(StorageObjectError, match="lock changed before acquisition completed"):
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    assert replaced
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) != initial_identity
    assert manifest_path.read_bytes() == prior_manifest
    assert payload.read_text(encoding="utf-8") == "payload\n"


def test_inventory_reports_uncertain_when_lock_is_replaced_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    lock_path = root / LOCK_NAME
    original_write = storage_inventory._write_manifest
    second_lock: object | None = None

    def _write_then_replace_lock(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal second_lock
        summary = original_write(*args, **kwargs)
        replacement = root / ".replacement-lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o644)
        replacement.replace(lock_path)
        second_lock = FileLock(lock_path, timeout=0)
        second_lock.acquire()
        return summary

    monkeypatch.setattr(storage_inventory, "_write_manifest", _write_then_replace_lock)

    try:
        with pytest.raises(
            StorageObjectPublicationUncertain,
            match="operation committed.*coordination lock changed.*revalidate",
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
    finally:
        if second_lock is not None:
            second_lock.release()

    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["producer_revision"] == "test-revision-1"
    assert verify_storage_object(root).manifest.producer_revision == "test-revision-1"


def test_refresh_reports_uncertain_when_lock_is_replaced_after_commit(
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
    prior_digest = _digest(manifest_path)
    lock_path = root / LOCK_NAME
    original_write = storage_inventory._write_manifest
    second_lock: object | None = None

    def _write_then_replace_lock(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal second_lock
        summary = original_write(*args, **kwargs)
        replacement = root / ".replacement-lock"
        replacement.write_bytes(b"")
        replacement.chmod(0o644)
        replacement.replace(lock_path)
        second_lock = FileLock(lock_path, timeout=0)
        second_lock.acquire()
        return summary

    monkeypatch.setattr(storage_inventory, "_write_manifest", _write_then_replace_lock)

    try:
        with pytest.raises(
            StorageObjectPublicationUncertain,
            match="operation committed.*coordination lock changed.*revalidate",
        ):
            refresh_storage_object(
                root,
                expected_manifest_digest=prior_digest,
                producer_revision="test-revision-2",
            )
    finally:
        if second_lock is not None:
            second_lock.release()

    assert _digest(manifest_path) != prior_digest
    assert verify_storage_object(root).manifest.producer_revision == "test-revision-2"


def test_inventory_release_error_reports_committed_manifest_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_release = storage_inventory._release_manifest_lock
    injected = False

    def _release_then_fail(descriptor: int) -> None:
        nonlocal injected
        original_release(descriptor)
        if manifest_path.exists() and not injected:
            injected = True
            raise OSError("injected post-commit lock release failure")

    monkeypatch.setattr(storage_inventory, "_release_manifest_lock", _release_then_fail)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="operation committed and verified.*lock release failed.*winning_manifest_digest",
    ) as caught:
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
    assert _digest(manifest_path) in str(caught.value)
    assert "dnadesign.contracts.storage_objects validate" in str(caught.value)
    assert verify_storage_object(root).manifest.producer_revision == "test-revision-1"


def test_manifest_lock_preserves_publication_uncertainty_when_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    lock_path = root / LOCK_NAME
    lock_path.touch(mode=0o644)
    original_release = storage_inventory._release_manifest_lock
    injected = False

    def _release_then_fail(descriptor: int) -> None:
        nonlocal injected
        original_release(descriptor)
        if not injected:
            injected = True
            raise OSError("injected uncertain-outcome lock release failure")

    monkeypatch.setattr(storage_inventory, "_release_manifest_lock", _release_then_fail)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="publication outcome is uncertain.*manifest lock.*release also failed",
    ) as caught:
        with storage_inventory._manifest_lock(root):
            raise StorageObjectPublicationUncertain("candidate recovery state retained")

    assert injected
    assert isinstance(caught.value.__cause__, StorageObjectPublicationUncertain)
    assert "candidate recovery state retained" in str(caught.value)


def test_refresh_release_error_reports_new_committed_manifest_digest(
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
    prior_digest = _digest(manifest_path)
    original_release = storage_inventory._release_manifest_lock
    injected = False

    def _release_then_fail(descriptor: int) -> None:
        nonlocal injected
        original_release(descriptor)
        if (
            json.loads(manifest_path.read_text(encoding="utf-8"))["producer_revision"] == "test-revision-2"
            and not injected
        ):
            injected = True
            raise OSError("injected post-commit lock release failure")

    monkeypatch.setattr(storage_inventory, "_release_manifest_lock", _release_then_fail)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="operation committed and verified.*lock release failed.*winning_manifest_digest",
    ) as caught:
        refresh_storage_object(
            root,
            expected_manifest_digest=prior_digest,
            producer_revision="test-revision-2",
        )

    committed_digest = _digest(manifest_path)
    assert injected
    assert committed_digest != prior_digest
    assert committed_digest in str(caught.value)
    assert "do not retry with the prior CAS digest" in str(caught.value)
    assert verify_storage_object(root).manifest.producer_revision == "test-revision-2"


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


def test_validation_rejects_transient_refresh_candidate_until_cas_finishes(
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
    competitor = json.loads(initial_bytes)
    competitor["producer_revision"] = "competing-revision"
    competitor_bytes = (json.dumps(competitor, indent=2, sort_keys=True) + "\n").encode("utf-8")
    original_exchange = storage_inventory._atomic_exchange
    original_replace = storage_inventory.os.replace
    exchange_paused = Event()
    allow_refresh_to_finish = Event()
    exchange_calls = 0

    def _pause_after_exchange(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            staged = destination.with_name(f".{MANIFEST_NAME}.competitor")
            staged.write_bytes(competitor_bytes)
            original_replace(staged, destination)
            original_exchange(source, destination)
            exchange_paused.set()
            if not allow_refresh_to_finish.wait(timeout=5):
                raise RuntimeError("test did not release paused refresh")
            return
        original_exchange(source, destination)

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _pause_after_exchange)

    with ThreadPoolExecutor(max_workers=1) as executor:
        refresh_future = executor.submit(
            refresh_storage_object,
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )
        assert exchange_paused.wait(timeout=5)
        try:
            with pytest.raises(
                StorageObjectError,
                match=r"ambiguous manifest staging state.*\.storage\.object\.json\.tmp-",
            ):
                verify_storage_object(root)
        finally:
            allow_refresh_to_finish.set()
        with pytest.raises(StorageObjectPublicationUncertain, match="displaced receipt changed.*retained"):
            refresh_future.result(timeout=5)

    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == competitor_bytes


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

    with pytest.raises(StorageObjectPublicationUncertain, match="displaced receipt changed.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert parsed_snapshots == [initial_bytes]
    assert injected
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == replacement_bytes


def test_refresh_rejects_unsupported_publication_platform_before_mutation(
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
    lock_path = root / LOCK_NAME
    lock_state = lock_path.stat()
    previous_recovery_paths = tuple(sorted(root.glob(f".{MANIFEST_NAME}.*-*")))
    monkeypatch.setattr(storage_inventory.sys, "platform", "unsupported")

    with pytest.raises(
        StorageObjectPublicationUnsupported,
        match="requires POSIX ownership.*atomic rename",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == previous_bytes
    assert tuple(sorted(root.glob(f".{MANIFEST_NAME}.*-*"))) == previous_recovery_paths
    assert lock_path.read_bytes() == b""
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == (lock_state.st_dev, lock_state.st_ino)


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
def test_writers_reject_missing_geteuid_before_lock_or_cleanup_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    payload_path = root / "payload.txt"
    payload_path.write_text("payload\n", encoding="utf-8")
    if operation == "refresh":
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
    lock_path = root / LOCK_NAME
    previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    previous_lock_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino) if lock_path.exists() else None
    previous_recovery_paths = tuple(sorted(root.glob(f".{MANIFEST_NAME}.*-*")))
    monkeypatch.delattr(storage_inventory.os, "geteuid")

    with pytest.raises(StorageObjectPublicationUnsupported, match="POSIX ownership.*geteuid"):
        if operation == "inventory":
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
        else:
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    assert payload_path.read_text(encoding="utf-8") == "payload\n"
    assert tuple(sorted(root.glob(f".{MANIFEST_NAME}.*-*"))) == previous_recovery_paths
    if operation == "inventory":
        assert not manifest_path.exists()
        assert not lock_path.exists()
    else:
        assert manifest_path.read_bytes() == previous_manifest
        assert lock_path.read_bytes() == b""
        assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == previous_lock_identity


@pytest.mark.parametrize("operation", ["inventory", "refresh"])
@pytest.mark.parametrize("capability", ["fcntl.flock", "O_EXCL"])
def test_writers_reject_missing_descriptor_lock_capability_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    capability: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    if operation == "refresh":
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
    lock_path = root / LOCK_NAME
    previous_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
    previous_lock_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino) if lock_path.exists() else None
    if capability == "fcntl.flock":
        monkeypatch.setattr(storage_locking, "fcntl", None)
    else:
        monkeypatch.delattr(storage_locking.os, capability)

    with pytest.raises(StorageObjectPublicationUnsupported, match=capability.replace(".", r"\.")):
        if operation == "inventory":
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
        else:
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )

    if operation == "inventory":
        assert not manifest_path.exists()
        assert not lock_path.exists()
    else:
        assert manifest_path.read_bytes() == previous_manifest
        assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == previous_lock_identity


@pytest.mark.parametrize("operation", ["inventory", "refresh", "refresh_rollback"])
def test_writers_use_held_descriptor_chmod_without_nofollow_path_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    if operation.startswith("refresh"):
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
    supported = set(storage_inventory.os.supports_follow_symlinks)
    supported.discard(storage_inventory.os.chmod)
    monkeypatch.setattr(storage_inventory.os, "supports_follow_symlinks", supported)
    original_chmod = Path.chmod

    def _chmod(path: Path, mode: int, *, follow_symlinks: bool = True) -> None:
        if not follow_symlinks:
            raise NotImplementedError("injected Linux no-follow chmod limitation")
        original_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", _chmod)

    if operation == "inventory":
        summary = inventory_storage_object(
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
        assert summary["status"] == "verified"
    elif operation == "refresh":
        manifest_path = root / MANIFEST_NAME
        summary = refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )
        assert summary["status"] == "verified"
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["producer_revision"] == "test-revision-2"
    else:
        manifest_path = root / MANIFEST_NAME
        previous_bytes = manifest_path.read_bytes()

        def _reject_published_receipt(*_args: object, **_kwargs: object) -> object:
            raise StorageObjectError("injected post-publication validation failure")

        monkeypatch.setattr(storage_inventory, "verify_storage_object", _reject_published_receipt)
        with pytest.raises(StorageObjectError, match="injected post-publication validation failure"):
            refresh_storage_object(
                root,
                expected_manifest_digest=_digest(manifest_path),
                producer_revision="test-revision-2",
            )
        assert manifest_path.read_bytes() == previous_bytes


@pytest.mark.parametrize("capability", ["fchmod", "stat_dir_fd", "unlink_dir_fd"])
def test_inventory_preflights_adjacent_posix_publication_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    if capability == "fchmod":
        monkeypatch.delattr(storage_inventory.os, "fchmod")
    else:
        function_name = capability.removesuffix("_dir_fd")
        supported = set(storage_inventory.os.supports_dir_fd)
        supported.discard(getattr(storage_inventory.os, function_name))
        monkeypatch.setattr(storage_inventory.os, "supports_dir_fd", supported)

    with pytest.raises(StorageObjectPublicationUnsupported, match=capability.split("_")[0]):
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
    assert not (root / LOCK_NAME).exists()
    assert not tuple(root.glob(f".{MANIFEST_NAME}.*-*"))


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


def test_inventory_preflights_create_rollback_support_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    link_calls = 0
    original_link = storage_inventory.os.link

    def _record_link(*args: object, **kwargs: object) -> None:
        nonlocal link_calls
        link_calls += 1
        original_link(*args, **kwargs)

    def _unsupported_move(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectPublicationUnsupported("injected unsupported no-replace move")

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory.os, "link", _record_link)
    monkeypatch.setattr(storage_inventory, "_atomic_move_no_replace", _unsupported_move)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectPublicationUnsupported, match="unsupported no-replace move"):
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

    assert link_calls == 0
    assert not (root / MANIFEST_NAME).exists()
    assert not tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))


def test_refresh_retains_both_receipts_when_displaced_receipt_changes(
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
        match="displaced receipt changed.*retained",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert exchange_calls == 1
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


def test_refresh_validation_rollback_rejects_same_byte_canonical_replacement(
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
    original_publish = storage_inventory._publish_refresh_manifest
    publish_calls = 0
    replacement_identity: tuple[int, int] | None = None

    def _publish_after_same_byte_replacement(*args: object, **kwargs: object) -> object:
        nonlocal publish_calls, replacement_identity
        publish_calls += 1
        if publish_calls == 2:
            replacement = manifest_path.with_name(f".{MANIFEST_NAME}.competitor")
            replacement.write_bytes(manifest_path.read_bytes())
            replacement.chmod(manifest_path.stat(follow_symlinks=False).st_mode & 0o777)
            replacement.replace(manifest_path)
            replacement_stat = manifest_path.stat(follow_symlinks=False)
            replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
        return original_publish(*args, **kwargs)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "_publish_refresh_manifest", _publish_after_same_byte_replacement)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectPublicationUncertain, match="displaced receipt changed.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert publish_calls == 2
    assert replacement_identity is not None
    assert manifest_path.read_bytes() == previous_bytes
    recovery = tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery) == 1
    recovery_stat = recovery[0].stat(follow_symlinks=False)
    assert (recovery_stat.st_dev, recovery_stat.st_ino) == replacement_identity
    assert json.loads(recovery[0].read_bytes())["producer_revision"] == "test-revision-2"


def test_refresh_validation_rollback_rejects_replaced_restore_staging(
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
    original_publish = storage_inventory._publish_refresh_manifest
    publish_calls = 0
    attacker_bytes: bytes | None = None

    def _publish_after_restore_replacement(temporary: Path, *args: object, **kwargs: object) -> object:
        nonlocal attacker_bytes, publish_calls
        publish_calls += 1
        if publish_calls == 2:
            attacker = json.loads(temporary.read_bytes())
            attacker["producer_revision"] = "attacker-restore-revision"
            attacker_bytes = (json.dumps(attacker, indent=2, sort_keys=True) + "\n").encode()
            replacement = temporary.with_name(f"{temporary.name}.competitor")
            replacement.write_bytes(attacker_bytes)
            replacement.replace(temporary)
        return original_publish(temporary, *args, **kwargs)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "_publish_refresh_manifest", _publish_after_restore_replacement)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry changed before refresh publication"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert publish_calls == 2
    assert attacker_bytes is not None
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    recovery = tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == attacker_bytes


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


def test_inventory_retains_staging_entry_replaced_before_publication_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    foreign_bytes = b"foreign staging bytes\n"
    foreign_identity: tuple[int, int] | None = None
    original_fsync = storage_inventory._fsync_directory
    injected = False

    def _replace_staging_after_publication_fsync(directory: Path) -> None:
        nonlocal foreign_identity, injected
        original_fsync(directory)
        staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
        if manifest_path.exists() and staging and not injected:
            replacement = staging[0].with_name(f"{staging[0].name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(staging[0])
            replacement_stat = staging[0].stat(follow_symlinks=False)
            foreign_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
            injected = True

    monkeypatch.setattr(storage_inventory, "_fsync_directory", _replace_staging_after_publication_fsync)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry.*changed.*(restored|retained)"):
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
    assert foreign_identity is not None
    assert not manifest_path.exists()
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    staging_stat = staging[0].stat(follow_symlinks=False)
    assert (staging_stat.st_dev, staging_stat.st_ino) == foreign_identity
    assert staging[0].read_bytes() == foreign_bytes


def test_inventory_retains_replaced_staging_entry_during_failed_write_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    foreign_bytes = b"foreign failed-write staging bytes\n"

    def _replace_staging_then_fail(source: Path, _destination: Path, **_kwargs: object) -> None:
        replacement = source.with_name(f"{source.name}.competitor")
        replacement.write_bytes(foreign_bytes)
        replacement.replace(source)
        raise FileExistsError("injected publication collision")

    monkeypatch.setattr(storage_inventory.os, "link", _replace_staging_then_fail)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry.*changed.*(restored|retained)"):
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

    assert not manifest_path.exists()
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == foreign_bytes


def test_owned_cleanup_atomically_restores_replacement_at_displacement_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / f".{MANIFEST_NAME}.tmp-owned"
    staging.write_bytes(b"owned staging bytes\n")
    staging_stat = staging.stat(follow_symlinks=False)
    owned_identity = (staging_stat.st_dev, staging_stat.st_ino)
    foreign_bytes = b"foreign staging bytes\n"
    original_move = storage_inventory._atomic_move_no_replace_into_directory
    injected = False

    def _replace_at_atomic_cleanup_boundary(source: Path, destination_directory: int, destination_name: str) -> None:
        nonlocal injected
        if not injected:
            replacement = source.with_name(f"{source.name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(source)
            injected = True
        original_move(source, destination_directory, destination_name)
        if injected:
            raise FileNotFoundError("injected error after cleanup displacement")

    monkeypatch.setattr(
        storage_inventory,
        "_atomic_move_no_replace_into_directory",
        _replace_at_atomic_cleanup_boundary,
    )

    with pytest.raises(StorageObjectPublicationUncertain, match="foreign entry restored"):
        storage_inventory._unlink_owned_entry(
            staging,
            expected_identity=owned_identity,
            context="manifest staging entry",
        )

    assert injected
    assert staging.read_bytes() == foreign_bytes
    cleanup_directories = tuple(tmp_path.glob(f".{MANIFEST_NAME}.cleanup-owner-*"))
    assert len(cleanup_directories) == 1
    assert not tuple(cleanup_directories[0].iterdir())


def test_owned_cleanup_classifies_error_after_successful_displacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / f".{MANIFEST_NAME}.tmp-owned"
    staging.write_bytes(b"owned staging bytes\n")
    staging_stat = staging.stat(follow_symlinks=False)
    owned_identity = (staging_stat.st_dev, staging_stat.st_ino)
    original_move = storage_inventory._atomic_move_no_replace_into_directory

    def _move_then_raise(source: Path, destination_directory: int, destination_name: str) -> None:
        original_move(source, destination_directory, destination_name)
        raise FileNotFoundError("injected error after cleanup displacement")

    monkeypatch.setattr(storage_inventory, "_atomic_move_no_replace_into_directory", _move_then_raise)

    storage_inventory._unlink_owned_entry(
        staging,
        expected_identity=owned_identity,
        context="manifest staging entry",
    )

    assert not staging.exists()
    cleanup_directories = tuple(tmp_path.glob(f".{MANIFEST_NAME}.cleanup-owner-*"))
    assert len(cleanup_directories) == 1
    assert not tuple(cleanup_directories[0].iterdir())


def test_owned_cleanup_private_quarantine_cannot_delete_shared_replacement_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / f".{MANIFEST_NAME}.tmp-owned"
    staging.write_bytes(b"owned staging bytes\n")
    staging_stat = staging.stat(follow_symlinks=False)
    owned_identity = (staging_stat.st_dev, staging_stat.st_ino)
    foreign_bytes = b"foreign staging bytes\n"
    original_identity = storage_inventory._directory_entry_identity
    injected = False

    def _replace_shared_path_after_private_verification(directory_descriptor: int, name: str) -> tuple[int, int]:
        nonlocal injected
        identity = original_identity(directory_descriptor, name)
        staging.write_bytes(foreign_bytes)
        injected = True
        return identity

    def _forbid_path_unlink(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("cleanup must not unlink a shared pathname")

    monkeypatch.setattr(storage_inventory, "_directory_entry_identity", _replace_shared_path_after_private_verification)
    monkeypatch.setattr(Path, "unlink", _forbid_path_unlink)

    storage_inventory._unlink_owned_entry(
        staging,
        expected_identity=owned_identity,
        context="manifest staging entry",
    )

    assert injected
    assert staging.read_bytes() == foreign_bytes
    cleanup_directories = tuple(tmp_path.glob(f".{MANIFEST_NAME}.cleanup-owner-*"))
    assert len(cleanup_directories) == 1
    assert not tuple(cleanup_directories[0].iterdir())


def test_owned_cleanup_rejects_group_writable_cleanup_boundary(tmp_path: Path) -> None:
    staging = tmp_path / f".{MANIFEST_NAME}.tmp-owned"
    staging.write_bytes(b"owned staging bytes\n")
    staging_stat = staging.stat(follow_symlinks=False)
    owned_identity = (staging_stat.st_dev, staging_stat.st_ino)
    cleanup_directory = tmp_path / f".{MANIFEST_NAME}.cleanup-owner-{os.geteuid()}"
    cleanup_directory.mkdir(mode=0o770)
    cleanup_directory.chmod(0o770)

    with pytest.raises(StorageObjectPublicationUncertain, match="not an owner-write-private.*boundary"):
        storage_inventory._unlink_owned_entry(
            staging,
            expected_identity=owned_identity,
            context="manifest staging entry",
        )

    assert staging.read_bytes() == b"owned staging bytes\n"
    assert not tuple(cleanup_directory.iterdir())


def test_inventory_rejects_staging_replaced_before_create_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_publish = storage_inventory._publish_create_only_manifest
    attacker_bytes: bytes | None = None

    def _publish_after_staging_replacement(temporary: Path, *args: object, **kwargs: object) -> object:
        nonlocal attacker_bytes
        attacker = json.loads(temporary.read_bytes())
        attacker["producer_revision"] = "attacker-revision"
        attacker_bytes = (json.dumps(attacker, indent=2, sort_keys=True) + "\n").encode()
        replacement = temporary.with_name(f"{temporary.name}.competitor")
        replacement.write_bytes(attacker_bytes)
        replacement.replace(temporary)
        return original_publish(temporary, *args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_publish_create_only_manifest", _publish_after_staging_replacement)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry changed before create-only publication"):
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

    assert attacker_bytes is not None
    assert not manifest_path.exists()
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == attacker_bytes


def test_inventory_rollback_retains_receipt_replaced_at_commit_boundary(
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

    with pytest.raises(StorageObjectPublicationUncertain, match="cannot identify the receipt moved"):
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
    assert not manifest_path.exists()
    recovery_paths = tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == competitor_bytes


def test_inventory_rollback_retains_same_byte_receipt_replaced_after_content_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    published_identity: tuple[int, int] | None = None
    replacement_identity: tuple[int, int] | None = None
    original_link = storage_inventory.os.link
    original_rollback = storage_inventory._rollback_create_only_manifest

    def _capture_publication_identity(
        source: Path,
        destination: Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal published_identity
        original_link(source, destination, follow_symlinks=follow_symlinks)
        published_stat = destination.stat(follow_symlinks=False)
        published_identity = (published_stat.st_dev, published_stat.st_ino)

    def _rollback_after_same_byte_replacement(path: Path, **kwargs: object) -> None:
        nonlocal replacement_identity
        assert kwargs["published_identity"] == published_identity
        replacement = path.with_name(f".{MANIFEST_NAME}.competitor")
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(path.stat(follow_symlinks=False).st_mode & 0o777)
        replacement.replace(path)
        replacement_stat = path.stat(follow_symlinks=False)
        replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
        original_rollback(path, **kwargs)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory.os, "link", _capture_publication_identity)
    monkeypatch.setattr(storage_inventory, "_rollback_create_only_manifest", _rollback_after_same_byte_replacement)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectPublicationUncertain, match="cannot identify the receipt moved"):
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

    assert published_identity is not None
    assert replacement_identity is not None
    assert replacement_identity != published_identity
    assert not manifest_path.exists()
    recovery_paths = tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))
    assert len(recovery_paths) == 1
    recovery_stat = recovery_paths[0].stat(follow_symlinks=False)
    assert (recovery_stat.st_dev, recovery_stat.st_ino) == replacement_identity


def test_inventory_rollback_preserves_quarantine_name_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    collision_bytes = b"competing recovery bytes\n"
    collision_path: Path | None = None
    original_move = storage_inventory._atomic_move_no_replace

    def _move_after_collision(source: Path, destination: Path) -> None:
        nonlocal collision_path
        if source == manifest_path and destination.name.startswith(f".{MANIFEST_NAME}.rollback-"):
            destination.write_bytes(collision_bytes)
            collision_path = destination
        original_move(source, destination)

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "_atomic_move_no_replace", _move_after_collision)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectError, match="manifest rollback failed"):
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

    assert collision_path is not None
    assert collision_path.read_bytes() == collision_bytes
    assert manifest_path.is_file()


def test_inventory_rollback_retains_quarantine_replaced_after_move(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    collision_path: Path | None = None
    replacement_identity: tuple[int, int] | None = None
    original_move = storage_inventory._atomic_move_no_replace

    def _move_then_replace(source: Path, destination: Path) -> None:
        nonlocal collision_path, replacement_identity
        original_move(source, destination)
        if source == manifest_path and destination.name.startswith(f".{MANIFEST_NAME}.rollback-"):
            replacement = destination.with_name(f"{destination.name}.competitor")
            replacement.write_bytes(destination.read_bytes())
            replacement.chmod(destination.stat(follow_symlinks=False).st_mode & 0o777)
            replacement.replace(destination)
            replacement_stat = destination.stat(follow_symlinks=False)
            replacement_identity = (replacement_stat.st_dev, replacement_stat.st_ino)
            collision_path = destination

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "_atomic_move_no_replace", _move_then_replace)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(StorageObjectPublicationUncertain, match="cannot identify the receipt moved"):
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

    assert collision_path is not None
    assert replacement_identity is not None
    assert (collision_path.stat().st_dev, collision_path.stat().st_ino) == replacement_identity
    assert not manifest_path.exists()


def test_create_rollback_preserves_same_inode_quarantine_rewritten_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    published_bytes = b"published receipt\n"
    replacement_bytes = b"modified! receipt\n"
    manifest_path.write_bytes(published_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    published_identity = (manifest_stat.st_dev, manifest_stat.st_ino)
    original_read_bytes = Path.read_bytes
    injected = False

    def _read_then_rewrite_quarantine(path: Path) -> bytes:
        nonlocal injected
        content = original_read_bytes(path)
        if path.name.startswith(f".{MANIFEST_NAME}.rollback-") and not injected:
            before = path.stat(follow_symlinks=False)
            with path.open("r+b") as handle:
                handle.write(replacement_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            injected = True
        return content

    monkeypatch.setattr(Path, "read_bytes", _read_then_rewrite_quarantine)

    with pytest.raises(StorageObjectError, match="refusing to remove unrelated receipt bytes"):
        storage_inventory._rollback_create_only_manifest(
            manifest_path,
            published_bytes=published_bytes,
            published_identity=published_identity,
            operation_error=StorageObjectError("injected validation failure"),
        )

    assert injected
    assert original_read_bytes(manifest_path) == replacement_bytes
    assert not tuple(tmp_path.glob(f".{MANIFEST_NAME}.rollback-*"))


@pytest.mark.parametrize(
    ("cleanup_context", "manifest_remains"),
    [
        ("create-only rollback quarantine placeholder", True),
        ("create-only rollback quarantine entry", False),
    ],
)
def test_inventory_rollback_retains_quarantine_replaced_at_cleanup_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_context: str,
    manifest_remains: bool,
) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    manifest_path = root / MANIFEST_NAME
    original_unlink_owned = storage_inventory._unlink_owned_entry
    injected = False
    replacement_bytes: bytes | None = None

    def _replace_before_owned_cleanup(
        path: Path,
        *,
        expected_identity: tuple[int, int],
        context: str,
        missing_ok: bool = False,
    ) -> None:
        nonlocal injected, replacement_bytes
        if context == cleanup_context and not injected:
            replacement_bytes = path.read_bytes()
            replacement = path.with_name(f"{path.name}.competitor")
            replacement.write_bytes(replacement_bytes)
            replacement.replace(path)
            injected = True
        original_unlink_owned(
            path,
            expected_identity=expected_identity,
            context=context,
            missing_ok=missing_ok,
        )

    def _fail_verification(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectError("injected validation failure")

    monkeypatch.setattr(storage_inventory, "_unlink_owned_entry", _replace_before_owned_cleanup)
    monkeypatch.setattr(storage_inventory, "verify_storage_object", _fail_verification)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="quarantine.*changed.*(restored|retained)|rollback cleanup is uncertain",
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

    assert injected
    assert replacement_bytes is not None
    assert manifest_path.exists() is manifest_remains
    quarantine = tuple(root.glob(f".{MANIFEST_NAME}.rollback-*"))
    assert len(quarantine) == 1
    assert quarantine[0].read_bytes() == replacement_bytes


def test_create_preflight_retains_staging_replaced_at_cleanup_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_unlink_owned = storage_inventory._unlink_owned_entry
    injected = False

    def _replace_before_preflight_cleanup(
        path: Path,
        *,
        expected_identity: tuple[int, int],
        context: str,
        missing_ok: bool = False,
    ) -> None:
        nonlocal injected
        if context == "create-only rollback preflight staging entry" and path.exists() and not injected:
            replacement = path.with_name(f"{path.name}.competitor")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            injected = True
        original_unlink_owned(
            path,
            expected_identity=expected_identity,
            context=context,
            missing_ok=missing_ok,
        )

    monkeypatch.setattr(storage_inventory, "_unlink_owned_entry", _replace_before_preflight_cleanup)

    with pytest.raises(StorageObjectPublicationUncertain, match="preflight cleanup is uncertain"):
        storage_inventory._preflight_create_only_rollback(tmp_path)

    assert injected
    retained = tuple(tmp_path.glob(f".{MANIFEST_NAME}.tmp-preflight-*"))
    assert len(retained) == 1


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


def test_manifest_byte_match_rejects_same_inode_rewrite_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    expected_bytes = b"verified bytes\n"
    replacement_bytes = b"modified bytes\n"
    manifest_path.write_bytes(expected_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    expected_identity = (manifest_stat.st_dev, manifest_stat.st_ino)
    original_read_bytes = Path.read_bytes
    injected = False

    def _read_then_rewrite_same_inode(path: Path) -> bytes:
        nonlocal injected
        content = original_read_bytes(path)
        if path == manifest_path and not injected:
            before = path.stat(follow_symlinks=False)
            with path.open("r+b") as handle:
                handle.write(replacement_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            injected = True
        return content

    monkeypatch.setattr(Path, "read_bytes", _read_then_rewrite_same_inode)

    assert not storage_inventory._entry_matches_regular_bytes(
        manifest_path,
        expected_identity=expected_identity,
        expected_bytes=expected_bytes,
    )
    assert injected
    assert original_read_bytes(manifest_path) == replacement_bytes


def test_refresh_swap_back_retains_candidate_after_same_inode_receipt_rewrite(
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
    expected_digest = _digest(manifest_path)
    previous_bytes = manifest_path.read_bytes()
    replacement_bytes = b"!" + previous_bytes[1:]
    original_exchange = storage_inventory._atomic_exchange
    original_fsync_directory = storage_inventory._fsync_directory
    original_read_bytes = Path.read_bytes
    exchange_calls = 0
    durability_failure_injected = False
    rewrite_injected = False

    def _record_exchange(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        original_exchange(source, destination)
        exchange_calls += 1

    def _fail_first_publication_fsync(directory: Path) -> None:
        nonlocal durability_failure_injected
        if exchange_calls == 1 and not durability_failure_injected:
            durability_failure_injected = True
            raise OSError("injected publication durability failure")
        original_fsync_directory(directory)

    def _read_then_rewrite_swapped_back_receipt(path: Path) -> bytes:
        nonlocal rewrite_injected
        content = original_read_bytes(path)
        if path == manifest_path and exchange_calls == 2 and not rewrite_injected:
            before = path.stat(follow_symlinks=False)
            with path.open("r+b") as handle:
                handle.write(replacement_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            rewrite_injected = True
        return content

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _record_exchange)
    monkeypatch.setattr(storage_inventory, "_fsync_directory", _fail_first_publication_fsync)
    monkeypatch.setattr(Path, "read_bytes", _read_then_rewrite_swapped_back_receipt)

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="displaced receipt changed.*retained candidate and recovery entries",
    ):
        refresh_storage_object(
            root,
            expected_manifest_digest=expected_digest,
            producer_revision="test-revision-2",
        )

    assert durability_failure_injected
    assert rewrite_injected
    assert exchange_calls == 3
    assert json.loads(original_read_bytes(manifest_path))["producer_revision"] == "test-revision-2"
    recovery = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(recovery) == 1
    assert original_read_bytes(recovery[0]) == replacement_bytes


def test_refresh_retains_candidate_when_initial_exchange_changes_canonical_then_fails(
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
    foreign_bytes = b"foreign canonical receipt\n"

    def _replace_canonical_then_fail(_source: Path, destination: Path) -> None:
        replacement = destination.with_name(f".{MANIFEST_NAME}.competitor")
        replacement.write_bytes(foreign_bytes)
        replacement.replace(destination)
        raise OSError("injected exchange failure after canonical replacement")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _replace_canonical_then_fail)

    with pytest.raises(StorageObjectPublicationUncertain, match="initial refresh exchange.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert manifest_path.read_bytes() == foreign_bytes
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert json.loads(staging[0].read_bytes())["producer_revision"] == "test-revision-2"


def test_refresh_rejects_staging_replaced_before_initial_exchange(
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
    original_publish = storage_inventory._publish_refresh_manifest
    attacker_bytes: bytes | None = None

    def _publish_after_staging_replacement(temporary: Path, *args: object, **kwargs: object) -> object:
        nonlocal attacker_bytes
        attacker = json.loads(temporary.read_bytes())
        attacker["producer_revision"] = "attacker-revision"
        attacker_bytes = (json.dumps(attacker, indent=2, sort_keys=True) + "\n").encode()
        replacement = temporary.with_name(f"{temporary.name}.competitor")
        replacement.write_bytes(attacker_bytes)
        replacement.replace(temporary)
        return original_publish(temporary, *args, **kwargs)

    monkeypatch.setattr(storage_inventory, "_publish_refresh_manifest", _publish_after_staging_replacement)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry changed before refresh publication"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert attacker_bytes is not None
    assert manifest_path.read_bytes() == previous_bytes
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == attacker_bytes


def test_refresh_retains_staging_entry_replaced_before_success_cleanup(
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
    foreign_bytes = b"foreign refresh staging bytes\n"
    original_fsync = storage_inventory._fsync_directory
    injected = False

    def _replace_staging_after_publication_fsync(directory: Path) -> None:
        nonlocal injected
        original_fsync(directory)
        staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
        if staging and not injected:
            replacement = staging[0].with_name(f"{staging[0].name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(staging[0])
            injected = True

    monkeypatch.setattr(storage_inventory, "_fsync_directory", _replace_staging_after_publication_fsync)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry changed.*(restored|retained)"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == foreign_bytes


def test_refresh_retains_staging_entry_replaced_before_rollback_cleanup(
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
    foreign_bytes = b"foreign rollback staging bytes\n"
    original_exchange = storage_inventory._atomic_exchange
    original_fsync = storage_inventory._fsync_directory
    exchange_calls = 0
    injected = False

    def _interrupt_after_initial_exchange(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        original_exchange(source, destination)
        if exchange_calls == 1:
            raise OSError("injected exchange completion error")

    def _replace_staging_after_rollback_fsync(directory: Path) -> None:
        nonlocal injected
        original_fsync(directory)
        staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
        if exchange_calls == 2 and staging and not injected:
            replacement = staging[0].with_name(f"{staging[0].name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(staging[0])
            injected = True

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _interrupt_after_initial_exchange)
    monkeypatch.setattr(storage_inventory, "_fsync_directory", _replace_staging_after_rollback_fsync)

    with pytest.raises(StorageObjectPublicationUncertain, match="staging entry changed.*(restored|retained)"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert manifest_path.read_bytes() == previous_bytes
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == foreign_bytes


def test_refresh_retains_candidate_when_displaced_receipt_is_replaced_before_inspection(
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
    foreign_bytes = b"foreign displaced receipt\n"
    original_exchange = storage_inventory._atomic_exchange
    injected = False

    def _exchange_then_replace_displaced(source: Path, destination: Path) -> None:
        nonlocal injected
        original_exchange(source, destination)
        if not injected:
            replacement = source.with_name(f"{source.name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(source)
            injected = True

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _exchange_then_replace_displaced)

    with pytest.raises(StorageObjectPublicationUncertain, match="displaced receipt changed.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == foreign_bytes


def test_refresh_retains_both_receipts_when_swap_back_fails_before_mutation(
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

    def _fail_before_swap_back(source: Path, destination: Path) -> None:
        nonlocal exchange_calls
        exchange_calls += 1
        if exchange_calls == 1:
            original_exchange(source, destination)
            raise OSError("injected exchange completion error")
        raise OSError("injected swap-back failure")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _fail_before_swap_back)

    with pytest.raises(StorageObjectPublicationUncertain, match="rollback failed.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert exchange_calls == 2
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == previous_bytes


@pytest.mark.parametrize("raise_after_foreign_swap", [False, True])
def test_refresh_retains_candidate_when_displaced_receipt_is_replaced_before_swap_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raise_after_foreign_swap: bool,
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
    foreign_bytes = b"foreign displaced receipt\n"
    original_exchange = storage_inventory._atomic_exchange
    injected = False
    exchange_calls = 0

    def _replace_displaced_at_swap_back(source: Path, destination: Path) -> None:
        nonlocal exchange_calls, injected
        exchange_calls += 1
        if exchange_calls == 2:
            replacement = source.with_name(f"{source.name}.competitor")
            replacement.write_bytes(foreign_bytes)
            replacement.replace(source)
            injected = True
        original_exchange(source, destination)
        if exchange_calls == 1:
            raise OSError("injected exchange completion error")
        if exchange_calls == 2 and raise_after_foreign_swap:
            raise OSError("injected foreign swap completion error")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _replace_displaced_at_swap_back)

    with pytest.raises(StorageObjectPublicationUncertain):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert exchange_calls == 3
    assert json.loads(manifest_path.read_bytes())["producer_revision"] == "test-revision-2"
    staging = tuple(root.glob(f".{MANIFEST_NAME}.tmp-*"))
    assert len(staging) == 1
    assert staging[0].read_bytes() == foreign_bytes


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

    with pytest.raises(StorageObjectPublicationUncertain, match="displaced receipt changed.*retained"):
        refresh_storage_object(
            root,
            expected_manifest_digest=_digest(manifest_path),
            producer_revision="test-revision-2",
        )

    assert injected
    assert manifest_path.read_bytes() == previous_bytes
    recovery_paths = tuple(root.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == competitor_bytes


def test_refresh_rollback_retains_both_receipts_when_displaced_receipt_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    previous_bytes = b"previous receipt\n"
    published_bytes = b"published receipt\n"
    competitor_bytes = b"competing receipt\n"
    manifest_path.write_bytes(published_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    published_identity = (manifest_stat.st_dev, manifest_stat.st_ino)
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
        match="displaced receipt changed.*retained",
    ):
        storage_inventory._rollback_manifest(
            manifest_path,
            published_bytes=published_bytes,
            previous_bytes=previous_bytes,
            previous_mode=0o644,
            operation_error=StorageObjectError("injected validation failure"),
            published_identity=published_identity,
        )

    assert exchange_calls == 1
    assert manifest_path.read_bytes() == previous_bytes
    recovery_paths = tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == competitor_bytes


def test_refresh_rollback_rejects_same_inode_previous_receipt_rewrite_after_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    previous_bytes = b"previous receipt\n"
    foreign_bytes = b"foreign! receipt\n"
    manifest_path.write_bytes(previous_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    published_identity = (manifest_stat.st_dev, manifest_stat.st_ino)
    original_read_bytes = Path.read_bytes
    injected = False

    def _read_then_rewrite_same_inode(path: Path) -> bytes:
        nonlocal injected
        content = original_read_bytes(path)
        if path == manifest_path and not injected:
            before = path.stat(follow_symlinks=False)
            with path.open("r+b") as handle:
                handle.write(foreign_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000),
            )
            injected = True
        return content

    monkeypatch.setattr(Path, "read_bytes", _read_then_rewrite_same_inode)

    with pytest.raises(StorageObjectPublicationUncertain, match="rollback classification.*uncertain"):
        storage_inventory._rollback_manifest(
            manifest_path,
            published_bytes=b"published receipt\n",
            previous_bytes=previous_bytes,
            previous_mode=0o644,
            operation_error=StorageObjectError("injected validation failure"),
            published_identity=published_identity,
        )

    assert injected
    assert original_read_bytes(manifest_path) == foreign_bytes
    assert not tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))


def test_refresh_rollback_fails_typed_unsupported_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    published_bytes = b"published receipt\n"
    manifest_path.write_bytes(published_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    published_identity = (manifest_stat.st_dev, manifest_stat.st_ino)

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
            published_identity=published_identity,
        )

    assert manifest_path.read_bytes() == published_bytes
    assert not tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))


def test_refresh_rollback_reports_uncertain_when_recovery_cleanup_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / MANIFEST_NAME
    published_bytes = b"published receipt\n"
    previous_bytes = b"previous receipt\n"
    manifest_path.write_bytes(published_bytes)
    manifest_stat = manifest_path.stat(follow_symlinks=False)
    published_identity = (manifest_stat.st_dev, manifest_stat.st_ino)

    def _unsupported_exchange(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectPublicationUnsupported("injected unsupported exchange")

    def _unsupported_cleanup(*_args: object, **_kwargs: object) -> None:
        raise StorageObjectPublicationUnsupported("injected unsupported cleanup")

    monkeypatch.setattr(storage_inventory, "_atomic_exchange", _unsupported_exchange)
    monkeypatch.setattr(
        storage_inventory,
        "_atomic_move_no_replace_into_directory",
        _unsupported_cleanup,
    )

    with pytest.raises(
        StorageObjectPublicationUncertain,
        match="rollback failed.*cleanup is uncertain.*inspect",
    ):
        storage_inventory._rollback_manifest(
            manifest_path,
            published_bytes=published_bytes,
            previous_bytes=previous_bytes,
            previous_mode=0o644,
            operation_error=StorageObjectError("injected validation failure"),
            published_identity=published_identity,
        )

    assert manifest_path.read_bytes() == published_bytes
    recovery_paths = tuple(tmp_path.glob(f".{MANIFEST_NAME}.restore-*"))
    assert len(recovery_paths) == 1
    assert recovery_paths[0].read_bytes() == previous_bytes


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
