"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/tests/storage_objects/test_cli.py

Tests deterministic storage inventory and validation commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from dnadesign.contracts.storage_objects import (
    MANIFEST_NAME,
    StorageObjectError,
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
    assert json.loads(completed.stdout)["status"] == "verified"
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
    checkout = tmp_path / "checkout"
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
    subprocess.run(
        ["git", "-C", str(checkout), "add", "examples/pilot/storage.object.json"],
        check=True,
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

    assert json.loads(completed.stdout)["status"] == "created-pending-git-add"
    assert (root / MANIFEST_NAME).is_file()
    assert before_add.returncode == 2
    assert "demo file is not tracked" in before_add.stderr
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
        "--json",
    ]

    stale = subprocess.run(
        [*refresh[:-2], "sha256:" + "0" * 64, "--json"],
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

    refresh_storage_object(root, expected_manifest_digest=expected_digest)

    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o640


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
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "storage object root is not a directory" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (root / LOCK_NAME).exists()


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


def test_inventory_does_not_delete_preexisting_manifest_temp(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    temporary = root / f".{MANIFEST_NAME}.tmp"
    temporary.write_text("pre-existing bytes\n", encoding="utf-8")

    with pytest.raises(StorageObjectError, match="cannot write storage object manifest"):
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
            )
        except StorageObjectError as exc:
            return str(exc)
        return "verified"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: _refresh(), range(2)))

    assert outcomes.count("verified") == 1
    assert sum("manifest changed before refresh" in outcome for outcome in outcomes) == 1
