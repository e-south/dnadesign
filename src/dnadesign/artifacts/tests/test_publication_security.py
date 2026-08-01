"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/tests/test_publication_security.py

Adversarial tests for immutable directory publication trust boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dnadesign.artifacts import (
    CreateOnlyDirectoryPublication,
    PublicationError,
)
from dnadesign.artifacts import owned_directory as owned_directory_module
from dnadesign.artifacts import publication as publication_module
from dnadesign.artifacts import recovery as recovery_module


@pytest.mark.parametrize(
    "nested_owner_name",
    [".dnadesign-publication-owner.json", ".DNADESIGN-PUBLICATION-OWNER.JSON"],
)
def test_publication_rejects_nested_owner_metadata_names_portably(
    tmp_path: Path,
    nested_owner_name: str,
) -> None:
    publication = CreateOnlyDirectoryPublication.prepare(tmp_path / "results" / "render-v1")
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        nested = publication.stage / "images"
        nested.mkdir()
        (nested / nested_owner_name).write_text("user artifact\n", encoding="utf-8")

        with pytest.raises(PublicationError, match="reserved.*owner|owner.*reserved"):
            publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()


def test_adjacent_stale_recovery_never_removes_a_swapped_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    bundle = parent / "render-v1"
    uid = os.getuid() if hasattr(os, "getuid") else None
    stale_pid = 91_337_551
    stale_name = f".{bundle.name}.staging-u{uid}-p{stale_pid}-{'0' * 32}"
    stale = parent / stale_name
    stale.mkdir()
    stale_owner = publication_module._owner_payload(bundle)
    stale_owner["pid"] = stale_pid
    publication_module._write_owner(stale / publication_module._OWNER_FILE, stale_owner)
    (stale / "stale.txt").write_text("stale\n", encoding="utf-8")

    replacement = parent / "unrelated"
    replacement.mkdir()
    (replacement / "keep.txt").write_text("keep\n", encoding="utf-8")
    displaced_stale = parent / "checked-stale"
    real_check = recovery_module._is_recoverable_directory

    def _check_then_swap(path: Path, **kwargs) -> bool:
        recoverable = real_check(path, **kwargs)
        if recoverable and path.name == stale_name:
            path.rename(displaced_stale)
            replacement.rename(path)
        return recoverable

    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    monkeypatch.setattr(recovery_module, "_is_recoverable_directory", _check_then_swap)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert (parent / stale_name / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert (displaced_stale / "stale.txt").read_text(encoding="utf-8") == "stale\n"


def test_rollback_stale_recovery_never_removes_a_swapped_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    bundle = parent / "render-v1"
    uid = os.getuid() if hasattr(os, "getuid") else None
    stale_pid = 91_337_553
    stale_name = f".{bundle.name}.rollback-u{uid}-p{stale_pid}-{'0' * 32}"
    stale = parent / stale_name
    stale.mkdir()
    stale_owner = publication_module._rollback_owner_payload(publication_module._owner_payload(bundle))
    stale_owner["pid"] = stale_pid
    publication_module._write_owner(stale / publication_module._OWNER_FILE, stale_owner)
    (stale / "stale.txt").write_text("stale\n", encoding="utf-8")

    replacement = parent / "unrelated"
    replacement.mkdir()
    (replacement / "keep.txt").write_text("keep\n", encoding="utf-8")
    displaced_stale = parent / "checked-rollback"
    real_check = recovery_module._is_recoverable_directory

    def _check_then_swap(path: Path, **kwargs) -> bool:
        recoverable = real_check(path, **kwargs)
        if recoverable and path.name == stale_name:
            path.rename(displaced_stale)
            replacement.rename(path)
        return recoverable

    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    monkeypatch.setattr(recovery_module, "_is_recoverable_directory", _check_then_swap)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert (parent / stale_name / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert (displaced_stale / "stale.txt").read_text(encoding="utf-8") == "stale\n"


def test_private_stale_recovery_never_removes_a_swapped_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    uid = os.getuid() if hasattr(os, "getuid") else None
    temp_root = tmp_path / "temp"
    temp_root.mkdir()
    monkeypatch.setattr(publication_module.tempfile, "gettempdir", lambda: temp_root.as_posix())
    private_parent = temp_root / f"dnadesign-artifact-publication-{uid}"
    private_parent.mkdir(mode=0o700)

    bundle = tmp_path / "results" / "render-v1"
    target_digest = publication_module._owner_payload(bundle)["target_sha256"]
    stale = private_parent / f"stage-{str(target_digest)[:16]}-stale"
    stale.mkdir(mode=0o700)
    stale_owner = publication_module._owner_payload(bundle)
    stale_owner["pid"] = 91_337_552
    publication_module._write_owner(stale / publication_module._OWNER_FILE, stale_owner)
    (stale / "stale.txt").write_text("stale\n", encoding="utf-8")

    replacement = private_parent / "unrelated"
    replacement.mkdir()
    (replacement / "keep.txt").write_text("keep\n", encoding="utf-8")
    displaced_stale = private_parent / "checked-stale"
    real_check = recovery_module._is_recoverable_directory

    def _check_then_swap(path: Path, **kwargs) -> bool:
        recoverable = real_check(path, **kwargs)
        if recoverable and path == stale:
            path.rename(displaced_stale)
            replacement.rename(path)
        return recoverable

    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    monkeypatch.setattr(recovery_module, "_is_recoverable_directory", _check_then_swap)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert (stale / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    assert (displaced_stale / "stale.txt").read_text(encoding="utf-8") == "stale\n"


def test_published_bundle_can_be_rolled_back_by_anchored_identity(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")

        assert bundle.is_dir()
        assert publication.rollback()
        assert not bundle.exists()
    finally:
        publication.close()


def test_publication_rollback_detaches_before_recursive_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    publication.publish(required_manifest="manifest.json")
    remove_directory = publication_module.remove_descriptor_anchored_directory

    def fail_hidden_cleanup(
        parent_descriptor: int,
        name: str,
        owned_descriptor: int,
        *,
        last_entry: str | None = None,
    ) -> bool:
        del parent_descriptor, owned_descriptor
        assert name.startswith(f".{bundle.name}.rollback-")
        assert last_entry == publication_module._OWNER_FILE
        assert not bundle.exists()
        detached = bundle.parent / name
        assert (detached / "manifest.json").read_text(encoding="utf-8") == "{}\n"
        raise OSError("injected recursive cleanup failure")

    monkeypatch.setattr(
        publication_module,
        "remove_descriptor_anchored_directory",
        fail_hidden_cleanup,
    )
    try:
        with pytest.raises(OSError, match="injected recursive cleanup failure"):
            publication.rollback()
        assert not bundle.exists()
        assert len(list(bundle.parent.glob(f".{bundle.name}.rollback-*"))) == 1
    finally:
        monkeypatch.setattr(
            publication_module,
            "remove_descriptor_anchored_directory",
            remove_directory,
        )
        publication.close()

    assert not list(bundle.parent.glob(f".{bundle.name}.rollback-*"))


def test_prepare_recovers_owned_rollback_after_process_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    publication.publish(required_manifest="manifest.json")
    remove_directory = publication_module.remove_descriptor_anchored_directory

    def fail_cleanup(
        _parent_descriptor: int,
        _name: str,
        _owned_descriptor: int,
        *,
        last_entry: str | None = None,
    ) -> bool:
        assert last_entry == publication_module._OWNER_FILE
        raise OSError("injected persistent cleanup failure")

    monkeypatch.setattr(
        publication_module,
        "remove_descriptor_anchored_directory",
        fail_cleanup,
    )
    with pytest.raises(OSError, match="persistent cleanup failure"):
        publication.rollback()
    with pytest.raises(OSError, match="persistent cleanup failure"):
        publication.close()

    detached = list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    assert len(detached) == 1
    assert (detached[0] / publication_module._OWNER_FILE).is_file()

    monkeypatch.setattr(
        publication_module,
        "remove_descriptor_anchored_directory",
        remove_directory,
    )
    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    recovered = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        assert not list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    finally:
        recovered.close()


def test_prepare_recovers_final_after_exit_between_owner_and_detach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    publication.publish(required_manifest="manifest.json")
    ensure_owner = publication_module._ensure_owner_on_descriptor

    def ensure_then_exit(*args, **kwargs) -> None:
        ensure_owner(*args, **kwargs)
        raise SystemExit("injected exit after rollback owner publication")

    monkeypatch.setattr(publication_module, "_ensure_owner_on_descriptor", ensure_then_exit)
    with pytest.raises(SystemExit, match="after rollback owner publication"):
        publication.rollback()
    with pytest.raises(SystemExit, match="after rollback owner publication"):
        publication.close()

    assert bundle.is_dir()
    assert (bundle / publication_module._OWNER_FILE).is_file()

    monkeypatch.setattr(publication_module, "_ensure_owner_on_descriptor", ensure_owner)
    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    recovered = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        assert not bundle.exists()
    finally:
        recovered.close()


def test_final_recovery_detaches_before_partial_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    (bundle / "data.txt").write_text("data\n", encoding="utf-8")
    stale_owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    stale_owner["pid"] = 91_337_554
    recovery_module._write_owner(bundle / recovery_module._OWNER_FILE, stale_owner)
    unlink = owned_directory_module.os.unlink
    payload_removed = False

    def unlink_then_fail(name, *args, **kwargs) -> None:
        nonlocal payload_removed
        unlink(name, *args, **kwargs)
        if name != recovery_module._OWNER_FILE and not payload_removed:
            payload_removed = True
            raise OSError("injected failure after partial recovery cleanup")

    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    monkeypatch.setattr(owned_directory_module.os, "unlink", unlink_then_fail)
    with pytest.raises(OSError, match="after partial recovery cleanup"):
        CreateOnlyDirectoryPublication.prepare(bundle)

    assert not bundle.exists()
    detached = list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    assert len(detached) == 1
    assert (detached[0] / recovery_module._OWNER_FILE).is_file()

    monkeypatch.setattr(owned_directory_module.os, "unlink", unlink)
    recovered = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        assert not list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    finally:
        recovered.close()


def test_prepare_preserves_unauthenticated_empty_rollback_after_exit_before_rmdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    publication.publish(required_manifest="manifest.json")
    remove_directory = owned_directory_module.os.rmdir

    def exit_before_rollback_rmdir(name, *args, **kwargs) -> None:
        if isinstance(name, str) and name.startswith(f".{bundle.name}.rollback-"):
            raise SystemExit("injected exit before rollback rmdir")
        remove_directory(name, *args, **kwargs)

    monkeypatch.setattr(owned_directory_module.os, "rmdir", exit_before_rollback_rmdir)
    with pytest.raises(SystemExit, match="before rollback rmdir"):
        publication.rollback()
    with pytest.raises(SystemExit, match="before rollback rmdir"):
        publication.close()

    detached = list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    assert len(detached) == 1
    assert not list(detached[0].iterdir())

    monkeypatch.setattr(owned_directory_module.os, "rmdir", remove_directory)
    monkeypatch.setattr(recovery_module, "_owner_process_is_active", lambda _pid, _token: False)
    recovered = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        assert detached[0].is_dir()
        assert not list(detached[0].iterdir())
    finally:
        recovered.close()


def test_prepare_preserves_owner_bundle_with_unrepresentable_pid(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.parent.mkdir()
    candidate = bundle.parent / f".{bundle.name}.rollback-untrusted"
    candidate.mkdir()
    owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    owner["pid"] = int("9" * 100)
    recovery_module._write_owner(candidate / recovery_module._OWNER_FILE, owner)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert candidate.is_dir()


def test_prepare_recovers_owner_bundle_after_pid_reuse(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.parent.mkdir()
    candidate = bundle.parent / f".{bundle.name}.rollback-reused-pid"
    candidate.mkdir()
    owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    owner["process_start_token"] = recovery_module._format_process_start_token(
        float(str(owner["process_start_token"])) + 1.0
    )
    recovery_module._write_owner(candidate / recovery_module._OWNER_FILE, owner)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert not candidate.exists()


def test_prepare_preserves_owner_bundle_while_original_process_is_active(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.parent.mkdir()
    candidate = bundle.parent / f".{bundle.name}.rollback-active-owner"
    candidate.mkdir()
    owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    recovery_module._write_owner(candidate / recovery_module._OWNER_FILE, owner)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert candidate.is_dir()


@pytest.mark.parametrize("invalid_token", ["garbage", "nan", "0", "", None])
def test_prepare_preserves_owner_bundle_with_malformed_process_identity(
    tmp_path: Path,
    invalid_token: object,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.parent.mkdir()
    candidate = bundle.parent / f".{bundle.name}.rollback-malformed-owner"
    candidate.mkdir()
    owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    owner["process_start_token"] = invalid_token
    recovery_module._write_owner(candidate / recovery_module._OWNER_FILE, owner)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert candidate.is_dir()


@pytest.mark.parametrize("invalid_pid", [True, "123", 0, -1, 2**31])
def test_prepare_preserves_owner_bundle_with_malformed_pid(
    tmp_path: Path,
    invalid_pid: object,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.parent.mkdir()
    candidate = bundle.parent / f".{bundle.name}.rollback-malformed-pid"
    candidate.mkdir()
    owner = recovery_module._rollback_owner_payload(recovery_module._owner_payload(bundle))
    owner["pid"] = invalid_pid
    recovery_module._write_owner(candidate / recovery_module._OWNER_FILE, owner)

    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    publication.close()

    assert candidate.is_dir()


def test_publication_rollback_preserves_a_swapped_replacement(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")
        displaced = bundle.parent / "displaced"
        bundle.rename(displaced)
        bundle.mkdir()
        (bundle / "keep.txt").write_text("keep\n", encoding="utf-8")

        assert not publication.rollback()
        assert (bundle / "keep.txt").read_text(encoding="utf-8") == "keep\n"
        assert (displaced / "manifest.json").read_text(encoding="utf-8") == "{}\n"
    finally:
        publication.close()


def test_publication_rollback_restores_replacement_swapped_during_detach(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")
        displaced = bundle.parent / "displaced"
        rename_create_only = publication_module._rename_create_only
        swapped = False

        def swap_then_rename(parent_descriptor: int, source: str, destination: str) -> None:
            nonlocal swapped
            if source == bundle.name and not swapped:
                swapped = True
                os.rename(
                    source,
                    displaced.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                os.mkdir(source, dir_fd=parent_descriptor)
                (bundle / "keep.txt").write_text("keep\n", encoding="utf-8")
            rename_create_only(parent_descriptor, source, destination)

        monkeypatch.setattr(publication_module, "_rename_create_only", swap_then_rename)

        with pytest.raises(PublicationError, match="identity changed"):
            publication.rollback()

        assert (bundle / "keep.txt").read_text(encoding="utf-8") == "keep\n"
        assert (displaced / "manifest.json").read_text(encoding="utf-8") == "{}\n"
        assert not (displaced / publication_module._OWNER_FILE).exists()
        assert not list(bundle.parent.glob(f".{bundle.name}.rollback-*"))
    finally:
        publication.close()


def test_post_rename_termination_after_owner_removal_cleans_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
    real_unlink = publication_module.os.unlink
    terminated = False

    def unlink_then_terminate(path, *args, **kwargs) -> None:
        nonlocal terminated
        real_unlink(path, *args, **kwargs)
        if path == publication_module._OWNER_FILE and not terminated:
            terminated = True
            raise SystemExit("injected termination after owner removal")

    monkeypatch.setattr(publication_module.os, "unlink", unlink_then_terminate)
    try:
        with pytest.raises(SystemExit, match="after owner removal"):
            publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()

    assert not bundle.exists()
    assert not list(bundle.parent.glob(".render-v1.staging-*"))
