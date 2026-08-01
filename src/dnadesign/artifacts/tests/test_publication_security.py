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

from dnadesign.artifacts import CreateOnlyDirectoryPublication, PublicationError
from dnadesign.artifacts import publication as publication_module


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
    real_check = publication_module._is_recoverable_adjacent_stage

    def _check_then_swap(path: Path, **kwargs) -> bool:
        recoverable = real_check(path, **kwargs)
        if recoverable and path.name == stale_name:
            path.rename(displaced_stale)
            replacement.rename(path)
        return recoverable

    monkeypatch.setattr(publication_module, "_pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(publication_module, "_is_recoverable_adjacent_stage", _check_then_swap)

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
    real_check = publication_module._is_recoverable_stale_stage

    def _check_then_swap(path: Path, **kwargs) -> bool:
        recoverable = real_check(path, **kwargs)
        if recoverable and path == stale:
            path.rename(displaced_stale)
            replacement.rename(path)
        return recoverable

    monkeypatch.setattr(publication_module, "_pid_is_alive", lambda _pid: False)
    monkeypatch.setattr(publication_module, "_is_recoverable_stale_stage", _check_then_swap)

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
