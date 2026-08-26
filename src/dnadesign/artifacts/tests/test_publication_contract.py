"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/tests/test_publication_contract.py

Public contracts for create-only directory publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import stat
from pathlib import Path

import pytest

from dnadesign.artifacts import (
    CreateOnlyDirectoryPublication,
    PublicationError,
    PublicationExistsError,
    preflight_create_only_directory_publication,
)


def test_existing_destination_raises_typed_create_only_conflict(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.mkdir(parents=True)

    with pytest.raises(PublicationExistsError) as raised:
        CreateOnlyDirectoryPublication.prepare(bundle)

    assert isinstance(raised.value, PublicationError)
    assert isinstance(raised.value, FileExistsError)


def test_destination_preflight_rejects_existing_target_without_mutation(tmp_path: Path) -> None:
    parent = tmp_path / "results"
    parent.mkdir()
    bundle = parent / "render-v1"
    bundle.write_text("keep\n", encoding="utf-8")
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    with pytest.raises(PublicationExistsError, match="already exists; publication is create-only"):
        preflight_create_only_directory_publication(bundle)

    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after
    assert bundle.read_text(encoding="utf-8") == "keep\n"


def test_destination_preflight_accepts_missing_parents_without_creating_them(tmp_path: Path) -> None:
    bundle = tmp_path / "uncreated" / "results" / "render-v1"

    final = preflight_create_only_directory_publication(bundle)

    assert final == bundle
    assert not bundle.parent.parent.exists()


def test_destination_race_raises_typed_create_only_conflict(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        bundle.mkdir()
        (bundle / "keep.txt").write_text("keep\n", encoding="utf-8")

        with pytest.raises(PublicationExistsError) as raised:
            publication.publish(required_manifest="manifest.json")

        assert isinstance(raised.value, PublicationError)
        assert isinstance(raised.value, FileExistsError)
        assert (bundle / "keep.txt").read_text(encoding="utf-8") == "keep\n"
    finally:
        publication.close()


def test_private_published_root_mode_is_applied(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle, published_root_mode=0o700)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()

    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700


def test_private_sensitivity_keeps_the_complete_published_tree_owner_only(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle, sensitivity="private")
    try:
        nested = publication.stage / "images"
        nested.mkdir(mode=0o755)
        (nested / "render.svg").write_text("sensitive\n", encoding="utf-8")
        (nested / "render.svg").chmod(0o644)
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()

    for path in (bundle, *bundle.rglob("*")):
        expected = 0o700 if path.is_dir() else 0o600
        assert stat.S_IMODE(path.stat().st_mode) == expected, path


def test_publication_copy_enforces_file_budget_before_verification(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        (publication.stage / "backend.log").write_bytes(b"x" * 32)

        with pytest.raises(PublicationError, match="32-byte copy limit"):
            publication.publish(
                required_manifest="manifest.json",
                copy_file_size_limit_bytes=32,
                copy_aggregate_size_limit_bytes=128,
            )
    finally:
        publication.close()

    assert not bundle.exists()


def test_publication_copy_enforces_entry_budget_before_verification(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        (publication.stage / "empty-artifact").touch()

        with pytest.raises(PublicationError, match="2-entry copy limit"):
            publication.publish(
                required_manifest="manifest.json",
                copy_entry_count_limit=2,
            )
    finally:
        publication.close()

    assert not bundle.exists()


def test_publication_copies_the_prepared_stage_descriptor_after_path_replacement(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    displaced_stage = publication.stage.with_name(f"{publication.stage.name}-displaced")
    try:
        (publication.stage / "manifest.json").write_text("original\n", encoding="utf-8")
        publication.stage.rename(displaced_stage)
        publication.stage.mkdir(mode=0o700)
        (publication.stage / "manifest.json").write_text("replacement\n", encoding="utf-8")

        publication.publish(required_manifest="manifest.json")
    finally:
        publication.close()
        shutil.rmtree(publication.stage, ignore_errors=True)

    assert (bundle / "manifest.json").read_text(encoding="utf-8") == "original\n"
    assert not displaced_stage.exists()


def test_close_removes_the_anchored_stage_after_owner_sentinel_corruption(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    stage = publication.stage
    (stage / ".dnadesign-publication-owner.json").write_text("corrupt\n", encoding="utf-8")

    publication.close()

    assert not stage.exists()


def test_published_path_identity_rejects_a_replacement_directory(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    displaced = tmp_path / "results" / "displaced-render-v1"
    publication = CreateOnlyDirectoryPublication.prepare(bundle)
    try:
        (publication.stage / "manifest.json").write_text("{}\n", encoding="utf-8")
        publication.publish(required_manifest="manifest.json")
        publication.assert_published_path_identity()
        bundle.rename(displaced)
        bundle.mkdir()

        with pytest.raises(PublicationError, match="path identity changed after publication"):
            publication.assert_published_path_identity()
    finally:
        publication.close()

    assert displaced.is_dir()
    assert bundle.is_dir()


@pytest.mark.parametrize("mode", [True, "700", -1, 0o1000, 0o600])
def test_invalid_published_root_mode_fails_before_filesystem_mutation(
    tmp_path: Path,
    mode: int,
) -> None:
    bundle = tmp_path / "uncreated" / "results" / "render-v1"

    with pytest.raises(PublicationError, match="published root mode"):
        CreateOnlyDirectoryPublication.prepare(bundle, published_root_mode=mode)

    assert not bundle.parent.parent.exists()
