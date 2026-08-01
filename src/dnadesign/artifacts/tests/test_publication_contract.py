"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/artifacts/tests/test_publication_contract.py

Public contracts for immutable directory publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from dnadesign.artifacts import (
    CreateOnlyDirectoryPublication,
    PublicationError,
    PublicationExistsError,
)


def test_existing_destination_raises_typed_create_only_conflict(tmp_path: Path) -> None:
    bundle = tmp_path / "results" / "render-v1"
    bundle.mkdir(parents=True)

    with pytest.raises(PublicationExistsError) as raised:
        CreateOnlyDirectoryPublication.prepare(bundle)

    assert isinstance(raised.value, PublicationError)
    assert isinstance(raised.value, FileExistsError)


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


@pytest.mark.parametrize("mode", [True, "700", -1, 0o1000, 0o600])
def test_invalid_published_root_mode_fails_before_filesystem_mutation(
    tmp_path: Path,
    mode: int,
) -> None:
    bundle = tmp_path / "uncreated" / "results" / "render-v1"

    with pytest.raises(PublicationError, match="published root mode"):
        CreateOnlyDirectoryPublication.prepare(bundle, published_root_mode=mode)

    assert not bundle.parent.parent.exists()
