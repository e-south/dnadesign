"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_banner_render_preflight.py

Tests banner rendering preconditions and fail-before-mutation behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from dnadesign.devtools.docs.banners import render as banner_render
from dnadesign.devtools.docs.banners.catalog import BANNERS, REPOSITORY_BANNER_PATH
from dnadesign.devtools.docs.banners.render import render_banners


def _write_repo_markers(repo_root: Path) -> None:
    (repo_root / "src" / "dnadesign").mkdir(parents=True)
    (repo_root / "src" / "dnadesign" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text('[project]\nname = "dnadesign"\n', encoding="utf-8")
    (repo_root / "README.md").write_text("# dnadesign\n", encoding="utf-8")
    for spec in BANNERS:
        readme_path = repo_root / spec.readme_path
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(f"# {spec.name}\n", encoding="utf-8")


def test_render_rejects_wrong_repo_root_without_mutation(tmp_path: Path) -> None:
    wrong_root = tmp_path / "not-dnadesign"

    with pytest.raises(ValueError, match="dnadesign repository root"):
        render_banners(wrong_root)

    assert not wrong_root.exists()


def test_render_rejects_existing_wrong_project_without_mutation(tmp_path: Path) -> None:
    wrong_root = tmp_path / "not-dnadesign"
    (wrong_root / "src" / "dnadesign").mkdir(parents=True)
    (wrong_root / "src" / "dnadesign" / "__init__.py").write_text("", encoding="utf-8")
    (wrong_root / "pyproject.toml").write_text('[project]\nname = "another-project"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="dnadesign repository root"):
        render_banners(wrong_root)

    assert not (wrong_root / REPOSITORY_BANNER_PATH).exists()


def test_render_preflights_every_output_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    repository_banner = repo_root / REPOSITORY_BANNER_PATH
    repository_banner.parent.mkdir(parents=True)
    repository_banner.write_text("preserve me", encoding="utf-8")
    outside = tmp_path / "escaped.svg"
    malicious = banner_render.BannerSpec(
        path="../escaped.svg",
        readme_path="src/dnadesign/escaped/README.md",
        name="escaped",
        capability="ESCAPE",
        description="Must not be written.",
        glyph="align",
    )
    monkeypatch.setattr(banner_render, "BANNERS", (*BANNERS, malicious))

    with pytest.raises(ValueError, match="canonical repository-relative POSIX path"):
        render_banners(repo_root)

    assert not outside.exists()
    assert repository_banner.read_text(encoding="utf-8") == "preserve me"


def test_render_rejects_noncanonical_output_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    repository_banner = repo_root / REPOSITORY_BANNER_PATH
    repository_banner.parent.mkdir(parents=True)
    repository_banner.write_text("preserve me", encoding="utf-8")
    spec = BANNERS[0]
    noncanonical = replace(
        spec,
        path=f"{Path(spec.path).parent.as_posix()}/missing/../{Path(spec.path).name}",
    )
    monkeypatch.setattr(banner_render, "BANNERS", (noncanonical, *BANNERS[1:]))

    with pytest.raises(ValueError, match="canonical repository-relative POSIX path"):
        render_banners(repo_root)

    assert repository_banner.read_text(encoding="utf-8") == "preserve me"


def test_render_rejects_non_banner_output_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    original_pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    spec = BANNERS[0]
    unrelated = replace(spec, path="pyproject.toml")
    monkeypatch.setattr(banner_render, "BANNERS", (unrelated, *BANNERS[1:]))

    with pytest.raises(ValueError, match="must end with"):
        render_banners(repo_root)

    assert (repo_root / "pyproject.toml").read_text(encoding="utf-8") == original_pyproject
    assert not (repo_root / REPOSITORY_BANNER_PATH).exists()


def test_render_rejects_output_outside_owning_readme_before_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    spec = BANNERS[0]
    wrong_owner = replace(
        spec,
        path=f"src/dnadesign/notify/assets/{spec.name}-banner.svg",
    )
    monkeypatch.setattr(banner_render, "BANNERS", (wrong_owner, *BANNERS[1:]))

    with pytest.raises(ValueError, match="owning README directory"):
        render_banners(repo_root)

    assert not (repo_root / REPOSITORY_BANNER_PATH).exists()


def test_render_rejects_non_directory_output_parent_without_partial_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    repository_banner = repo_root / REPOSITORY_BANNER_PATH
    repository_banner.parent.mkdir(parents=True)
    repository_banner.write_text("preserve me", encoding="utf-8")
    blocking_parent = repo_root / "src" / "dnadesign" / "aligner" / "assets"
    blocking_parent.parent.mkdir(parents=True, exist_ok=True)
    blocking_parent.write_text("blocking file", encoding="utf-8")

    with pytest.raises(ValueError, match="parent component is not a directory"):
        render_banners(repo_root)

    assert repository_banner.read_text(encoding="utf-8") == "preserve me"
    assert blocking_parent.read_text(encoding="utf-8") == "blocking file"


def test_render_rejects_non_file_output_without_partial_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    repository_banner = repo_root / REPOSITORY_BANNER_PATH
    repository_banner.parent.mkdir(parents=True)
    repository_banner.write_text("preserve me", encoding="utf-8")
    blocking_output = repo_root / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg"
    blocking_output.mkdir(parents=True)

    with pytest.raises(ValueError, match="output is not a regular file"):
        render_banners(repo_root)

    assert repository_banner.read_text(encoding="utf-8") == "preserve me"
    assert blocking_output.is_dir()


def test_render_rejects_symlinked_output_parent_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo_root / "assets").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert list(outside.iterdir()) == []


def test_render_rejects_in_repo_symlinked_output_parent_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    alternate_assets = repo_root / "alternate-assets"
    alternate_assets.mkdir()
    (repo_root / "assets").symlink_to(alternate_assets, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert list(alternate_assets.iterdir()) == []


def test_render_rejects_symlinked_output_file_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    unrelated = repo_root / "unrelated.svg"
    unrelated.write_text("preserve me", encoding="utf-8")
    output = repo_root / REPOSITORY_BANNER_PATH
    output.parent.mkdir(parents=True)
    output.symlink_to(unrelated)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert output.is_symlink()
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_render_rejects_hard_linked_output_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    pyproject_path = repo_root / "pyproject.toml"
    original_pyproject = pyproject_path.read_text(encoding="utf-8")
    output = repo_root / REPOSITORY_BANNER_PATH
    output.parent.mkdir(parents=True)
    os.link(pyproject_path, output)

    with pytest.raises(ValueError, match="multiple hard links"):
        render_banners(repo_root)

    assert pyproject_path.read_text(encoding="utf-8") == original_pyproject
    assert output.read_text(encoding="utf-8") == original_pyproject
