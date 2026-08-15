"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_root_resolution_contract.py

Contract tests for the lightweight shared USR root-resolution helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.usr as usr_roots


def test_default_usr_root_matches_packaged_usr_datasets_dir() -> None:
    expected = (usr_roots.pkg_usr_root() / "datasets").resolve()
    assert usr_roots.default_usr_root() == expected


def test_resolve_usr_root_from_env_normalizes_usr_package_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_USR_ROOT", str(usr_roots.pkg_usr_root()))
    assert usr_roots.resolve_usr_root_from_env() == usr_roots.default_usr_root()


def test_resolve_usr_root_from_config_normalizes_relative_usr_package_root(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "job.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job: {}\n", encoding="utf-8")
    usr_pkg_root = tmp_path / "shared_usr"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# test package root\n", encoding="utf-8")

    resolved = usr_roots.resolve_usr_root_from_config(
        "../shared_usr",
        config_path=config_path,
        label="job.input.root",
    )

    assert resolved == (usr_pkg_root / "datasets").resolve()


def test_resolve_usr_root_from_config_rejects_empty_string(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "job.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job: {}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="job.input.root must be a non-empty string"):
        usr_roots.resolve_usr_root_from_config(
            "   ",
            config_path=config_path,
            label="job.input.root",
        )


def test_explicit_operator_root_requires_an_absolute_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "operator-data"
    existing.mkdir()

    assert usr_roots.require_explicit_usr_root(existing) == existing.resolve()
    with pytest.raises(ValueError, match="must be absolute"):
        usr_roots.require_explicit_usr_root(Path("relative-data"))
    with pytest.raises(ValueError, match="existing directory"):
        usr_roots.require_explicit_usr_root(tmp_path / "missing")


def test_explicit_operator_root_rejects_symbolic_links(tmp_path: Path) -> None:
    existing = tmp_path / "operator-data"
    existing.mkdir()
    linked = tmp_path / "linked-data"
    linked.symlink_to(existing, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        usr_roots.require_explicit_usr_root(linked)
