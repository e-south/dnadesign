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

from dnadesign import usr_roots


def test_default_usr_root_matches_packaged_usr_datasets_dir() -> None:
    expected = (Path(usr_roots.__file__).resolve().parent / "usr" / "datasets").resolve()
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
