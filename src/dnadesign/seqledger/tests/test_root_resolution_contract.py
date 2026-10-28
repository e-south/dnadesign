"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/seqledger/tests/test_root_resolution_contract.py

Contract tests for the lightweight shared SeqLedger root-resolution helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import dnadesign.seqledger as usr_roots


def test_explicit_operator_root_contract_is_owned_by_usr_contracts() -> None:
    assert usr_roots.require_explicit_seqledger_root.__module__ == "dnadesign.seqledger.src.contracts.roots"


def test_default_usr_root_is_project_local_and_outside_the_package(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "seqledger_datasets"
    assert usr_roots.default_seqledger_root() == expected
    assert not expected.exists()


def test_resolve_usr_root_from_env_normalizes_usr_package_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DNADESIGN_SEQLEDGER_ROOT", str(usr_roots.pkg_seqledger_root()))
    assert usr_roots.resolve_seqledger_root_from_env() == usr_roots.pkg_seqledger_root() / "datasets"


def test_resolve_usr_root_from_config_normalizes_relative_usr_package_root(tmp_path: Path) -> None:
    config_path = tmp_path / "configs" / "job.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("job: {}\n", encoding="utf-8")
    usr_pkg_root = tmp_path / "shared_usr"
    usr_pkg_root.mkdir(parents=True, exist_ok=True)
    (usr_pkg_root / "__init__.py").write_text("# test package root\n", encoding="utf-8")

    resolved = usr_roots.resolve_seqledger_root_from_config(
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
        usr_roots.resolve_seqledger_root_from_config(
            "   ",
            config_path=config_path,
            label="job.input.root",
        )


def test_explicit_operator_root_requires_an_absolute_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "operator-data"
    existing.mkdir()

    assert usr_roots.require_explicit_seqledger_root(existing) == existing.resolve()
    with pytest.raises(ValueError, match="must be absolute"):
        usr_roots.require_explicit_seqledger_root(Path("relative-data"))
    with pytest.raises(ValueError, match="existing directory"):
        usr_roots.require_explicit_seqledger_root(tmp_path / "missing")


def test_explicit_operator_root_rejects_symbolic_links(tmp_path: Path) -> None:
    existing = tmp_path / "operator-data"
    existing.mkdir()
    linked = tmp_path / "linked-data"
    linked.symlink_to(existing, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        usr_roots.require_explicit_seqledger_root(linked)


def test_seqledger_preserves_stored_schema_and_lock_protocol():
    schema = importlib.import_module("dnadesign.seqledger.src.contracts.schema")
    locks = importlib.import_module("dnadesign.seqledger.src.storage.locking")
    assert schema.META_SCHEMA_VERSION == "usr:schema_version"
    assert locks.LOCK_FILENAME == ".usr.lock"
