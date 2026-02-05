"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_remotes_config.py

Tests for remotes config discovery behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest

from dnadesign.usr.src import config


def _write_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("remotes: {}\n", encoding="utf-8")


def test_locate_config_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "remotes.yaml"
    _write_empty(cfg)
    monkeypatch.setenv("USR_REMOTES_PATH", str(cfg))

    assert config.locate_config() == cfg


def test_locate_config_default_repo_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("USR_REMOTES_PATH", raising=False)
    with pytest.raises(config.RemoteConfigError, match="USR_REMOTES_PATH"):
        config.locate_config()


def test_load_all_missing_config_is_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = tmp_path / "missing.yaml"
    monkeypatch.setenv("USR_REMOTES_PATH", str(cfg))
    with pytest.raises(config.RemoteConfigError, match="not found"):
        config.load_all()
