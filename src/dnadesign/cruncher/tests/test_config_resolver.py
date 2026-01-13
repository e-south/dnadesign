"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_config_resolver.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import pytest

from dnadesign.cruncher.cli.config_resolver import (
    CANDIDATE_CONFIG_FILENAMES,
    DEFAULT_WORKSPACE_ENV_VAR,
    INVOCATION_CWD_ENV_VAR,
    NONINTERACTIVE_ENV_VAR,
    WORKSPACE_ENV_VAR,
    WORKSPACE_ROOTS_ENV_VAR,
    ConfigResolutionError,
    parse_config_and_value,
    resolve_config_path,
)


@pytest.fixture(autouse=True)
def _clear_workspace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        INVOCATION_CWD_ENV_VAR,
        WORKSPACE_ENV_VAR,
        DEFAULT_WORKSPACE_ENV_VAR,
        WORKSPACE_ROOTS_ENV_VAR,
        NONINTERACTIVE_ENV_VAR,
        "INIT_CWD",
    ):
        monkeypatch.delenv(var, raising=False)


def test_resolve_config_from_cwd_single(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config_path = tmp_path / "cruncher.yaml"
    config_path.write_text("cruncher: {}\n")
    caplog.set_level(logging.INFO, logger="dnadesign.cruncher.cli.config_resolver")

    resolved = resolve_config_path(None, cwd=tmp_path)

    assert resolved == config_path.resolve()
    assert any("Using config from CWD: ./cruncher.yaml" in record.message for record in caplog.records)


def test_resolve_config_from_cwd_none(tmp_path: Path) -> None:
    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "No config argument provided" in message
    for name in CANDIDATE_CONFIG_FILENAMES:
        assert name in message


def test_resolve_config_from_cwd_multiple(tmp_path: Path) -> None:
    (tmp_path / "cruncher.yaml").write_text("cruncher: {}\n")
    (tmp_path / "config.yaml").write_text("cruncher: {}\n")
    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "Multiple config files found" in message
    assert "cruncher.yaml" in message
    assert "config.yaml" in message


def test_resolve_config_explicit_path(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    resolved = resolve_config_path(config_path, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_explicit_path_uses_init_cwd_when_cwd_diff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    workspace = repo_root / "workspaces" / "demo"
    workspace.mkdir(parents=True)
    config_path = workspace / "config.yaml"
    config_path.write_text("cruncher: {}\n")
    monkeypatch.setenv("INIT_CWD", str(workspace))
    monkeypatch.chdir(repo_root)

    resolved = resolve_config_path(Path("config.yaml"), log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_invalid_invocation_cwd_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv(INVOCATION_CWD_ENV_VAR, str(missing))
    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None)
    assert "CRUNCHER_CWD points to a missing directory" in str(excinfo.value)


def test_parse_config_and_value_single_config_path_errors(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    with pytest.raises(ConfigResolutionError) as excinfo:
        parse_config_and_value(
            [str(config_path)],
            None,
            value_label="RUN",
            command_hint="cruncher report <run_name>",
            cwd=tmp_path,
        )
    message = str(excinfo.value)
    assert "Missing RUN" in message
    assert "config.yaml" in message


def test_parse_config_and_value_single_value_uses_cwd_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    resolved_config, value = parse_config_and_value(
        ["sample_run_1"],
        None,
        value_label="RUN",
        command_hint="cruncher report <run_name>",
        cwd=tmp_path,
    )

    assert resolved_config == config_path.resolve()
    assert value == "sample_run_1"


def _make_workspace(root: Path, name: str) -> Path:
    workspace_dir = root / name
    workspace_dir.mkdir(parents=True)
    config_path = workspace_dir / "config.yaml"
    config_path.write_text("cruncher: {}\n")
    return config_path


def test_resolve_config_from_parent_dir(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    config_path = parent / "config.yaml"
    config_path.write_text("cruncher: {}\n")

    resolved = resolve_config_path(None, cwd=child, log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_single_workspace_auto_select(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    config_path = _make_workspace(root, "demo")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(NONINTERACTIVE_ENV_VAR, "1")

    resolved = resolve_config_path(None, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_multiple_workspaces_error_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    _make_workspace(root, "alpha")
    _make_workspace(root, "beta")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(NONINTERACTIVE_ENV_VAR, "1")

    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "Discovered 2 workspace configs" in message
    assert "[1]" in message
    assert "[2]" in message


def test_resolve_config_workspace_selector_by_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    config_path = _make_workspace(root, "demo")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, "demo")

    resolved = resolve_config_path(None, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_resolve_config_workspace_selector_by_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    _make_workspace(root, "alpha")
    config_path = _make_workspace(root, "beta")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(WORKSPACE_ENV_VAR, "2")

    resolved = resolve_config_path(None, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_default_workspace_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    config_path = _make_workspace(root, "demo")
    _make_workspace(root, "other")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(DEFAULT_WORKSPACE_ENV_VAR, "demo")
    monkeypatch.setenv(NONINTERACTIVE_ENV_VAR, "1")

    resolved = resolve_config_path(None, cwd=tmp_path, log=False)

    assert resolved == config_path.resolve()


def test_noninteractive_does_not_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "workspaces"
    root.mkdir()
    _make_workspace(root, "alpha")
    _make_workspace(root, "beta")
    monkeypatch.setenv(WORKSPACE_ROOTS_ENV_VAR, str(root))
    monkeypatch.setenv(NONINTERACTIVE_ENV_VAR, "1")

    with pytest.raises(ConfigResolutionError) as excinfo:
        resolve_config_path(None, cwd=tmp_path, log=False)
    message = str(excinfo.value)
    assert "Choose one" in message
