from __future__ import annotations

import os
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path

from dnadesign.densegen.src.cli.main import DEFAULT_CONFIG_FILENAME, _workspace_command


@contextmanager
def _chdir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def test_workspace_command_uses_cd_when_config_present(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    with _chdir(tmp_path):
        cmd = _workspace_command("dense run", run_root=workspace)
    assert cmd == "cd ws && uv run dense run"


def test_workspace_command_omits_cd_when_already_in_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    with _chdir(workspace):
        cmd = _workspace_command("dense run", run_root=workspace)
        assert cmd == "uv run dense run"


def test_workspace_command_falls_back_to_config_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("x")
    other = tmp_path / "other"
    other.mkdir()

    with _chdir(tmp_path):
        cmd = _workspace_command("dense run", cfg_path=cfg_path, run_root=other)
        assert cmd == "uv run dense run -c config.yaml"


def test_workspace_command_external_workspace_hint_is_executable(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="densegen_ws_cmd_", dir="/tmp") as workspace_dir:
        workspace = Path(workspace_dir)
        (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

        with _chdir(tmp_path):
            cmd = _workspace_command("dense run", run_root=workspace)

        assert cmd.startswith("cd ")
        assert cmd.endswith(" && uv run dense run")
        cd_target = cmd.split(" && ", 1)[0].removeprefix("cd ").strip()
        resolved = (tmp_path / cd_target).resolve()
        assert resolved == workspace.resolve()


def test_workspace_command_external_workspace_hint_uses_absolute_path(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="densegen_ws_cmd_", dir="/tmp") as workspace_dir:
        workspace = Path(workspace_dir)
        (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

        with _chdir(tmp_path):
            cmd = _workspace_command("dense run", run_root=workspace)

        expected = f"cd {shlex.quote(str(workspace))} && uv run dense run"
        assert cmd == expected


def test_workspace_command_external_config_hint_uses_absolute_path(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="densegen_ws_cfg_", dir="/tmp") as workspace_dir:
        cfg_path = Path(workspace_dir) / "config.yaml"
        cfg_path.write_text("x")

        with _chdir(tmp_path):
            cmd = _workspace_command("dense run", cfg_path=cfg_path)

        expected = f"uv run dense run -c {shlex.quote(str(cfg_path))}"
        assert cmd == expected


def test_workspace_command_external_workspace_with_config_prefers_config_flag(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="densegen_ws_cfg_", dir="/tmp") as workspace_dir:
        workspace = Path(workspace_dir)
        cfg_path = workspace / "config.yaml"
        cfg_path.write_text("x")

        with _chdir(tmp_path):
            cmd = _workspace_command("dense run", cfg_path=cfg_path, run_root=workspace)

        expected = f"uv run dense run -c {shlex.quote(str(cfg_path))}"
        assert cmd == expected


def test_workspace_command_quotes_workspace_path_with_spaces(tmp_path: Path) -> None:
    workspace = tmp_path / "ws with space"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    with _chdir(tmp_path):
        cmd = _workspace_command("dense run", run_root=workspace)

    assert cmd == "cd 'ws with space' && uv run dense run"


def test_workspace_command_quotes_config_path_with_spaces(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg dir" / "config with space.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text("x")
    run_root = tmp_path / "other"
    run_root.mkdir()

    with _chdir(tmp_path):
        cmd = _workspace_command("dense run", cfg_path=cfg_path, run_root=run_root)

    assert cmd == "uv run dense run -c 'cfg dir/config with space.yaml'"


def test_workspace_command_uses_config_flag_when_run_root_is_omitted(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("x")

    with _chdir(tmp_path):
        cmd = _workspace_command("dense run", cfg_path=cfg_path)

    assert cmd == "uv run dense run -c config.yaml"


def test_workspace_command_uses_uv_launcher_by_default(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")

    with _chdir(workspace):
        cmd = _workspace_command("dense run", run_root=workspace)

    assert cmd == "uv run dense run"


def test_workspace_command_uses_pixi_launcher_when_running_under_pixi(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / DEFAULT_CONFIG_FILENAME).write_text("x")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", "/tmp/pixi.toml")

    with _chdir(workspace):
        cmd = _workspace_command("dense run", run_root=workspace)

    assert cmd == "pixi run dense run"


def test_workspace_command_uses_config_flag_for_pixi_workspace_hints(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg_path = workspace / DEFAULT_CONFIG_FILENAME
    cfg_path.write_text("x")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", "/tmp/pixi.toml")

    with _chdir(tmp_path):
        cmd = _workspace_command("dense notebook run", cfg_path=cfg_path, run_root=workspace)

    assert cmd == "pixi run dense notebook run -c ws/config.yaml"
