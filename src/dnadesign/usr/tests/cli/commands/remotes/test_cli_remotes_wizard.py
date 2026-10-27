"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/cli/commands/remotes/test_cli_remotes_wizard.py

Tests for USR remotes wizard and doctor commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

import dnadesign.usr.src.cli as cli_module
from dnadesign.usr.src.sync.remote.remote import SSHControlSessionStatus


def _write_remotes(path: Path, text: str = "remotes: {}\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_remotes_wizard_bu_scc_writes_remote_and_prints_ssh_snippet(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(remotes_path)
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "remotes",
            "wizard",
            "--preset",
            "bu-scc",
            "--name",
            "bu-scc",
            "--user",
            "alice",
            "--base-dir",
            "/project/alice/usr_datasets",
            "--host",
            "scc1.bu.edu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Saved remote 'bu-scc'" in result.output
    assert "Host bu-scc" in result.output
    assert "HostName scc1.bu.edu" in result.output

    payload = yaml.safe_load(remotes_path.read_text(encoding="utf-8"))
    assert payload["remotes"]["bu-scc"]["host"] == "scc1.bu.edu"
    assert payload["remotes"]["bu-scc"]["user"] == "alice"
    assert payload["remotes"]["bu-scc"]["base_dir"] == "/project/alice/usr_datasets"
    assert payload["remotes"]["bu-scc"]["batch_mode"] is True


def test_remotes_wizard_requires_copy_first_config(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    result = CliRunner().invoke(
        cli_module.app,
        [
            "remotes",
            "wizard",
            "--preset",
            "bu-scc",
            "--user",
            "alice",
            "--base-dir",
            "/project/alice/usr_datasets",
        ],
    )

    assert result.exit_code != 0
    assert result.exception is not None
    assert "Remote config not found" in str(result.exception)
    assert "Copy remotes.example.yaml" in str(result.exception)
    assert not remotes_path.exists()


def test_remotes_add_can_disable_batch_mode(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(remotes_path)
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    runner = CliRunner()
    result = runner.invoke(
        cli_module.app,
        [
            "remotes",
            "add",
            "bu-scc",
            "--host",
            "scc1.bu.edu",
            "--user",
            "alice",
            "--base-dir",
            "/project/alice/usr_datasets",
            "--no-batch-mode",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = yaml.safe_load(remotes_path.read_text(encoding="utf-8"))
    assert payload["remotes"]["bu-scc"]["batch_mode"] is False


def test_remotes_doctor_reports_success(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  bu-scc:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
            "    batch_mode: false\n"
        ),
    )
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    def _fake_which(name: str):
        if name in {"ssh", "rsync"}:
            return f"/usr/bin/{name}"
        return None

    commands: list[str] = []

    class _FakeRemote:
        def __init__(self, _cfg):
            pass

        def _ssh_run(self, remote_cmd: str, check: bool = True):
            commands.append(remote_cmd)
            if "command -v rsync" in remote_cmd or "command -v flock" in remote_cmd:
                return 0, "", ""
            if "test -d" in remote_cmd:
                return 0, "", ""
            return 0, "ok\n", ""

    monkeypatch.setattr(cli_module.shutil, "which", _fake_which)
    monkeypatch.setattr(cli_module, "SSHRemote", _FakeRemote)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["remotes", "doctor", "--remote", "bu-scc"])
    assert result.exit_code == 0, result.output
    assert "doctor checks passed" in result.output.lower()
    assert "Remote process identity: ok" in result.output
    assert 'test -n "$(LC_ALL=C TZ=UTC0 ps -o lstart= -p "$$" 2>/dev/null | tr -d \'[:space:]\')"' in commands


def test_remotes_doctor_rejects_missing_process_start_identity(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  bu-scc:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
        ),
    )
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    monkeypatch.setattr(
        cli_module.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"ssh", "rsync"} else None,
    )

    class _FakeRemote:
        def __init__(self, _cfg):
            pass

        def _ssh_run(self, remote_cmd: str, check: bool = True):
            del check
            if "ps -o lstart=" in remote_cmd:
                return 1, "", ""
            return 0, "", ""

    monkeypatch.setattr(cli_module, "SSHRemote", _FakeRemote)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["remotes", "doctor", "--remote", "bu-scc"])

    assert result.exit_code != 0
    assert "Remote process-start identity is unavailable" in result.output
    assert "LC_ALL=C TZ=UTC0 ps -o lstart= -p $$" in result.output
    assert "Doctor checks passed" not in result.output


def test_remotes_doctor_guides_keyboard_interactive_auth(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  bu-scc:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
            "    batch_mode: true\n"
        ),
    )
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    def _fake_which(name: str):
        if name in {"ssh", "rsync"}:
            return f"/usr/bin/{name}"
        return None

    class _FakeRemote:
        def __init__(self, _cfg):
            pass

        def _ssh_run(self, _remote_cmd: str, check: bool = True):
            del check
            return 255, "", "Permission denied (keyboard-interactive,hostbased)"

    monkeypatch.setattr(cli_module.shutil, "which", _fake_which)
    monkeypatch.setattr(cli_module, "SSHRemote", _FakeRemote)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["remotes", "doctor", "--remote", "bu-scc"])
    assert result.exit_code != 0
    assert "keyboard-interactive" in result.output
    assert "--no-batch-mode" in result.output
    assert "ssh scc1" in result.output


def test_global_remotes_config_option_wires_commands_without_env(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  cluster:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
            "    batch_mode: false\n"
        ),
    )
    monkeypatch.delenv("USR_REMOTES_PATH", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["--remotes-config", str(remotes_path), "remotes", "list"])
    assert result.exit_code == 0, result.output
    assert "cluster" in result.output
    assert "interactive-auth" in result.output


def test_remotes_status_emits_json(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  cluster:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
            "    batch_mode: false\n"
        ),
    )
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    class _FakeRemote:
        def __init__(self, _cfg):
            pass

        def control_session_status(self):
            return SSHControlSessionStatus(
                host="scc1.bu.edu",
                user="alice",
                ssh_target="alice@scc1.bu.edu",
                batch_mode=False,
                control_master="auto",
                control_path="/tmp/cm.sock",
                control_persist="600",
                multiplex_enabled=True,
                socket_exists=True,
                socket_live=True,
            )

    monkeypatch.setattr(cli_module, "SSHRemote", _FakeRemote)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["remotes", "status", "--remote", "cluster", "--json"])
    assert result.exit_code == 0, result.output
    assert '"remote":"cluster"' in result.output
    assert '"socket_live":true' in result.output
    assert '"recommendation":"ready for sync"' in result.output


def test_remotes_warm_auth_reports_started_state(tmp_path: Path, monkeypatch) -> None:
    remotes_path = tmp_path / "config" / "usr-remotes.yaml"
    _write_remotes(
        remotes_path,
        text=(
            "remotes:\n"
            "  cluster:\n"
            "    type: ssh\n"
            "    host: scc1.bu.edu\n"
            "    user: alice\n"
            "    base_dir: /project/alice/usr_datasets\n"
            "    batch_mode: false\n"
        ),
    )
    monkeypatch.setenv("USR_REMOTES_PATH", str(remotes_path))

    class _FakeRemote:
        def __init__(self, _cfg):
            pass

        def control_session_status(self):
            return SSHControlSessionStatus(
                host="scc1.bu.edu",
                user="alice",
                ssh_target="alice@scc1.bu.edu",
                batch_mode=False,
                control_master="auto",
                control_path="/tmp/cm.sock",
                control_persist="600",
                multiplex_enabled=True,
                socket_exists=False,
                socket_live=False,
            )

        def warm_auth_session(self):
            return SSHControlSessionStatus(
                host="scc1.bu.edu",
                user="alice",
                ssh_target="alice@scc1.bu.edu",
                batch_mode=False,
                control_master="auto",
                control_path="/tmp/cm.sock",
                control_persist="600",
                multiplex_enabled=True,
                socket_exists=True,
                socket_live=True,
            )

    monkeypatch.setattr(cli_module, "SSHRemote", _FakeRemote)

    runner = CliRunner()
    result = runner.invoke(cli_module.app, ["remotes", "warm-auth", "--remote", "cluster", "--json"])
    assert result.exit_code == 0, result.output
    assert '"bootstrap_state":"started"' in result.output
    assert '"socket_live":true' in result.output
