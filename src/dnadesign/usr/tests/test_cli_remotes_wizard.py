"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_remotes_wizard.py

Tests for USR remotes wizard and doctor commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

import dnadesign.usr.src.cli as cli_module


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

        def _ssh_run(self, remote_cmd: str, check: bool = True):
            if "command -v rsync" in remote_cmd:
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
