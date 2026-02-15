"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_profiles.py

Profile workflow tests for secure, fail-fast Notify CLI setup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.notify.cli import _ensure_private_directory, app
from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.secrets import resolve_secret_ref


def _event(action: str = "write_overlay_part") -> dict:
    return {
        "event_version": 1,
        "timestamp_utc": "2026-02-06T00:00:00+00:00",
        "action": action,
        "dataset": {"name": "demo", "root": "/tmp/datasets"},
        "args": {"namespace": "densegen"},
        "metrics": {"rows_written": 3},
        "artifacts": {"overlay": {"namespace": "densegen"}},
        "fingerprint": {"rows": 1, "cols": 2, "size_bytes": 128, "sha256": None},
        "registry_hash": "abc123",
        "actor": {"tool": "densegen", "run_id": "run-1", "host": "host", "pid": 123},
        "version": "0.1.0",
    }


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _set_ssl_cert_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle))
    return ca_bundle


def test_profile_init_writes_densegen_policy_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--policy",
            "densegen",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["provider"] == "slack"
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert data["policy"] == "densegen"
    assert data["only_tools"] == "densegen"
    assert "densegen_health" in data["only_actions"]
    assert "url_env" not in data


def test_profile_init_persists_progress_tunables(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--progress-step-pct",
            "20",
            "--progress-min-seconds",
            "90",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["progress_step_pct"] == 20
    assert data["progress_min_seconds"] == 90.0


def test_profile_init_writes_infer_evo2_policy_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event(action="attach")])

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--policy",
            "infer_evo2",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert data["policy"] == "infer_evo2"
    assert data["only_tools"] == "infer"
    assert data["only_actions"] == "attach,materialize"


def test_profile_init_resolves_runtime_paths_from_cli_cwd(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "profiles" / "notify.profile.json"
    _write_events(events, [_event()])

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workspace)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "init",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--events",
            str(events),
            "--cursor",
            "outputs/notify/densegen/cursor",
            "--spool-dir",
            "outputs/notify/densegen/spool",
        ],
    )

    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["cursor"] == str((workspace / "outputs" / "notify" / "densegen" / "cursor").resolve())
    assert data["spool_dir"] == str((workspace / "outputs" / "notify" / "densegen" / "spool").resolve())


def test_profile_wizard_writes_generic_policy_without_filters(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event(action="attach")])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "generic",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["policy"] == "generic"
    assert "only_actions" not in data
    assert "only_tools" not in data


def test_profile_wizard_writes_v2_profile_with_env_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["profile_version"] == 2
    assert data["provider"] == "slack"
    assert data["events"] == str(events)
    assert data["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert "url_env" not in data
    assert "Next steps:" in result.stdout
    assert "notify profile doctor --profile" in result.stdout
    assert "notify usr-events watch --profile" in result.stdout
    assert "--dry-run" in result.stdout
    assert "export DENSEGEN_WEBHOOK" in result.stdout
    assert "NOTIFY_PROFILE" in result.stdout
    assert "docs/bu-scc/jobs/notify-watch.qsub" in result.stdout


def test_profile_wizard_persists_progress_tunables(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--progress-step-pct",
            "12",
            "--progress-min-seconds",
            "75",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["progress_step_pct"] == 12
    assert data["progress_min_seconds"] == 75.0


def test_profile_wizard_env_uses_default_webhook_env_when_not_set(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}


def test_profile_wizard_defaults_to_namespaced_profile_and_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    assert profile.exists()
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}
    assert data["cursor"] == str((tmp_path / "outputs" / "notify" / "densegen" / "cursor").resolve())
    assert data["spool_dir"] == str((tmp_path / "outputs" / "notify" / "densegen" / "spool").resolve())


def test_profile_wizard_requires_policy_or_profile_for_default_events_mode_profile(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 1
    assert "pass --policy or --profile" in result.stdout.lower()


def test_profile_wizard_stores_tls_ca_bundle_path(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy", encoding="utf-8")
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
            "--tls-ca-bundle",
            str(ca_bundle),
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["tls_ca_bundle"] == str(ca_bundle.resolve())


def test_profile_wizard_can_emit_json_output(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--policy",
            "densegen",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["profile"] == str(profile.resolve())
    assert payload["events"] == str(events)
    assert payload["events_exists"] is True
    assert payload["policy"] == "densegen"
    assert payload["webhook"] == {"source": "env", "ref": "DENSEGEN_WEBHOOK"}
    assert isinstance(payload.get("next_steps"), list)


def test_profile_wizard_stores_secret_ref_when_requested(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: list[tuple[str, str]] = []

    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda _backend: True)
    monkeypatch.setattr(
        "dnadesign.notify.cli.store_secret_ref",
        lambda secret_ref, secret_value: captured.append((secret_ref, secret_value)),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "keychain",
            "--secret-ref",
            "keychain://dnadesign.notify/demo",
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 0
    assert captured == [("keychain://dnadesign.notify/demo", "https://example.invalid/webhook")]
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"}
    assert "Next steps:" in result.stdout
    assert "notify profile doctor --profile" in result.stdout
    assert "notify usr-events watch --profile" in result.stdout
    assert "NOTIFY_PROFILE" in result.stdout
    assert "docs/bu-scc/jobs/notify-watch.qsub" in result.stdout
    assert "export " not in result.stdout


def test_profile_wizard_auto_falls_back_to_file_secret_ref(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda backend: backend == "file")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"]["source"] == "secret_ref"
    assert data["webhook"]["ref"].startswith("file://")


def test_profile_wizard_rejects_yaml_path_for_events(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(cfg_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout
    assert "inspect run --usr-events-path" in result.stdout


def test_profile_wizard_reports_actionable_cursor_directory_error(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    def _fail_private_dir(path: Path, *, label: str) -> None:
        raise NotifyConfigError(f"failed to set secure permissions on {label}: {path}")

    monkeypatch.setattr("dnadesign.notify.cli._ensure_private_directory", _fail_private_dir)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            str(profile),
            "--provider",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "--cursor" in result.stdout
    assert "--spool-dir" in result.stdout


def test_ensure_private_directory_wraps_mkdir_permission_errors(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "restricted" / "dir"

    def _deny_mkdir(self, *args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", _deny_mkdir)

    with pytest.raises(NotifyConfigError, match="failed to create cursor directory"):
        _ensure_private_directory(target, label="cursor directory")


def test_profile_wizard_stores_absolute_events_path(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "outputs" / "notify" / "generic" / "profile.json"
    _write_events(events, [_event()])
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "profile",
            "wizard",
            "--profile",
            "outputs/notify/generic/profile.json",
            "--provider",
            "slack",
            "--events",
            "events.log",
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str(events.resolve())


def test_setup_slack_can_resolve_densegen_events_from_config_without_existing_events(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str(resolved_events.resolve())
    assert data["policy"] == "densegen"
    assert data["events_source"] == {"tool": "densegen", "config": str(config_path.resolve())}
    assert f"notify usr-events watch --tool densegen --config {config_path.resolve()} --follow" in result.stdout
    assert "--wait-for-events" in result.stdout


def test_setup_slack_next_steps_use_tool_config_watch_when_events_exist(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    resolved_events.parent.mkdir(parents=True, exist_ok=True)
    _write_events(resolved_events, [_event()])
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    assert f"notify usr-events watch --tool densegen --config {config_path.resolve()} --dry-run" in result.stdout
    assert f"notify usr-events watch --tool densegen --config {config_path.resolve()} --follow" in result.stdout
    assert "notify usr-events watch --profile" not in result.stdout


def test_setup_slack_next_steps_quote_paths_with_spaces(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "workspace with space" / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    resolved_events.parent.mkdir(parents=True, exist_ok=True)
    _write_events(resolved_events, [_event()])
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    quoted_config = shlex.quote(str(config_path.resolve()))
    assert f"--config {quoted_config} --dry-run" in result.stdout
    assert f"--config {quoted_config} --follow" in result.stdout


def test_setup_slack_defaults_to_tool_namespaced_profile_and_runtime_paths(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    assert profile.exists()
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str(resolved_events.resolve())
    assert data["policy"] == "densegen"
    assert data["cursor"] == str((tmp_path / "outputs" / "notify" / "densegen" / "cursor").resolve())
    assert data["spool_dir"] == str((tmp_path / "outputs" / "notify" / "densegen" / "spool").resolve())


def test_setup_slack_defaults_profile_under_config_directory(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "workspaces" / "demo" / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_cwd)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    expected_profile = config_path.parent / "outputs" / "notify" / "densegen" / "profile.json"
    assert expected_profile.exists()
    assert not (outside_cwd / "outputs" / "notify" / "densegen" / "profile.json").exists()


def test_setup_slack_accepts_workspace_shorthand_for_config(tmp_path: Path, monkeypatch) -> None:
    workspace = "demo_workspace"
    config_path = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces" / workspace / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_workspace_config_path",
        lambda *, tool, workspace, search_start: config_path,
    )
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--tool",
            "densegen",
            "--workspace",
            workspace,
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    expected_profile = config_path.parent / "outputs" / "notify" / "densegen" / "profile.json"
    assert expected_profile.exists()
    assert f"--config {config_path.resolve()} --follow" in result.stdout


def test_setup_slack_stores_tls_ca_bundle_path(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    ca_bundle = tmp_path / "ca.pem"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    ca_bundle.write_text("dummy", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--tls-ca-bundle",
            str(ca_bundle),
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["tls_ca_bundle"] == str(ca_bundle.resolve())


def test_setup_slack_persists_progress_tunables(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--progress-step-pct",
            "18",
            "--progress-min-seconds",
            "120",
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "densegen" / "profile.json"
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["progress_step_pct"] == 18
    assert data["progress_min_seconds"] == 120.0


def test_setup_resolve_events_emits_plain_events_path(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "resolve-events",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == str(resolved_events.resolve())


def test_setup_resolve_events_accepts_workspace_shorthand(tmp_path: Path, monkeypatch) -> None:
    workspace = "demo_workspace"
    config_path = tmp_path / "src" / "dnadesign" / "densegen" / "workspaces" / workspace / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_workspace_config_path",
        lambda *, tool, workspace, search_start: config_path,
    )
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "resolve-events",
            "--tool",
            "densegen",
            "--workspace",
            workspace,
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == str(resolved_events.resolve())


def test_setup_resolve_events_can_emit_json_output(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "resolve-events",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["tool"] == "densegen"
    assert payload["config"] == str(config_path.resolve())
    assert payload["events"] == str(resolved_events.resolve())
    assert payload["policy"] == "densegen"


def test_setup_list_workspaces_emits_names_and_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "dnadesign.notify.cli._list_tool_workspaces",
        lambda *, tool, search_start: ["demo_a", "demo_b"],
    )

    runner = CliRunner()
    text_result = runner.invoke(
        app,
        [
            "setup",
            "list-workspaces",
            "--tool",
            "densegen",
        ],
    )
    assert text_result.exit_code == 0
    assert text_result.stdout.splitlines() == ["demo_a", "demo_b"]

    json_result = runner.invoke(
        app,
        [
            "setup",
            "list-workspaces",
            "--tool",
            "densegen",
            "--json",
        ],
    )
    assert json_result.exit_code == 0
    payload = json.loads(json_result.stdout)
    assert payload["ok"] is True
    assert payload["workspaces"] == ["demo_a", "demo_b"]


def test_setup_resolve_events_can_print_policy_only(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (resolved_events, "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "resolve-events",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--print-policy",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == "densegen"


def test_setup_webhook_auto_falls_back_to_file_secret_ref(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str((tmp_path / "config_home").resolve()))
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda backend: backend == "file")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "webhook",
            "--secret-source",
            "auto",
            "--name",
            "densegen-shared",
            "--webhook-url",
            "https://example.invalid/webhook",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["webhook"]["source"] == "secret_ref"
    secret_ref = str(payload["webhook"]["ref"])
    assert secret_ref.startswith("file://")
    assert resolve_secret_ref(secret_ref) == "https://example.invalid/webhook"


def test_setup_webhook_auto_falls_back_to_file_when_secretservice_write_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str((tmp_path / "config_home").resolve()))
    monkeypatch.setattr(
        "dnadesign.notify.cli.is_secret_backend_available",
        lambda backend: backend in {"secretservice", "file"},
    )

    def _store(secret_ref: str, secret_value: str) -> None:
        if secret_ref.startswith("secretservice://"):
            raise NotifyConfigError("secret backend keyring write failed")
        from dnadesign.notify.secrets import store_secret_ref as _store_secret_ref

        _store_secret_ref(secret_ref, secret_value)

    monkeypatch.setattr("dnadesign.notify.cli.store_secret_ref", _store)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "webhook",
            "--secret-source",
            "auto",
            "--name",
            "densegen-shared",
            "--webhook-url",
            "https://example.invalid/webhook",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["webhook"]["source"] == "secret_ref"
    secret_ref = str(payload["webhook"]["ref"])
    assert secret_ref.startswith("file://")
    assert resolve_secret_ref(secret_ref) == "https://example.invalid/webhook"


def test_setup_webhook_auto_reuses_existing_secret_without_prompt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str((tmp_path / "config_home").resolve()))
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda backend: backend == "file")
    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "setup",
            "webhook",
            "--secret-source",
            "auto",
            "--name",
            "densegen-shared",
            "--webhook-url",
            "https://example.invalid/webhook",
            "--json",
        ],
    )
    assert first.exit_code == 0
    monkeypatch.setattr(
        "dnadesign.notify.profile_flows.typer.prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt should not be called")),
    )
    second = runner.invoke(
        app,
        [
            "setup",
            "webhook",
            "--secret-source",
            "auto",
            "--name",
            "densegen-shared",
            "--json",
        ],
    )
    assert second.exit_code == 0


def test_setup_webhook_env_mode_uses_default_env_ref(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str((tmp_path / "config_home").resolve()))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "webhook",
            "--secret-source",
            "env",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}


def test_setup_slack_requires_events_or_tool_config(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "notify.profile.json"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "pass either --events or --tool with --config" in result.stdout.lower()


def test_setup_slack_requires_policy_or_profile_for_default_events_mode_profile(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--events",
            str(events),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 1
    assert "pass --policy or --profile" in result.stdout.lower()


def test_setup_slack_events_mode_defaults_profile_to_policy_namespace(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    _write_events(events, [_event()])
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--events",
            str(events),
            "--policy",
            "infer_evo2",
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    profile = tmp_path / "outputs" / "notify" / "infer_evo2" / "profile.json"
    assert profile.exists()
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str(events.resolve())
    assert data["policy"] == "infer_evo2"
    assert data["cursor"] == str((tmp_path / "outputs" / "notify" / "infer_evo2" / "cursor").resolve())
    assert data["spool_dir"] == str((tmp_path / "outputs" / "notify" / "infer_evo2" / "spool").resolve())


def test_setup_slack_env_uses_default_webhook_env_when_not_set(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (tmp_path / "usr" / "demo" / ".events.log", "densegen"),
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"] == {"source": "env", "ref": "NOTIFY_WEBHOOK"}


def test_setup_slack_auto_falls_back_to_file_secret_ref(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (tmp_path / "usr" / "demo" / ".events.log", "densegen"),
    )
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda backend: backend == "file")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "auto",
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["webhook"]["source"] == "secret_ref"
    assert data["webhook"]["ref"].startswith("file://")


def test_setup_slack_auto_reuses_existing_secret_without_prompt(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli._resolve_tool_events_path",
        lambda *, tool, config: (tmp_path / "usr" / "demo" / ".events.log", "densegen"),
    )
    monkeypatch.setattr("dnadesign.notify.cli.is_secret_backend_available", lambda backend: backend == "file")
    stored: dict[str, str] = {}

    def _store(secret_ref: str, secret_value: str) -> None:
        stored[secret_ref] = secret_value

    def _resolve(secret_ref: str) -> str:
        if secret_ref not in stored:
            raise NotifyConfigError("missing secret")
        return stored[secret_ref]

    monkeypatch.setattr("dnadesign.notify.cli.store_secret_ref", _store)
    monkeypatch.setattr("dnadesign.notify.cli.resolve_secret_ref", _resolve)
    monkeypatch.setattr(
        "dnadesign.notify.profile_flows.typer.prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("prompt should not be called")),
    )

    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "auto",
            "--webhook-url",
            "https://example.invalid/webhook",
        ],
    )
    assert first.exit_code == 0

    second = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--secret-source",
            "auto",
            "--force",
        ],
    )
    assert second.exit_code == 0


def test_setup_slack_rejects_unsupported_tool(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "unknown_tool",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--url-env",
            "DENSEGEN_WEBHOOK",
        ],
    )
    assert result.exit_code == 1
    assert "unsupported tool" in result.stdout.lower()


def test_setup_slack_can_resolve_infer_evo2_events_from_config(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "infer.yaml"
    profile = tmp_path / "notify.profile.json"
    usr_root = tmp_path / "usr_root"
    config_path.write_text(
        "\n".join(
            [
                "model:",
                "  id: evo2",
                "  device: cpu",
                "  precision: fp32",
                "  alphabet: dna",
                "jobs:",
                "  - id: j1",
                "    operation: generate",
                "    ingest:",
                "      source: usr",
                "      dataset: infer_demo",
                f"      root: {usr_root}",
                "    params:",
                "      max_new_tokens: 8",
                "    io:",
                "      write_back: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "setup",
            "slack",
            "--profile",
            str(profile),
            "--tool",
            "infer_evo2",
            "--config",
            str(config_path),
            "--secret-source",
            "env",
            "--url-env",
            "INFER_WEBHOOK",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(profile.read_text(encoding="utf-8"))
    assert data["events"] == str((usr_root / "infer_demo" / ".events.log").resolve())
    assert data["policy"] == "infer_evo2"
    assert data["events_source"] == {"tool": "infer_evo2", "config": str(config_path.resolve())}


def test_profile_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])

    runner = CliRunner()
    args = [
        "profile",
        "init",
        "--profile",
        str(profile),
        "--provider",
        "slack",
        "--url-env",
        "DENSEGEN_WEBHOOK",
        "--events",
        str(events),
    ]
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)
    assert first.exit_code == 0
    assert second.exit_code == 1
    assert "already exists" in second.stdout


def test_profile_doctor_fails_when_webhook_env_is_missing(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "is not set or empty" in result.stdout


def test_profile_doctor_rejects_non_usr_events_file(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(cfg_path),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout


def test_profile_doctor_rejects_non_usr_events_file_v2_secret_ref(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile = tmp_path / "notify.profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(cfg_path),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "USR .events.log" in result.stdout
    assert "inspect run --usr-events-path" in result.stdout


def test_profile_doctor_rejects_legacy_profile_version_1(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 1,
                "provider": "slack",
                "url_env": "DENSEGEN_WEBHOOK",
                "events": str(events),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "profile_version must be 2" in result.stdout


def test_profile_doctor_rejects_legacy_preset_field(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
                "preset": "densegen",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "legacy profile field 'preset' is not supported" in result.stdout


def test_profile_doctor_rejects_policy_without_explicit_filters(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "policy": "densegen",
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "requires explicit only_actions and only_tools" in result.stdout


def test_profile_doctor_passes_when_wiring_is_valid(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "cursor": str(tmp_path / "notify.cursor"),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_profile_doctor_can_emit_json_output(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "cursor": str(tmp_path / "notify.cursor"),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["profile"] == str(profile.resolve())
    assert payload["events"] == str(events)


def test_profile_doctor_json_reports_structured_error(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert "is not set or empty" in str(payload["error"])


def test_profile_doctor_rejects_non_slack_webhook_for_slack_provider(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 1
    assert "slack provider requires webhook host" in result.stdout


def test_profile_doctor_passes_with_secret_ref_profile(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "cursor": str(tmp_path / "notify.cursor"),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://hooks.slack.com/services/T000/B000/XXX"
    )
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "Profile wiring OK." in result.stdout


def test_profile_doctor_allows_missing_events_when_profile_has_events_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "future" / ".events.log"
    config_path = tmp_path / "densegen.config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "events_source": {"tool": "densegen", "config": str(config_path)},
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile)])
    assert result.exit_code == 0
    assert "events file not created yet" in result.stdout


def test_profile_doctor_json_reports_missing_events_as_pending_for_events_source(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "future" / ".events.log"
    config_path = tmp_path / "densegen.config.yaml"
    profile = tmp_path / "notify.profile.json"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "slack",
                "events": str(events),
                "events_source": {"tool": "densegen", "config": str(config_path)},
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://hooks.slack.com/services/T000/B000/XXX")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["profile", "doctor", "--profile", str(profile), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["events_exists"] is False
    assert payload["events"] == str(events)


def test_usr_events_watch_can_use_profile_defaults(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"tool": "densegen"' in result.stdout


def test_usr_events_watch_can_use_profile_context_flags(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "include_context": True,
                "include_raw_event": True,
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"host": "host"' in result.stdout
    assert '"cwd": "/tmp/datasets"' in result.stdout
    assert "usr_event" in result.stdout


def test_usr_events_watch_can_use_profile_secret_ref(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "webhook": {"source": "secret_ref", "ref": "keychain://dnadesign.notify/demo"},
            }
        ),
        encoding="utf-8",
    )

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr(
        "dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.invalid/webhook"
    )
    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
        ],
    )
    assert result.exit_code == 0
    assert len(sent) == 1


def test_usr_events_watch_rejects_profiles_with_plain_url(tmp_path: Path) -> None:
    events = tmp_path / "events.log"
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "url": "https://example.invalid/webhook",
                "events": str(events),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 1
    assert "must not store plain webhook URLs" in result.stdout


def test_spool_drain_can_use_profile_defaults(tmp_path: Path, monkeypatch) -> None:
    events = tmp_path / "events.log"
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir(parents=True, exist_ok=True)
    profile = tmp_path / "notify.profile.json"
    _write_events(events, [_event()])
    payload = {
        "status": "running",
        "tool": "densegen",
        "run_id": "run-1",
        "timestamp": "2026-02-06T00:00:00+00:00",
        "host": "host",
        "cwd": "/tmp",
        "meta": {"usr_action": "write_overlay_part"},
    }
    (spool_dir / "item.json").write_text(
        json.dumps({"provider": "generic", "payload": payload, "event": _event()}),
        encoding="utf-8",
    )
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": str(events),
                "spool_dir": str(spool_dir),
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    sent: list[dict] = []

    def _fake_post(_url: str, body: dict, **_kwargs) -> None:
        sent.append(body)

    monkeypatch.setattr("dnadesign.notify.cli.post_json", _fake_post)
    monkeypatch.setenv("DENSEGEN_WEBHOOK", "https://example.invalid/webhook")
    _set_ssl_cert_file(monkeypatch, tmp_path)

    runner = CliRunner()
    result = runner.invoke(app, ["spool", "drain", "--profile", str(profile)])
    assert result.exit_code == 0
    assert len(sent) == 1
    assert not list(spool_dir.glob("*.json"))


def test_usr_events_watch_resolves_profile_relative_events_path(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    events = workspace / "events.log"
    profile = workspace / "notify.profile.json"
    _write_events(events, [_event()])
    profile.write_text(
        json.dumps(
            {
                "profile_version": 2,
                "provider": "generic",
                "events": "events.log",
                "webhook": {"source": "env", "ref": "DENSEGEN_WEBHOOK"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "usr-events",
            "watch",
            "--profile",
            str(profile),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert '"tool": "densegen"' in result.stdout
