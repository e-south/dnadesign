"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_setup.py

Setup command tests for notify CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.notify.cli import app
from dnadesign.notify.delivery.secrets import resolve_secret_ref
from dnadesign.notify.errors import NotifyConfigError


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


def test_setup_slack_can_resolve_densegen_events_from_config_without_existing_events(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    profile = tmp_path / "notify.profile.json"
    resolved_events = tmp_path / "usr" / "demo" / ".events.log"
    config_path.write_text("densegen:\n  run:\n    id: demo\n", encoding="utf-8")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_workspace_config_path",
        lambda *, tool, workspace, search_start: config_path,
    )
    monkeypatch.setattr(
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_workspace_config_path",
        lambda *, tool, workspace, search_start: config_path,
    )
    monkeypatch.setattr(
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._list_tool_workspaces",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda backend: backend == "file")

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
        "dnadesign.notify.cli.bindings.is_secret_backend_available",
        lambda backend: backend in {"secretservice", "file"},
    )

    def _store(secret_ref: str, secret_value: str) -> None:
        if secret_ref.startswith("secretservice://"):
            raise NotifyConfigError("secret backend keyring write failed")
        from dnadesign.notify.delivery.secrets import store_secret_ref as _store_secret_ref

        _store_secret_ref(secret_ref, secret_value)

    monkeypatch.setattr("dnadesign.notify.cli.bindings.store_secret_ref", _store)

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
    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda backend: backend == "file")
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
        "dnadesign.notify.profiles.flow_webhook.typer.prompt",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
        lambda *, tool, config: (tmp_path / "usr" / "demo" / ".events.log", "densegen"),
    )
    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda backend: backend == "file")

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
        "dnadesign.notify.cli.bindings._resolve_tool_events_path",
        lambda *, tool, config: (tmp_path / "usr" / "demo" / ".events.log", "densegen"),
    )
    monkeypatch.setattr("dnadesign.notify.cli.bindings.is_secret_backend_available", lambda backend: backend == "file")
    stored: dict[str, str] = {}

    def _store(secret_ref: str, secret_value: str) -> None:
        stored[secret_ref] = secret_value

    def _resolve(secret_ref: str) -> str:
        if secret_ref not in stored:
            raise NotifyConfigError("missing secret")
        return stored[secret_ref]

    monkeypatch.setattr("dnadesign.notify.cli.bindings.store_secret_ref", _store)
    monkeypatch.setattr("dnadesign.notify.cli.bindings.resolve_secret_ref", _resolve)
    monkeypatch.setattr(
        "dnadesign.notify.profiles.flow_webhook.typer.prompt",
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
