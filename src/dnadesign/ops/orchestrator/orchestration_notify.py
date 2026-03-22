"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/orchestrator/orchestration_notify.py

Shared notify runtime and orchestration-notify command contracts for batch
workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dnadesign._contracts.notify_webhook_profile import parse_notify_profile_webhook, resolve_file_secret_ref_path
from dnadesign._contracts.tls_ca_bundle import (
    DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
    resolve_tls_ca_bundle_path,
)


def _notify_webhook_file_path_from_profile(profile_path: Path) -> str | None:
    if not profile_path.is_file():
        return None
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"notify profile is not valid JSON: {profile_path}") from exc
    except OSError as exc:
        raise ValueError(f"notify profile is not readable: {profile_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"notify profile root must be an object: {profile_path}")
    source, ref = parse_notify_profile_webhook(payload)
    if source != "secret_ref":
        return None
    path = resolve_file_secret_ref_path(
        ref,
        source_label=f"notify profile webhook secret_ref (profile={profile_path})",
    )
    return str(path)


def _notify_webhook_file_path(webhook_env_name: str, *, profile_path: Path | None = None) -> str | None:
    webhook_file_env_name = f"{webhook_env_name}_FILE"
    webhook_file = os.environ.get(webhook_file_env_name, "").strip()
    if not webhook_file:
        if profile_path is None:
            return None
        return _notify_webhook_file_path_from_profile(profile_path)
    return str(Path(webhook_file).expanduser().resolve())


def _require_notify_webhook_file_path(webhook_env_name: str, *, profile_path: Path | None = None) -> str:
    webhook_file_env_name = f"{webhook_env_name}_FILE"
    webhook_file = _notify_webhook_file_path(webhook_env_name, profile_path=profile_path)
    if webhook_file is None:
        profile_hint = ""
        if profile_path is not None:
            profile_hint = (
                f", or configure {profile_path} with webhook.source=secret_ref and a file:// secret reference"
            )
        raise ValueError(
            "notify webhook secret file is required for batch notify workflows. "
            f"Set {webhook_file_env_name} to a readable file path{profile_hint}."
        )
    if not os.path.isfile(webhook_file):
        raise ValueError(
            f"notify webhook secret file does not exist or is not a file: {webhook_file} (from {webhook_file_env_name})"
        )
    if not os.access(webhook_file, os.R_OK):
        raise ValueError(f"notify webhook secret file is not readable: {webhook_file} (from {webhook_file_env_name})")
    return webhook_file


def _resolve_tls_ca_bundle() -> str:
    return str(
        resolve_tls_ca_bundle_path(
            explicit_path=None,
            env_var_name="SSL_CERT_FILE",
            allow_system_candidates=True,
            system_candidates=DEFAULT_SYSTEM_TLS_CA_BUNDLE_CANDIDATES,
            not_configured_error=(
                "notify TLS CA bundle is not configured. "
                "Set SSL_CERT_FILE to a readable CA bundle path before running notify workflows."
            ),
            source_label="notify TLS CA bundle path",
        )
    )


def build_notify_setup_secret_contract(secret_ref: str) -> tuple[str, ...]:
    return (
        "--secret-source",
        "file",
        "--secret-ref",
        secret_ref,
        "--no-store-webhook",
    )


@dataclass(frozen=True)
class NotifyRuntimeContract:
    webhook_file: str
    secret_ref: str
    tls_ca_bundle: str


def resolve_notify_runtime_contract(
    webhook_env_name: str,
    *,
    profile_path: Path | None = None,
) -> NotifyRuntimeContract:
    webhook_file = _require_notify_webhook_file_path(webhook_env_name, profile_path=profile_path)
    tls_ca_bundle = _resolve_tls_ca_bundle()
    return NotifyRuntimeContract(
        webhook_file=webhook_file,
        secret_ref=Path(webhook_file).as_uri(),
        tls_ca_bundle=tls_ca_bundle,
    )


@dataclass(frozen=True)
class OrchestrationNotifySpec:
    tool: str
    provider: str
    webhook_env: str
    secret_ref: str
    run_id: str
    tls_ca_bundle: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "provider": self.provider,
            "webhook_env": self.webhook_env,
            "secret_ref": self.secret_ref,
            "run_id": self.run_id,
            "tls_ca_bundle": self.tls_ca_bundle,
        }


def build_orchestration_notify_spec(
    *,
    tool: str,
    provider: str,
    webhook_env: str,
    run_id: str,
    profile_path: Path | None = None,
) -> OrchestrationNotifySpec:
    runtime = resolve_notify_runtime_contract(
        webhook_env,
        profile_path=profile_path,
    )
    return OrchestrationNotifySpec(
        tool=tool,
        provider=provider,
        webhook_env=webhook_env,
        secret_ref=runtime.secret_ref,
        run_id=run_id,
        tls_ca_bundle=runtime.tls_ca_bundle,
    )


def build_orchestration_notify_argv(
    *,
    notify: OrchestrationNotifySpec,
    status: str,
    message: str,
) -> tuple[str, ...]:
    command_parts = [
        "uv",
        "run",
        "notify",
        "send",
        "--status",
        status,
        "--tool",
        notify.tool,
        "--run-id",
        notify.run_id,
        "--provider",
        notify.provider,
    ]
    if not notify.secret_ref.strip():
        raise ValueError("orchestration notify secret_ref is required")
    command_parts.extend(["--secret-ref", notify.secret_ref])
    command_parts.extend(
        [
            "--tls-ca-bundle",
            notify.tls_ca_bundle,
            "--message",
            message,
        ]
    )
    return tuple(command_parts)
