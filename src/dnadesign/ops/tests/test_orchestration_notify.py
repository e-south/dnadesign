"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_orchestration_notify.py

Contract tests for orchestration notify runtime and send-command construction.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.ops.orchestrator import orchestration_notify


def test_build_orchestration_notify_argv_requires_secret_ref() -> None:
    notify = orchestration_notify.OrchestrationNotifySpec(
        tool="infer",
        provider="slack",
        webhook_env="NOTIFY_WEBHOOK",
        secret_ref="",
        run_id="infer_demo",
        tls_ca_bundle="/etc/ssl/certs/ca-certificates.crt",
    )

    with pytest.raises(ValueError, match="secret_ref is required"):
        orchestration_notify.build_orchestration_notify_argv(
            notify=notify,
            status="started",
            message="ops submit requested workflow=infer_batch_submit project=dunlop",
        )


def test_resolve_notify_runtime_contract_prefers_profile_secret_ref_when_env_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOTIFY_WEBHOOK_FILE", raising=False)
    ca_bundle = tmp_path / "ca-bundle.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle))

    webhook_file = tmp_path / "notify.secret"
    webhook_file.write_text("https://example.invalid/webhook\n", encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(
        f'{{"webhook": {{"source": "secret_ref", "ref": "{webhook_file.as_uri()}"}}}}\n',
        encoding="utf-8",
    )

    runtime = orchestration_notify.resolve_notify_runtime_contract(
        "NOTIFY_WEBHOOK",
        profile_path=profile_path,
    )

    assert runtime.webhook_file == str(webhook_file)
    assert runtime.secret_ref == webhook_file.as_uri()
    assert runtime.tls_ca_bundle == str(ca_bundle)
