"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_validation.py

Tests for notifier input validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.validation import resolve_tls_ca_bundle, resolve_webhook_url, validate_provider_webhook_url


def test_resolve_webhook_url_requires_source(monkeypatch) -> None:
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with pytest.raises(NotifyConfigError):
        resolve_webhook_url(url=None, url_env=None)


def test_resolve_webhook_url_rejects_multiple_sources(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with pytest.raises(NotifyConfigError):
        resolve_webhook_url(url="https://example.com/other", url_env="WEBHOOK_URL", secret_ref=None)


def test_resolve_webhook_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    resolved = resolve_webhook_url(url=None, url_env="WEBHOOK_URL", secret_ref=None)
    assert resolved == "https://example.com/hook"


def test_resolve_webhook_url_from_secret_ref(monkeypatch) -> None:
    monkeypatch.setattr("dnadesign.notify.validation.resolve_secret_ref", lambda _ref: "https://example.com/hook")
    resolved = resolve_webhook_url(  # pragma: allowlist secret
        url=None,
        url_env=None,
        secret_ref="keychain://dnadesign.notify/demo",  # pragma: allowlist secret
    )
    assert resolved == "https://example.com/hook"


def test_resolve_webhook_url_rejects_env_plus_secret_ref(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with pytest.raises(NotifyConfigError):
        resolve_webhook_url(  # pragma: allowlist secret
            url=None,
            url_env="WEBHOOK_URL",
            secret_ref="keychain://dnadesign.notify/demo",  # pragma: allowlist secret
        )


def test_resolve_tls_ca_bundle_requires_explicit_source_for_https(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    with pytest.raises(NotifyConfigError, match="requires an explicit CA bundle"):
        resolve_tls_ca_bundle(webhook_url="https://example.com/hook", tls_ca_bundle=None)


def test_resolve_tls_ca_bundle_uses_ssl_cert_file_env_for_https(tmp_path: Path, monkeypatch) -> None:
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("dummy", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_bundle))

    resolved = resolve_tls_ca_bundle(webhook_url="https://example.com/hook", tls_ca_bundle=None)
    assert resolved == ca_bundle.resolve()


def test_resolve_tls_ca_bundle_rejects_missing_file(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "missing-ca.pem"
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    with pytest.raises(NotifyConfigError, match="CA bundle file not found"):
        resolve_tls_ca_bundle(webhook_url="https://example.com/hook", tls_ca_bundle=missing)


def test_resolve_tls_ca_bundle_returns_none_for_http(monkeypatch) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    assert resolve_tls_ca_bundle(webhook_url="http://example.com/hook", tls_ca_bundle=None) is None


def test_validate_provider_webhook_url_allows_valid_slack_host() -> None:
    validate_provider_webhook_url(
        provider="slack",
        webhook_url="https://hooks.slack.com/services/T000/B000/XXX",
    )


def test_validate_provider_webhook_url_rejects_non_slack_host_for_slack_provider() -> None:
    with pytest.raises(NotifyConfigError, match="slack provider requires webhook host"):
        validate_provider_webhook_url(
            provider="slack",
            webhook_url="https://example.invalid/webhook",
        )
