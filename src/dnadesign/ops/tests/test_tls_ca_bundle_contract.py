"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_tls_ca_bundle_contract.py

Tests for shared TLS CA bundle resolution contract used by ops and notify.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign._contracts.tls_ca_bundle import resolve_tls_ca_bundle_path


def test_resolve_tls_ca_bundle_path_prefers_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ca_bundle = tmp_path / "explicit-ca.pem"
    ca_bundle.write_text("ca", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    resolved = resolve_tls_ca_bundle_path(
        explicit_path=ca_bundle,
        allow_system_candidates=False,
        not_configured_error="missing bundle",
        source_label="test bundle",
    )

    assert resolved == ca_bundle.resolve()


def test_resolve_tls_ca_bundle_path_uses_env_before_system_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_bundle = tmp_path / "env-ca.pem"
    env_bundle.write_text("env", encoding="utf-8")
    fallback_bundle = tmp_path / "fallback-ca.pem"
    fallback_bundle.write_text("fallback", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(env_bundle))

    resolved = resolve_tls_ca_bundle_path(
        explicit_path=None,
        allow_system_candidates=True,
        system_candidates=(str(fallback_bundle),),
        not_configured_error="missing bundle",
        source_label="test bundle",
    )

    assert resolved == env_bundle.resolve()


def test_resolve_tls_ca_bundle_path_rejects_missing_env_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_bundle = tmp_path / "missing-ca.pem"
    monkeypatch.setenv("SSL_CERT_FILE", str(missing_bundle))

    with pytest.raises(ValueError, match="SSL_CERT_FILE"):
        resolve_tls_ca_bundle_path(
            explicit_path=None,
            allow_system_candidates=True,
            not_configured_error="missing bundle",
            source_label="test bundle",
        )


def test_resolve_tls_ca_bundle_path_uses_system_candidates_when_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_bundle = tmp_path / "fallback-ca.pem"
    fallback_bundle.write_text("fallback", encoding="utf-8")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    resolved = resolve_tls_ca_bundle_path(
        explicit_path=None,
        allow_system_candidates=True,
        system_candidates=(str(fallback_bundle),),
        not_configured_error="missing bundle",
        source_label="test bundle",
    )

    assert resolved == fallback_bundle.resolve()


def test_resolve_tls_ca_bundle_path_requires_configuration_when_no_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)

    with pytest.raises(ValueError, match="missing bundle"):
        resolve_tls_ca_bundle_path(
            explicit_path=None,
            allow_system_candidates=False,
            not_configured_error="missing bundle",
            source_label="test bundle",
        )

