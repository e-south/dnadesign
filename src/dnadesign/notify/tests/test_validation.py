"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_validation.py

Tests for notifier input validation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.notify.errors import NotifyConfigError
from dnadesign.notify.validation import resolve_webhook_url


def test_resolve_webhook_url_requires_source(monkeypatch) -> None:
    monkeypatch.delenv("WEBHOOK_URL", raising=False)
    with pytest.raises(NotifyConfigError):
        resolve_webhook_url(url=None, url_env=None)


def test_resolve_webhook_url_rejects_multiple_sources(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    with pytest.raises(NotifyConfigError):
        resolve_webhook_url(url="https://example.com/other", url_env="WEBHOOK_URL")


def test_resolve_webhook_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")
    resolved = resolve_webhook_url(url=None, url_env="WEBHOOK_URL")
    assert resolved == "https://example.com/hook"
