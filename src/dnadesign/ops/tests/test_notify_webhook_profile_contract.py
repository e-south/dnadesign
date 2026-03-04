"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_notify_webhook_profile_contract.py

Contract tests for shared notify profile webhook parsing helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign._contracts.notify_webhook_profile import (
    parse_notify_profile_webhook,
    resolve_file_secret_ref_path,
)


def test_parse_notify_profile_webhook_returns_source_and_ref() -> None:
    source, ref = parse_notify_profile_webhook(
        {
            "profile_version": 2,
            "provider": "slack",
            "events": "/tmp/x.events.log",
            "webhook": {"source": "secret_ref", "ref": "file:///tmp/notify.secret"},
        },
        required_profile_version=2,
    )

    assert source == "secret_ref"
    assert ref == "file:///tmp/notify.secret"


def test_resolve_file_secret_ref_path_returns_normalized_absolute_path(tmp_path: Path) -> None:
    secret_path = tmp_path / "notify.secret"
    resolved = resolve_file_secret_ref_path(secret_path.as_uri(), source_label="profile webhook secret_ref")

    assert resolved == secret_path.resolve()


def test_resolve_file_secret_ref_path_rejects_non_file_scheme() -> None:
    with pytest.raises(ValueError, match="file://"):
        resolve_file_secret_ref_path(
            "keychain://dnadesign.notify/default",
            source_label="profile webhook secret_ref",
        )
