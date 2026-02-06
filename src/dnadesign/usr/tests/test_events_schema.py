"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_events_schema.py

Event schema validation for USR JSONL logs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.usr import Dataset
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_event_schema_includes_actor_and_registry_hash(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["event_version"] == 1
    assert payload["dataset"]["name"] == "demo"
    assert payload["dataset"]["root"] == str(root)
    assert payload["registry_hash"]
    assert payload["actor"]["tool"] == "usr"
    assert isinstance(payload["metrics"], dict)
    assert isinstance(payload["artifacts"], dict)
    assert isinstance(payload["maintenance"], dict)


def test_event_schema_redacts_sensitive_args(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    ds.log_event(
        "custom_action",
        args={
            "token": "secret-token",
            "nested": {"client_secret": "very-secret"},  # pragma: allowlist secret
            "safe_value": "ok",
        },
    )
    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["args"]["token"] == "***REDACTED***"
    assert payload["args"]["nested"]["client_secret"] == "***REDACTED***"
    assert payload["args"]["safe_value"] == "ok"


def test_event_schema_redacts_sensitive_cli_style_args(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    ds.log_event(
        "custom_action",
        args={
            "argv": ["--token", "secret-token", "--mode", "fast", "--api-key=abc123"],
        },
    )
    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["args"]["argv"] == [
        "--token",
        "***REDACTED***",
        "--mode",
        "fast",
        "--api-key=***REDACTED***",
    ]
