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

import pyarrow as pa

from dnadesign.usr import USR_EVENT_VERSION, Dataset
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_event_schema_includes_actor_and_registry_hash(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["event_version"] == USR_EVENT_VERSION
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


def test_import_rows_uses_explicit_actor_when_provided(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    actor = {"tool": "densegen", "run_id": "run-1", "host": "host-a", "pid": 101}
    ds.import_rows([{"sequence": "ACGT"}], source="unit-test", actor=actor)

    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["action"] == "import_rows"
    assert payload["actor"] == actor


def test_write_overlay_part_uses_explicit_actor_when_provided(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    ds.import_rows([{"sequence": "ACGT"}], source="unit-test")
    target_id = str(ds.head(n=1)["id"].iloc[0])
    table = pa.table({"id": [target_id], "mock__score": [1.25]})

    actor = {"tool": "densegen", "run_id": "run-2", "host": "host-b", "pid": 202}
    ds.write_overlay_part("mock", table, key="id", actor=actor)

    payload = json.loads(ds.events_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["action"] == "write_overlay_part"
    assert payload["actor"] == actor


def test_usr_public_api_exports_event_version() -> None:
    assert isinstance(USR_EVENT_VERSION, int)
    assert USR_EVENT_VERSION >= 1
