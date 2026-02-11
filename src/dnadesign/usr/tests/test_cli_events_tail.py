"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/test_cli_events_tail.py

CLI events tail output tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.usr import Dataset
from dnadesign.usr.src.cli import app
from dnadesign.usr.tests.registry_helpers import register_test_namespace


def test_events_tail_json(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "events",
            "tail",
            "demo",
            "--format",
            "json",
            "--n",
            "1",
        ],
    )
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["action"] == "init"


def test_events_tail_accepts_dataset_path_outside_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_root"
    register_test_namespace(dataset_root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(dataset_root, "densegen/demo")
    ds.init(source="unit-test")

    unrelated_root = tmp_path / "unrelated_root"
    unrelated_root.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(unrelated_root),
            "events",
            "tail",
            str(ds.dir),
            "--format",
            "json",
            "--n",
            "1",
        ],
    )
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["action"] == "init"


def test_events_tail_does_not_read_full_file_for_tail_n(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    register_test_namespace(root, namespace="mock", columns_spec="mock__score:float64")
    ds = Dataset(root, "demo")
    ds.init(source="unit-test")
    events_path = ds.events_path.resolve()
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event_version": 1, "action": "event_a"}) + "\n")
        handle.write(json.dumps({"event_version": 1, "action": "event_b"}) + "\n")

    original_read_text = Path.read_text

    def _guarded_read_text(path: Path, *args, **kwargs):
        if path.resolve() == events_path:
            raise AssertionError("events tail should stream lines instead of calling read_text on .events.log")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read_text)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--root",
            str(root),
            "events",
            "tail",
            "demo",
            "--format",
            "json",
            "--n",
            "1",
        ],
    )
    assert result.exit_code == 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    assert payload["action"] == "event_b"
