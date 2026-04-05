"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/tests/test_api.py

Public API boundary tests for construct.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.construct import preflight_from_config, run_from_config
from dnadesign.construct.src.errors import ExecutionError, ValidationError
from dnadesign.usr import SchemaError as USRSchemaError


def _raise_usr_schema_error(path: str | Path) -> None:
    raise USRSchemaError("registry schema mismatch")


def _raise_usr_schema_error_on_write(planned: object) -> None:
    raise USRSchemaError("overlay attach rejected")


def test_preflight_api_wraps_usr_errors_as_construct_validation_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "dnadesign.construct.src.api._runtime_preflight_from_config",
        _raise_usr_schema_error,
    )

    with pytest.raises(ValidationError, match="construct preflight failed while reading USR inputs"):
        preflight_from_config(tmp_path / "config.yaml")


def test_run_api_wraps_usr_write_errors_as_construct_execution_errors(monkeypatch, tmp_path: Path) -> None:
    planned = object()
    monkeypatch.setattr(
        "dnadesign.construct.src.api._planned_run_from_config",
        lambda path: planned,
    )
    monkeypatch.setattr(
        "dnadesign.construct.src.api._persist_construct_run",
        _raise_usr_schema_error_on_write,
    )

    with pytest.raises(ExecutionError, match="construct run failed while writing USR outputs"):
        run_from_config(tmp_path / "config.yaml")
