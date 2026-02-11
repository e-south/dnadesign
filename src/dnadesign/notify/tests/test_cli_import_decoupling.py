"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/tests/test_cli_import_decoupling.py

Import decoupling tests for notify CLI startup dependencies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_notify_cli_import_does_not_load_usr_dataset_module() -> None:
    script = """
import importlib
import sys

importlib.import_module("dnadesign.notify.cli")
raise SystemExit(1 if "dnadesign.usr.src.dataset" in sys.modules else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_notify_cli_import_does_not_depend_on_root_event_schema_module() -> None:
    script = """
import importlib
import sys

importlib.import_module("dnadesign.notify.cli")
try:
    importlib.import_module("dnadesign.event_schema")
except ModuleNotFoundError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
