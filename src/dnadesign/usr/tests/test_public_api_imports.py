"""
--------------------------------------------------------------------------------
dnadesign
dnadesign/src/dnadesign/usr/tests/test_public_api_imports.py

Public API import hygiene tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_public_api_does_not_import_typer_or_rich() -> None:
    code = "import sys\nimport dnadesign.usr\nprint('typer' in sys.modules)\nprint('rich' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines = proc.stdout.strip().splitlines()
    assert lines == ["False", "False"]


def test_public_api_keeps_write_session_type_internal() -> None:
    code = "import dnadesign.usr as usr\nprint(hasattr(usr, 'DatasetWriteSession'))\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.strip() == "False"
