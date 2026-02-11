"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_cli_imports.py

CLI import hygiene tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_cli_import_does_not_import_torch() -> None:
    code = "import sys\nimport dnadesign.usr.src.cli\nprint('torch' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.strip() == "False"
