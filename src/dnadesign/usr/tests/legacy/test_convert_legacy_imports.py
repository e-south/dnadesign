"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/usr/tests/test_convert_legacy_imports.py

Import-time contract tests for legacy conversion helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_convert_legacy_import_is_lazy() -> None:
    code = "import sys\nimport dnadesign.usr.src.convert_legacy\nprint('torch' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.strip() == "False"
