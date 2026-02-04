"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/test_native_stderr_filters.py

Native stderr filter installation tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_densegen_import_installs_native_stderr_filters() -> None:
    code = """
import sys
from dnadesign.densegen.src.utils import logging_utils
import dnadesign.densegen.src  # triggers stderr filter installation
flag = getattr(logging_utils._install_native_stderr_deduper, "_installed", False)
sys.stdout.write("1" if flag else "0")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "1"
