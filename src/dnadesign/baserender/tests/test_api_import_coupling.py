"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_api_import_coupling.py

Regression test preventing render-stack imports during API module import.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_api_module_import_does_not_preload_matplotlib() -> None:
    code = "import sys\nimport dnadesign.baserender.src.public.api\nprint('matplotlib' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip().endswith("False")


def test_package_root_import_does_not_preload_render_stack() -> None:
    code = "\n".join(
        [
            "import sys",
            "import dnadesign.baserender",
            "print('matplotlib' in sys.modules)",
            "print('numpy' in sys.modules)",
            "print('dnadesign.baserender.src.public.api' in sys.modules)",
            "print('dnadesign.baserender.src.render' in sys.modules)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines() == ["False", "False", "False", "False"]
