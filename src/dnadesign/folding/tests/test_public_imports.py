"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/tests/test_public_imports.py

Public Folding facade import-boundary tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def _fresh_process_lines(code: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def test_error_access_does_not_import_folding_runtime_or_plotting() -> None:
    lines = _fresh_process_lines(
        "\n".join(
            [
                "import sys",
                "import dnadesign.folding as folding",
                "folding.FoldingConfigError",
                "print('dnadesign.folding.src.errors' in sys.modules)",
                "print('dnadesign.folding.src.api' in sys.modules)",
                "print('dnadesign.folding.src.viennarna_plot' in sys.modules)",
            ]
        )
    )

    assert lines == ["True", "False", "False"]


def test_preflight_access_does_not_import_plotting() -> None:
    lines = _fresh_process_lines(
        "\n".join(
            [
                "import sys",
                "import dnadesign.folding as folding",
                "folding.preflight_request",
                "print('dnadesign.folding.src.api' in sys.modules)",
                "print('dnadesign.folding.src.viennarna_plot' in sys.modules)",
            ]
        )
    )

    assert lines == ["True", "False"]
