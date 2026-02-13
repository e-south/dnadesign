"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_cli_coupling.py

Regression tests keeping CLI free from plotting/runtime rendering dependencies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_module_import_does_not_preload_matplotlib() -> None:
    code = "import sys\nimport dnadesign.baserender.src.cli\nprint('matplotlib' in sys.modules)\n"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip().endswith("False")


def test_cli_source_has_no_plotting_tokens() -> None:
    cli_path = Path("src/dnadesign/baserender/src/cli.py")
    source = cli_path.read_text()

    for token in ("matplotlib", "render_record_figure", "render_parquet_record_figure", "plt."):
        assert token not in source
