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
    cli_paths = [
        Path("src/dnadesign/baserender/src/cli/__init__.py"),
        Path("src/dnadesign/baserender/src/cli/actions.py"),
        Path("src/dnadesign/baserender/src/cli/app.py"),
    ]
    source = "\n".join(path.read_text() for path in cli_paths)

    for token in ("matplotlib", "render_record_figure", "render_parquet_record_figure", "plt."):
        assert token not in source


def test_cli_help_uses_base_render_contract_language() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dnadesign.baserender.src.cli.app", "job", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "BaseRender v3 render commands" in proc.stdout
    assert "Sequence Rows v3 job commands" not in proc.stdout
