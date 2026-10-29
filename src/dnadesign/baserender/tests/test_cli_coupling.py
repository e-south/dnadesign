"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_cli_coupling.py

Regression tests keeping CLI free from plotting/runtime rendering dependencies.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cli_module_import_does_not_preload_numeric_or_render_stacks() -> None:
    code = "\n".join(
        [
            "import sys",
            "import dnadesign.baserender.src.cli",
            "print('matplotlib' in sys.modules)",
            "print('numpy' in sys.modules)",
            "print('dnadesign.baserender.src.integrations.densegen.adapter' in sys.modules)",
            "print('dnadesign.baserender.src.public.api' in sys.modules)",
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.splitlines() == ["False", "False", "False", "False"]


def test_cli_source_has_no_plotting_tokens() -> None:
    cli_paths = [
        Path("src/dnadesign/baserender/src/cli/__init__.py"),
        Path("src/dnadesign/baserender/src/cli/actions.py"),
        Path("src/dnadesign/baserender/src/cli/app.py"),
    ]
    source = "\n".join(path.read_text() for path in cli_paths)

    for token in ("matplotlib", "render_record_figure", "render_parquet_record_figure", "plt."):
        assert token not in source


def test_cli_help_uses_plain_job_language() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dnadesign.baserender.src.cli.app", "job", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Validate and run render jobs." in proc.stdout
    assert "BaseRender v3 render commands" not in proc.stdout
    assert "Sequence Rows v3 job commands" not in proc.stdout


def test_cli_catalog_is_machine_readable_and_complete() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "dnadesign.baserender.src.cli.app", "catalog", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema"] == "dnadesign.baserender.catalog.v1"
    assert {entry["kind"] for entry in payload["adapters"]} == set(
        __import__("dnadesign.baserender", fromlist=["list_adapters"]).list_adapters()
    )
    assert {entry["name"] for entry in payload["transforms"]} == {
        "attach_motifs_from_config",
        "attach_motifs_from_cruncher_lockfile",
        "attach_motifs_from_library",
        "sigma70",
    }
    assert {entry["name"] for entry in payload["style_profiles"]} == {
        "motif_showcase.v1",
        "promoter_compact_slide.v1",
    }
