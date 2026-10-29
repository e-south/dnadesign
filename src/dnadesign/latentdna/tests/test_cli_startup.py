"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_cli_startup.py

CLI discovery does not initialize numerical analysis services.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize("args", [["--help"], ["deliverable", "run", "--help"], ["plot", "render", "--help"]])
def test_help_does_not_import_analysis_libraries(args):
    script = """
import json
import sys
from typer.testing import CliRunner
from dnadesign.latentdna.src.cli.app import app
result = CliRunner().invoke(app, json.loads(sys.argv[1]))
assert result.exit_code == 0, result.stdout
print(json.dumps(sorted(name for name in ("numpy", "scipy", "pandas", "pyarrow", "umap") if name in sys.modules)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(args)], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == []
