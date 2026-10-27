"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/config/test_local_independence.py

Verify that local-table OPAL workflows do not load the optional USR integration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_local_config_and_cli_do_not_import_usr() -> None:
    config = Path(__file__).resolve().parents[2] / "campaigns/demo_rf_topn/configs/campaign.yaml"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import os
import sys

attempted = []

class NoUSR(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dnadesign.usr' or fullname.startswith('dnadesign.usr.'):
            attempted.append(fullname)
            raise ImportError('USR is unavailable for this standalone test')

sys.meta_path.insert(0, NoUSR())
os.environ.pop('OPAL_NOTEBOOK_USR_ROOT', None)
from dnadesign.opal import load_config
config = load_config(sys.argv[1])
assert config.data.location.kind == 'local'
from dnadesign.opal.src.cli.app import _build
app = _build()
names = {item.name for item in app.registered_commands}
assert {'validate', 'init', 'ingest-y', 'run', 'verify-outputs'} <= names
from dnadesign.opal.src.analysis.notebook_scope.usr_root import resolve_notebook_usr_root
assert resolve_notebook_usr_root(None) is None
assert attempted == [], attempted
""",
            str(config),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
