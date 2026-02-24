"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/notebooks/test_promoter_eda_notebook_smoke.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path("src/dnadesign/opal/notebooks/prom60_eda.py")


def _load_notebook_module() -> object:
    spec = importlib.util.spec_from_file_location("prom60_eda_smoke", NOTEBOOK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load prom60_eda notebook module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prom60_eda_headless(tmp_path: Path) -> None:
    if importlib.util.find_spec("marimo") is None:
        pytest.skip("marimo is not installed in this environment")
    env_backup = os.environ.copy()
    os.environ["DNADESIGN_HEADLESS"] = "1"
    os.environ["MPLBACKEND"] = "Agg"

    try:
        module = _load_notebook_module()
        outputs, defs = module.app.run(defs={"active_record": None, "active_record_id": None})
    finally:
        os.environ.clear()
        os.environ.update(env_backup)

    assert outputs is not None
