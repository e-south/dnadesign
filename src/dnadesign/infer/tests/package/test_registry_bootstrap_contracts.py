"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_registry_bootstrap_contracts.py

Registry bootstrap contracts for infer adapter/function registration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import subprocess
import sys


def test_registry_bootstrap_requires_explicit_initialize_call() -> None:
    script = """
from dnadesign.infer.src.registry import list_fns, list_models
print(f"before_models={len(list_models())}")
print(f"before_fns={len(list_fns())}")

from dnadesign.infer.src.bootstrap import initialize_registry
initialize_registry()
print(f"after_models={len(list_models())}")
print(f"after_fns={len(list_fns())}")
"""

    result = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout

    lines = {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}
    assert "before_models=0" in lines
    assert "before_fns=0" in lines
    after_models = [line for line in lines if line.startswith("after_models=")]
    after_fns = [line for line in lines if line.startswith("after_fns=")]
    assert len(after_models) == 1
    assert len(after_fns) == 1
    assert int(after_models[0].split("=", 1)[1]) >= 2
    assert int(after_fns[0].split("=", 1)[1]) >= 4
