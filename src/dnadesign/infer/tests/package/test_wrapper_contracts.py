"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/package/test_wrapper_contracts.py

Wrapper and public API contract tests for infer package entrypoints.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import dnadesign.infer as infer
from dnadesign.infer.cli import main as infer_cli_main


def test_infer_public_api_exports_callable_wrappers() -> None:
    assert callable(infer.run_extract)
    assert callable(infer.run_generate)
    assert callable(infer.run_job)


def test_infer_cli_wrapper_exposes_main_callable() -> None:
    assert callable(infer_cli_main)


def test_infer_module_entrypoint_exists() -> None:
    assert importlib.util.find_spec("dnadesign.infer.__main__") is not None


def test_python_module_wrapper_invokes_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dnadesign.infer", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "Model-agnostic sequence inference CLI" in result.stdout
