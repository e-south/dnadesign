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
from pathlib import Path

import dnadesign.infer as infer
import pytest
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


def test_infer_import_does_not_eagerly_load_gpu_runtime_modules() -> None:
    script = """
import sys
import dnadesign.infer
print(f"torch_loaded={'torch' in sys.modules}")
print(f"evo2_loaded={'evo2' in sys.modules}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    lines = {line.strip() for line in (result.stdout or "").splitlines() if line.strip()}
    assert "torch_loaded=False" in lines
    assert "evo2_loaded=False" in lines


def test_infer_public_contract_exposes_runbook_gpu_validation(tmp_path: Path) -> None:
    from dnadesign.infer import validate_runbook_gpu_resources

    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  id: evo2_40b
  device: cuda:0
  precision: bf16
  alphabet: dna
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CAPACITY_FAIL"):
        validate_runbook_gpu_resources(
            config_path=config,
            declared_gpus=1,
            gpu_capability="8.9",
            gpu_memory_gib=None,
        )


def test_runbook_gpu_validation_normalizes_config_contract_errors(tmp_path: Path) -> None:
    from dnadesign.infer import validate_runbook_gpu_resources

    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  id: evo2_7b
  device: cuda:0
  precision: bf16
  alphabet: dna
  parallelism:
    strategy: single_device
    gpu_ids: []
jobs: []
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="infer model contract invalid"):
        validate_runbook_gpu_resources(
            config_path=config,
            declared_gpus=1,
            gpu_capability=None,
            gpu_memory_gib=None,
        )
