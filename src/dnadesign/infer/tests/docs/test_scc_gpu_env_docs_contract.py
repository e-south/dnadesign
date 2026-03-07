"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/docs/test_scc_gpu_env_docs_contract.py

Docs contract checks for infer SCC Evo2 GPU environment runbook discoverability
and deterministic build controls.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_infer_scc_gpu_env_runbook_exists_and_covers_uv_stack_contract() -> None:
    doc = (
        _repo_root()
        / "src/dnadesign/infer/docs/operations/scc-evo2-gpu-uv-runbook.md"
    ).read_text(encoding="utf-8")

    assert "UV default groups" in doc
    assert "infer-evo2 extra" in doc
    assert "uv add" in doc
    assert "uv remove" in doc
    assert "FLASH_ATTENTION_FORCE_BUILD" in doc
    assert "FLASH_ATTN_CUDA_ARCHS" in doc
    assert "uv sync --locked --extra infer-evo2" in doc
    assert "flash-attn is sdist-only in `uv.lock`" in doc
    assert "MISSING_REQUIRED" in doc
    assert "raise SystemExit(1)" in doc
    assert "--reinstall-package flash-attn" in doc
    assert "--reinstall-package transformer-engine-torch" in doc
    assert "uv run infer adapters list" in doc
    assert "uv run infer validate config --config" in doc
    assert "ops runbook plan" in doc
    assert "### Capacity and build profile gate" in doc
    assert "TARGET_MODEL_ID" in doc
    assert "TARGET_PRECISION" in doc
    assert "RUN_CAPACITY_FAIL" in doc
    assert "RESOURCE_GATE_OK" in doc
    assert "FLASH_ATTN_CUDA_ARCHS" in doc
    assert "evo2_40b" in doc
    assert "400B model is out of scope" in doc
    assert "Quantized/offloaded 40B execution is not currently wired." in doc
