"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_pressure_runbook_docs_contract.py

Docs contract test for infer pressure-test runbook command coverage.

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


def test_pressure_runbook_docs_include_standalone_and_ops_paths() -> None:
    doc = (
        _repo_root()
        / "src/dnadesign/infer/docs/operations/pressure-test-agnostic-models.md"
    ).read_text(encoding="utf-8")

    assert "uv run infer validate config --config" in doc
    assert "uv run infer run --config" in doc
    assert "uv run ops runbook init" in doc
    assert "uv run ops runbook execute" in doc
    assert "--no-submit" in doc
    assert "--submit" in doc
    assert "--no-notify" in doc
    assert "--with-notify" in doc
    assert "outputs/logs/ops/audit/" in doc
    assert "infer__<model_id>__<job_id>__<out_id>" in doc
