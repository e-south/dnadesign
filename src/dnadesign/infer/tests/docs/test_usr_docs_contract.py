"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/docs/test_usr_docs_contract.py

Infer docs contract tests for USR write-back naming.

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


def test_infer_docs_document_usr_column_name_contract() -> None:
    for relative_path in (
        "src/dnadesign/infer/README.md",
        "src/dnadesign/infer/AGENTS.md",
    ):
        text = (_repo_root() / relative_path).read_text(encoding="utf-8")
        assert "infer__<model_id>__<job_id>__<out_id>" in text
        assert "infer__<model_id>**<job_id>**<out_id>" not in text
