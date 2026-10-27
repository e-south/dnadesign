"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/portfolio/test_portfolio_workflow_hardening.py

Focused hardening checks for Portfolio workflow boundaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

import dnadesign.cruncher.app.portfolio_workflow as portfolio_workflow
from dnadesign.cruncher.portfolio.layout import (
    portfolio_logs_dir,
    portfolio_meta_dir,
    portfolio_plots_dir,
    portfolio_tables_dir,
)


def test_ensure_portfolio_run_dirs_preserves_unrelated_outputs_metadata(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    run_dir = workspace_root / "outputs" / "portfolio" / "demo_run"
    unrelated_run = workspace_root / "outputs" / "sample" / "older_run"
    unrelated_run.mkdir(parents=True, exist_ok=True)
    ds_store = unrelated_run / ".DS_Store"
    ds_store.write_text("finder\n")

    portfolio_workflow._ensure_portfolio_run_dirs(
        run_dir=run_dir,
        workspace_root=workspace_root,
        force_overwrite=False,
    )

    assert ds_store.exists()
    assert portfolio_meta_dir(run_dir).exists()
    assert portfolio_logs_dir(run_dir).exists()
    assert portfolio_tables_dir(run_dir).exists()
    assert portfolio_plots_dir(run_dir).exists()


def test_ensure_portfolio_run_dirs_rejects_existing_file_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    run_dir = workspace_root / "outputs" / "portfolio" / "demo_run"
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.write_text("blocked\n")

    with pytest.raises(ValueError, match="not a directory"):
        portfolio_workflow._ensure_portfolio_run_dirs(
            run_dir=run_dir,
            workspace_root=workspace_root,
            force_overwrite=False,
        )
